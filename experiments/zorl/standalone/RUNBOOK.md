# ZORL Runbook

Standalone zeroth-order RL on top of SGLang. No trainer process — all
state (parent LoRA, perturbation seeds, ES update, snapshots) lives on the
SGLang server. The Python client is a thin orchestrator that drives the
loop over HTTP.

This is a runbook for the entire setup: what to deploy, what flags matter,
how to write task plugins, what's known to break, what hyperparameters
actually work, and how to recover when things go wrong.

---

## 1. Concept

**ZORL = Zeroth-Order RL.** Instead of computing gradients with respect to
LoRA weights via backprop, we estimate them with finite differences on the
LoRA-B matrix. Each step:

1. Sample `N` antithetic pairs of perturbations `ε_i, -ε_i` from
   `Normal(0, σ²)`.
2. Build `2N` candidate LoRAs = `parent + σ·ε_i` and `parent - σ·ε_i`.
3. Roll out each candidate on the train prompts; score with a verifiable
   or learned reward.
4. The ES gradient estimate is `(1 / 2Nσ) · Σ (r_i⁺ − r_i⁻) · ε_i`.
5. Apply `parent ← parent + lr · grad`.

The candidates differ from the parent only by a Gaussian B-perturbation
(`A` is kaiming-init at start, kept identical across candidates). This is
"B-only ES" — much cheaper than perturbing both `A` and `B`.

Compared to GRPO: ZORL doesn't need teacher logprobs, doesn't need
backprop, and scales naturally with the number of inference workers
(every candidate is just another `/generate` call). It does need a much
larger population than GRPO to estimate the gradient.

---

## 2. Architecture

```
┌─────────────────────────┐                    ┌──────────────────────────┐
│  zorl_client.py         │   HTTP             │  SGLang (xorl fork)      │
│  (single Python script) │  ───────────────►  │                          │
│                         │                    │  /load_lora_adapter      │
│  - load task plugin     │                    │  /start_zorl_session     │
│  - cold probe           │                    │  /start_zorl_generation  │
│  - for step:            │                    │  /apply_zorl_rewards     │
│      gen candidates     │                    │  /snapshot_zorl_parent   │
│      score on train     │                    │  /restore_zorl_parent    │
│      apply ES update    │                    │  /abort_zorl_generation  │
│      probe on eval      │                    │  /generate               │
└─────────────────────────┘                    │  /flush_cache            │
                                               └──────────────────────────┘

      ┌──────────────────────────────────────────────────────────┐
      │  init_zorl_adapter.py — one-shot before the run          │
      │  Produces adapter_model.safetensors with zero-B,         │
      │  kaiming-A. Same shapes the trainer would output.        │
      └──────────────────────────────────────────────────────────┘
```

The client does **not** talk to a trainer. It only talks to SGLang. The
"init adapter" is a static on-disk artifact produced once; SGLang loads
it as the starting parent for the ZORL session.

---

## 3. Quick start

```bash
# 1. Generate the starting adapter (one-time, ~30 s for 30B-A3B; 5 min for 235B).
python experiments/zorl/standalone/init_zorl_adapter.py \
    --model /shared/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/<sha> \
    --output-dir /shared/zorl/init-adapters/qwen3-30b-a3b-r16 \
    --rank 16 --alpha 16

# 2. Deploy SGLang with LoRA enabled (see model-specific recipes below).
kubectl apply -f experiments/zorl/k8s/generated/zorl-<exp>-sglang.yaml

# 3. Wait for "The server is fired up" in the SGLang pod logs.
kubectl logs -n apanda <pod-name> -f | grep "fired up"

# 4. Fire the run. NOTE: PYTHONUNBUFFERED + python -u for nohup, else
#    the log stays empty for hours.
SGLANG_IP=$(kubectl get pod -n apanda <pod-name> -o jsonpath='{.status.podIP}')
TS=$(date +%s)
PYTHONUNBUFFERED=1 nohup python -u -m experiments.zorl.standalone.zorl_client \
    --task wordle \
    --infer-url http://${SGLANG_IP}:30060 \
    --adapter-dir /shared/zorl/init-adapters/qwen3-30b-a3b-r16 \
    --model /shared/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/<sha> \
    --parent-lora-name "zorl-${TS}/parent" \
    --session-id "zorl-${TS}" \
    --steps 25 --num-pairs 16 \
    --b-sigma 0.012 --lr 0.01 \
    --max-update-norm 20000 \
    --train-size 16 --eval-size 32 \
    --rollout-temperature 0.6 --rollout-max-new-tokens 48 \
    --probe-temperature 0.6 --probe-n 16 \
    --score-max-workers 32 \
    --probe-interval 1 \
    > /shared/zorl/results/zorl-${TS}.log 2>&1 &
```

---

## 4. Components

### 4.1 The init adapter — `init_zorl_adapter.py`

Builds an `adapter_model.safetensors` + `adapter_config.json` pair in the
**xorl `sglang_shared_outer` 3D layout** for MoE models (PEFT 2D layout
for dense models). Init convention:
- `A` is kaiming-uniform (non-zero) — same as `torch.nn.Linear` default.
- `B` is all zeros.

So `A·B = 0` at start: the parent contributes nothing to the base forward
pass on step 0. ZORL's B-only perturbations produce non-zero candidate
deltas because `A` is non-zero.

**MoE layout (per layer)**:

| module | sglang name | A shape (shared dim first) | B shape |
|---|---|---|---|
| `gate_proj` | `w1` | `[1, r, hidden]` shared | `[E, moe_int, r]` per-expert |
| `up_proj`   | `w3` | `[1, r, hidden]` shared | `[E, moe_int, r]` per-expert |
| `down_proj` | `w2` | `[E, r, moe_int]` per-expert | `[1, hidden, r]` shared |

Attention always uses standard 2D PEFT shapes: `A: [r, in]`, `B: [out, r]`.

**Flags**:
- `--rank` / `--alpha`: LoRA rank/scale. We've been using 16 throughout.
- `--target-modules`: list, e.g. `qkv_proj o_proj gate_proj up_proj down_proj`.
  **The generator respects this filter** — pass attention modules only to
  emit a smaller adapter (fix landed during the 235B-FP8 work).
- `--fused-qkv`: auto-detected. Qwen3 family is fused.
- `--dtype`: bf16/fp16/fp32. Default bf16. Use bf16 even for fp8 base
  models — LoRA tensors stay bf16.

### 4.2 The SGLang stack

Use the `xorl-sglang-internal` fork (commit on `main` as of 2026-06-02
has all 7 ZORL endpoints). Key launch flags:

```
--enable-lora                                # turn on LoRA path
--lora-backend triton                        # only backend ZORL is tested with
--max-lora-rank 16                           # must >= adapter rank
--max-loras-per-batch N                      # batch size of LoRAs SGLang serves concurrently
--max-loaded-loras N                         # total LoRA slots in pool; must >= parent + 2*num_pairs + snapshot buffer
--lora-target-modules qkv_proj o_proj gate_proj up_proj down_proj
--lora-moe-format hybrid_shared              # MoE LoRA in 3D shared_outer layout
--experts-shared-outer-loras                 # routes through the shared-outer kernel
--disable-radix-cache                        # ZORL invalidates KV between gens
--disable-overlap-schedule                   # required: ZORL endpoints can't tolerate scheduler races
--disable-cuda-graph                         # piecewise-graph + LoRA = unstable
--disable-custom-all-reduce                  # safer with multi-LoRA
--attention-backend fa3
--sampling-backend flashinfer
--skip-server-warmup
```

The MoE-specific flags (`--lora-moe-format hybrid_shared` +
`--experts-shared-outer-loras`) are only valid when the adapter targets MoE
modules. For **attention-only LoRA** (no MoE targets) drop both flags.

### 4.3 The task plugin (`tasks/<name>.py`)

Duck-typed module with three required pieces and one optional. See
`tasks/base.py` for the full contract.

```python
is_multi_turn = False  # or True for Wordle-style games

def build_examples(tokenizer, *, train_size, eval_size, seed) -> (list[Example], list[Example]):
    """Return train + held-out eval. Each Example has:
       - project: stable id (used in per-project logs)
       - prompt_ids: chat-template-rendered input_ids
       - metadata: task-specific (gold answers, etc.) — client never inspects
    """

def score_completion(example, generated_text) -> dict[str, float]:
    """Returns at minimum {"reward", "exact_match"}. Extra keys (format_rate,
    info_gain, ...) flow through to per-step logs unchanged."""

# Optional, only when is_multi_turn=True:
def rollout_completion(example, *, generate_turn, lora_path, tokenizer, args) -> dict[str, float]:
    """Play the full multi-turn rollout. generate_turn(input_ids, lora_path=, ...) -> str
    is provided by the client; call it once per turn, splice the feedback into the chat
    history, and return the same score blob shape as score_completion."""
```

Existing tasks:
- `countdown.py` — single-shot, 32-puzzle pool with rotation
- `gsm8k.py` — single-shot, math word problems with `#### N` answer marker
- `alphabet_sort.py` — single-shot, sort names with similarity^8 reward
- `wordle.py` — **multi-turn** (6 guesses with G/Y/B feedback)
- `opd_multiplication.py` — single-shot OPD pause-filler 3/4-digit mult

Register new tasks in `tasks/base.py`'s `load_task(name)`.

### 4.4 The client (`zorl_client.py`)

Single file, ~420 lines. Key call sequence per run:

```
1. load_lora_adapter(parent_lora_name, adapter_dir)         # register init parent
2. start_zorl_session(session_id, parent_lora_name, num_pairs, b_sigma, seed)
3. cold probe                                               # baseline
4. for step in steps:
     a. start_zorl_generation(session_id)                   # → 2*num_pairs candidates
     b. score_candidates() over train set                   # parallel /generate
     c. apply_zorl_rewards(rewards, lr, max_update_norm)    # ES update, parent ← parent + lr*grad
     d. probe_parent() on eval set                          # optional, gated by --probe-interval
     e. snapshot/restore on elitist-rollback                # optional
```

**Multi-turn dispatch** (`is_multi_turn = True`): `score_candidates` and
`probe_parent` branch on `task.is_multi_turn`. When True, they hand
`task.rollout_completion(...)` a `generate_turn(...)` callable and let
the task own the per-turn loop. Tokenizer is passed through so the task
can re-render the chat history each turn.

---

## 5. Configuration reference

All flags accepted by `zorl_client.py`:

| flag | default | what it does |
|---|---|---|
| `--task` | (req) | one of `countdown|gsm8k|alphabet_sort|wordle|opd_multiplication` |
| `--infer-url` | (req) | SGLang URL, e.g. `http://<pod-ip>:30060` |
| `--adapter-dir` | (req) | local path with `adapter_model.safetensors` + `adapter_config.json` |
| `--model` | (req) | HF path for `AutoTokenizer.from_pretrained` |
| `--parent-lora-name` | (req) | LoRA name to register parent under — **unique per run** |
| `--session-id` | (req) | ZORL session id — **unique per run** |
| `--snapshot-id` | `best` | name for elitist-rollback snapshot |
| `--steps` | 50 | ZORL outer-loop steps |
| `--num-pairs` | 32 | antithetic pairs; population_size = `2*num_pairs` |
| `--b-sigma` | 0.012 | LoRA-B perturbation std |
| `--lr` | 0.01 | ES learning rate |
| `--max-update-norm` | 3500.0 | L2 cap on the update direction |
| `--seed` | 1234 | RNG seed for session |
| `--rollouts-per-puzzle` | 1 | `n` per `/generate` call in scoring |
| `--rollout-temperature` | 0.6 | sampling temp during scoring |
| `--rollout-max-new-tokens` | 64 | per-turn max tokens during scoring |
| `--score-max-workers` | 32 | thread-pool size for the candidate × example grid |
| `--train-size` | 8 | prompts in train set |
| `--eval-size` | 128 | prompts in held-out eval |
| `--probe-interval` | 5 | probe every N steps; 0 disables |
| `--probe-temperature` | 0.0 | probe temp (0 = greedy) |
| `--probe-n` | 1 | rollouts per prompt during probe |
| `--elitist-rollback` | off | restore parent if probe drops below best |

**Hyperparam recipe that works** (Q3-30B-A3B-Instruct, 3-digit OPD-mult,
Q3-235B-A22B-FP8 wordle, both validated): `--b-sigma 0.012 --lr 0.01
--max-update-norm 20000 --num-pairs 16`. Don't use `--max-update-norm
3500` — observed to clip the natural ES step on these tasks.

---

## 6. Model-specific recipes

What we've actually run + the gotchas per model.

### 6.1 Qwen3-30B-A3B-Instruct-2507 (BF16)

- **Path**: `/shared/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`
- **Topology**: single-node, TP=2, 2× H100
- **Init adapter**: `/shared/zorl/init-adapters/qwen3-30b-a3b-r16`
  (full MoE LoRA, 480 tensors)
- **SGLang flags**: standard set above, `--max-loaded-loras 96`,
  `--max-loras-per-batch 64`, `--mem-fraction-static 0.85`
- **Status**: ✅ all five tasks run cleanly. Baseline for everything.

### 6.2 Qwen3-235B-A22B-Instruct-2507-FP8 (pre-quantized)

- **Path**: `/shared/research-coder-data/model_cache/transformers/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/c0eb82898e3da8fb6dd017e3e6698a5e37b3a3e6`
- **Topology**: single-node, **TP=4** (not 8 — see below), 4 of 8 H100s
- **Init adapter**: `/shared/zorl/init-adapters/qwen3-235b-a22b-fp8-attn-r16`
  (**attention-only**, 376 tensors)
- **SGLang flags**: standard set, **drop** `--lora-moe-format hybrid_shared`
  and `--experts-shared-outer-loras` (no MoE LoRA), `--lora-target-modules
  qkv_proj o_proj`, `--max-loaded-loras 48`, `--mem-fraction-static 0.85`
- **Status**: ✅ wordle 15 steps in 29 min, 1/16 exact at 4 checkpoints
  vs cold 0/16 (see [[zorl-wordle-235b]]).

**Two non-obvious constraints**:

1. **TP=4 not TP=8.** The pre-quant FP8 model uses block_size `[128, 128]`.
   At TP=8: `moe_intermediate=1536 / 8 = 192`, **not divisible by 128** →
   `ValueError: gate's and up's weight=192 is not divisible by block_n=128`.
   TP=4: `1536/4=384` ✓. TP=2: `1536/2=768` ✓.
2. **Attention-only LoRA.** The FP8 MoE GEMM kernel in SGLang doesn't
   compose with LoRA-Triton on MoE modules — Triton compilation fails:
   `Unsupported rhs dtype fp8e4nv`. Restrict adapter to `qkv_proj o_proj`
   and drop the MoE LoRA flags.

**Storage**: `/shared/research-coder-data/` is ~10× slower than
`/shared/huggingface/hub/` (~3 min/safetensors shard, ~70 min total load
for 235B FP8 first time). Copy the model to `/shared/huggingface/hub/`
once if you'll iterate.

### 6.3 Qwen3-235B-A22B-Instruct-2507 (BF16) — what we know

- **Path**: `/shared/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/snapshots/ac9c66cc9b46af7306746a9250f23d47083d689e`
- **Single-node TP=8 fits weights** (~59 GB/GPU after BF16 shard) but **the
  combination of full MoE LoRA pool + KV cache exceeds available memory**.
  Two failure modes:
  - `mem-fraction-static=0.85` → `OutOfMemoryError` at LoRA pool init
    (Pytorch alloc 76 GB / 79 GB total).
  - `mem-fraction-static=0.78` → "Not enough memory; please try to
    increase mem-fraction-static" (KV pool can't fit under static budget).
- **Workaround**: multi-node TP=16 (2 nodes × 8 H100s) — weights drop to
  29 GB/GPU, plenty of room for KV+LoRA. **Untested end-to-end** as of
  2026-06-02; yaml lives at `experiments/zorl/k8s/generated/zorl-wordle-235b-tp16.yaml`.

Multi-node TP requires these NCCL env vars on the SGLang pods:

```yaml
- name: NCCL_IB_GID_INDEX
  value: "0"
- name: NCCL_IB_HCA
  value: "^mlx5_4,mlx5_7,mlx5_8"   # exclude known-bad HCAs on apanda cluster
- name: NCCL_SOCKET_IFNAME
  value: "bond0"
```

**Do NOT set these on pods that init Mooncake P2P** (e.g., OPD trainer
pods). They force the user-specified GID path that fails when IB HCAs
have no IPoIB netdev. For ZORL inference SGLang doesn't use Mooncake, so
they're safe.

### 6.4 Models tested for adapter generation but not run

- Qwen3-0.6B-Base (dense) — GSM8K ZORL validated: 9.4% → 45.3% in 30 steps
- Qwen3-4B-Instruct-2507 (dense) — alphabet sort validated
- Qwen3-Coder-30B-A3B-Instruct (MoE) — Countdown validated

---

## 7. Operational gotchas

### Always-on

- **`PYTHONUNBUFFERED=1` + `python -u` for nohup'd runs.** Without this
  stdout buffers indefinitely and the log file stays empty for hours.
  Smoke tests in foreground TTY work fine; the bug only bites nohup.
- **Privileged k8s pods + nvidia-container-toolkit need explicit
  `CUDA_VISIBLE_DEVICES`.** The runtime exposes all 8 host GPUs in
  `nvidia-smi`; if `CUDA_VISIBLE_DEVICES` is unset, PyTorch picks 0/1
  (often another tenant's GPUs). Set it explicitly in the pod env.
- **Use `nodeSelector` not `nodeName`** for hostname binding. `nodeName`
  bypasses the scheduler and gets `NodeAffinity` failures from kubelet
  when the node is contended.
- **Parallelize the probe.** `probe_parent` uses `score_max_workers`.
  Sequential probing on 30B is 80+ min for a 128-prompt eval. Parallel
  drops it to <5 min.

### ZORL-specific

- **Always sample-N on probe (`--probe-temperature 0.6 --probe-n 16`)**
  for small-LoRA tasks. Greedy probe (`temp=0 n=1`) is sticky — small
  LoRA deltas redistribute probability mass without flipping argmaxes,
  so greedy locks at cold value while sampled-N actually moves. Verified
  on OPD-mult.
- **Elitist rollback can lock the run.** If the probe ties cold exactly
  (which it does whenever the LoRA delta hasn't crossed the argmax
  threshold) and rollback is on, it restores the parent every step → no
  net movement. Either turn rollback off, or use sampled probe so ties
  are vanishingly rare.
- **`max_update_norm=3500` (the default) clips too aggressively** on
  most setups we've tested. Natural ES update norm at `num_pairs=8-16,
  sigma=0.012` lands ~3000-6000 depending on the model and target
  modules. Use `--max-update-norm 20000` to let the natural scale apply.
- **`max_loaded_loras` must fit `1 parent + 2*num_pairs candidates +
  1 snapshot + buffer`.** Otherwise SGLang evicts mid-step and the next
  `/start_zorl_generation` thrashes.

### SGLang-side

- **Don't use FP8 quantization + LoRA-Triton on MoE modules.** Triton
  fails to compile with `Unsupported rhs dtype fp8e4nv`. Either: use
  BF16, or restrict LoRA to attention-only on FP8 models (see §6.2).
- **`--disable-radix-cache`, `--disable-overlap-schedule`,
  `--disable-cuda-graph`, `--disable-custom-all-reduce`** are all required.
  Without them ZORL hits various scheduler / cache races.
- **Pre-quantized FP8 model TP must keep block dims divisible.** Block
  size 128 + module dim X → require X / TP % 128 == 0. For 235B,
  `moe_intermediate=1536`, so TP ∈ {1, 2, 4, 12} (skip 8, 16, ...).

### Storage / k8s

- `/shared/huggingface/hub/` is the **fast PVC**.
  `/shared/research-coder-data/` is ~10× slower.
- Cluster GPU availability is unpredictable — nodes that show free at
  `kubectl get nodes` can be taken in the seconds between query and
  apply. For multi-node setups use `nodeSelector` + `podAntiAffinity` or
  schedule serially.
- **Memory note `[NCCL IB env vars on apanda cluster]`** says these vars
  are required on every pod, but the exception `[NCCL_IB vars break
  Mooncake P2P init]` says strip them on trainer pods that init Mooncake.
  ZORL inference doesn't use Mooncake — apply the IB vars freely.

---

## 8. Reading the logs

Per-step line emitted by the client:

```
step N/STEPS: reward_mean=X best_cand=X update_norm=X used_pairs=N
              t_score=Xs t_apply=Xs, probe_reward=X exact_rate=X
              [task-specific metrics]
```

- `reward_mean` — average train reward across all candidates this step.
- `best_cand` — the best single candidate's train reward this step.
  This is the **ceiling** if you could just pick the best candidate as
  the new parent. ZORL averages instead.
- `update_norm` — L2 norm of the applied update direction. Should be
  stable across steps (gradient estimator variance is dominated by
  noise geometry, not loss landscape). If it's bouncing wildly, sigma
  is probably wrong.
- `probe_reward` — eval reward (uses `--probe-temperature` /
  `--probe-n`).
- `exact_rate` — eval `exact_match` rate.

**What "signs of life" looks like**:
1. `reward_mean` climbs monotonically (with noise) across steps.
2. `best_cand` consistently above cold.
3. `probe_reward` lifts above cold over time (with sampled probe).
4. Task-specific metrics (`format_rate`, `info_gain`, `exact_rate`)
   trend in the right direction.

**What "broken" looks like**:
- `probe_reward` ≡ cold for all steps + greedy probe → run might be
  fine; switch to sampled probe to see (verified on OPD-mult: greedy
  showed locked, sampled showed +24%).
- `reward_mean` < cold consistently → something wrong with adapter
  setup, perturbation seeding, or apply path. Check that the adapter
  was registered (look for `/load_lora_adapter` line) and the session
  opened cleanly.
- `update_norm = max_update_norm` exactly → cap is clipping. Raise the
  cap to let the natural scale apply.

---

## 9. Known-good hyperparameters (per validated setup)

| Setup | num_pairs | b_sigma | lr | max_update_norm | probe |
|---|---|---|---|---|---|
| Q3-30B GSM8K | 16 | 0.012 | 0.01 | 20000 | greedy |
| Q3-30B OPD-mult 3-digit | 16 | 0.012 | 0.01 | 20000 | t=0.6 n=16 |
| Q3-30B Countdown 32-puzzle | 16 | 0.012 | 0.01 | 20000 | greedy |
| Q3-235B-FP8 Wordle | 8 | 0.012 | 0.01 | 20000 | greedy |
| Q3-4B Alphabet Sort | 16 | 0.012 | 0.01 | 20000 | greedy |

**Pattern**: `sigma=0.012, lr=0.01, max_update_norm=20000` works
everywhere we've tried. Only knob that varies is `num_pairs` (drop to 8
on 235B for cost, push to 16-32 for smaller models).

The researcher's blog setting was `sigma=0.012, lr=0.01, population=64`
(= `num_pairs=32`). Our prior `sigma=0.1` was 8× too high; that's the
likely cause of the Countdown 5/32 plateau ([[zorl-5of32-plateau]]).

---

## 10. Files

- `experiments/zorl/standalone/zorl_client.py` — client orchestrator
- `experiments/zorl/standalone/init_zorl_adapter.py` — adapter generator
- `experiments/zorl/standalone/tasks/base.py` — task protocol + dispatcher
- `experiments/zorl/standalone/tasks/<name>.py` — per-task plugins
- `experiments/zorl/k8s/generated/zorl-*-sglang.yaml` — SGLang pod yamls
- `/shared/zorl/init-adapters/<model>-r<rank>/` — generated init adapters
- `/shared/zorl/results/*.log` — per-run logs

External:
- `xorl-sglang-internal` on `main` (>= commit `8bcf2653e`) has all 7
  ZORL endpoints.

---

## 11. References (memory entries)

The lessons here are distilled from:

- `[zorl-standalone-e1-gsm8k]` — 9.4% → 45.3% on Q3-0.6B-Base; sanity check
  that the architecture works at all.
- `[zorl-standalone-e3-alphabet-sort]` — train climbs 3×, eval flat; sharp
  similarity^8 reward + 16-prompt train is overfit-prone.
- `[zorl-opd-mult-signs-of-life]` — train robust 0.58→0.74; greedy probe
  misleads (use sampled-N). 3-digit is the sweet spot.
- `[zorl-wordle-235b]` — wordle on 235B-FP8 TP=4 + attn-only LoRA: 1/16
  exact at 4 ckpts vs 0/16 cold. Step-15 format regressed.
- `[zorl-5of32-plateau]` — old Countdown runs with `sigma=0.1` plateaued at
  5/32. Hyperparams were wrong.
- `[zorl-countdown]` — first reproducible parent improvement on
  30B-A3B MoE LoRA with verifiable reward.
- `[feedback-set-cuda-visible-devices]` — privileged k8s pod gotcha.
- `[feedback-nccl-ib-env-vars]` + `[feedback-nccl-ib-breaks-mooncake-init]`
  — when to set IB env vars.
