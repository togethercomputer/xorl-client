# OPD Pause-Distillation — Complete Runbook (Qwen3.6-35B-A3B)

_Last updated: 2026-05-28. Supersedes `run_b_opd_2node_teacher.md` for end-to-end runs._

This is the single source of truth for running an OPD ("on-policy distillation")
pause-buffer training run on Qwen3.6-35B-A3B and **knowing whether it actually
worked**. It folds in everything validated on 2026-05-28: the 8× weight-sync
speedup, the windowed dataset, and — most importantly — the diagnosis that the
prior recipe **reward-hacked the loss and produced a degenerate model**, plus
the recipe change required to fix it.

---

## 0. TL;DR

- **Goal:** teach the student that `prompt + " pause"×100` (a content-free
  "thinking buffer") should yield the *same answer* the teacher produces from
  `prompt + real CoT`. Success = the pause-conditioned student approaches the
  teacher's CoT accuracy (Q3.6 4dmult 10-shot: base pause **63.6%**, CoT
  ceiling **91.3%**).
- **Infra is solved** (§4): warm weight sync **127s → 16s**, step **583s → 45s**,
  windowed disjoint dataset, all clean. Reuse it as-is.
- **The recipe is NOT solved** (§2–§3). The 2026-05-28 run (`opdb-condinval-052812`,
  loss 0.26→0.019) produced a model that **emits the pause prefill then
  immediately EOSes — 0% accuracy**. The converging loss was a reward-hack, not
  learning. **Do not trust loss convergence as a success signal.** §3 is the fix.
- **Always eval generation** (§7), in the *training* prompt format, before
  declaring success.

---

## 1. Objective & data flow

Teacher (frozen, real reasoning): `prompt + CoT → answer`, emits hidden states.
Student (trained, filler reasoning): `prompt + " pause"×100 → answer`.
OPD loss aligns the student's answer-position hidden states (conditioned on
pause) with the teacher's (conditioned on CoT), so the student learns to use the
pause tokens as a latent thinking buffer.

Prompt format actually used in training (verified from live student samples):
```
<|im_start|>user\n/no_think Calculate: {A} * {B}<|im_end|>\n<|im_start|>assistant\n<think>\n pause pause … (×100)
```
The student then generates `</think>` + the answer. **Any eval must reproduce
this exact prefix** (thinking-mode assistant turn, pause inside `<think>`), not
the filler-RL `/no_think … Answer:` format — see §7.

---

## 2. CRITICAL: why the 2026-05-28 run failed (read before re-running)

**Symptom:** loss converged 0.26 → 0.019 over 64 steps and checkpointed cleanly,
but the trained model scores **0%** (4dmult) in every eval setting:
- pause prompt (exact training format), temp 0 or 1.0 → **empty completion (EOS)**
- pause prompt + forced generation (`ignore_eos`, `min_new_tokens`) → **garbage
  loops** (`2051 * 7548. 2051 * 7548. …`)
- even a plain no-pause prompt now EOSes after ~15 tokens (generation broadly
  damaged)

**Root cause — loss reward-hack via early EOS:**
- The OPD loss distills teacher hidden states at the student's **self-sampled**
  answer positions (`target_tokens = _opd_causal_pair(student_sampled_sequence)`
  in `on_policy_distillation.py`).
- The `K` student-filler (pause) positions are **masked** (`target = -100`).
- The student's sampling params have **no `min_new_tokens` / no `ignore_eos`**,
  so the lowest-loss action is to emit `<eos>` immediately after the pause
  prefill → **zero answer positions → trivially near-zero KL**.
- Nothing in the loss forces the student to actually *produce* the teacher's
  answer tokens. It opts out, the KL collapses, and the model learns "after
  pause, stop" — which generalizes into premature-EOS everywhere.

**Detection rule:** a healthy run must show the student **generating answer
tokens** in its on-policy samples (watch the per-step sample log — see §6). If
late-step samples are still just `<think> pause pause … <eos>` with no digits,
the run is hacking the loss. Loss value alone is meaningless here.

---

## 3. The recipe fix — CONFIRMED ROOT CAUSE: prompt format (2026-05-28 PM)

**It was primarily a prompting bug, confirmed against the base model.** The OPD
training forced the student prefix to `<think>\n pause×100` — an open think
block full of pause with **no answer cue**. Probed on the **base** Qwen3.6-35B
(0-shot, 20 problems):

| Student prefix | base accuracy |
|---|---|
| `<think>\n pause×100` (what OPD trained) | **0/20** — base just echoes `pause`, never answers |
| `…pause×100` + `Answer: ` | 10–11/20 |
| `…pause×100</think>Answer: ` | **11/20** |
| `…pause×100\n</think>\n\nAnswer: ` | 11/20 |

So even the *base* model produces no answer in the trained format → the on-policy
student had nothing to imitate → it collapsed to EOS. **The fix is to close the
think block and add an explicit answer lead-in after the pause buffer.** Newlines
don't matter; the `Answer:` cue is what matters.

**Implemented fix (client-side only, no server change):**
- New `student_prefill_suffix` config in `on_policy_distillation.py`, set to
  `"</think>Answer: "`. The forced student prefix becomes
  `<think>\n pause×100</think>Answer: ` (the chat template auto-adds `<think>\n`).
  The masked filler region `K` is extended to the full prefix length (pause +
  suffix) — threaded through `_sample_student_batch` → `_teacher_hidden_cache_data`
  / `_opd_loss_data` as `k_filler`. Generator flag: `--student-prefill-suffix "</think>Answer: "`.
- Now the on-policy student emits real answer tokens (base ~55% at 0-shot), so
  the distillation has answer positions to align and can lift pause-conditioned
  accuracy toward the teacher's CoT ceiling (91%).

### 3.0 (superseded hypotheses — kept for context)

The student must be trained on **present, correct answer tokens**, not on a
self-sample it can truncate to empty. Two compatible levers — do **at least**
the first; the second is belt-and-suspenders:

### 3a. Teacher-force / ground-truth the answer target (primary fix)
Build the student training sequence as
`prompt + " pause"×100 + ANSWER`, where `ANSWER` is the **teacher's answer
tokens** (extract the post-`</think>` answer span from each
`randnum_4digit_*_cot_mt8192.json` entry) or the **ground-truth product**, not
the student's sampled continuation. Compute the OPD KL (and a CE/NLL term, see
3c) on those answer positions. This guarantees the answer positions exist and
are correct every step, eliminating the EOS opt-out.

Implementation sketch (`on_policy_distillation.py`, `_opd_loss_data` / `_opd_*`
construction): after the pause prefill boundary, append the teacher/GT answer
token ids and set them as `target_tokens` (unmasked), instead of slicing the
student's sampled tail. Keep the pause positions masked. This makes the run
**off-policy / SFT-style distillation** for the answer span — which is the
correct objective for "make pause→answer mimic CoT→answer".

### 3b. Floor the student generation (if keeping any on-policy sampling)
Add `min_new_tokens` (≈ expected answer length, e.g. 12–16 for a product) and
disable early `ignore_eos` during the student sampling call so on-policy
trajectories always contain answer positions. Note: on the *already-degenerate*
checkpoint this only yields garbage; the value is during fresh training, paired
with 3a, to prevent the collapse from forming.

### 3c. Add an answer-token CE term
Pure hidden-state/distribution KL doesn't directly pressure *generation*. Add a
cross-entropy term on the answer tokens (weight ~0.1–1.0) so the student learns
to emit them autoregressively, not just match hidden states under teacher
forcing. `opd_loss.py` already has the loss-mode plumbing
(`reverse_kl_full` default, `forward_kl_full`, `abs`, PG mode); the CE term is a
small addition alongside the chosen KL mode.

### 3d. Sanity gate before a long run
Run **2–3 steps, then immediately eval generation** (§7) on ~50 problems. If the
student isn't emitting digit answers after pause, stop and fix — do not let it
run to a checkpoint on a hacked loss again.

> Status: 3a is the recommended primary change and has **not yet been
> implemented/validated** — it is the open work item. 3b/3c are supporting.

---

## 4. Validated infrastructure (reuse as-is)

### Topology (8+4+8+1 = 21 pods, ~14 nodes, nccl pool)
- **Trainer:** 1 head + 7 workers (FSDP=64, EP=8) — `qwen3_6_35b_a3b_opd_opdb_8node.yaml`
- **Teacher:** 4-node (FSDP=32, EP=8), `sample_packing_sequence_len: 16384`,
  `--max-running-requests 16` — `qwen3_6_35b_a3b_teacher_4node.yaml`
- **Sampler:** 8 SGLang TP2 pods across 2 nodes (`--sampling-node nodeA,nodeB`),
  FP8 receiver checkpoint, DeepGEMM-disabled (`--fp8-gemm-backend triton
  --moe-runner-backend triton`)
- **Dispatch:** 1 CPU pod, round-robins sampling across the 8 SGLang pods

### Dataset (windowed, disjoint)
- Prompts: `/shared/opd-coord/randnum_4digit_16384_combined.json` (v1+v2, 16384,
  disjoint, index-aligned)
- Teacher CoT: `/shared/opd-coord/randnum_4digit_16384_combined_cot_mt8192.json`
  (mt=8192, avg 2728 tok/prompt, 1.4% truncation — use the mt8192 files, never
  the old `_cot.json` which amputated reasoning)
- `--prompts-per-step 128` → step N trains on a distinct 128-prompt window
  (steps 0–63 = v1, 64–127 = v2; wraps at 128). Needs the OPD-client
  `prompts_per_step` config (added 2026-05-28).

### Weight-sync fixes (the 8× — all validated, in PRs)
| Knob | Effect | Where |
|---|---|---|
| `XORL_P2P_FP8_QUANTIZE_DEVICE=gpu` (now the **default** when CUDA present) | CPU→GPU block-FP8 quant, **85s→10.7s/sync** | xorl-internal #330 |
| `XORL_WEIGHT_SYNC_BATCH_DENSE=1` | batch per-layer dense RDMA into one flush | xorl-internal #330 |
| conditional post-process cache-invalidation | warm-cache survives FP8 sync, **backend_init 34s→0.05s** | xorl-sglang-internal #43 |
| `XORL_P2P_CPU_POOL_MIN_BYTES=0` | Bug-7 small-entries stability | PR #309/#330 |

Result: warm sync **127s → 16s**; step **583s → 45s**.
**Footgun:** never skip the `/prepare_weights_update` call on warm syncs
(`XORL_P2P_SKIP_CACHED_PREPARE`) — it re-arms the receiver RDMA buffers and
skipping it → `ret=-1` × 50 retries → 735s failure.

### Required env (every trainer/sender pod)
```
XORL_P2P_CPU_POOL_MIN_BYTES=0
XORL_WEIGHT_SYNC_BATCH_MOE=1
XORL_WEIGHT_SYNC_BATCH_DENSE=1
XORL_WEIGHT_SYNC_DENSE_BUCKET_BYTES=134217728
XORL_WEIGHT_SYNC_MOE_BUCKET_BYTES=1073741824
XORL_P2P_CPU_SCRATCH_POOL_BYTES=2147483648
XORL_WEIGHT_SYNC_QUANTIZATION={"quant_method":"fp8","fmt":"e4m3","weight_block_size":[128,128]}
P2P_TRAINER_HOSTNAME=$(POD_IP)   XORL_WEIGHT_SYNC_MASTER_ADDRESS=$(POD_IP)
NCCL_SOCKET_IFNAME=bond0
# NOTE: trainer pods that init Mooncake must NOT set NCCL_IB_GID_INDEX / NCCL_IB_HCA
```
The generator injects all of these (see §5).

---

## 5. Launch (exact command)

Generate + apply via `experiments/opd_profile/k8s/generate_opd_manifest.py`
(self-contained; injects the env above, the GPU-quant + batch-dense knobs, the
teacher `--max-running-requests 16`, and the FP8 receiver flags):

```bash
python3 experiments/opd_profile/k8s/generate_opd_manifest.py \
  --run-name er-opd-<DATE> --num-trainer-nodes 8 --num-sglang-pods 8 \
  --sampling-node <nodeA>,<nodeB> --teacher-num-nodes 4 \
  --trainer-config experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node.yaml \
  --teacher-config  experiments/opd_profile/configs/qwen3_6_35b_a3b_teacher_4node.yaml \
  --wandb-run-name opd-<DATE> \
  --prompts-json-path      /shared/opd-coord/randnum_4digit_16384_combined.json \
  --teacher-cot-json-path  /shared/opd-coord/randnum_4digit_16384_combined_cot_mt8192.json \
  --num-prompts 16384 --prompts-per-step 128 --num-steps 64 \
  --student-prefill-text " pause" --student-prefill-count 100 \
  --opd-microbatch-size 128 --opd-prepare-batch-size 128 --opd-prepare-concurrency 8 \
  --profile-warmup-steps 1 --profile-sync-cuda \
  --save-every 50 --save-name-prefix opd-<DATE> \
  --sglang-model-path /shared/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/95a723d08a9490559dae23d0cff1d9466213d989 \
  --fp8-sync-quantization --instrument \
  --output experiments/opd_profile/k8s/generated/er-opd-<DATE>.yaml

kubectl -n apanda apply -f experiments/opd_profile/k8s/generated/er-opd-<DATE>.yaml
```
Pre-flight checks on the generated yaml: `XORL_WEIGHT_SYNC_QUANTIZATION` ×12,
`XORL_WEIGHT_SYNC_BATCH_DENSE` ×(sender pods), `num_prompts=16384
prompts_per_step=128`, no `XORL_P2P_SKIP_CACHED_PREPARE`.

Cluster hygiene: `backoffLimit≥8`; pin `CUDA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES`;
multi-node teacher must be on the `nccl` pool (default pool has no IB → NCCL
store-init deadlock). Cold bring-up ≈ 8 min (model load).

---

## 5b. In-loop eval (IMPLEMENTED 2026-05-28) — the success signal

`on_policy_distillation.py` now computes generation health + accuracy every step
from the student's own on-policy completions (free; this is the signal that
would have caught the original collapse) and logs decoded samples to wandb:
- `eval/empty_frac` (immediate-EOS rate), `eval/mean_completion_tokens`,
  `eval/has_think_close_frac`, `eval/has_digit_frac`
- `eval/accuracy` — operands parsed from the prompt (`eval_task=multiplication`),
  comma-robust product match against the completion
- `eval/samples` wandb.Table + per-step log lines with decoded completions
- **Auto-abort**: `eval_abort_patience>0` stops the run if `empty_frac ≥
  eval_abort_empty_frac` for N consecutive steps (kills a reward-hacking run
  fast). Generator flag: `--eval-abort-patience 3`.

A healthy run shows `empty_frac` near 0, `has_digit_frac` high, and
`eval/accuracy` ≥ base (~55% at 0-shot) and rising.

## 6. Monitoring (loss is NOT the signal)

Run dir: `experiments/encoded_reasoning/results/qwen3_30b_a3b_full_weight_real_reward/<run>/<ts>/`
- `opd_profile.jsonl` — per-step `sync_inference_weights_s` (~16s warm),
  `step_total_s` (~45s), `forward_backward_s`, `loss`.
- `trainer_job.log` — per-step `student sample (last N tokens)` line (the
  `OPD_LOG_SAMPLES=1` log added 2026-05-28). **This is the success gate**: late
  samples must contain `</think>` + digit answers. If they are
  `<think> pause … <eos>`, the run is hacking the loss → stop.
- `server.log` — `Timing breakdown` (backend_init should be ~0.05s warm),
  receiver `retaining warm P2P cache`.
- Set `httpx` to WARNING (`OPD_HTTPX_LOG_LEVEL` already defaulted) so samples
  aren't drowned by 200-OK spam.

Healthy warm step: `backend_init≈0.05s, transfer≈16s, fb≈9s, teacher≈10s`.

---

## 7. Eval methodology (the part that caught the failure)

**The filler-RL harness (`tomi`) default format does NOT match OPD training.** It
prefills `/no_think … Answer:`; OPD trains `<think>\n pause×100` (answer after
`</think>`). Evaluating in the wrong format gives a false 0% even on a good
model, and (as seen) the real model also fails — so use the training format and
distinguish the two.

### 7a. Quick direct probe (fastest signal, run after a few steps)
Hit a live SGLang pod's `/generate` with the **exact training prefix** and a
generous budget:
```python
SGIP=<sglang pod IP>   # kubectl -n apanda get pod <run>-sglang-0 -o jsonpath='{.status.podIP}'
prompt = f"<|im_start|>user\n/no_think Calculate: {a} * {b}<|im_end|>\n<|im_start|>assistant\n<think>\n" + " pause"*100
POST http://$SGIP:30060/generate  {"text":prompt,"sampling_params":{"temperature":0.0,"max_new_tokens":256}}
```
Expect `</think>` + the correct product. Empty/loops ⇒ the recipe is still
collapsing (§2/§3). Also probe `ignore_eos:true, min_new_tokens:64` to tell
"EOS-blocked-but-knows-answer" from "doesn't know answer".

### 7b. Full accuracy sweep (`tomi`, N=1000)
Framework at `/old-data/apanda/tomi`; venv `/old-data/apanda/tomi/.venv/bin/python`;
entry `examples/run_fresh_filler0_eval_passk.py`. Must set
`TOMI_USE_LITE_QWEN_RENDERER=1` (full renderer lacks `_render_message`),
`--tokenizer-model <Q3.6 bf16 snapshot>`, `--training-output-dir <any existing
filler-rl-multiplication_4digit-* config dir>`, `--inference-base-url
http://<sglang-pod-ip>:30060`, `--prompt-format training --api-mode generate`.
- Baseline: `--prefill-answer`
- Pause: `--prefill-filler-type pause --filler-tokens 100`
- `--problem-count 1000 --samples-per-problem 1 --max-in-flight 64`

**Caveat:** the harness's `--prefill-filler-type pause` uses the `/no_think …
Answer:` structure, NOT the `<think>` structure OPD trains. For a faithful OPD
eval, prefer 7a (exact format) or adapt the harness to emit the thinking-mode
pause prefix. Compare to the documented base numbers (4dmult 10-shot: baseline
59.4%, pause 63.6%, CoT 91.3%) from
`/old-data/apanda/tomi/outputs/multimodel_filler_and_cot_summary_20260528.md`.

### 7c. Success criteria
- **Working:** trained+pause accuracy materially **> base+pause (63.6%)** and
  trending toward CoT (91.3%); plain generation unharmed.
- **Failing:** trained+pause ≤ base, or empty/garbage completions, or plain
  generation degraded (premature EOS) — the §2 reward-hack.

---

## 8. Known footguns (all hit on 2026-05-28)

1. **Loss convergence ≠ success** — the headline lesson. Gate on generation (§6/§7).
2. `XORL_P2P_SKIP_CACHED_PREPARE` → 735s sync failure. Never use.
3. tomi eval format mismatch → false 0%. Use training format (§7a).
4. tomi renderer needs `TOMI_USE_LITE_QWEN_RENDERER=1`.
5. CPU FP8 quant default was an 8× sync footgun — fixed (now GPU default).
6. Teacher `max_running_requests` defaults to 2 → serializes prefill; set 16.
7. `XORL_WEIGHT_SYNC_QUANTIZATION` env must reach **all** sender pods (the
   generator post-pass handles this; verify count==12).

---

## 9. PRs / open items

- **xorl-internal #330** — GPU-quant default + `BATCH_DENSE` + README (merged-ready).
- **xorl-sglang-internal #43** — conditional post-process cache-invalidation
  (base: `feat/hybrid-shared-rl-stack`; that's where the P2P receiver code lives).
- **OPEN (recipe):** implement §3a (teacher-forced/GT answer target) + §3b/3c,
  then validate with §3d sanity gate. This is the work that makes OPD actually
  train a useful model. Until then, treat any "converged" OPD run as suspect
  until eval'd.

---

## 10. 2026-05-29 findings — the pause buffer is a NO-OP + scaling/infra

### 10.1 THE BIG RESULT: the pause buffer carries zero computation
Served the finished `opd-8x8-step50` checkpoint (trainer load-DCP → sglang →
p2p sync → probe) and ran with/without-pause ablations (greedy, held-out):

| arm | acc | |
|---|---|---|
| no_pause (0 pause, just `</think>Answer: ` cue) | **82.8%** | the control |
| pause50 / pause100 / pause200 / pause400 | 82.8 / 82.4 / 81.2 / 81.2 | **flat** |
| freegen (pause, NO answer cue, 320 tok) | **0.0%** | emits `"pause pause…"` to the cap |
| offdist 3-digit mult / 4-digit add | 99.2 / 100 | learned GENERAL arithmetic |

→ Removing the buffer changes nothing; the curve is flat 0→400; uncued it just
continues "pause". **OPD lifted intrinsic *direct* arithmetic (59.4%→82.8%
no-pause) and generalizes — a real result — but NOT via latent reasoning in the
pause tokens.** Encoded-reasoning is **falsified for this recipe**. The earlier
"49→77%" was real accuracy but entirely the direct-answer gain.

**Root cause (by construction):** the OPD loss MASKS the pause positions
(`_teacher_hidden_cache_data` insert mode: `mask_len = C + K_student` masks both
teacher CoT and student pause → cache keeps only prompt+answer hiddens;
`_opd_loss_data` masks the K student-pause positions). Only answer-position
hiddens are distilled → the buffer is never given a gradient to reason.

### 10.2 supervise_student_cot variant (implemented, gated, default off)
Client flag `supervise_student_cot=true` (needs `teacher_cot_mode=insert`):
sets `mask_len=C` (keep pause in teacher cache), student remap becomes identity,
no masking → KL on student-CoT(pause)+answer. **Diagnostic verdict (16 steps):**
`buffer_delta` rose from −0.115 (base: pause HURTS) → **~0** (oscillating ±0.02);
it removed the *distraction* but did NOT make the buffer load-bearing. logit-KL
on pause is too weak — the teacher's pause next-token target is trivially "pause".
→ For a real effect, need **hidden-STATE matching** (regress student pause hiddens
onto teacher post-CoT pause hiddens), or try supervising the no-CoT path. Watch
`buffer_delta` over the 122k run; if it stays ~0, switch loss.

**5-DIGIT (harder-task) RESULT (2026-05-29, er-opd5train-053002): buffer only
MARGINALLY load-bearing even where direct-answer is hard.** Moved to 5-digit×
5-digit (base direct ~6%, vs 78% on 4-digit) — the regime where the buffer
*should* have a job. Teacher scaled to 4 nodes (ep_fsdp=4) to clear the 5k-token
CoTs (2-node hung). Findings: (a) exact-match is uninformative on 5-digit
(10-digit products are close-but-not-exact → both arms ~0-15%, delta=noise) →
use a GRADED metric (leading-correct-digit fraction; `eval/buffer_lead_delta`
now in the client + `graded_buffer_probe.py`). (b) Model learns approximate
5-digit fast (lead-frac 0.40→0.53, rel-err 99.999%→0.001% by step 1). (c) GRADED
buffer verdict on the step-40 model (n=192): pause lead-frac 0.532 vs nopause
0.522 → **+0.011** (and exact +0.010) — a *marginal* ~+1pp benefit, within noise,
NOT a strong unlock. Caveat: only 40 steps / <1 epoch, model still climbing — a
longer run would show if +1pp grows. Bottom line so far: OPD pause-distillation
does NOT robustly produce buffer-encoded reasoning for multiplication on either
the easy (no-op) or hard (marginal) task at these training lengths. Next levers:
longer 5-digit run (graded metric live), hidden-state matching, more pause tokens.

**HEADLINE CONCLUSION (2026-05-29, after both supervision recipes): the buffer
won't be load-bearing on a task the model can solve DIRECTLY.** Tested logit-KL
supervise (100 steps, buffer_delta≈0, direct acc 0.78→0.96) AND hidden-state
matching (coef=0.5, 20 steps, buffer_delta≈0 AND direct acc 0.77→0.72 — strictly
worse). 4dmult is too easy: the trained model does it in ONE forward pass at 96%,
so there is no gradient pressure to use the ~100 pause tokens — the buffer stays
dead weight under any supervision. **For encoded reasoning to emerge the task
must sit where DIRECT (single-pass) answer FAILS but buffer-assisted SUCCEEDS**
(e.g. 6–8 digit mult / multi-step problems where base direct acc < ~40%). Next
experiments should move to a harder task, not keep tuning supervision on 4dmult.
The `opd_hidden_match_coef` + with/without-pause `buffer_delta` eval are the
reusable instruments for that. ↓ implementation details:

**Hidden-state matching IMPLEMENTED (2026-05-29, gated, default off).** Client
flag `opd_hidden_match_coef` (→ loss_param `opd_hidden_match_coef` → model_runner
→ `opd_loss_function(hidden_match_coef=)`). Adds `coef · (1 − cosine(student_h,
teacher_h[detached]))` at valid positions (= pause+answer under supervise mode),
emits `opd_hidden_match_loss`. Cosine = scale-invariant. Self-distill → dims
match. Run: `supervise_student_cot=true opd_hidden_match_coef=0.5` (tune by the
`opd_hidden_match_loss` vs `opd_kl` magnitudes). The 122k run (`opd-sup-big`)
confirmed logit-KL alone leaves `buffer_delta`≈0 through step 25 while direct acc
climbs — so hidden-matching is the next experiment.

### 10.3 In-loop with/without-pause control eval = the genuine-improvement judge
Ported from tomi `filler_tokens_rl.run_filler_eval`. Flags `eval_accuracy_every`
/ `eval_num_problems` → logs `eval/acc_pause`, `eval/acc_nopause`,
`eval/buffer_delta`. **This is the metric to gate on** — it would have caught the
no-op live. `buffer_delta>0` = buffer load-bearing; `~0` = OPD only improved
direct answering.

### 10.4 Teacher FSDP topology: ep_fsdp must divide hidden=7168
6-node teacher (`data_parallel_shard_size=48`, EP=8 → **ep_fsdp=6**) **crashes**
in `fully_shard(experts)` because 7168/6 isn't integer. Use **2/4/8** nodes
(ep_fsdp 2/4/8 all divide 7168=2^10·7); **6 and 7 do NOT**. `teacher_8node.yaml`
(ep_fsdp=8) is correct.

### 10.5 Empty-CoT filtering (trainer hard-aborts)
The trainer raises `teacher_cot_json_path entry N produced 0 tokens (empty cot)`
on ANY empty CoT. The 122,880 set had 4 (idx 53541/84494/96289/122837). Filter
**both** prompts+CoT JSONs at the same indices (they're index-aligned) →
`/shared/opd-coord/randnum_4digit_122876_filtered{,_cot_mt8192}.json`.

### 10.6 Weight-sync: 4-shards-on-1-node is the new footgun (NOT cpu-quant)
Big run sync = **33s** (bf16, 69GB, 2.1 GB/s) vs the 2-shard diagnostic **2.4s**
(SAME 69GB, 29 GB/s) → 14× slower purely from packing 4 sglang TP=2 shards on one
node (RDMA ingress contention; note `P2P_TRAINER_GPU_TO_IB_DEVICE_MAP` maps GPU
4&5 both → mlx5_9). **FP8 sync quant is NOT a drop-in fix (footgun, confirmed 2026-05-29):** passing
`--fp8-sync-quantization` with the generator-default **bf16** sglang
(`Qwen3.6-35B-A3B`) makes the trainer send FP8 → the bf16 receiver's tensor_map
**size-mismatches** (`expert gate_proj source=524288 vs receiver expects=1048576`)
→ sync crashes at step 0. FP8-on-wire only works when sglang serves the **FP8**
model (`Qwen3.6-35B-A3B-FP8`) — that's why opd-8x8 (FP8 sglang) synced FP8 fine
at 18s but a bf16-sglang run crashes. So **only pair `--fp8-sync-quantization`
with `--sglang-model-path <…-FP8>`**. The REAL low-sync lever is **shard
placement**: 2 shards (bf16) = 2.4s; **4-on-1-node (bf16) = 33s** (RDMA ingress
contention); spread shards across nodes for parallel sampling without contention.
Sampling is NOT the bottleneck (step is teacher-bound), so few shards is fine.

### 10.7 load_weights_mode=skip → SILENT gibberish without buffer reinit
`skip`+DCP does `to_empty()` which leaves non-persistent buffers (RoPE `inv_freq`)
as garbage; `dcp.load` never restores them → random RoPE → gibberish, NO warning
(loss 14 vs 12). PR #329 fixed `trainers/trainer.py`; **OPD uses the SERVER path
(`server/runner/checkpoint/manager.py`)** which had the same bug — ported the fix
there (pushed to #329, commit 709ffa9a). `all_ranks`/`grouped` (module_utils)
snapshot+restore buffer_dict and the from-scratch parallelize path calls
init_weights(), so only skip+DCP (Trainer+server) were affected. To use skip
(saves ~2min startup by skipping the redundant HF base load before the DCP
overwrite): flip config + **validate step-1 loss matches all_ranks** (silent bug).

### 10.8 Servable checkpoints: use `save_weights_for_sampler`, NOT save_full_weights_safetensors
The latter 404s on this server build. `TrainingClient.save_weights_for_sampler(name)`
returns a `model_path` directly usable as an sglang `--model-path`/SamplingClient
source (designed to be called every batch → fast). Client flag
`save_hf_safetensors=true` now fires it fire-and-forget (drained at end). DCP
`save_state` is resume-only (not sglang-loadable) → serving a DCP needs the
trainer-load+p2p-sync dance.

### 10.9 Engine startup ~9 min — breakdown + the lever
8-node startup: ~5 min multi-node rendezvous + JIT-heavy python imports + NCCL/IB
init (fixed tax, model-independent), then ~2 min base HF shard load + ~2 min DCP
overwrite. The model is loaded TWICE (HF base via all_ranks, then DCP overwrite);
`load_weights_mode=skip` (§10.7) skips the HF load.

### 10.10 generate_opd_manifest.py gaps (post-edits required every time)
The generator produces the **python dispatch** (not SMG) and lacks
`supervise_student_cot` / `eval_accuracy_every` / `eval_num_problems` /
`save_hf_safetensors` flags. After generating, post-edit: (a) **SMG swap** —
replace the dispatch pod command with the `smg` binary
(`/home/apanda/smg-together-thunderagent-port/target/debug/smg launch --policy
cache_aware --worker-urls …`) AND change the head's dispatch wait from
`/health` to a `/v1/models` grep (SMG doesn't serve `/health`); (b) inject the
chz flags after `teacher_cot_mode=insert`; (c) **drop `node-pool=compute`** from
nodeSelectors (it needlessly limits trainer-eligible nodes — keep `node-group=nccl`
for IB); (d) pass `--fp8-sync-quantization` to the generator (§10.6).

### 10.11 sglang-as-teacher (OPEN, the big lever) — FEASIBILITY: HIGH
Teacher prefill (~117s/step at 8 FSDP nodes, the 5800-char CoTs) dominates the
step (fwd-bwd is ~39s). The xorl teacher uses the TRAINING framework (FSDP fwd)
for an inference-only task. Replacing with an SGLang microservice is feasible:
- **SGLang already returns per-position hiddens.** `enable_return_hidden_states`
  + per-req `return_hidden_states` → `CaptureHiddenMode.FULL`
  (`model_executor/model_runner.py`) → `output_hidden_states` (all positions,
  not just last; `scheduler_output_processor_mixin` appends
  `logits_output.hidden_states[i]`).
- **The OPD cache is exactly that.** `teacher_hidden_cache` (model_runner ~1055)
  stores per-position last-layer hiddens (`hidden.reshape(-1,H)`, bf16
  safetensors); `opd_streaming_kl` makes teacher logits via
  `teacher_hidden_states @ t_weight.t()` (shared head applied in the loss). So
  the teacher just needs to emit those hiddens.
- **Representation match RESOLVED (traced xorl-sglang-internal, 2026-05-29): NO
  norm step needed.** `Qwen3MoeModel(Qwen2MoeModel)` applies the final `self.norm`
  inside `self.model`, so the hidden handed to the LogitsProcessor is
  POST-final-norm; `_get_hidden_states_to_store` FULL mode returns it directly
  (the `hidden_states_before_norm` override is EAGLE/spec-decode-only,
  `logits_processor.py:71`, and Qwen3MoE.forward never passes it). The xorl
  teacher cache stores the head input and `opd_streaming_kl` does a pure
  `teacher_hidden @ head_weight` matmul (no norm) → ALSO post-final-norm. So
  SGLang FULL hiddens == xorl cache representation; match directly.
- **Remaining challenges:** (1) confirm bf16/TP numerical equality via a quick
  A/B (cosine ~1.0) before trusting. (2) Transfer: returning full hiddens over HTTP JSON
  is impractical (~1 GB+/batch even for kept positions) → write a custom sglang
  endpoint that prefills and writes ONLY the kept positions (prompt+pause+answer,
  ~130 pos, NOT the ~2000-tok CoT) straight to the shared safetensors cache in
  the OPD format. Then swap `_teacher_cache_from_xorl` → `_teacher_cache_from_sglang`.
Tracked in task #27.

**BUILT + VALIDATED 2026-05-29 (another agent built it; validated here).** Endpoint:
`xorl-sglang-internal/.../entrypoints/teacher_hidden_cache.py` (`/teacher_hidden_cache`,
reuses `return_hidden_states` generate path, writes the safetensors cache in-process).
Client: `_teacher_cache_from_sglang` + A/B gate `examples/ab_teacher_cache.py` in the
`xorl-client-opd` worktree.
- **A/B numerical gate: PASS.** Same fixed token seqs through xorl teacher vs sglang
  teacher → cosine(kept-position hiddens) mean **0.998**, p01 0.992, 0.38% below 0.99
  (bf16/TP noise). Confirms post-final-norm representations match directly (no align step).
- **Teacher MUST launch with `--chunked-prefill-size 16384` (≥ max CoT len).** The
  endpoint reads only `chunks[0]` (the last forward chunk's hiddens). `chunked_prefill_size`
  DEFAULTS TO 2048, so any sequence >2048 tok returns only ~one chunk → early CoT
  positions missing → it raises "N kept positions served from the radix cache (cached
  prefix = X tokens)". **That message MISATTRIBUTES the cause — it is NOT radix.** Proof:
  the error persisted with `--disable-radix-cache`; setting `--chunked-prefill-size 16384`
  fixed it (8×6k and 8×11k seqs return ALL kept rows). Real 5-digit CoTs are ~8–12k tok,
  so the 2048 default would break every real teacher prefill.
- **Keep `--disable-radix-cache` ON.** Radix is fundamentally incompatible with this
  endpoint: it returns only FRESH hiddens, but radix skips recomputing matched prefixes,
  so any KEPT position radix serves from cache → no fresh hidden → the same `cached_kept`
  error. Distinct seqs (one epoch, prompts diverge at the digits) are fine, but OPD reuses
  fixed-CoT prompts across epochs → `prompt+CoT+pause` recurs verbatim → kept *pause*
  positions land in the cached prefix → error. (Realizing radix's epoch-2 win would need
  the endpoint to RETRIEVE cached hiddens, not just read fresh ones — future work.)
- **Throughput: ~3000–3400 tok/s on ONE TP=2 replica (2 GPUs)** vs the 4-node xorl
  teacher's ~690 tok/s wall on 32 GPUs (41k tok / ~60s/step) → **~5× wall-clock on 1/16
  the GPUs.** Scale-up = swap the 4-node xorl teacher for 1–2 TP=2 sglang replicas
  (`teacher_backend=sglang`): speeds the step's dominant phase AND frees ~28–30 GPUs.
- **Validated teacher launch:** `--tp-size 2 --dtype bfloat16 --enable-return-hidden-states
  --disable-radix-cache --chunked-prefill-size 16384 --mem-fraction-static 0.85
  --skip-server-warmup`; mount /shared + /home (writes cache to files, NO Mooncake/IB).
- **k8s gotcha:** the apanda namespace injects `nodeSelector: node-group=default`; to pin a
  teacher to an nccl node set `nodeSelector: {node-group: nccl}` + `tolerations:[{operator:
  Exists}]` explicitly, else the pod fails the NodeAffinity predicate. Use a node with REAL
  free GPUs (req=0 across ALL namespaces, not just "0 apanda pods") so GPUs 0,1 don't collide.
SCALE-UP STATUS: validated at small scale; gated on free nodes (nccl pool tenant-saturated)
— apply to the NEXT OPD run (don't disrupt the live er-opd5big science run).

### 10.12 Rapid-relaunch footguns (hit twice 2026-05-29)
Tearing down a run and immediately re-applying on the same nodes hits two races:
- **ZMQ `Address already in use (tcp://127.0.0.1:54080)`** — the trainer engine's
  orchestrator output port is deterministic; a leftover process from the prior
  run on a reused node still holds it. The trainer-head Errors with "process
  exited before engine became ready." FIX: after teardown, wait for 0 pods THEN
  settle ~60s before re-applying (let the node release the port/process).
- **`UnexpectedAdmissionError` (Allocate failed, devices unavailable)** — sglang
  pods pinned to a `--sampling-node` that another tenant grabbed in the
  teardown→relaunch window. FIX: pick a currently-free `--sampling-node`; the
  cluster churns fast (other-namespace tenants), so re-check free nodes right
  before launch.

### 10.13 LR sweep on 5-digit (2026-05-29): 3e-5 COLLAPSES, 1e-5 is the rate
Swept learning_rate on the 5-digit supervise run (mt16384 CoTs, same topology):
- **1e-5**: stable. lead_pause climbs (~0.40 → 0.53 over 40 steps). The proven rate.
- **3e-5**: **on-policy generation COLLAPSE by step 10.** KL/loss drops FAST
  (0.30 → 0.13) but the student's sampled generations degrade into runaway digit
  garbage: e.g. `31275 * 74501 -> 233006051586715551512248780020580027638` (39
  digits, true is 10), `mean_completion_tokens` shoots to the 192 cap, train_acc=0,
  graded `lead_pause` DROPS (0.45→0.36). The too-fast update pushes the student off
  the answer-then-stop manifold → bad self-samples → distills from garbage → spiral.
  Loss going DOWN while generation breaks is the signature — never trust loss alone
  (same lesson as [[project-opd-reward-hack-eos-collapse]], different mechanism).
- **2e-5**: untested (no node room for a parallel arm; nccl pool was saturated).
VERDICT for unattended runs: use **1e-5** (stable, guaranteed progress). 2e-5 is the
only remaining sweet-spot candidate worth a *monitored* daytime test; do NOT exceed
1e-5 on an overnight run. Catch a collapse early via `mean_completion_tokens` (healthy
~11, collapse → 192 cap) + graded `lead_pause` trend, not loss.

### 10.14 sglang OOM = node co-location, not just CUDA_VISIBLE_DEVICES=0,1
The generator pins EVERY pod to an explicit `nodeName` (from `--sampling-node` etc.),
bypassing the scheduler's GPU accounting. Two failure modes seen 2026-05-29 cloning a
working manifest:
- **Teacher/sglang co-location**: the clone put `teacher-worker-1` + `dispatch` + BOTH
  `sglang-0/1` on the same node. The teacher (FSDP, 8 GPUs/node) grabs all 8; sglang
  (CUDA_VISIBLE_DEVICES=0,1, mem-fraction 0.8) then OOMs at MoE weight alloc and exits
  0 → STATUS=**Completed** (looks like a clean finish, isn't). The original run only
  survived by winning the GPU-init race.
- **`UnexpectedAdmissionError`**: repinning sglang to a node with "0 apanda pods" still
  fails — those nodes are full of OTHER-namespace tenants. `nodeName` bypasses the
  scheduler so the kubelet device-plugin rejects the 2-GPU request at admission.
FIX (the reliable check): compute REAL free GPUs per node =
`allocatable.nvidia.com/gpu` − (sum of `nvidia.com/gpu` requests across ALL namespaces,
from `kubectl describe node ... | grep nvidia.com/gpu` in the Allocated-resources block).
Pin the 2 sglang pods to DISTINCT nodes with req=0 (truly free). "0 apanda pods" is NOT
"free". The 35B model needs TP=2 (won't fit 1 GPU: 70 GB bf16 weights > 0.8×80 GB), so
you need 2 free GPUs per replica on a node where physical GPUs 0,1 are free (the
`--mooncake-ib-device {"0":"mlx5_2","1":"mlx5_3"}` map is keyed to physical 0,1).
