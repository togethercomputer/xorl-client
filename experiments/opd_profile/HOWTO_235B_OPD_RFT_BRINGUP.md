# HOWTO: bring up a Qwen3-235B OPD / RFT-on-filler training run (correctly)

Last updated 2026-06-24. Written after debugging a 6-failure bring-up gauntlet for the
`er-opd-235b-fillerrft` RFT-on-filler run. This is the operational source of truth for
standing up a 235B OPD-stack training job (student trainer + sampler + dispatch, optional
teacher). Science context: `FILLER_235B_EXPLOIT_RUNBOOK_20260620.md`. Read this BEFORE launching —
every section is a real failure mode that cost a ~20–30 min relaunch.

## 0. TL;DR working recipe (RFT-on-filler, sft_mode, no real teacher)
- **Stack = 8 trainer nodes (head + 7 workers) + 1 sglang sampler + 1 dispatch(CPU).** = **9 full GPU nodes.**
- **Proven repo pairing** (don't change): engine `XORL_REPO=/home/apanda/xorl-apanda-dev-opd-port`,
  client `XORL_CLIENT_REPO=/home/apanda/xorl-client-chat-completions` (both have sft_mode + opd_correct_prefix_only),
  sglang `/home/apanda/xorl-sglang-internal`. These deprecated-looking paths STILL RESOLVE and are the only
  combination verified to init the 235B engine.
- Trainer config: `/shared/apanda/filler_rft/trainer_rft.yaml` (pr83 muon+fp32 recipe, ce_mode=compiled).
- Manifest: `/shared/apanda/filler_rft/er-opd-235b-fillerrft.yaml` (built from `xorl-infra/.../er-opd-235b-clean4d.yaml`).
- Launch: free pool → `kubectl apply -f <manifest>` → babysit head log to `=== OPD step 0 ===` then `forward_backward`.

## 1. Capacity — you need 9 FULL nodes; they do NOT need to be `nccl`
- 8 trainer pods + 1 sampler each request a full `nvidia.com/gpu: 8` → need **9 nodes with ≥8 free GPUs**.
  Dispatch is CPU-only (SMG router) and schedules anywhere.
- **Nodes do not need node-group=nccl.** Default nodes have `rdma/infiniband` and work for both inter-node
  training NCCL and the P2P (mooncake) weight sync. Strip `nodeSelector: {node-group: nccl}` from all pods so
  the scheduler can use the whole pool (the nccl pool alone is usually too contended).
- Count full-free nodes (any group):
  ```bash
  python3 - <<'PY'
  import subprocess,json,collections
  pods=json.loads(subprocess.check_output(["kubectl","get","pods","-A","-o","json"]))
  nodes=json.loads(subprocess.check_output(["kubectl","get","nodes","-o","json"]))
  u=collections.Counter()
  for p in pods["items"]:
      if p["status"].get("phase") not in ("Running","Pending"): continue
      n=p["spec"].get("nodeName");  c0=p["spec"]["containers"]
      if n:
          for c in c0:
              g=c.get("resources",{}).get("limits",{}).get("nvidia.com/gpu");  u[n]+= int(g) if g else 0
  print(sum(1 for nd in nodes["items"] if not nd["spec"].get("unschedulable")
        and int(nd["status"]["allocatable"].get("nvidia.com/gpu",0))-u[nd["metadata"]["name"]]>=8))
  PY
  ```

## 2. The eleven bring-up failure modes (symptom → cause → fix)
(#1–#6 hit the sft/RFT OPD bring-up; #7–#11 additionally hit the GRPO `filler_tokens_rl.py` harness swap. #7–#8 are
config; #9–#10 are harness code edits; #11 is an SMG dispatch flag — all things the sft baseline never exercised:
free-form sampling, full-weights forward_backward registration, and heavy concurrent sampling through a sync pause.)

**(1) `ValueError: Unrecognized config fields: ['empty_cache_steps', 'skip_param_upcast']`**
- Cause: the OPD-server config schema is STRICTER than the standalone-trainer args. pr83's
  `empty_cache_steps`/`skip_param_upcast` exist in `arguments.py` but the OPD config loader rejects them.
- Fix: **delete both keys** from the trainer config. `skip_param_upcast: false` is the DEFAULT, so dropping it
  PRESERVES fp32 master. `empty_cache_steps` is a memory tweak (moot at packing 1024).

**(2) `NotImplementedError: FSDP does not support uneven sharding on dim 1: torch.Size([16, 4096, 3072]) (world size: 7)`**
- Cause: expert FSDP shards the expert weight (hidden dim 4096) across `ep_fsdp = world/EP`. 4096 must divide
  evenly → ep_fsdp ∈ {powers of 2}. 8 nodes → world 64, EP 8, ep_fsdp 8 ✓. 7 nodes → ep_fsdp 7 ✗.
- Fix: **use exactly 8 trainer nodes** (`data_parallel_replicate_size: 8`, world 64). You CANNOT shrink the trainer
  to fit fewer nodes — the 235B expert mesh requires the 8-node (ep_fsdp=8) layout. (This is why §1 says 9 total.)

**(3) `ValueError: teacher_head must point to the teacher prediction head or teacher model path`**
- Cause: `main()` validates `teacher_head` UNCONDITIONALLY (even in sft_mode, which uses no teacher).
- Fix: set `teacher_head="${MODEL_PATH}"` (validated, unused in sft_mode).

**(4) Client hangs on `Waiting for teacher (xorl): http://127.0.0.1:30002` (then dies)**
- Cause: defaults are `teacher_backend=xorl`, `teacher_base_url=127.0.0.1:30002`; nothing serves that. The client
  ALWAYS health-checks a teacher endpoint at bring-up, even in sft_mode.
- Fix: point it at an endpoint that IS up — reuse the sampler: `teacher_backend=sglang
  teacher_base_url=http://<run>-sglang-0:30060`. sft_mode never CALLS the teacher for loss, so this is harmless.

**(5) `RuntimeError: Sample 0: teacher cache rows 433 != expected 30 (p=24, ans=7, K=403)`**
- Cause: with a filler buffer (K>0) the teacher-cache-row check has 3 layouts. Default (supervise_student_cot=false)
  expects `(p-1)+ans` (filler EXCLUDED). But pointing the teacher at a real sglang made it forward the FULL student
  sequence → `(p-1)+K+ans` rows. Mismatch → crash.
- Fix: set **`supervise_student_cot=true`** → expected becomes `(p-1)+K+ans`, matching the full forward. sft_mode
  still re-maps the loss to CE on the answer region only (filler masked), so it remains RFT.

**(6) `torch.distributed.DistStoreError: Timed out after 1801 seconds waiting for clients. 4/8 clients joined.`**
- Cause: 8-node gang rendezvous timed out because only 4/8 trainer pods reached torchrun in time — usually because
  a rapid delete+re-apply left TERMINATING pods occupying nodes, so the late workers couldn't schedule.
- Fix: **clean slate before relaunch.** Delete the old stack, WAIT until `kubectl get pods | grep <run>` is empty,
  confirm ≥9 full-free nodes, THEN apply. With a clean slate all 8 pods go Running within seconds and rendezvous
  succeeds. RDZV timeout is `XORL_TORCHRUN_RDZV_CONF=timeout=1800`.

**(7) Client crashes immediately on `ModuleNotFoundError: No module named 'chz'` (or `tinker_cookbook`)**
- Cause: the manifest's `PYTHON_BIN` find-logic prefers the **sglang venv** (`${SGLANG_REPO}/.venv/bin/python`), which
  has `xorl_client`/`torch`/`wandb`/`sglang` but NOT `chz`/`tinker_cookbook`/`tinker`. The launcher (engine) is fine with
  it, and `on_policy_distillation.py` doesn't import those — but the **GRPO harness `filler_tokens_rl.py` hard-imports
  `chz` + `tinker_cookbook` + `tinker`** (`chz.nested_entrypoint` is its entrypoint). So an sft/OPD manifest that works
  will crash the moment you swap the client to the GRPO harness.
- Fix: run the **client** under a venv that has the harness deps, leaving the launcher on `PYTHON_BIN`. Add a
  `CLIENT_PYTHON` resolver (prefer `${XORL_REPO}/.venv/bin/python` — the engine venv has chz+tinker_cookbook+tinker+
  xorl_client+torch+wandb; fall back to `/home/apanda/xorl-internal/.venv/bin/python`) and invoke the harness with
  `"${CLIENT_PYTHON}"`. Client↔engine communicate over HTTP (`127.0.0.1:26050`), so the two processes can use different
  interpreters with no coupling. Verify with: `<venv>/bin/python -c "import chz,xorl_client,torch,wandb; from
  tinker_cookbook import renderers"` BEFORE launch.

**(8) GRPO harness crashes at `get_renderer` with `KeyError: 'Qwen3-235B-A22B'` (in `tinker_cookbook/model_info.py`)**
- Cause: `filler_tokens_rl.py` picks the chat renderer via `model_info.get_recommended_renderer_name(renderer_model_name)`
  where `renderer_model_name = config.renderer_model_name or config.model_name`. tinker_cookbook's bundled Qwen registry
  (`get_qwen_info()`) has `Qwen3-32B`, `Qwen3-30B-A3B`, `Qwen3-235B-A22B-Instruct-2507`, … but **NOT the base/thinking
  `Qwen3-235B-A22B`** → KeyError. Happens AFTER engine-ready + endpoint-register + session-create (so it looks late), but
  before step 0 — no GPU compute wasted.
- Fix: pass **`renderer_model_name=Qwen/Qwen3-32B`** (maps to renderer `qwen3` — the thinking, non-Instruct Qwen3 renderer,
  so the harness's K3-fix `build_generation_prompt` patch fires). The renderer only selects the chat-template FORMAT class;
  the real 235B tokenizer (`tokenizer_name=${MODEL_PATH}`) is still what's passed to `renderers.get_renderer(name, tokenizer)`,
  and all Qwen3 thinking models share the same template, so this is exact. (`Qwen/Qwen3-30B-A3B` also works → `qwen3`.) Do
  NOT use `…-Instruct-2507` (→ `qwen3_instruct`, patch won't fire) or a `…-Base` (→ `role_colon`, wrong format). Line 2180
  is the harness's ONLY `model_info` consumer, so this one arg resolves the whole registry issue.

**(9) Step-0 sampling loops forever: `HTTP 404 model_not_found: No worker available for model 'default'`**
- Cause: in full-weights mode `filler_tokens_rl.py` creates `tomi.SamplingClient(base_url=url)` with **no `model=`**, so
  the request omits the model field and the SMG dispatch (8080) defaults to model `'default'` — which no worker serves
  (the sampler serves `served_model_name=Qwen/Qwen3-235B-A22B`). Every sample 404s; the harness retries indefinitely
  (no crash, just a stuck loop flooding the log). `on_policy_distillation.py` (baseline) doesn't hit this because it
  passes `model=config.model_name` to its SamplingClients.
- Fix: edit the full-weights SamplingClient (one site, ~line 2331) to `tomi.SamplingClient(base_url=url,
  model=config.model_name)` — `config.model_name=Qwen/Qwen3-235B-A22B` matches what the dispatch serves. Eval/control
  sampling reuses the same client list, so this one edit fixes sampling everywhere. (Routing parity check: both baseline
  and GRPO use `inference_base_urls=http://<run>-dispatch:8080` + `model_name=Qwen/Qwen3-235B-A22B`.)
- Detection: a `model_not_found 'default'` flood is a STUCK loop, not a crash — monitor for it explicitly (the pod stays
  Running, so a crash-only watcher reports nothing). Tear down + relaunch after the harness edit (the running client
  won't reload the file).

**(10) `forward_backward` 400s: `model_id '...' has not been registered. Call create_model or create_session first`**
- Cause: `filler_tokens_rl.py`'s full-weights path registers the model via `POST /api/v1/create_model`
  (`{model_id, base_model, lora_config:{}}`), which returns a `future_...` id but does NOT register the model_id for
  training ops on the `xorl.server.launcher` engine — that endpoint targets the tinker/tomi server. So sampling works,
  then the first `forward_backward` 400s (non-retryable) and the run dies. Engine did NOT restart; the registration just
  never took.
- Fix: register via `POST /api/v1/create_session` (`{session_id: model_id, base_model: config.model_name}`) — the call
  `on_policy_distillation.py`'s `_ensure_xorl_session` uses and that this engine accepts (proven: the baseline did
  full-weights `forward_backward`+`optim_step`+p2p-sync via create_session). Edit ~L1513 of the harness:
  `create_model {model_id,base_model,lora_config}` → `create_session {session_id,base_model}`, keep
  `TrainingClient(model_id=model_id)`. (NB the `optim_step(tomi.AdamParams(lr))` call needs NO change — the engine
  applies its config optimizer (Muon); AdamParams just carries the LR. Baseline-proven.)

**(11) Step-0 sampling stuck looping on `HTTP 503 no_available_workers (circuit breaker open or unhealthy)`**
- Cause: the SMG dispatch (`smg launch`) circuit breaker opens after **10 failures** and stays open **60s**
  (`--cb-failure-threshold 10 --cb-timeout-duration-secs 60` defaults). The per-batch weight-sync pauses the (single)
  sampler for ~9s; >10 of the batch's 1024 free-form requests (group_size 64 × batch 16) fail during the pause → breaker
  opens → the whole batch 503s and the GRPO group never completes (sampler is HEALTHY the whole time — check its log:
  `restartCount=0`, `Decode batch #running-req: 9-11` — so it's purely the dispatch breaker, not sampler overload).
  Self-healed once (relaunch #3 batch 0 squeaked through in 1:46) but is not reliable.
- Fix: add **`--disable-circuit-breaker`** to the SMG dispatch launch command
  (`smg launch ... --policy cache_aware --disable-circuit-breaker --worker-urls ...`). SMG keeps its 5× retry, so
  sync-pause failures just retry against the resumed sampler instead of tripping a 60s lockout. (Bypassing SMG —
  `inference_base_urls=http://<run>-sglang-0:30060` — also works since `get_inference_urls` uses a full http URL as-is,
  but disabling the breaker keeps the router for future multi-sampler.) Diagnose sampler-vs-dispatch first: if the
  sampler log shows it's actively decoding, the breaker is the culprit, not capacity.

## 2b. Running on FEWER than 8 nodes (capacity-constrained fallback)
The 235B EP mesh needs `ep_fsdp = world/EP` to evenly divide the expert hidden dim 4096 (failure #2). Valid trainer
node counts (EP=8, 8 GPU/node): **8** (ep_fsdp 8, proven), **4** (ep_fsdp 4, 4096/4=1024 ✓), 2 (ep_fsdp 2), 1.
**3,5,6,7 nodes FAIL** (ep_fsdp non-integer-divisor). A k-node trainer needs k+1 full nodes total (+1 sampler; dispatch
is CPU). **Memory caveat:** per-GPU param+optimizer memory ≈ total/world, so 4 nodes (world 32) ≈ **2× the 8-node
(world 64)** footprint — with fp32 master + Muon momentum the 235B may OOM on 4 nodes even with recompute+offload (the
savers cut activations, not param/optimizer state). `skip_param_upcast` (to drop fp32 master) is REJECTED by the OPD
config loader (failure #1), so you can't easily shrink the optimizer state. Treat 4-node as best-effort.
- 4-node config: `/shared/apanda/filler_grpo/trainer_grpo_4node.yaml` (`data_parallel_replicate_size: 4`, recompute+offload
  ON, packing 4096). Manifest: `/shared/apanda/filler_grpo/er-grpo-235b-4node.yaml` (head + 3 workers, `--nnodes 4`,
  `--nproc-per-node=8` kept). Built from the 8-node manifest by dropping workers 4-7 + `--nnodes 8→4` + CONFIG_PATH swap.
- **EMPIRICAL RESULT (2026-06-25): 4-node does NOT train stably.** It loads the 235B fine (no load-OOM) and step 0's
  `forward_backward` ran in 42s — BUT the run died at step ~1 with `504: Forward-backward timeout after 1800.0s`. Tell:
  it failed only AFTER the first `optim_step`, i.e. once Muon's fp32 momentum buffers were allocated the memory headroom
  vanished and the next `forward_backward` thrashed near-OOM (a hang, not a clean OOM error). Confirms the 2×-optimizer-
  memory ceiling: **4-node is not viable for 235B training with fp32 master.** Don't loop-retry it; wait for 9 nodes
  (8-node) instead. Would only work if fp32 master could be dropped (config loader rejects `skip_param_upcast`) or with
  far more parallelism (≥8 nodes).

## 3. Editing the manifest — use a YAML parser, NOT string replace
The trainer command is a giant YAML-escaped scalar. A raw `str.replace` that inserts `"..."` (e.g. adding
`teacher_head="${MODEL_PATH}"`) **corrupts the quoting and breaks the YAML**. Always parse + edit + re-dump:
```python
import yaml
docs=[d for d in yaml.safe_load_all(open(SRC)) if d]
# drop teacher pods + sglang-1 for the no-teacher single-sampler RFT stack:
docs=[d for d in docs if "teacher" not in d["metadata"]["name"] and "sglang-1" not in d["metadata"]["name"]]
# edit container.command[2] as a normal python string; strip nodeSelector; rename clean4d->fillerrft
yaml.safe_dump_all(docs, open(OUT,"w"), sort_keys=False, width=10000)
```
The reusable generator that bakes in all fixes is in `FILLER_235B_EXPLOIT_RUNBOOK_20260620.md` PHASE-2 section.
Always `kubectl apply --dry-run=server -f <manifest>` and `python3 -c "yaml.safe_load_all(...)"` before launch.

## 4. Launch + babysit
```bash
kubectl apply -n apanda -f /shared/apanda/filler_rft/er-opd-235b-fillerrft.yaml
HEAD=$(kubectl get pods -n apanda | grep -oE "er-opd-235b-fillerrft-trainer-head-[a-z0-9]+")
kubectl logs -n apanda "$HEAD" -f
```
Healthy bring-up sequence (head log): `SGLang 0 healthy` → `SMG ready` → `Starting xorl training server` →
`xorl trainer engine ready` (235B weight load ~15–20 min) → `Registering SGLang inference endpoint` →
`Waiting for teacher (sglang) ... ` (passes fast) → `=== OPD step 0 ===` → on-policy `sample correct=...` →
`OPD step 0 profile: {...}` (control eval acc_pause/acc_nopause) → `=== OPD step 1 ===`. ~400 s/step.
Gate step 0: `empty_frac=0`, `has_digit_frac=1`, `opd_gold_answer_replacements`=batch size, control eval emits
`eval/acc_pause` & `eval/acc_nopause`. NB for sft_mode the OPD `"loss"` field reads 0.0 (training signal is the
trainer's `cross_entropy`, not opd_loss) — that's expected, not a failure.

## 5. Gotchas / notes
- Single sampler under the concurrent 3-arm control eval (1536 reqs) shows high `control_request_failure_frac`
  (~0.4–0.7) and `heldout` is unreliable; the pause/nopause control arms still score ~500/512 so `buffer_delta`
  is usable. Add a 2nd sampler (10th node) for clean held-out numbers.
- Teardown: `kubectl delete -f <manifest>` (or delete the job + worker/sampler/dispatch pods). Always confirm
  pods are GONE before relaunching (see failure #6).
- Do NOT preempt other live runs to make room without explicit owner authorization (cluster is shared).

## 6. K3 reconciliation for the GRPO run (the "insanely high K3" fix chain)
After the §2 harness bring-up the GRPO run trained but K3 was ~2 (should be ~1e-3). Fixes, by impact:
- **(K3-1) THE big one — packing pad token.** `src/xorl/server/orchestrator/packing.py` padded packed
  `target_tokens` with token-id **0** (a real token) instead of `IGNORE_INDEX`, so the trainer recomputed
  logprobs at those pad positions and compared them to the sampler → a fake K3 tail. Fix:
  `pad_value = IGNORE_INDEX if key=="target_tokens" else 0` then `value[0].extend([pad_value]*pad_length)`. Took
  K3 2.7 → ~2.3e-4. The engine repo `XORL_REPO=/home/apanda/xorl-qwen-k3-reconciliation` already has it; if you
  launch from a different `XORL_REPO`, port it. Also prefer `target_tokens` over `labels` in `training_utils.py`
  count_valid/active (labels can be extra-masked by advantages=0 → undercount).
- **(K3-2) The reconciliation recipe** (sampler + trainer numerics must match). Sampler:
  `--rl-on-policy-target xorl-batch-invariant --enable-fp32-lm-head --enable-fp32-router --attention-backend fa3`
  + env `SGLANG_DISABLE_ROPE_COMPILE=1 SGLANG_RMSNORM_FP32_WEIGHT_MUL=1 SGLANG_RETURN_ORIGINAL_LOGPROB=1`. Trainer
  config: `router_fp32:true lm_head_fp32:true rmsnorm_mode:native ce_mode:eager` (eager not compiled — compiled
  lowers the lm-head matmul to cuBLAS and escapes the fp32 override). See the `xorl-train-serve-parity` skill +
  `K3_GRPO_235B_HANDOFF_20260628.md`.
  - **UPDATE 2026-06-29 — batch-invariant dropped (optional, not required for GRPO).** `--rl-on-policy-target
    xorl-batch-invariant` was dropped (apanda's direction). Fuller data over 25 steps: WITHOUT it the live k3 mean
    BOUNCES ~1e-4–4.5e-3 (mean ~1.4e-3, worst-token tail up to ~4.4); WITH it the earlier run was ~2-7e-4
    (worst-token ~0.1-0.3). So batch-invariant DOES tighten k3 ~3-4× and shrinks the token tail — it is a real
    lever, just not REQUIRED: GRPO trains fine at ~1e-3 mean (reward climbing, clipfrac ~0), because the
    packing-pad fix (K3-1) + fp32 lm-head/router + routing replay (K3-3) carry the bulk of reconciliation.
    Trade-off: batch-inv = tighter k3 but slower determinism kernels. Keep it OFF for outcome-reward GRPO (k3
    rigor not critical); turn it ON for OPD/KL-distillation or if you need sub-1e-3 k3. (NB: do not trust a single
    step's k3 — it bounces ~20× step-to-step; judge off the multi-step series.)
- **(K3-3) MoE routing replay (R3)** — sampler `--enable-return-routed-experts`; the client pads/truncates each
  datum's base64 routed_experts (int32, 94 layers × 8 topk) to the model_input length and re-encodes; the trainer
  replays the sampler's expert routing so the recompute matches the decode path.
  - **R3 over LONG prompts HANGS the fwd/bwd** (2026-06-30): 10-shot prompts (~1645 tok, ~0.6GB routing/step)
    broadcast the routing INLINE through Gloo `broadcast_object_list`, which deadlocks (100% util / ~120W spin).
    FIX = `externalize_r3_payloads: true` in the trainer config (out-of-band shared-FS transport + uuid-unique
    dirs; keeps full routing → low k3, NOT answer-only). Engine branch `k3-recon-r3-hangfix-merge-20260630`
    (`fix/q36-live-k3-behavior-replay` merged). Step-0 k3 reads transient-high then drops to ~1.5e-4 by step 2.
    Full detail: `R3_EXTERNALIZE_PAYLOADS_HANGFIX_HANDOFF_20260630.md` + memory `q235-r3-externalize-payloads-hang-fix`.
- **NOT the cause** (don't chase): TP8-sampler-vs-EP8-trainer layout, the fp32/batch-invariant flags alone, or
  weight-sync. The pad token was the tail.

## 7. p2p weight sync: the mooncake CUDA-12/13 fix
Symptom: step-1 dies `Weight sync failed: Failed to initialize p2p backend` (serial AND pipeline). Cause: the
cu132 trainer venv (`.venv-cu132-latest-probe`) lacks `libcudart.so.12`, which mooncake's `engine.so` dlopens —
so `import mooncake.engine` fails. The real exception is in `server.log`, not the client error string. Fix
(env-independent, NFS-shared so all 8 nodes see it): copy `libcudart.so.12` into mooncake's rpath dir
`<venv>/lib/python3.12/site-packages/mooncake_transfer_engine.libs/`. Survives run.sh rewrites. After this, p2p
inits and syncs ~470GB at ~45-57 GB/s. Detail: `PIPELINE_RL_P2P_SYNC_HANDOFF_20260629.md`.

## 8. pipeline_rl + p2p sync (overlap) via client-level pause
To weight-sync over p2p without corrupting in-flight generation, the CLIENT pauses the engine first (NOT an
engine change): `_pause_inference(mode="retract")` → POST `/pause_generation` to each inference URL →
`sync_weights_to_inference(p2p)` → `finally: _continue_inference()` → POST `/continue_generation`. Added in
`filler_tokens_rl.py` around the pipeline-sync branch. Steady-state sync ~8.5s, hidden under the step.

## 9. EP dispatch: use all-to-all, NOT DeepEP (live training)
A frozen-trace K3 sweep (`experiments/k3_tests/run_q235_ep_dispatch_sweep.py`, cu129 venv) shows DeepEP
numerically best at EP8 (K3 4.5e-5 vs all-to-all 1.3e-4). **Do not swap it into a live run:** (a) DeepEP
**deadlocks the live weight-sync loop** — the sweep was forward_backward-only, so it never synced weights and
never hit the deadlock; (b) `deep_ep` isn't installed in the cu132 trainer venv (only cu129); (c) EP16+ DeepEP is
additionally blocked on cross-node NVSHMEM bootstrap. all-to-all's K3 (~1.3e-4 static, ~2-7e-4 live with weight
drift) is already far below the 1e-3 gate. Live config: `ep_dispatch: alltoall`. DeepEP is a future *throughput*
lever (faster MoE dispatch) gated on fixing the sync deadlock + matching the venv — not a correctness win.

## 10. Running on reprogrammable slots (not a one-off manifest)
The 235B GRPO/RFT now runs on long-lived GPU-holding slot pods (`er-opd-q235-fillerrft-slots`): 8 trainer +
TP8 sampler + CPU dispatch = the same 9 full GPU nodes, held warm. Swap the workload by rewriting
`/shared/opd-control/<stack>/<role>/run.sh` (the slot agent re-execs the child on run.sh hash change — NO pod
reschedule). **NEVER `kubectl delete` slot pods** (loses the scarce whole-node capacity). Memory:
`reprogrammable-slots-235b-launch`.
- **Capacity reality:** you CANNOT spin up a second 235B stack on demand — the cluster typically has ~1 whole
  free node + fragmented partials (~28 GPUs across 7 nodes is a real snapshot); the 9 whole nodes you need ARE
  the ones already in the slots. That is precisely why the slots are held warm rather than re-created.
- **Relaunch-war hazard:** if two sessions both rewrite the same stack's run.sh (e.g. a K3 sweep vs the GRPO run)
  they flap the slots and neither completes (each bring-up ≈13 min). Resolution is human: stop the other workload
  in its own session. Cross-session `pkill` is correctly blocked by the auto-mode classifier — do NOT work around
  it; surface to the user. A watcher that only relaunches once the contender is gone 60s avoids premature
  re-grabs and resets if it reappears.
- **Static rendezvous re-trigger:** verify `:29610` free + 0 stale procs, sequence head-first then workers.
- **Dispatch stale-health killer:** a restarted SMG dispatch can hold a stale "unhealthy" worker view and
  fast-503 every rollout → 0 datums → cold EP → step-1 sync wedges. Restart dispatch+sampler fresh and verify
  BOTH sampler `/generate=200` AND dispatch `/v1/chat=200` before re-triggering the trainer.
