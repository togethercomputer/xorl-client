# Shared-outer (attention+experts) LoRA-MoE forward hangs on Qwen3.6-35B-A3B (TP=2)

**Status: RESOLVED 2026-06-10** — root cause found + fixed in `xorl-sglang-internal` @ `apanda-dev`
`bc1ffb583` ("Fix LoRA serving on Qwen3.5/3.6-MoE hybrid models"). See **Resolution** at the bottom.
**The pool must be restarted** (`kubectl rollout restart statefulset/zorl-ar-sglang -n apanda`) to
pick up the fix, and the candidate must use the regenerated init adapter
(`/shared/zorl/init-adapters/qwen3_6-35b-a3b-r16-eggroll-hybridattn` — already repointed in
`autoresearch/candidates/MULTOPSD-EGGROLL-35B.yaml`).

**Status (original):** OPEN — blocks the rung-3 EggRoll-ES experts-LoRA run. Written 2026-06-10.
**Owner:** SGLang LoRA-MoE serving (the `xorl-sglang-internal` subsystem). This is a *serving-kernel*
bug, not an ES-logic bug — everything on the ES side (pool config, EggRoll merge-every-step client,
candidate) is ready the moment the experts-LoRA forward serves.

---

## TL;DR

Serving an **attention + experts** LoRA (`--lora-moe-format hybrid_shared --experts-shared-outer-loras`)
on **Qwen3.6-35B-A3B** (TP=2): the LoRA adapter **loads fine**, but the **first `/generate` that uses
it hangs**. After the 300 s running-phase watchdog it SIGQUITs → all 16 replicas recycle synchronously
→ the parent LoRA is gone → every subsequent `/generate` returns *"Got LoRA adapter that has never been
loaded"*. **Base (non-LoRA) `/generate` works fine.** Confirmed **repeatable** (warm re-launch hangs
identically; pool restart count went 16→32). So it is a real, persistent forward hang on the
shared-outer LoRA-MoE path — not a cold-start/JIT timeout, not OOM, not the base model.

There is also a **secondary bug in the SIGQUIT handler** (`aclose()` on an already-running async gen)
that makes the recycle messy — worth fixing independently.

---

## Environment

- **SGLang:** `xorl-sglang-internal` @ `apanda-dev` HEAD `5e9acb7db` ("MTP native multi-token decode +
  OPD/ZORL serving; fix overlap seq_lens desync") — the post-17:14 fixed build (has the batched-decode +
  chat-rendering fixes; quack EP #355; etc.).
- **Pool:** StatefulSet `zorl-ar-sglang`, 16× **TP=2** (32 GPUs), namespace `apanda`.
  Manifest: `experiments/zorl/k8s/qwen3_6-35b-a3b-zorl-sglang-tp2-shard.yaml`.
- **Launch flags (the relevant ones):**
  ```
  --model-path Qwen/Qwen3.6-35B-A3B --tp 2 --dtype bfloat16 --mem-fraction-static 0.75
  --enable-lora --lora-backend triton --max-lora-rank 16
  --max-loras-per-batch 64 --max-loaded-loras 80
  --lora-target-modules qkv_proj o_proj gate_proj up_proj down_proj
  --lora-moe-format hybrid_shared --experts-shared-outer-loras
  --disable-cuda-graph --watchdog-timeout (default 300)
  ```
- **Model MoE geometry:** Qwen3.6-35B-A3B — `num_experts=256`, `num_experts_per_tok=8`,
  `moe_intermediate_size=512`, `shared_expert_intermediate_size=512`.
- The pool log at startup confirms the path:
  `Shared outer LoRA mode enabled: gate_up lora_A and down lora_B will be shared across experts (expert_dim=1)`
  and `Using triton as backend of LoRA kernels.`

---

## Symptom / reproduction

1. Pool loads clean — all 16 replicas reach `Application startup complete`, **no OOM**, shared-outer
   mode enabled (above).
2. Client (`zorl_client.py`) loads the parent LoRA — a fresh **rank-16 attention+experts** adapter
   (`adapter_config.json`: `r=16`, targets `qkv_proj o_proj gate_proj up_proj down_proj`,
   `_sglang_lora_format=shared_outer`, `moe_hybrid_shared_lora=true`). The pool logs
   `LoRA adapter loading completes` on **both** TP0 and TP1.
3. The first `/generate` that uses the LoRA (the cold greedy probe — a **short** 4×4-mult prompt,
   `max_new_tokens=64`, `temperature=0`) **hangs**. It never returns.
4. At 300 s the **running-phase watchdog** fires → SIGQUIT.
5. **All 16 replicas recycle synchronously** (`exit 0` via SIGQUIT; restart count 1 → then 2 on a
   repeat run, i.e. pool restart sum 16 → 32).
6. After recycle the parent LoRA is no longer resident → every probe `/generate` returns HTTP 500
   `Got LoRA adapter that has never been loaded` → the client exits `Error`.

**Base (non-LoRA) `/generate` works fine** (e.g. `2+2=` → `5` returns instantly), so the base model
and TP mesh are healthy. Only the **shared-outer LoRA-MoE forward** path hangs.

---

## The crash signature (logs)

Replica `--previous` (the SIGQUIT path):
```
File ".../managers/tokenizer_manager.py", line 2836, in running_phase_sigquit_handler   # the watchdog handler
RuntimeError: aclose(): asynchronous generator is already running                         # SECONDARY bug
```
The `aclose()` error is the handler trying to close a still-running async generator (the hung
streaming `/generate`). It's a **secondary** bug — the root is the forward hang that triggered the
watchdog — but it should be fixed independently (the handler shouldn't double-close a running gen).

Replica current log after recycle:
```
[http_server] Error: Got LoRA adapter that has never been loaded: <session>/parent
```

---

## What is ruled out

- **Not cold-start / triton JIT timeout.** A warm re-launch (pool already up, kernels presumably
  cached) hung identically and recycled again (restart sum 16 → 32). Persistent.
- **Not OOM.** No `OutOfMemory` / `OutOfResources` / `shared memory` in any log; the pool loaded at
  `mem-fraction 0.75` and base generation runs.
- **Not the base model / TP mesh.** Base (non-LoRA) `/generate` works.
- **Not the adapter shape.** An *earlier* failure ("index 40 is out of range" on `/load_lora_adapter`)
  was a **separate, fixed** issue: the manifest's `MODEL_PATH` defaulted to Qwen3-30B, so the
  init adapter was built 30B-shaped (48 layers) and out-of-range on the 35B. Now `MODEL_PATH=
  Qwen/Qwen3.6-35B-A3B`, the adapter is 35B-correct and **loads cleanly**; the hang is downstream
  at the forward.

---

## Leads / hypotheses

- The shared-outer LoRA-MoE **forward** path (`gate_up lora_A` / `down lora_B` shared across experts,
  `expert_dim=1`) on the **Qwen3.6-35B** expert geometry (256 experts, top-8, `moe_intermediate=512`)
  appears to deadlock/hang (not OOM, not error — it just never returns).
- **CONFIRMED contrast (2026-06-10) — rank-16 shared-outer works on the 30B, hangs on the 35B.**
  Brought up the 30B/Qwen3 pool (`qwen3-coder-30b-a3b-zorl-sglang-tp2-shard.yaml`) with the **exact same
  flags** (`--lora-moe-format hybrid_shared --experts-shared-outer-loras`, `--max-lora-rank 16`,
  triton backend, TP=2) and a **rank-16 attention+experts** adapter (`init_zorl_adapter` emitted 480
  shared_outer tensors: `is_moe=True`, 128 experts, moe_int 768). The cold probe — a **128-example LoRA
  `/generate`** — **completed cleanly** (`exact_count=11/128`), pool stayed `rst=0`, and the EggRoll-ES
  scoring loop is running. So the 35B hang is **NOT** rank-16 and **NOT** the shared-outer/hybrid_shared
  format in general — both are exercised end-to-end on the 30B. It is **specific to Qwen3.6-35B-A3B**.
  The strongest remaining suspect is the **expert geometry in the shared-outer expand**: 35B has
  **256 experts / moe_int 512** vs the 30B's **128 experts / moe_int 768**. (Both hidden=2048, TP=2.)
  Worth checking: does the expand kernel's grid/tiling assume `moe_int` ≥ some bound, or mis-handle
  256-expert sharding across TP=2 (128 experts/rank)?
- Known prior MoE-LoRA issues in this stack (memory): `hybrid_shared` shape-mismatch; `per_expert`
  OOM; chunked-SGMV multi-LoRA OOB (`_ebi_expert_ranks` sized at first call, never resized as
  num_loras grows). The OOB ones manifested as crashes; this one manifests as a **hang**.

---

## Suggested next steps (SGLang owner)

1. **Minimal repro (no ES):** bring up the 35B pool (manifest above), `POST /load_lora_adapter` with a
   rank-16 attention+experts adapter, then a single `POST /generate` with `lora_path=<that adapter>`
   and a short prompt. It should hang; the watchdog SIGQUITs at 300 s. (Lower `--watchdog-timeout` to,
   say, 60 for faster iteration, or temporarily disable it to attach a profiler/py-spy to the hung
   forward.)
2. **Localize:** `py-spy dump` the hung scheduler/worker during the forward to find the stuck frame in
   the shared-outer expand / chunked-SGMV LoRA-MoE kernel or its host-side launch. Hang (not OOM/error)
   points at a kernel that never returns or a host↔device sync that never completes.
3. **A/B to isolate:** (a) `per_expert` vs `hybrid_shared`; (b) rank-4 vs rank-16 on the 35B;
   (c) ~~the 30B/Qwen3 pool with rank-16 (rank vs arch)~~ — **DONE 2026-06-10: rank-16 shared-outer
   WORKS on the 30B** (see CONFIRMED contrast above), so rank and format are ruled out; the variable
   is the Qwen3.6-35B arch / 256-expert·moe_int-512 geometry; (d) attention-only targets (no experts)
   on the 35B — that path is known to work (rung-2 OPD-LoRA used attention-only successfully).
4. **Fix the secondary `aclose()` bug** in `tokenizer_manager.py:2836` `running_phase_sigquit_handler`
   so the SIGQUIT recycle doesn't error on the running async gen.

---

## Repro artifacts (this repo / cluster)

- Pool manifest: `experiments/zorl/k8s/qwen3_6-35b-a3b-zorl-sglang-tp2-shard.yaml`
- ES candidate: `experiments/zorl/autoresearch/candidates/MULTOPSD-EGGROLL-35B.yaml`
  (rank-16 attention+experts, `MERGE_EVERY_STEPS=1`, `ELITIST_ROLLBACK=0`, mult-OPSD forward-KL)
- Client (incl. the new EggRoll merge-every-step wiring): `experiments/zorl/standalone/zorl_client.py`
  (`merge_zorl_parent_into_base_all` + `--merge-every-steps`)
- Init adapter (35B-correct, rank 16): `/shared/zorl/init-adapters/qwen3_6-35b-a3b-r16-eggroll`
- Failed client pods: `zorl-ar-multopsd-eggroll-35b-*` (Error). Pool: `zorl-ar-sglang-*`
  (restart count climbing on each LoRA-`/generate` attempt).

---

## Context: why this matters

This is rung 3 of the no-filler-OPSD reproduction ladder (full-FT → LoRA → **ES**). Rungs 1-2 reproduce
the fast rise (full-FT 0.51→0.625; attention-only LoRA held-out 0.66→0.68). Rung 3 is EggRoll-style ES
(low-rank noise + **merge-into-base every step** via `merge_zorl_parent_into_base`, the fold op already
built+validated) on an **attention + experts** LoRA — which requires the shared-outer LoRA-MoE forward
to serve. See `experiments/zorl/OPSD_ZORL_RUNBOOK.md`.

---

## Resolution (2026-06-10, fixed in `xorl-sglang-internal` @ `apanda-dev` `bc1ffb583`)

**It was never a kernel hang.** Reproduced locally (TP=2, same tree/flags/adapter): the first LoRA
batch crashes BOTH schedulers with a buffer-shape assert in
`mem_pool.load_lora_weight_to_buffer` (`[16,256]` vs `[16,512]` on `down_proj_moe` A;
`[16,2048]` vs `[16,4096]` on `o_proj` A). On the pod the server is PID 1, and the broken SIGQUIT
path (bug 4 below) swallowed the crash → the pool looked "hung" for 300 s and recycled with exit 0.
The cluster's `aclose()` traceback is exactly that path.

Four stacked root causes:

1. **No module was LoRA-wrapped on this model at all.** `Qwen3_5MoeForConditionalGeneration`
   inherited `Qwen3VLForConditionalGeneration._lora_pattern`, which requires
   `model.layers.N.(self_attn|mlp).(qkv_proj|o_proj|down_proj|gate_up_proj)`. On the Qwen3.5/3.6-MoE
   stack the attention projections sit **directly on the decoder layer** (`model.layers.N.qkv_proj`,
   no `self_attn`) and the MLP routes through `mlp.experts` (FusedMoE). So `should_apply_lora`
   rejected everything: experts LoRA would never have applied, and adapter weights were never
   TP-sliced → per-rank buffer copy assert at first use. (The 30B/Qwen3 pool worked because
   `Qwen3MoeForCausalLM` defines no `should_apply_lora` — no filter.) Fixed by overriding
   `_lora_pattern` (attention direct-on-layer + `mlp.experts`).
2. **`attn_output_gate` not modeled in LoRA buffer sizing.** Qwen3.6 fuses a per-head output gate
   into qkv (q section doubled: full out 9216, not 5120). The generic `get_hidden_dim` fallback
   undersized the qkv B buffer. Fixed by implementing `get_hidden_dim` on the model class.
3. **Adapter weights for modules that don't exist on a layer crash the load.** The init adapter had
   `self_attn.{qkv,o}_proj` for all 40 layers, but 30 layers are GatedDeltaNet (no attention
   modules). `mem_pool` now drops such weights with an aggregate warning instead of crashing.
4. **Secondary `aclose()` bug fixed**: `running_phase_sigquit_handler` now guards the crash dump and
   exits via `os._exit(1)` after killing children — on PID 1, self-SIGKILL is ignored and the old
   `sys.exit(0)` fallback raised SystemExit inside the interrupted streaming async generator
   (`aclose(): asynchronous generator is already running`), masking crashes with exit 0.

**Adapter side:** regenerated init adapter at
`/shared/zorl/init-adapters/qwen3_6-35b-a3b-r16-eggroll-hybridattn` — attention keys only on the 10
full-attention layers (3,7,…,39), qkv `lora_B` zeros at the gated shape `[9216,16]`. The candidate
yaml is repointed. The original adapter dir is untouched (it still loads post-fix, with warnings,
but its ungated `[5120,16]` qkv B will be rejected — use the `-hybridattn` one).

**Validation (local TP=2, pool flags):** adapter loads clean (no warnings), LoRA `/generate`
returns, and since all `lora_B` are zero the LoRA output is token-exact equal to base on 3/3 greedy
prompts. Regression test added: `test/registered/lora/test_moe_lora_mem_pool.py`
(`test_weights_without_wrapped_module_are_skipped`).

**Ops to unblock rung 3:** restart the pool on the fixed build
(`kubectl rollout restart statefulset/zorl-ar-sglang -n apanda` — pods run from the same
`~/xorl-sglang-internal` PVC tree), then relaunch the EggRoll client.
