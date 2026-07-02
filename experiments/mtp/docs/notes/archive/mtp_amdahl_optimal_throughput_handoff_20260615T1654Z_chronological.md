# Handoff: OPD-MTP throughput improvement runbook

Date: 2026-06-13
Stack: `er-opd-q36-mtp-ss-0605c`
Model: Qwen3.6-35B-A3B, native SingleShot-MTP, ConfAdapt, k=2
Historical validation trainer code: `/home/apanda/xorl-mtp-commitlen-fix-20260612`
Dedicated MTP canonical-engine checkout: `/home/apanda/xorl-mtp-fix-merge-20260613`
SGLang code: `/home/apanda/xorl-sglang-internal`

> 🏷️ **RENAMED 2026-06-14 (READ FIRST — overrides every dir/branch below):** the canonical MTP line is now
> **DIR `/home/apanda/xorl-mtp`**, **BRANCH `apanda-dev-mtp @ 21653953`** (`origin/apanda-dev-mtp`); it = the prior
> `5cf35d68` + the **clean-region** engine port (now committed, no longer a no-op). `origin/fix/mtp-fix-plus-upstream`
> is DELETED and the other two `~/xorl-mtp*` worktrees were removed. Set `OPD_XORL_REPO=/home/apanda/xorl-mtp`.
> NAME-REUSE: the OLD wrong `apanda-dev-mtp @ 3a6a6a02` is retired to `origin/backup/apanda-dev-mtp-b2-20260614` —
> avoid the SHA `3a6a6a02`, not the name. ⚠ The live control `run.sh` are STALE (point at deleted worktrees) and
> the supervisor was stood down — re-render the control from `/home/apanda/xorl-mtp` before any resume.

Post-consolidation canonical locations:

- MTP engine start point: `/home/apanda/xorl-mtp-fix-merge-20260613`,
  branch `fix/mtp-fix-plus-upstream`, commit `08041ed7`, tracking
  `origin/fix/mtp-fix-plus-upstream`.
- client docs and non-k8s harness: `/home/apanda/xorl-client/experiments/mtp`
  (`xorl-client` `internal/apanda-dev @ 0c29943` contains the migrated harness;
  the local `opd-battery-consolidation` checkout may also contain uncommitted
  closeout doc edits)
- k8s/manifests: `/home/apanda/xorl-infra/k8s/opd_profile`
  (current consolidation branch: `opd-battery-consolidation`, intended to fold into infra PR #1)
- SGLang runtime: `/home/apanda/xorl-sglang-internal`, branch `apanda-dev`, containing
  `18c2357de Fix per-request MTP draft verification`
- run outputs: `/shared`

Do **not** start new throughput work from `/home/apanda/xorl-stage-mtp`, `pr/mtp`,
`mtp-merge-apanda-dev`, `/home/apanda/xorl-mtp-singleshot-port-20260602` branch
`apanda-dev-mtp@3a6a6a02`, or the old local pointer branch
`throughput/mtp-fwdbwd-profile-20260614@3a6a6a02`. That line had B2/current-apanda-dev
content and the `b3876456` review fixes, but missed the live-proven `08041ed7`
emit-window/fix-line content. Use it only as a reference branch for focused review-fix
porting, not as an execution base. Do **not** start from `/home/apanda/xorl-mtp-commitlen-fix-20260612`
except to inspect historical validation artifacts.
New controls should set `OPD_XORL_REPO=/home/apanda/xorl-mtp-fix-merge-20260613`
until the canonical branch is renamed.

This replaces the old incremental override log. The pre-consolidation version is
archived at:

`docs/notes/mtp_amdahl_optimal_throughput_handoff_archive_20260613_pre_consolidation.md`

## Current conclusion

The current promotable throughput control is:

- student samplers: k=2 with canonical q-banding enabled
- trainer: clean-region replay context enabled
- trainer checkpointing: `recompute_before_dispatch`
- trainer topology: EP32, `alltoall` dispatch, triton MoE
- pipeline: 64 prompts per step, two 32-prompt prepare chunks, no trainer coalescing
- checkpoints disabled for short validation

Do not chase OPD KL/top-k. CUDA-synced profiling put the model-side time in
trainer forward/loss, backward/recompute, and communication. OPD loss/KL is
small relative to f/b.

Do not compare full-stack throughput runs across different sampler states as if
they are same-workload A/Bs. The student changes the workload over time. In the
late runs, commit length and sampler tok/s changed enough to dominate wall time.
Use either same-checkpoint/same-sampler-state full-stack runs or trainer-only
replays of captured payloads.

## Latest validation

The stack was reprogrammed to the current q-band + clean-region control and the
capped validation completed cleanly:

| item | value |
|---|---|
| run | `q36mtp-20260613T223322Z-2s1t` |
| W&B | `btqzrlm6` |
| profile | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260613T223322Z-2s1t/artifacts/opd_profile.jsonl` |
| trainer log | `/shared/opd-control/er-opd-q36-mtp-ss-0605c/trainer-head/logs/20260613T223321Z-run.log` |
| control root | `/shared/opd-control/er-opd-q36-mtp-ss-0605c` |
| supervisor | paused |
| terminal log | `OPD pipeline validation succeeded`; W&B synced |

The first row was a cold/low-accept row:

| step | wall | trainer f/b | sample | teacher | sync | consumed tok/s | commit_len |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 112.00s | 64.05s | 35.91s | 16.57s | 12.05s | 413 | 1.156 |

The warm windows show the sampler-state effect directly:

| window | wall | trainer f/b | sample | teacher | prefetch wait | sync | consumed tok/s | sample tok/s | commit_len |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 501-509 | 37.83s | 28.30s | 19.65s | 25.43s | 3.06s | 2.99s | 1333 | 772 | 1.686 |
| 502-509 | 35.02s | 25.10s | 19.18s | 25.93s | 3.44s | 2.99s | 1406 | 802 | 1.761 |

Step 501 was still a transition row (`commit_len=1.089`). By step 502, commit
length had climbed to the 1.73-1.80 range, which is why the steadier 502-509
window is the better comparison to earlier q-band + clean runs. This validates
the control and confirms the user's hypothesis: as the student changes,
confidence and sampler output change too.

The live control summary currently says:

- `trainer_ep_dispatch=alltoall`
- `trainer_moe_implementation=triton`
- `trainer_gradient_checkpointing_method=recompute_before_dispatch`
- `trainer_expert_parallel_size=32`
- `gdn_replay_plan_use_stateful_prefix_cache=True`
- `trainer_clean_replay_context=True`
- `prompts_per_step=64`
- `pipeline_chunk_size=32`
- `trainer_coalesce_chunks=1`
- `k_toks=2`
- `max_opd_steps=510`
- `checkpoint_interval_steps=0`
- `checkpoint_save_best=False`

## Where the next agents should start

The immediate next MTP agent should complete the canonical line before any new
throughput experiment launches:

```bash
cd /home/apanda/xorl-mtp-fix-merge-20260613
git status --short --branch
git rev-parse --short HEAD
```

Expected:

```text
## fix/mtp-fix-plus-upstream
08041ed7
```

Tell that agent exactly:

```text
/goal Implement @docs/notes/mtp_canonical_line_completion_runbook_20260614.md
```

That completion runbook merges current `origin/apanda-dev`, ports the B2 layout
test plus the rest of focused review-fix commit `b3876456`, validates CPU + GPU
MTP smoke, and pushes `origin/fix/mtp-fix-plus-upstream`. Only after that lands
should the next throughput agent make live-stack throughput/control writes from
this same checkout/branch, unless the branch has been renamed to `apanda-dev-mtp`.
Before then, throughput work should be limited to offline analysis of existing
logs/payloads that does not touch the shared stack.

Then tell the throughput agent exactly:

```text
/goal Continue MTP throughput from @/home/apanda/xorl-client/experiments/mtp/docs/notes/mtp_amdahl_optimal_throughput_handoff.md. Use the completed canonical branch only; capture current-step f/b payloads and run trainer-only replay before touching full-stack controls. Keep outputs under /shared.
```

Use `/home/apanda/xorl-client/experiments/mtp` only for client-side harness docs/scripts once the closeout push lands.
Use `/home/apanda/xorl-infra/k8s/opd_profile` for Kubernetes manifests/generators, not for engine edits.

Before launching a new throughput run, make the control-plane repo selection explicit:

```bash
export OPD_XORL_REPO=/home/apanda/xorl-mtp-fix-merge-20260613
export OPD_SGLANG_REPO=/home/apanda/xorl-sglang-internal
```

## How to inspect the historical validation run

```bash
cd /home/apanda/xorl-mtp-fix-merge-20260613

python experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py status \
  $(tr '\n' ' ' < experiments/opd_profile/k8s/launch_args_er-opd-q36-mtp-ss-0605c.txt)

tail -n 120 /shared/opd-control/er-opd-q36-mtp-ss-0605c/trainer-head/logs/20260613T223321Z-run.log

python - <<'PY'
import json, statistics
from pathlib import Path
p = Path("/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260613T223322Z-2s1t/artifacts/opd_profile.jsonl")
rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
print("rows", len(rows))
for row in rows:
    print({
        "step": row.get("iter", row.get("step")),
        "wall": round(row.get("iter_time", 0.0), 2),
        "trainer_fb": round(row.get("trainer_forward_backward_s", 0.0), 2),
        "prefetch_wait": round(row.get("opd_async_prefetch_wait_s", 0.0), 2),
        "sample_s": round(row.get("student_sampling_s", 0.0), 2),
        "sample_tok_s": round(row.get("student_sampling_output_tok_per_s", 0.0), 1),
        "commit_len": round(row.get("mtp/commit_len_mean", 0.0), 3),
        "consumed_tok_s": round(row.get("consumed_toks_per_sec_world", 0.0), 1),
    })
PY
```

## How to relaunch the current canonical validation

Use this only if the current run has exited or must be re-rendered. Keep the
supervisor paused while doing diagnostic throughput runs.

```bash
cd /home/apanda/xorl-mtp-fix-merge-20260613

export OPD_XORL_REPO=/home/apanda/xorl-mtp-fix-merge-20260613
export OPD_SGLANG_REPO=/home/apanda/xorl-sglang-internal
export OPD_START_STEP=500
export OPD_LOAD_CHECKPOINT_PATH=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260613T110939Z-2s1t/server_output/weights/default/q36mtp-coderforge-v1-step000500

python experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py \
  write-student-inference-control \
  $(tr '\n' ' ' < experiments/opd_profile/k8s/launch_args_er-opd-q36-mtp-ss-0605c.txt)

python experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py \
  write-trainer-control \
  $(tr '\n' ' ' < experiments/opd_profile/k8s/launch_args_er-opd-q36-mtp-ss-0605c.txt) \
  --max-opd-steps 510 \
  --checkpoint-interval-steps 0 \
  --no-checkpoint-save-best
```

The canonical args file now includes `--k-toks 2`,
`--student-mtp-canonical-q-banding`, `--trainer-clean-replay-context`,
`--trainer-ep-dispatch alltoall`, `--trainer-moe-implementation triton`,
`--prompts-per-step 64`, and `--trainer-coalesce-chunks 1`.

## Evidence table

Warm windows exclude cold step 500 and terminal cleanup rows unless noted.

| experiment | run / artifact | result | decision |
|---|---|---|---|
| k=2 science baseline, no q-band/clean | `q36mtp-20260613T110939Z-2s1t`, steps 508-521 | 62.53s/step, sampling 85.15s, teacher 18.34s, trainer f/b 23.63s, consumed 730 tok/s | Baseline was sampling-bound. |
| q-banding only | `q36mtp-20260613T194614Z-2s1t` | 43.33s/step, prefetch wait 0.19s, sampling 19.01s, trainer f/b 36.00s, consumed 1147 tok/s | Fixed sampling; exposed/inflated trainer f/b. Needs clean-region. |
| q-banding + clean-region | `q36mtp-20260613T200325Z-2s1t`, W&B `tu5ffak6` | 34.62s/step, sampling 18.55s, teacher 25.00s, trainer f/b 24.98s, sync 3.37s, consumed 1460 tok/s | Best completed full-stack result. |
| q-band + clean CUDA-synced profile | `q36mtp-20260613T201834Z-2s1t`, W&B `krofvfsq` | 35.34s/step, trainer f/b 26.96s, forward/loss 9.20s, backward 15.10s, OPD total 0.379s, KL 0.140s | Bottleneck is model f/b and comm/recompute, not KL/top-k. |
| no recompute | `q36mtp-20260613T203140Z-2s1t`, W&B `h57g4v8g` | OOM on first step-500 f/b in stateful GDN suffix path; zero profile rows | Do not retry as-is. |
| q-band + clean capture | `q36mtp-20260613T204443Z-2s1t` | Warm steps 501-506: 32.16s/step, trainer f/b 22.18s, sampler 830 tok/s, commit_len 1.80 | Strong q-band + clean reference window. |
| trainer-only two-payload replay | `fb_capture_qband_clean_step500_20260613T204431Z`, payloads 11 + 8 | Sequential warmed two calls: 18.21s trainer f/b, 1356 valid tok/s, actual MFU proxy 1.59% | Model-side replay harness works; useful for same-payload A/B. |
| combined 64-sample trainer replay | `q36mtp-20260613T211419Z-2s1t` | One combined call warmed ~10.11s, 2440 valid tok/s, actual MFU proxy 2.44% | Positive trainer-call amortization signal, trainer-only only. |
| full pipeline chunk size 64 | `q36mtp-20260613T211900Z-2s1t`, W&B `i3ji92nq` | 40.90s/step, trainer f/b 25.74s, prefetch wait 8.71s, consumed 1172 tok/s | Do not promote; loses prepare parallelism. |
| trainer coalesce-2 with DeepEP | `q36mtp-20260613T215028Z-2s1t`, W&B `75424yvk` | Failed first coalesced f/b with `DeepEP error: timeout (dispatch CPU)` | DeepEP cannot handle this merged k=2 payload as-is. |
| EP8/G-shrink replay | `q36mtp-20260613T220513Z-2s1t` plus replay JSONL | EP8 base-weight replay: 26.41s vs EP32 replay 18.21s; per-GPU MFU proxy improves, wall throughput worsens | Cost-efficiency future branch, not wall-clock fix. |
| EP32 checkpoint into EP8 | `q36mtp-20260613T220042Z-2s1t` | Optimizer state shape mismatch on expert momentum buffer | Needs model-only load or optimizer conversion before EP shrink science run. |
| trainer coalesce-2 with alltoall/triton | `q36mtp-20260613T221209Z-2s1t`, W&B `qm62obaw` | Clean, but warm 59.37s/step, trainer f/b 27.31s, sampler 139 tok/s, commit_len 1.07 | Not promotable; sampler state changed and trainer did not improve. |
| current q-band + clean validation | `q36mtp-20260613T223322Z-2s1t`, W&B `btqzrlm6` | Completed cleanly. Warm 501-509: 37.83s/step, trainer f/b 28.30s, sample 19.65s, teacher 25.43s, sync 2.99s, consumed 1333 tok/s, commit_len 1.686. Steadier 502-509: 35.02s/step, trainer f/b 25.10s, consumed 1406 tok/s, commit_len 1.761. | Confirms current control and sampler confidence drift; do not over-rank against older sampler states. |

## Why OPSD microbench still helps, but only as a checklist

The OPSD low-MFU microbench decomposes visible MFU into:

- denominator choice: executed tokens vs valid target tokens
- dummy-fill / rank occupancy
- above-model server overhead
- in-model small-GEMM and communication shape

Do not copy the OPSD numeric answer to MTP. MTP q-band + clean-region payloads
already execute large GDN replay work per call, and trainer-only replay showed
that some slow live calls warm away. For MTP, the latest profile points to
backward/recompute/FSDP/EP communication rather than OPD KL/top-k or a fixed bad
payload.

## Clean-region replay status

Clean-region replay is enabled by:

```bash
XORL_SINGLESHOT_MTP_CLEAN_REPLAY_CONTEXT=1
```

The proof run showed clean-region and current ordering are top-1
prediction-equivalent for the sampled proof batch:

- current top-1 agreement: 0.500
- clean top-1 agreement: 0.476
- current == clean in 8/8 examples

Caveat: this is an argmax-level proof. It does not prove the full KL/logit
distribution is identical. Treat it as safe enough for throughput validation,
not as a mathematical equivalence proof for all science claims.

The live launcher now has `--trainer-clean-replay-context` and validates that it
is only used with stateful GDN replay prefix cache.

## SGLang dependency

Canonical q-banding previously crashed the sampler with:

```text
ValueError: Invalid verified MTP commit length. verified=2 planned=1 phase=steady
```

The SGLang fix is now committed and pushed on `apanda-dev`:

```text
18c2357de Fix per-request MTP draft verification
```

It adds per-request `mtp_hf_exact_accept_len` verification and unit tests in
`test/srt/zorl/test_mtp_decode_verify.py`. The MTP engine landing should be
paired with this SGLang commit.

## Rules for the next throughput agent

1. Do not promote from cross-state full-stack A/Bs.
   If commit_len or sampler tok/s changed, the workload changed.

2. Do not retry no-recompute without a memory plan.
   It OOMed in stateful GDN suffix replay on the first f/b.

3. Do not promote global `pipeline_chunk_size=64`.
   It reduced prepare parallelism and regressed wall-clock.

4. Do not retry `trainer_coalesce_chunks=2` on DeepEP.
   The merged k=2 payload hit DeepEP dispatch CPU timeout.

5. If revisiting coalescing, start from alltoall/triton and compare same
   captured payloads first.
   The alltoall full-stack coalesce run completed but was not faster.

6. Use trainer-only replay for model-side changes.
   It removes sampler drift and isolates f/b, FSDP, EP dispatch, GDN suffix, and
   compile effects.

7. Preserve q-banding + clean-region + recompute_before_dispatch as the baseline
   until a same-workload run beats it.

8. Keep outputs under `/shared`.
   The OPD generator now sets future `WANDB_DIR=${RUN_DIR}/artifacts/wandb`, and
   local-benchmark k8s rendering defaults to `/shared/xorl-local-benchmark`.

## Next useful work

1. If optimizing throughput next, capture current-step f/b payloads and replay
   them trainer-only before touching full-stack controls.

2. Candidate model-side code targets:
   reduce stateful GDN suffix work, reduce FSDP all-gather/reduce-scatter
   frequency, or find a safe coalescing implementation that does not hit DeepEP
   timeout and does not lose sampler/teacher prepare overlap.

3. Candidate science path:
   if science wants EXP-1 continuation, use q-band + clean only if they accept
   that sampler behavior is now part of the training state. Otherwise render a
   science-control run that preserves their intended sampler setting and do not
   compare its wall-clock to q-band runs as a throughput A/B.

## Consolidation notes

The MTP engine path is no longer a PR #371 landing flow. PR #371 is closed, and the 52-commit research history is
backed up at `origin/backup/apanda-dev-mtp-research-20260614`. The live-proven canonical engine line is
`origin/fix/mtp-fix-plus-upstream@08041ed7` until it is completed and possibly renamed to `apanda-dev-mtp`.

Branch identity:

| branch | current role | start new work? |
|---|---|---|
| `origin/fix/mtp-fix-plus-upstream@08041ed7` | canonical live-proven line with emit-window fix | yes, after completing apanda-dev merge + tests |
| `origin/apanda-dev-mtp@3a6a6a02` | wrong durable-name line; B2/current-apanda-dev but missing emit-window fix | no |
| `origin/backup/apanda-dev-mtp-research-20260614@f61ebad3` | historical research backup | no |

Harness/infra closeout status as of 2026-06-14:

- `xorl-client` branch `opd-battery-consolidation` carries ZORL, OPD, and Wordle harness commits; `experiments/mtp`
  is still an untracked local import backed up under `/shared` until the final path-scoped MTP client commit lands.
- `xorl-infra` branch `opd-battery-consolidation` carries the OPD/Wordle/MTP infra consolidation and is intended to
  supersede/fold into infra PR #1.

For the next throughput generation, treat this file as the handoff for measured throughput decisions, but treat
`/home/apanda/xorl-mtp-fix-merge-20260613` on `fix/mtp-fix-plus-upstream` as the executable source of truth.

---

## SESSION 2026-06-14 (throughput/perf agent) — trainer-only replay findings on canonical engine /home/apanda/xorl-mtp @ apanda-dev-mtp

Method: isolated trainer-only `/forward_backward` replay of captured k=2 step-500 q-band+clean payloads, on
dedicated stacks (1/2/4-node), ZERO interference with the live science baseline. Harness + raw data:
`/shared/opd-control/er-opd-q36-mtp-perf-replay/{launch_replay.sh,harvest.sh,AB_RESULTS.md,SESSION_LOG_throughput_20260614.md,TOPOLOGY_FINDING_and_PROPOSAL.md}`.

GOTCHA fixed: the k8s generator's singleshot defaults are k=4/conf0.6/static1280→1536 — WRONG for this stack
(traces + cfg-of-record are k=2/conf0.3/static1280). Any replay/microbench MUST pass --k-toks 2 --conf-threshold
0.3 --prompt-len 512 --max-new-tokens 256 or it measures a different workload.

Faithful k=2 EP8-1node baseline warm f/b = 13.4s (fwd 3.3s + bwd 8.4s [sync-cuda confirmed] + ~1.3s prep/comm gap).

1. ALL trainer-only EXECUTION levers are WITHIN NOISE (~±0.5s on 13.4s) at k=2: defer_grad_sync+reshard,
   capture_align 512, their combo, clean-region ON/OFF, k=2-vs-4. NONE promotable. (The handoff's clean-region
   "~26x FB win" is the STATEFUL-GDN prefix cache — ON in both arms — NOT clean-region; clean-region is
   throughput-neutral, keep ON for replay semantics.) pack9216 (bigger micro-batch) OOMs at EP8-1node.
2. The GDN stateful prefix cache already executes only ~1% of logical GDN replay tokens, so GDN is NOT the
   wall-clock bottleneck. Backward (recompute_before_dispatch recomputes attn+norm+router) dominates (~65%).
3. TOPOLOGY (the only real lever): EP size is a pure parallelism layout — EP8 vs EP32 is MATHEMATICALLY
   IDENTICAL training (same routing/GEMMs/loss), only the MoE all-to-all placement differs. ep_intranode=True
   (engine default) makes EP groups consecutive ranks => EP8 on N nodes = N intra-node (NVLink) groups; EP32
   = 1 cross-node (IB) group. Same-payload 4-node: EP8x4 8.17s vs EP32x4 8.99s (EP8 ~10% faster, mfu 1.9% vs
   1.72%). Modest at this 32-prompt payload (small alltoall volume); grows at production volume (235B precedent:
   EP8-intranode ~75s/step vs EP64 cross-node ~35min/step). EP8 also FITS at 4-node (1/8 experts + 1/4 batch).
4. STRONG-SCALING: for fixed work, fewer GPUs = higher MFU: EP8-1node 4.1%, EP8x2 3.4%, EP8x4 1.9%. So the
   trainer is most MFU-efficient at FEWER GPUs. 10% MFU at 4-node needs much more work/GPU (big batch).
5. AMDAHL: live full stack is teacher/prepare-bound on light steps (prefetch_wait>0) and fb-bound on heavy
   steps (fb 35s, prefetch_wait 0). Net recommendation: SHRINK the trainer (EP8, 1-2 node, 2-4x better MFU)
   and reallocate freed GPUs to samplers/teachers (the prepare bottleneck the other 2 agents own).
6. big256 (256-prompt, EP8x4): warm f/b 26.8s => mfu_actual 3.44% (3.7x live EP32 0.94%); pack9216 no better (3.35%); +defer_grad_sync 25.9s/3.56%. 10% NOT reached (natural batch too small for 32 GPUs).
   4-node MFU at a realistic large batch.

## SESSION 2026-06-14 continuation — engine prep microbench and replay-plan patch

Scope: offline CPU-only trainer prep investigation on `/home/apanda/xorl-mtp @ apanda-dev-mtp`; no live-stack
control writes and no config promotion. Science launched k=4 separately from current promoted config v1; this
engine patch still needs same-workload trainer replay at k=4 before it can be treated as part of any promoted
launch line.

Artifacts:

- harness: `/shared/opd-control/perf-replay-common/prep_microbench.py`
- result note: `/shared/opd-control/perf-replay-common/PREP_MICROBENCH_RESULTS_20260614.md`
- before: `/shared/opd-control/perf-replay-common/prep_microbench_fb0_full32_before.json`
- trace-reuse after: `/shared/opd-control/perf-replay-common/prep_microbench_fb0_full32_after_trace_reuse_repeat3.json`
- trace-reuse + plan-scatter after:
  `/shared/opd-control/perf-replay-common/prep_microbench_fb0_full32_after_dense_scatter_repeat3.json`
- trace-reuse + plan-scatter + assignment/scalar pack after:
  `/shared/opd-control/perf-replay-common/prep_microbench_fb_request_0_after_assignment_scalar_pack_repeat3.json`
- trace-reuse + plan materialization pack after:
  `/shared/opd-control/perf-replay-common/prep_microbench_fb_request_0_after_repeatinterleave_rows_repeat3.json`
- trace-reuse + full plan/schedule materialization pack after:
  `/shared/opd-control/perf-replay-common/prep_microbench_fb_request_0_after_schedule_repeatinterleave_rows_repeat3.json`
- trace-reuse + fast CPU int-list conversion after:
  `/shared/opd-control/perf-replay-common/prep_microbench_fb_request_0_after_fast_long_buffer_repeat3.json`
- trace-reuse + native fill slicing after:
  `/shared/opd-control/perf-replay-common/prep_microbench_fb_request_0_after_slice_native_fill_repeat3.json`
- cross-payload after:
  `/shared/opd-control/perf-replay-common/prep_microbench_fb_request_{0,1,2}_after_dense_scatter_repeat3.json`
- negative flat-fill result: `/shared/opd-control/perf-replay-common/prep_microbench_fb0_full32_after_flatfill.json`

Finding: `prepare_singleshot_mtp_opd_batch` was normalizing/parsing native trace rows twice for native-trace replay:
once to decide the native replay path and again inside `_prepare_singleshot_mtp_native_trace_replay_opd_batch`.
Patch in `/home/apanda/xorl-mtp/src/xorl/mtp/singleshot.py` passes the already-normalized
`native_trace_steps_by_row` into the native replay preparer. Follow-up patch in the same file optimizes
`build_rollout_replay_linear_plan`: it uses prefix lookup for committed-context visibility when source indices are
sorted, materializes dense plan rows/cols/valid bits by scattering from the already-required
`packed_rows`/`packed_cols`, builds output assignment tensors from one packed assignment tensor instead of four small
conversions, and generates `packed_rows` from compact per-sequence row IDs with `repeat_interleave` instead of a huge
repeated Python list. Native replay metadata now also packs scalar MTP metadata into one tensor and stores scalar
views. The stateful GDN replay schedule now uses the same compact row-ID materialization for `ctx_packed_rows` and
`suffix_packed_rows`. Large CPU integer-list conversions now use a guarded `array('q')` + `torch.frombuffer(...).clone()`
fast path; small lists and non-CPU devices retain the regular `torch.tensor` path. The latest native replay fill patch
replaces per-token scalar writes with per-row slice assignments and small row tensors in the native replay assembly
loop.

Same captured k=2 step-500 payload `fb_request_0`, 32 micro-batches, repeat=3:

| path | total prep mean | prep ms / micro-batch | plan ms / micro-batch |
|---|---:|---:|---:|
| before patch | 1.5625s | 48.83 | 29.09 |
| trace reuse patch | 1.3765s | 43.02 | 25.85 |
| trace reuse + plan scatter | 1.1586s | 36.21 | 18.81 |
| trace reuse + plan scatter + assignment/scalar pack | 1.1029s | 34.47 | 17.66 |
| trace reuse + plan materialization pack | 0.9716s | 30.36 | 13.08 |
| trace reuse + full plan/schedule materialization pack | 0.9420s | 29.44 | 12.00 |
| trace reuse + fast CPU int-list conversion | 0.7981s | 24.94 | 8.12 |
| trace reuse + native fill slicing | 0.7031s | 21.97 | 8.77 |

Measured offline prep reduction for the full patch stack: 55.0% versus the original path on this payload
(~26.9 ms per micro-batch), 48.9% lower than trace reuse alone, and 39.3% lower than plan scatter alone. Sequential
cross-payload repeat=3 after the native fill slicing patch: `fb_request_0` 21.97 ms/mb, `fb_request_1` 28.19 ms/mb,
`fb_request_2` 30.25 ms/mb. Relative to v6, this is +11.9% / +23.8% / +13.6% on fb0/fb1/fb2 respectively. This is a
code-level reduction in trainer prep overhead, not a training-math change.

Validation:

```bash
PYTHONPATH=/home/apanda/xorl-mtp/src pytest \
  tests/mtp/test_singleshot.py tests/ops/test_linear_attention_singleshot_mask.py -q

PYTHONPATH=/home/apanda/xorl-mtp/src ruff check \
  src/xorl/mtp/singleshot.py tests/mtp/test_singleshot.py

PYTHONPATH=/home/apanda/xorl-mtp/src ruff format --check \
  src/xorl/mtp/singleshot.py tests/mtp/test_singleshot.py
```

Observed: `65 passed, 2 skipped`; ruff check and format check clean.

Negative result: a prototype flat-fill rewrite of `build_rollout_replay_linear_plan` regressed full-payload prep to
2.0684s / 64.64 ms per micro-batch by trading small tensor allocations for huge Python index-list construction. Do
not pursue that implementation shape.

Current live-state read after Kubernetes auth came back: `kubectl auth can-i get pods -n apanda` returns `yes`
under context `apanda-admin@research-common-h100`. Use `kubectl get pods -n apanda -l stack=er-opd-q36-mtp-ss-0605c`
for the whole stack; there is no shared `app=er-opd-q36-mtp-ss-0605c` label. Namespace `apanda` has the 0605c
dispatch, student SGLang, teacher, and all four trainer pods running with zero restarts. The k=4 live run is
`q36mtp-20260614T192418Z-2s1t`, launched on the current promoted EP32 config v1. Read-only profile tail shows steps
1600-1605 completed with `sync_success=true`; step1605 had `commit_len_mean=2.3166`, `effective_k=3.4845`,
`trainer_forward_backward_s=49.66`, and `opd_singleshot_mtp_mfu_actual=1.2168%`. This is live science state, not a
throughput promotion. No fresh k=4 `/forward_backward` replay payload JSONs are present yet, so the patch above is
still validated only on captured k=2 payloads. Next throughput gate remains: capture/replay k=4 same-workload trainer
payloads before treating this code patch, EP8, or defer as part of a promoted k=4 line.

Isolated GPU replay status after auth recovery: the trainer-only pod
`er-opd-q36-mtp-perf-replay-trainer-head` is still `Pending` due to `Insufficient nvidia.com/gpu`, so no
`replay_results.jsonl` exists yet. If it schedules, harvest with:

```bash
bash /shared/opd-control/er-opd-q36-mtp-perf-replay/harvest.sh prep_patch_v4_ep8_1node 9
```

The queued tag name says `prep_patch_v4_ep8_1node`, but the pod has not started and the mounted worktree now includes
the v7 prep patch. Check trainer log provenance before labeling the replay result.

### 2026-06-14 continuation: compact linear-plan prep candidate v8

Added an opt-in engine flag:

```bash
XORL_SINGLESHOT_MTP_COMPACT_LINEAR_PLAN=1
```

When clean-region replay yields a stateful GDN replay schedule, this skips dense fallback
`sequence_rows` / `sequence_cols` / `sequence_valid` materialization. Default behavior is unchanged when the env is
unset. This is aligned with the live stateful-GDN path: the stateful executor consumes the stateful schedule and packed
row/column tensors, not the dense fallback matrices. GDN packed fallback is still supported with dense tensors absent
and has a focused regression test.

Clean-region ON offline A/B, same captured k=2 step-500 payloads, repeat=3:

| payload | v7 clean ON ms/mb | v8 compact-plan ms/mb | delta |
|---|---:|---:|---:|
| `fb_request_0` | 24.23 | 21.12 | 12.9% lower |
| `fb_request_1` | 30.69 | 24.76 | 19.3% lower |
| `fb_request_2` | 29.13 | 26.70 | 8.3% lower |

Validation: `67 passed, 2 skipped` on `tests/mtp/test_singleshot.py` and
`tests/ops/test_linear_attention_singleshot_mask.py`; ruff check/format clean on
`src/xorl/mtp/singleshot.py`, `src/xorl/ops/linear_attention/layers/gated_deltanet.py`, and the touched tests.

This is not promoted. The isolated trainer-only GPU replay was re-rendered as
`prep_patch_v8_compact_ep8_1node` with `COMPACT_PLAN=1`; generated trainer control exported
`XORL_SINGLESHOT_MTP_COMPACT_LINEAR_PLAN=1`. Kubernetes dry-run/apply passed, the pod later scheduled, and the replay
completed:

- result JSONL:
  `/shared/opd-control/er-opd-q36-mtp-perf-replay/payloads_sub3/replay_results.jsonl`
- harvest command:
  `bash /shared/opd-control/er-opd-q36-mtp-perf-replay/harvest.sh prep_patch_v8_compact_ep8_1node 9`
- first-step compile rows:
  `157.697/49.790/48.840s` excluded
- warm replay mean (`step>=1`, n=6):
  `13.20s`, per-payload `13.07/13.11/13.43s`
- comparable `baseline_k2`:
  `13.43s`, per-payload `13.03/13.07/14.19s`

Conclusion: the compact-plan CPU-prep improvement is real offline, but it does not materially move same-workload GPU
f/b. Treat v8 as guarded opt-in code cleanup, not a throughput lever or live config promotion. The isolated replay pod
was deleted after harvest to release its 8 GPUs.

### 2026-06-14 continuation: opt-in forward_backward defrag-skip candidate

Prepared a narrow candidate in `/home/apanda/xorl-mtp` but did not promote it:

```bash
XORL_SKIP_FORWARD_BACKWARD_DEFRAG=1
```

Default behavior is unchanged. When unset, `ModelRunner.forward_backward()` still runs the current outer
`gc.collect(); torch.cuda.empty_cache()` at request entry and the second outer `torch.cuda.empty_cache()` before main
f/b. When set, those two outer defrag calls are skipped; the existing `XORL_FB_PER_CALL_DEFRAG=1` fallback remains
separate. This is a throughput candidate only, not a science/config change.

Validation so far:

- `PYTHONPATH=/home/apanda/xorl-mtp/src pytest tests/server/runner/test_model_runner_profile.py -q` => 7 passed.
- Combined focused tests over MTP/GDN/ModelRunner => 74 passed, 2 skipped.
- Ruff check/format clean on touched engine/test files.
- Render-only stack verified `SKIP_FB_DEFRAG=1` bakes `export XORL_SKIP_FORWARD_BACKWARD_DEFRAG=1` into trainer
  `run.sh`.

GPU proof is now bracket-validated on isolated trainer-only replay. Because the original `default` node pool had no
single 8-GPU node free, the isolated replay pod only was retargeted to `node-group=nccl`; the live science stack was
not control-written. All measurements below use the same `nccl` pod/node pool, EP8/alltoall/triton,
`COMPACT_PLAN=0`, `DEEPEP_SMS=48`, same captured k=2 payloads, and warm rows `step>=1` (n=6 per run).

| run | artifact | skip defrag | warm f/b mean | warm roundtrip mean | outer defrag mean |
|---|---|---:|---:|---:|---:|
| pre-control | `replay_results_control_current_alltoall_triton_ep8_1node_nccl_20260614T211810Z.jsonl` | 0 | 13.318941s | 13.496675s | 0.405727s |
| candidate | `replay_results_candidate_skip_fb_defrag_alltoall_triton_ep8_1node_nccl_20260614T212719Z.jsonl` | 1 | 12.661684s | 12.832331s | 0.000000s |
| post-control | `replay_results_control_repeat_after_skip_defrag_alltoall_triton_ep8_1node_nccl_20260614T213210Z.jsonl` | 0 | 13.105937s | 13.281719s | 0.369772s |

Candidate deltas: -4.93% vs pre-control, -3.39% vs post-control, and -4.17% vs the average of both controls on
`trainer_forward_backward_s`; roundtrip is -4.16% vs average control. The measured outer defrag bucket is removed
entirely in the candidate.

Conclusion: `XORL_SKIP_FORWARD_BACKWARD_DEFRAG=1` is a real, same-workload trainer f/b win and training math is
unchanged. It is still not live-promoted. The remaining gate is allocator stability over a longer live-like run: no
OOM, no fragmentation drift, and no monotonic memory rise across many forward_backward calls.

Follow-up instrumentation added at 21:10Z so that replay/profile JSON can report the defrag bucket directly:

- server metrics: `forward_backward_entry_defrag_s`, `forward_backward_pre_main_defrag_s`,
  `forward_backward_outer_defrag_s`, `forward_backward_skip_defrag`
- replay/profile aliases: `trainer_fb_entry_defrag_s`, `trainer_fb_pre_main_defrag_s`,
  `trainer_fb_outer_defrag_s`, `trainer_fb_skip_defrag`

Validation after instrumentation: ModelRunner focused test => 7 passed; combined focused MTP/GDN/ModelRunner tests =>
74 passed, 2 skipped; ruff check/format clean on touched files; `scripts/opd/run_opd_pipeline.py --help` imports
cleanly.

21:14Z propagation follow-up: the defrag timing fields are now explicitly passed through the orchestrator/API response
builders, so they will actually reach replay/profile JSON. Added CPU tests for request-processor and API pass-through.
Validation: focused propagation tests => 2 passed; full touched server/API/request-processor/ModelRunner files =>
42 passed; MTP/GDN focused suite => 67 passed, 2 skipped; ruff check/format clean. The isolated replay pod was deleted
after the GPU A/B to release its 8 GPUs.

22:01Z stability follow-up: allocator-stability gate PASSED for `XORL_SKIP_FORWARD_BACKWARD_DEFRAG=1`, still not
live-promoted. Ran isolated replay tag
`candidate_skip_fb_defrag_memtelemetry_stability20_alltoall_triton_ep8_1node_nccl_20260614T2200Z` with
`SKIP_FB_DEFRAG=1`, `MEMORY_TELEMETRY=1`, `NUM_STEPS=20`, EP8/alltoall/triton, `COMPACT_PLAN=0`, `DEEPEP_SMS=48`,
same captured k=2 payloads. Result: 60/60 forward_backward calls, controller rc=0, replay pod deleted after
completion. Artifact:
`/shared/opd-control/er-opd-q36-mtp-perf-replay/payloads_sub3/replay_results_candidate_skip_fb_defrag_memtelemetry_stability20_alltoall_triton_ep8_1node_nccl_20260614T2200Z.jsonl`.

Warm rows `step>=1` (n=57): 12.884064s f/b mean / 13.030954s roundtrip mean, defrag buckets 0, skip flag 1. Against
the earlier pre/post-control average: f/b -0.328375s (-2.49%), roundtrip -0.358243s (-2.68%); with `step>=2`, f/b
-0.384744s (-2.90%). Memory telemetry showed no allocator drift: warm start/end allocated flat at 18531.792 MB, end
reserved 18902-18906 MB, post-backward allocated flat at 35049.547 MB, and same-payload peak memory flat/bounded.
Activation boundary: this is a validated opt-in engine candidate, not the current live config. Enable only in an
announced control-write/relaunch window and monitor first live steps for OOM or memory drift.

22:18Z checkpointing follow-up: `no_recompute` is rejected as-is for the captured k=2 EP8 replay workload. Ran
isolated tag `candidate_no_recompute_skipdefrag_memtelemetry_ep8_1node_nccl_20260614T220542Z` with
`CKPT=no_recompute`, `SKIP_FB_DEFRAG=1`, `MEMORY_TELEMETRY=1`, `NUM_STEPS=5`, EP8/alltoall/triton,
`COMPACT_PLAN=0`, `DEEPEP_SMS=48`. The trainer became healthy and issued the first replay request, but the first
`/forward_backward` future failed with `504: Forward-backward timeout after 600.0s`; zero rows were produced and no
`replay_results.jsonl` exists. The isolated pod was deleted after failure and the live science stack remained Running.
Do not retry `no_recompute` without a separate memory/compile plan; keep `recompute_before_dispatch` as the
checkpointing baseline.

22:28Z MoE execution follow-up: `deepep/quack` with `DEEPEP_SMS=24` is rejected as a throughput lever for the current
captured k=2 EP8 replay. Ran isolated tag
`candidate_quack_deepep24_skipdefrag_ep8_1node_nccl_20260614T221915Z` with `CKPT=recompute_before_dispatch`,
`SKIP_FB_DEFRAG=1`, `MEMORY_TELEMETRY=1`, `NUM_STEPS=3`, `COMPACT_PLAN=0`. Artifact:
`/shared/opd-control/er-opd-q36-mtp-perf-replay/payloads_sub3/replay_results_candidate_quack_deepep24_skipdefrag_ep8_1node_nccl_20260614T221915Z.jsonl`.
The replay completed 9/9 rows and the pod was deleted; live science stack remained Running. Warm rows `step>=1`
(n=6): 13.020460s f/b mean / 13.204389s roundtrip mean, versus the long skip-defrag alltoall/triton EP8 baseline
at 12.884064s / 13.030954s (n=57), i.e. +1.059% f/b and +1.331% roundtrip slower. Cold rows were expensive
(79.614694s roundtrip mean; first row 142.135075s). Keep alltoall/triton as the model execution baseline for this
workload. Any future DeepEP/Quack speed win would still need K3 gating before promotion because it changes MoE
execution/dispatch.

23:05Z MoE split follow-up: Quack is rejected, but DeepEP dispatch with the existing Triton MoE backend is now a
speed-positive candidate. All runs used the same isolated trainer-only k=2 EP8 replay, `recompute_before_dispatch`,
`SKIP_FB_DEFRAG=1`, memory telemetry, and no live-stack control write.

Short split matrix, warm `step>=1`: alltoall/triton long baseline 12.884064s f/b / 13.030954s roundtrip;
alltoall/quack 13.028582s / 13.214780s (+1.122% / +1.411%); deepep/quack 13.020460s / 13.204389s (+1.059% /
+1.331%); deepep/triton 12.639671s / 12.817754s (-1.897% / -1.636%). The short deepep/triton smoke artifact is
`/shared/opd-control/er-opd-q36-mtp-perf-replay/payloads_sub3/replay_results_candidate_triton_deepep24_skipdefrag_ep8_1node_nccl_20260614T2237Z.jsonl`.

Long deepep/triton stability artifact:
`/shared/opd-control/er-opd-q36-mtp-perf-replay/payloads_sub3/replay_results_candidate_triton_deepep24_skipdefrag_memtelemetry_stability20_ep8_1node_nccl_20260614T2245Z.jsonl`.
It completed 60/60 rows, controller rc=0, pod deleted. Warm `step>=1` (n=57): 12.576904s f/b / 12.720916s
roundtrip vs long alltoall/triton baseline 12.884064s / 13.030954s, i.e. -2.384% f/b and -2.379% roundtrip;
`step>=2` remains -2.375% / -2.370%. Memory telemetry is clean: start/end allocated flat at 18531.792 MB,
post-backward allocated flat at 35049.547 MB, reserved bounded 18902-18908 MB, same-payload peak memory bounded.
Cold row remains expensive (first row 156.803s roundtrip), so this is a steady-state f/b candidate.

Status: K3 FAILED / NOT PROMOTABLE AS-IS. Because `ep_dispatch=deepep` changes MoE dispatch, the speed-positive
`deepep/triton` SMS24 replay was correctness-gated before promotion. The K3 launcher was repaired for the
research-common-h100 cluster (`team: turbo`, configurable node/pool selection, CUDA UUID-to-index handling), a
Qwen3.6 CoderForge pilot static trace bundle was regenerated, and the exact SMS24 config was replayed.

Trace bundle:
`/shared/opd-control/er-opd-q36-mtp-perf-replay/k3/q36-coderforge-pilot-opd32-static-traces-20260614T233149Z.json`
(32 pilot CoderForge OPD traces, prompt suffix 2048 tokens, dataset completion 128 tokens, top-logprobs 5).
Result:
`/shared/opd-control/er-opd-q36-mtp-perf-replay/k3/triton_deepep_ep8_20260614T233439Z/k3_result.json`.
The run covered 32 prompts / 4096 tokens and failed the configured threshold: token-level K3 mean 336.039091, p95
0.800540, max 1361095.501391. Worst token was `train.parquet:rg0:row29`, position 53, token_id 85, where SGLang had
logprob -0.000626725 and XoRL had logprob -14.124438.

Keep alltoall/triton as the promoted model execution baseline. Do not promote `deepep/triton` SMS24 unless a later,
separate candidate fixes the divergence and passes the static K3 gate.

### 2026-06-15 continuation: k=4/files64 same-payload gate

Scope: isolated replay/CPU only after science reclaimed the main trainer. No further main-trainer control writes
after the clean capture.

Clean-region infra correction:

- Root-cause: the files64 k=4 relaunch had baked `XORL_SINGLESHOT_MTP_CLEAN_REPLAY_CONTEXT=''`, so GDN replay fell
  back to non-stateful rows despite `OPD_GDN_REPLAY_PLAN_USE_STATEFUL_PREFIX_CACHE=true`.
- Fix in infra checkout:
  `/home/apanda/xorl-infra-opd-wordle-pr1-20260614/k8s/opd_profile/q36_singleshot_reprogrammable_slots.py`
  now exposes `--trainer-clean-replay-context`, and the canonical args file includes that flag. The flag bakes
  `XORL_SINGLESHOT_MTP_CLEAN_REPLAY_CONTEXT=1` into trainer `run.sh`.
- Clean capture: run `q36mtp-20260614T234817Z-2s1t`, resume step 1900, capped at one OPD step, saved under
  `/shared/opd-control/er-opd-q36-mtp-perf-replay/k4_files64_capture_clean_20260614T234233Z/`
  (`fb_request_0/1.json`, `teacher_hidden_0/1.safetensors`). Science then reclaimed the trainer for the files64
  epoch-target run.

Same-payload k=4/files64 replay, EP8/alltoall/triton, clean-region on, GDN stateful on, warm `step>=1`:

| arm | artifact suffix | skip defrag | warm n | f/b mean | outer defrag mean |
|---|---|---:|---:|---:|---:|
| control first | `k4files64_control_alltoall_triton_ep8_1node_nccl_20260614T2357Z` | 0 | 4 | 9.0947s | 0.3673s |
| skip-defrag | `k4files64_skipfbdefrag_alltoall_triton_ep8_1node_nccl_20260615T0003Z` | 1 | 4 | 8.1663s | 0.0000s |
| control repeat | `k4files64_control_repeat_after_skipfbdefrag_alltoall_triton_ep8_1node_nccl_20260615T0006Z` | 0 | 4 | 8.5475s | 0.3935s |

Conclusion: `XORL_SKIP_FORWARD_BACKWARD_DEFRAG=1` is now validated beyond k=2: -0.3813s/call versus the repeat
control (-4.46% f/b time, +4.67% speed), and the delta matches the removed outer defrag bucket. It remains an
opt-in candidate, not a live promotion, until a coordinated science/infra relaunch window.

CPU prep attribution on the same clean payloads:

| payload | artifact | prep mean | plan ms / batch | plan fraction |
|---|---|---:|---:|---:|
| 0 | `prep_microbench_payload0_clean_20260615T0012Z.json` | 0.9985s | 16.91 | 54.2% |
| 1 | `prep_microbench_payload1_clean_20260615T0012Z.json` | 0.9956s | 14.89 | 47.9% |

Plan-builder optimization is a real but bounded target: at most roughly 0.5s/payload if eliminated. Do not promote
the earlier compact-plan mixed non-clean result without a fresh clean+compact replay gate.

DeepEP/Triton SMS24 status is unchanged: speed-positive in replay, failed static K3, not promotable.

### 2026-06-15 continuation: compact linear-plan GPU gate

The existing guarded `XORL_SINGLESHOT_MTP_COMPACT_LINEAR_PLAN=1` path was retested on the same clean k=4/files64
capture because clean CPU prep showed replay-plan build is still a bounded cost.

CPU-only sequential microbench, clean env, repeat=3:

| payload | control prep mean | compact prep mean | control plan ms/batch | compact plan ms/batch |
|---|---:|---:|---:|---:|
| 0 | 0.6860s | 0.5893s | 9.27 | 6.18 |
| 1 | 0.7364s | 0.6253s | 9.87 | 6.61 |

GPU replay rejected it as a trainer f/b lever. Same clean k=4/files64 payloads, EP8/alltoall/triton,
`SKIP_FB_DEFRAG=1`, warm `step>=1`:

| arm | artifact suffix | warm n | f/b mean | payload means |
|---|---|---:|---:|---:|
| compact + skip-defrag | `k4files64_compact_skipfbdefrag_alltoall_triton_ep8_1node_nccl_20260615T0019Z` | 4 | 8.2656s | 8.1325s / 8.3986s |
| skip-defrag order-control | `k4files64_skipfbdefrag_control_after_compact_alltoall_triton_ep8_1node_nccl_20260615T0025Z` | 4 | 8.1491s | 8.1090s / 8.1892s |
| prior skip-defrag control | `k4files64_skipfbdefrag_alltoall_triton_ep8_1node_nccl_20260615T0003Z` | 4 | 8.1663s | 8.1560s / 8.1766s |

Decision: do not promote compact linear plan. It is a CPU-prep cleanup, not a measured f/b/MFU improvement on this
workload. The validated opt-in f/b lever remains skip-defrag; DeepEP/Triton remains K3-failed.

### 2026-06-15 continuation: optional native-trace parser cleanup

Engine change in `/home/apanda/xorl-mtp/src/xorl/mtp/singleshot.py`: native MTP trace strings now use
`orjson.loads` when available, with stdlib `json.loads` fallback and identical malformed-trace behavior. This is a
CPU-prep cleanup, not a config knob.

Validation:

- py_compile for `src/xorl/mtp/singleshot.py`
- full `tests/mtp/test_singleshot.py -q`: 43 passed
- ruff check and format-check on `src/xorl/mtp/singleshot.py` and `tests/mtp/test_singleshot.py`

CPU-only clean k=4/files64 prep, compact off, repeat=3:

| payload | previous sequential control | optional-orjson |
|---|---:|---:|
| 0 | 0.6860s | 0.5995s |
| 1 | 0.7364s | 0.6606s |

GPU replay did not turn this into an f/b/MFU win. Same clean k=4/files64 payloads, EP8/alltoall/triton,
`SKIP_FB_DEFRAG=1`, compact off:

| arm | artifact suffix | warm n | f/b mean | payload means |
|---|---|---:|---:|---:|
| optional-orjson + skip-defrag | `k4files64_orjson_skipfbdefrag_alltoall_triton_ep8_1node_nccl_20260615T0032Z` | 10 | 8.5278s | 8.6058s / 8.4498s |
| prior skip-defrag order-control | `k4files64_skipfbdefrag_control_after_compact_alltoall_triton_ep8_1node_nccl_20260615T0025Z` | 4 | 8.1491s | 8.1090s / 8.1892s |

Decision: keep as safe CPU-prep hygiene, but do not promote or count as an MFU win. The validated opt-in f/b lever
remains `XORL_SKIP_FORWARD_BACKWARD_DEFRAG=1`; compact-plan remains rejected; DeepEP/Triton remains K3-failed.
