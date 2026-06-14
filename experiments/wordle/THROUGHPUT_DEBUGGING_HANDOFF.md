# Wordle OPSD throughput debugging — handoff (2026-06-13)

**Audience**: an agent picking up "why is our OPSD-Wordle trainer at 0.13% MFU and
how do we fix it." The user's (correct) instinct: 0.13% makes no sense; it's a
diagnosable pathology, not a floor. This doc is the honest ledger of what was
measured, what was tried, what did/didn't work, and - most importantly - the
**next diagnostic that can end the guessing.**

Stack: Qwen3.6-35B-A3B (40 layers, 3:1 GDN-linear-attn:full-attn, 256 experts/8
active, hidden 2048, vocab 248320), full-weight server-mode OPSD, EP=8 on 8×H100,
think-contract rollouts (~3k think tokens/turn), full-vocab reverse-KL teacher.

---
## Consolidation Status — read before running more jobs

`CONSOLIDATION_HANDOFF.md` is currently the operational owner for this worktree.
Part 1 has been executed:

- This checkout is now on branch `exp/opsd-wordle`.
- Commit `746c71fd` (`Add OPSD microbatch diagnostics`) contains only the two engine
  files requested by the consolidation handoff:
  `src/xorl/server/orchestrator/request_processor.py` and
  `src/xorl/server/runner/runner_dispatcher.py`.
- Those committed engine edits are the Wordle-side diagnostic half that must be
  unioned with the sibling `xorl-apanda-dev-opd-port` `runner_dispatcher.py`
  minimal-dummy-batch edits when B1 is assembled.
- The remaining throughput harness/docs/config changes in this checkout are **not**
  part of that engine commit. Treat them as local experiment handoff material until
  the post-B1 relocation decides what moves to `xorl-client` or `xorl-infra`.

Do not launch more jobs that write under `experiments/zorl/results` from this
checkout. The checkout already contains about 1.9T of run artifacts that Part 2 says
must be moved to `/shared`. If a post-consolidation agent needs to resume a replay
before relocation is complete, set `RESULT_ROOT=/shared/apanda/opsd-wordle-mfu-replays`
or another `/shared/apanda/...` path explicitly.

Non-result to avoid misreading: a no-checkpoint direct tensor replay attempt
`opsd-wordle-q36-fullft-8g-hbrhw` wrote only an `init` row under
`/shared/apanda/opsd-wordle-mfu-replays/20260613T223713Z-opsd-wordle-q36-fullft-8g-hbrhw-zcpm5-wordle-r16-asymmetric_opsd/`
and was deleted during consolidation focus. It produced no warmup or measured f/b
row and must not be counted as evidence for or against checkpointing.

---
## TL;DR — no-checkpoint was neutral; replay f/b before launching full OPSD jobs

**Do not keep chasing the one-line `enable_packing: true` hypothesis in the current
server path without a steady-state measurement. It was smoke-tested on 2026-06-13
and did not improve the warmup chunk.**

- Packed smoke job: `opsd-wordle-q36-fullft-8g-mlw76`, run dir
  `experiments/zorl/results/opsd_wordle_native_baseline/20260613T165902Z-opsd-wordle-q36-fullft-8g-mlw76-lm78r-wordle-r16-asymmetric_opsd`,
  W&B run `ofq3rbeu`, `RB=8`, `chunk=16`, `wordle_reasoning_token_weight=1.0`,
  `sample_eval_interval=0`, `teacher_sample_interval=0`.
- Server-side proof it was actually packed:
  `server.log` has `Using packing config: seq_len=4096, enabled=True` and
  `RequestProcessor initialized ... packing=enabled`.
- First packed train chunk: `rows=38`, `tokens=69749`, `forward_backward_requests=5`,
  `forward_backward_time_s=434.04`, or **6.22 s / 1k train tokens**. Important:
  comparable checkpointed baseline first chunks are also around **5.7-6.0 s / 1k**,
  while later full `train` events settle near **1.55 s / 1k**. So this first-chunk
  smoke is **not a fair steady-state rejection by itself**.
- Comparable unpacked public-reframe RB8/chunk16 runs are much faster:
  `20260613T133109Z-...PUBLICREFRAME-chunk16-S1` and
  `20260613T133111Z-...PUBLICREFRAME-chunk16-S0` both have `packing=disabled`,
  8 train events each, and about **1.55 s / 1k train tokens**.
- Why the hypothesis is not promoted: the packed smoke did not improve the warmup
  chunk, and observed requests packed 8 samples into only 6-8 batches at ~54-71%
  bin utilization, then spent 78-130s per forward/backward request. Re-reading the
  code matters here: `sample_packing_sequence_len=4096` is a bin capacity in
  `server/orchestrator/packing.py`; finalized packed batches are padded only to
  `pad_to_multiple_of`, not blindly to 4096 in this path. So do **not** claim the
  packing test failed solely because of static 4096 padding without a tensor-shape
  capture proving that later collation reintroduced it.
- No-checkpoint first smoke job: `opsd-wordle-q36-fullft-8g-rlp8p`, run dir
  `experiments/zorl/results/opsd_wordle_native_baseline/20260613T172339Z-opsd-wordle-q36-fullft-8g-rlp8p-5xgqj-wordle-r16-asymmetric_opsd`,
  W&B run `j6mhf8np`. It fit, and first train chunk was `rows=36`,
  `tokens=66912`, `forward_backward_requests=5`, `forward_backward_time_s=395.85`,
  or **5.92 s / 1k train tokens**. As with packing, this first-chunk result is
  warmup-contaminated and must be compared against baseline first chunks, not later
  steady `train` averages.
- Corrected no-checkpoint full-step job: `opsd-wordle-q36-fullft-8g-94jcp`, run dir
  `experiments/zorl/results/opsd_wordle_native_baseline/20260613T174727Z-opsd-wordle-q36-fullft-8g-94jcp-pc65h-wordle-r16-asymmetric_opsd`,
  W&B run `xi2ran8j`, `steps=1`, `sample_eval_interval=0`,
  `teacher_sample_interval=0`. It completed one full train event:
  `opd_pipeline_forward_backward_s=1045.61`, `train_tokens=285507`,
  `valid_tokens=284396`, `forward_backward_requests=20`, or **3.66 s / 1k train
  tokens**. Checkpointed public-reframe first train events were **3.70** and
  **3.74 s / 1k**, so checkpointing OFF is **neutral on first full step**, not a
  measured throughput win. It also uses much more memory, with observed f/b samples
  near ~81GB on one rank.
- A direct warmed bare-tensor no-checkpoint A/B is still unresolved. The only replay
  attempt after the consolidation handoff was `hbrhw`, and it was stopped after the
  init row to avoid continuing perf work during consolidation. Resume it only from
  `/shared`, not with the manifest default result root.

**Current best direction:** keep the production baseline at `enable_packing: false`
and gradient checkpointing ON. Checkpointing OFF did not improve the first full step.
The CUDA-synchronized profile now says the next target is not OPD KL, HTTP, or
client-side orchestration; it is the transformer train fwd/bwd path inside
`/forward_backward`. Important nuance: `opd_profile_backward_compute_s` is the
server timer around `local_loss_sum.backward()` and is checkpoint-recompute
contaminated. The active configs use `gradient_checkpointing_method:
recompute_before_dispatch`, so this is not a pure backward-kernel bucket and also
not the full-layer recompute case. Do not hunt for a "95% backward-only" magic
kernel; profile or bisect the whole trunk fwd/bwd path.

**Do not keep launching full OPSD jobs for every f/b hypothesis.** There is now a
static replay path:

- Capture once from a real train chunk with
  `--dump-forward-backward-replay forward_backward_replay.jsonl --skip-initial-eval`.
  This writes API-shaped `/forward_backward` requests plus copied teacher-cache files
  beside the artifact.
- Replay many times with `FB_REPLAY_ARTIFACT=<artifact>` in the canonical k8s job.
  Replay mode starts only the XORL training server, skips sampler/SMG readiness,
  skips SGLang generation, skips teacher-cache materialization, and does no weight
  sync. It posts the captured f/b requests and runs `optim_step` at `lr=0` by default
  to clear gradients without changing weights.
- For the next localization pass, run the same replay with
  `FB_REPLAY_MICROBATCH_DIAGNOSTIC_SUMMARY_ONLY=1` and `FB_REPLAY_MAX_REQUESTS=1`.
  That writes per-rank JSON summaries after server collation/packing and
  `_shard_and_slice_batches`, then returns a fake loss without executing model f/b
  or `optim_step`. Set `FB_REPLAY_MICROBATCH_DIAGNOSTIC_TENSORS=1` only when you
  really need `.pt` tensor payloads; summary JSON is enough to inspect rank
  occupancy, padding, position segments, and valid/input token counts.
- For a lower-level split, capture tensor payloads with
  `FB_REPLAY_MICROBATCH_DIAGNOSTIC_TENSORS=1`, then run
  `FB_TENSOR_REPLAY_DIR=<microbatch_diagnostics_dir>` in the canonical k8s job.
  That launches `experiments/zorl/standalone/replay_microbatch_tensors.py` under
  `torchrun`, loads each rank's post-slice `.pt` tensors, and calls
  `ModelRunner.forward_backward()` directly without starting the HTTP training
  server, sampler, SMG, teacher-cache materializer, or weight-sync path.
- Use the replay loop for EP=8 vs EP=1, checkpointing, packing, request shape,
  profiler, and kernel-level changes. Only return to full OPSD when the replay loop
  shows a real f/b win.

2026-06-13 status: the replay artifact exists; the first EP A/B, the post-slice
microbatch shape diagnostic, and the bare tensor replay are measured.
Capture job `opsd-wordle-q36-fullft-8g-r8shd` wrote
`experiments/zorl/results/opsd_wordle_native_baseline/20260613T202921Z-opsd-wordle-q36-fullft-8g-r8shd-8tbkw-wordle-r16-asymmetric_opsd/forward_backward_replay.jsonl`
with 5 captured `/forward_backward` requests (`datums` = 8, 8, 8, 8, 4) and a
copied teacher cache. Replaying the first request with no sampler, no SGLang
generation, no teacher-cache materialization, and no weight sync shows a huge
first-call warmup/autotune tax, then a still-bad warmed server f/b path:

| config | job | pass | f/b wall | valid tokens | valid tok/s | ms / 1k valid tok | rough MFU, N_active=3B |
|---|---|---:|---:|---:|---:|---:|---:|
| EP=8 | `opsd-wordle-q36-fullft-8g-tr75v` | cold measured | 291.54s | 13,986 | 47.97 | 20,845 | 0.0109% |
| EP=1 | `opsd-wordle-q36-fullft-8g-rwnxt` | cold measured | 259.68s | 13,986 | 53.86 | 18,567 | 0.0123% |
| EP=8 | `opsd-q36-fb-warm-ep8-5frvq` | warmup | 291.69s | 13,986 | 47.95 | 20,856 | 0.0109% |
| EP=8 | `opsd-q36-fb-warm-ep8-5frvq` | warmed measured | 7.17s | 13,986 | 1,949.55 | 513 | 0.4435% |
| EP=1 | `opsd-q36-fb-warm-ep1-f4njj` | warmup | 261.05s | 13,986 | 53.58 | 18,665 | 0.0122% |
| EP=1 | `opsd-q36-fb-warm-ep1-f4njj` | warmed measured | 13.93s | 13,986 | 1,004.21 | 996 | 0.2285% |

Interpretation: do **not** compare cold replay rows to steady-state MFU; the first
replayed request is dominated by warmup/autotune/JIT-style effects. After one warmup,
EP=8 is about 1.94x faster than EP=1 on the exact captured request, so EP=1 is not the
fix. The warmed API replay is still far below the MTP bare-f/b 1.85% floor at the
small-token point and far below the 12-14% token-per-rank knee. Server logs for the
warm replay also show `pack≈0.001s`, API/ZMQ wait ≈ backend time, so the remaining
time is inside the rank backend f/b path, not JSON/HTTP/request packing.

Summary-only microbatch diagnostic job `opsd-wordle-q36-fullft-8g-q56kz`
replayed the same first captured request with
`FB_REPLAY_MICROBATCH_DIAGNOSTIC_SUMMARY_ONLY=1`, so it skipped model f/b and
`optim_step` after `_shard_and_slice_batches`. Output:
`experiments/zorl/results/opsd_wordle_native_baseline/20260613T211636Z-opsd-wordle-q36-fullft-8g-q56kz-4c4bc-wordle-r16-asymmetric_opsd/microbatch_diagnostics/`.

| rank | real microbatches | `input_ids` shape | input positions | valid target tokens | position segments |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | `[1, 479]` | 479 | 185 | 1 |
| 1 | 1 | `[1, 931]` | 931 | 597 | 1 |
| 2 | 1 | `[1, 2104]` | 2,104 | 1,721 | 1 |
| 3 | 1 | `[1, 3455]` | 3,455 | 3,072 | 1 |
| 4 | 1 | `[1, 813]` | 813 | 520 | 1 |
| 5 | 1 | `[1, 3394]` | 3,394 | 3,072 | 1 |
| 6 | 1 | `[1, 2073]` | 2,073 | 1,747 | 1 |
| 7 | 1 | `[1, 3402]` | 3,402 | 3,072 | 1 |

Totals: 8 real microbatches, 0 dummy microbatches, 16,651 input positions, and
13,986 valid target tokens. This refutes the "static 4096 padding" and
"some ranks dummy-filled" explanations for this captured unpacked request. The
real issue visible in this artifact is that every rank runs one tiny,
imbalanced single-sample forward/backward: average input positions/rank is only
2,081 and the largest rank has 3,455, far below the MTP note's ~16k
tokens/rank knee.

Tensor-payload dump job `opsd-wordle-q36-fullft-8g-mzh9t` replayed the same first
captured request with `FB_REPLAY_MICROBATCH_DIAGNOSTIC_TENSORS=1`, writing 8 JSON
summaries and 8 rank-local `.pt` files to
`experiments/zorl/results/opsd_wordle_native_baseline/20260613T212614Z-opsd-wordle-q36-fullft-8g-mzh9t-mw6vs-wordle-r16-asymmetric_opsd/microbatch_diagnostics/`.
Bare tensor replay job `opsd-wordle-q36-fullft-8g-ngzrz` then loaded those tensors
directly through `experiments/zorl/standalone/replay_microbatch_tensors.py` and
wrote
`experiments/zorl/results/opsd_wordle_native_baseline/20260613T213048Z-opsd-wordle-q36-fullft-8g-ngzrz-mzshx-wordle-r16-asymmetric_opsd/bare_tensor_replay_metrics.jsonl`.

| path | wall | valid tok/s | input tok/s | rough valid MFU | rough input MFU | forward bucket | backward bucket |
|---|---:|---:|---:|---:|---:|---:|---:|
| bare tensor cold warmup | 291.81s | 47.93 | 57.06 | 0.0109% | 0.0130% | 165.97s | 124.29s |
| bare tensor warmed measured | 8.03s | 1,741.57 | 2,073.42 | 0.3962% | 0.4717% | 0.91s | 5.64s |

Interpretation: this is the decisive server-tax split. Warmed API replay was
`7.17s` and roughly `0.4435%` valid-token MFU on the same 13,986 valid tokens;
warmed bare tensor replay is `8.03s` and `0.3962%` valid-token MFU
(`0.4717%` input-token MFU). Those are close enough that the huge remaining tax is
not HTTP, ZMQ, sampler orchestration, teacher-cache creation, request packing, or
weight sync. The cold `~292s` behavior also reproduces in bare tensor replay, so
the first-call 0.01% MFU row is model-runner/autotune/kernel warmup, not server
machinery. The bug is now localized to `ModelRunner`/transformer fwd+bwd on these
tiny exact rank-local shapes: MoE/DeepEP communication, FSDP all-gather or
reduce-scatter, activation-checkpoint recompute, attention/GDN-linear-attn kernels,
or their interaction with one short sample per rank.

Coalesced tensor replay support was added next (`FB_TENSOR_REPLAY_TARGET_INPUT_TOKENS`
and `FB_TENSOR_REPLAY_COALESCE_REPEAT`). Job `opsd-wordle-q36-fullft-8g-q2tdt`
targeted the MTP note's ~16k input-token/rank knee using the same rank-local tensor
payloads, repeated with `position_ids` resets so attention still sees independent
segments. Its init row was exactly the intended shape: `global_input_tokens=135267`,
`dumped_global_valid_tokens=101965`, `min_local_input_tokens=16584`,
`max_local_input_tokens=17275`, and repeat factors `5-35` depending on original
rank length. It did **not** produce a speed row: rank 7 OOMed during the first
coalesced warmup in Triton autotune for GDN `l2norm_bwd`
(`src/xorl/ops/linear_attention/ops/gated_delta_rule/chunk.py:263` ->
`src/xorl/ops/linear_attention/modules/l2norm.py:213`, `RuntimeError: Triton Error
[CUDA]: out of memory`). The job was deleted after failure. Interpretation: the
16k/rank coalescing target is still the right question, but this exact target does
not fit the current memory/autotune envelope. The next coalesced replay should try
an intermediate target such as 8k/rank, or change memory/autotune settings before
retrying 16k.

---
## Measured facts (trust these)

Per-step at the canonical RB8+chunk16, 8×H100, warm:
- `step_time_s` ~116-150 (varies with rollout think-length); `opd_pipeline_forward_backward_s` (fb) **~91% of step**.
- `opd_pipeline_prepare_wait_s` ~20-30s — rollout+teacher-cache are mostly hidden
  behind fb. **We are trainer/fb-bound**, NOT teacher-bound (opposite of the sibling
  prefill-time-compute stack — see their `~/xorl-apanda-dev-opd-port/docs/notes/amdahl_allocation_20260613.md`).
- **MFU ≈ 0.13%** (6·N_active·tokens / fb / 8 / 989e12, N_active=3e9). fb-ms/1k-tok ≈ 2100.
- **GPU power ~120-160W of ~700W TDP at 75-94% util** → memory-bandwidth/latency-bound
  tiny kernels, NOT compute-bound. (Sampled via `kubectl exec <trainer> -- nvidia-smi
  --query-gpu=utilization.gpu,power.draw`.)
- **The OPD loss (lm_head + full-vocab reverse-KL) is NOT the bottleneck.** The
  CUDA-synchronized one-step profile `opsd-wordle-q36-fullft-8g-sgjgj`
  (`20260613T190309Z-...sgjgj-tklgl...`, W&B `12e18ji8`) used
  `--opd-profile-timings --opd-profile-sync-cuda` and summed profile durations
  across 20 f/b calls. Train f/b was `1015.77s` for `267,288` train tokens
  (**3.80 s/1k**). The measured split was `opd_profile_backward_compute_s=969.91`
  (**95.5% of f/b**), `opd_profile_forward_compute_s=19.49` (**1.9%**),
  `opd_profile_total_ms=6086.76` for the OPD loss region (**6.09s, 0.6%**), and
  `opd_profile_kl_compute_ms=5396.42` (**5.40s, 0.53%**). So cheaper/top-k KL
  would save almost nothing; the expensive block is the transformer training path.
  Because the configs use `gradient_checkpointing_method: recompute_before_dispatch`,
  the `backward_compute_s` bucket includes recompute of the pre-dispatch path and
  backward/comm work, not just backward kernels. It also does **not** imply
  `recompute_full_layer` is hiding the entire trunk forward in backward; that is not
  the active checkpointing method in these runs.
- The same run shows huge warmup decay inside a single step. Train chunk f/b times
  were `416.40s`, `252.70s`, `204.85s`, `141.81s` for `72,991`, `67,048`,
  `67,979`, `59,270` tokens: **5.70 → 3.77 → 3.01 → 2.39 s/1k**. The first two
  f/b calls were `125.32s` and `101.45s`; warmed full-token calls later reached
  `24-30s`. Do not compare first chunks to steady-state train averages.
- `fb_requests = train_rows / request_batch_size` (each is one `/api/v1/forward_backward`
  HTTP call → one engine fwd/bwd). Multi-turn Wordle can emit more train rows than
  sampled games; the `sgjgj` profile had 138 train rows at RB8 → 20 f/b calls.
- Code-level batching nuance: with `enable_packing: false`,
  `server/orchestrator/packing.py` creates **one micro-batch per sample**. Increasing
  `request_batch_size` therefore gives each rank more sequential one-sample
  micro-batches inside a request; it does not automatically create denser GEMMs. With
  `enable_packing: true`, samples are concatenated into fewer packed micro-batches,
  but `runner_dispatcher.py` slices micro-batches across the stage-local ranks. If a
  request packs into fewer than 8 micro-batches on this 8-GPU run, some ranks get
  dummy/no-valid-token work. A real packing fix must coalesce enough real tokens per
  rank while preserving rank occupancy; plain "bigger RB" or plain stock packing does
  not guarantee that.
- Current Wordle OPSD configs are **not** forwarding mostly no-gradient tokens. The
  profiled EP8 train event had `valid_tokens=266,187` and `train_tokens=267,288`
  (`valid/train=99.6%`); the EP1 train event had `valid_tokens=263,287` and
  `train_tokens=264,437` (`valid/train=99.6%`). Cutting think tokens can still
  reduce wall-clock work or change the recipe, but the current pathology is not
  "96% forwarded tokens carry no gradient."
- Sibling MTP microbench note
  `~/xorl-mtp-singleshot-port-20260602/docs/notes/opsd_low_mfu_microbench_20260613.md`
  is the right corrected framing: a **bare** local `xorl.cli.train` fwd/bwd on
  synthetic static data at OPSD-ish shape measured 1.85% MFU at 1,024 tokens/rank
  and rose to roughly 12-14% MFU around 16k-64k tokens/rank. That does not
  contradict our 0.1% OPSD MFU, but it also does not prove a single "14x pipeline
  tax." The updated note splits the visible MFU into stack-specific denominator,
  dummy-fill/rank-occupancy, and above-model overhead terms. On current
  OPSD-Wordle the denominator term is **not** the issue (`valid/train=99.6%`), and
  cutting pause/think tokens is not an MFU lever because masked and unmasked tokens
  ride the same GEMMs. Warmed API replay keeps the server f/b path and improves to
  ~0.44% MFU, which is still about 4x below the MTP note's smallest bare-f/b point
  and about 27x below the larger-token knee. The exact post-pack/post-slice tensor
  replay now measures the same order of slowness outside the server path:
  `8.03s`, `0.3962%` valid-token MFU, and `0.4717%` input-token MFU. The
  summary-only diagnostic also shows the first captured OPSD request is not padded
  to 4096 and does not leave ranks dummy-filled; it is one small, imbalanced sample
  per rank (479-3,455 input positions/rank). So the above-model server tax is no
  longer the primary explanation; the remaining pathology is the model-runner
  fwd/bwd path on tiny exact shapes. The note's "decisive open experiment:
  bare-TENSOR replay" is now resolved for OPSD-Wordle in this checkout: bare tensor
  replay is slow too. Its "pack/coalesce to ~16k/rank" recommendation remains the
  right lever to test, but it does **not** currently explain the whole OPSD-Wordle
  gap. A direct 16k/rank coalesced replay OOMed during GDN backward autotune. An
  8k/rank coalesced replay fit and improved warmed input-token MFU from `0.4717%`
  to `0.7060%`, but that is only about a 1.5x gain and is still roughly 9x below
  the sibling synthetic 8k/rank point (`6.6%`). EP=1 is also now directly tested in
  the bare-tensor harness: it is slower than EP8 on the same exact tensors
  (`14.06s` vs `8.03s`, `0.2695%` vs `0.4717%` input-token MFU). The remaining
  pathology is not just "too few tokens per rank" or "EP all-to-all overhead."
  Quack MoE was also tested on the same exact tensors and gave only a modest
  warmed improvement (`7.49s`, `0.5061%` input-token MFU) over Triton
  (`8.03s`, `0.4717%`). The issue is the current model-runner real-shape path, not
  a single MoE implementation switch.

## What was TRIED and the RESULT

| lever | result | verdict |
|---|---|---|
| **EP-dedup** (`ff262108`, merged) | valid/train tokens 7.89→0.99 (removed 8× redundant FLOPs at EP=8). Wall-NEUTRAL at 8-GPU single-EP-group (the dup ran in parallel across ranks). | KEEP (correct, gradient-identical); not a wall win here |
| **chunk size** (`OPD_PIPELINE_CHUNK_SIZE`) | 2→16 looked like ~2× but was confounded (vs no-dedup baseline + autotune). 16 vs 32 token-normalized identical. | chunk16 fine; not a real lever |
| **request_batch_size sweep** | RB8 (8 calls) MFU **0.13%** > RB64+chunk16 (4 calls) 0.10% > chunk64+RB64 (1 call) 0.054%. Fewer/bigger fb calls **monotonically WORSE** (token-normalized). | RB8 is best; **disproved the FSDP-amortization hypothesis for our regime** |
| **FSDP-amortization hypothesis** (from sibling `ada302bc`: "1.25MB chunk cap → 96 calls, each pays FSDP param-movement; raise cap → fewer calls → faster") | Their fix is for SHORT completions (mnt=64). On our long-think rows, bigger calls are worse (padding, above). Does NOT transfer. | dead end for us |
| **KL backend** | `streaming` optimal; `torch_compile` WORSE (recompiles on variable-length think rows). | keep streaming |
| **quack vs triton MoE** | Direct bare-tensor Quack job `2mnrn` replayed the exact `mzh9t` tensors with the `qwen3_6_35b_a3b_opsd_wordle_fullft_ep8_warmstart_sft48_quack.yaml` config. Cold warmup was `292.08s` (`0.0109%` valid MFU, `0.0130%` input MFU). Warmed measured row was `7.49s`, `1,868` valid tok/s, `2,224` input tok/s, `0.4251%` valid MFU, and `0.5061%` input MFU. Baseline Triton exact bare replay was `8.03s`, `0.3962%` valid MFU, `0.4717%` input MFU. | **Small win only (~7% input-MFU)**; useful default, not the root cause of <1% MFU |
| **`ce_mode: quack_linear`** (user-requested) | set; didn't crash; likely inert on the reverse-KL path (it's a CE-path mode; requires token-count÷8). | harmless; unverified benefit |
| **`enable_packing: true`** | 2026-06-13 packed smoke `mlw76`: first train chunk `434.04s` fb for `69,749` train tokens = `6.22 s/1k`. Baseline first chunks are also ~`5.7-6.0 s/1k`; later train averages are ~`1.55 s/1k`. | **INCONCLUSIVE / no warmup win**; needs full `train` event before a fair verdict |
| **grad-checkpointing OFF** | 2026-06-13 full-step no-ckpt run `94jcp` completed: `1045.61s` fb for `285,507` train tokens = `3.66 s/1k` on its first train event. Checkpointed first train events were `3.70`/`3.74 s/1k`, and checkpointed later average is ~`1.55 s/1k`. Peak observed f/b memory was ~81GB on one rank. | **NEUTRAL on first full step; do not promote** |
| **CUDA-synced OPD/f-b profile** | 2026-06-13 profile run `sgjgj` completed with summed profile aggregation: f/b `1015.77s`, train tokens `267,288`, 20 f/b requests, `3.80 s/1k`. Synchronized phase totals: backward bucket `969.91s` (95.5%), forward bucket `19.49s` (1.9%), OPD loss region `6.09s`, KL `5.40s` (0.53%). Active configs use `recompute_before_dispatch`, so the backward bucket includes checkpoint recompute and comm, not only backward kernels. | **Transformer fwd/bwd path is the bottleneck**; split server path vs bare tensor replay before chasing kernel families |
| **static API f/b replay harness** | Added `train_opsd_baseline.py --dump-forward-backward-replay`, `--skip-initial-eval`, and `standalone/replay_forward_backward_artifact.py`; canonical k8s manifest now supports `FB_REPLAY_ARTIFACT=...`. Capture job `r8shd` produced 5 real OPSD f/b requests. Cold first request is ~260-292s and should be treated as warmup/autotune/JIT tax. After one warmup, EP8 measured `7.17s`/13,986 valid tok (`0.4435%` MFU at N_active=3B); EP1 measured `13.93s` (`0.2285%`). Summary-only shape job `q56kz` shows the first request gives each rank one real sample, no dummy rank, no 4096 padding, and only 479-3,455 input positions/rank. | **Use warmed replay as the server f/b benchmark**; EP1 is slower and first-call replay is not steady-state evidence |
| **bare tensor f/b replay** | Tensor dump job `mzh9t` wrote rank-local `.pt` payloads after `_shard_and_slice_batches`. Direct `ModelRunner.forward_backward()` replay job `ngzrz` skipped the HTTP training server entirely. Cold warmup was `291.81s` (`0.0109%` valid MFU, `0.0130%` input MFU); warmed measured was `8.03s` for 13,986 valid / 16,651 input tokens (`0.3962%` valid MFU, `0.4717%` input MFU). | **No big above-model server tax once warm**; the remaining issue is inside model-runner fwd/bwd on tiny per-rank shapes |
| **bare tensor EP=1 A/B** | Job `qzz6n` replayed the same exact tensor dump with `expert_parallel_size: 1`. Shape stayed identical (`16,651` input / `13,986` valid tokens; ranks `479-3,455` input tokens). Cold warmup was `262.94s` (`0.0121%` valid MFU, `0.0144%` input MFU). Warmed measured row was `14.06s`, `995` valid tok/s, `1,185` input tok/s, `0.2264%` valid MFU, and `0.2695%` input MFU. | **EP=1 is worse in the clean harness too**; do not promote the "remove EP all-to-all" theory for this captured shape |
| **coalesced bare tensor replay to ~16k input/rank** | Added `replay_microbatch_tensors.py --coalesce-target-input-tokens`. Job `q2tdt` repeated each rank's exact tensor segment to `16.6k-17.3k` input tokens/rank (`135,267` global input, `101,965` valid) with position resets. It OOMed before a measured row during Triton autotune in GDN `l2norm_bwd` on the first warmup. | **16k/rank is not currently a drop-in fit**; reduce autotune/memory pressure before retrying |
| **coalesced bare tensor replay to ~8k input/rank** | Job `f9jrm` repeated each rank's exact tensor segment to `8.3k-10.4k` input tokens/rank (`73,405` global input, `55,943` valid). Cold warmup was `309.94s` (`0.0411%` valid MFU, `0.0539%` input MFU). Warmed measured row was `23.65s`, `2,365` valid tok/s, `3,103` input tok/s, `0.5381%` valid MFU, and `0.7060%` input MFU. | **Coalescing helps only modestly**; it is still far below the MTP synthetic 8k/rank `6.6%` point, so the remaining issue is deeper than token-count/rank alone |

## The honest gaps (what still needs measurement)

1. **No kernel-level profiler has split the synchronized transformer train path.** We
   now know the slow region is transformer fwd/bwd plus recompute/comm, but the kernel-level split
   between attention, MoE dispatch/expert GEMMs, GDN-linear-attn, FSDP/all-gather or
   reduce-scatter, activation checkpoint recompute, and padding is still UNKNOWN.
   Run `nsys profile` or torch profiler on one warmed replayed unpacked f/b call.
   Capture a cold call only if you explicitly want the warmup pathology.
2. **The 8k coalesced replay did not close the MTP gap.** It proves larger real
   per-rank work helps, but only from `0.4717%` to `0.7060%` input-token MFU. The
   same rough token/rank regime is `6.6%` in the sibling bare synthetic sweep, so
   there is still a real model-runner/topology/shape difference to localize.
3. **Direct warmed no-checkpoint replay is still not closed.** First full OPSD step
   was neutral and memory-heavy, but the clean bare-tensor A/B did not finish before
   consolidation took priority. If someone resumes it, use the exact `mzh9t` tensor
   dump and write results under `/shared`, not this checkout.
4. **Packed steady-state is still not fully closed.** Packed first chunk did not beat
   unpacked first chunks, but there is no packed full `train` event. This is a lower
   priority than backward profiling unless someone wants to close the config ledger.
5. **No kernel profile has been run on the bare tensor harness yet.** The decisive
   server-tax split is done: API replay and bare tensor replay are both slow once
   warm. The missing follow-up is a kernel or torch-profiler split on the direct
   `replay_microbatch_tensors.py` path so profiler overhead and HTTP/API machinery
   cannot be blamed.

## Next-step diagnostic plan (in priority order)

1. **Profile the warmed bare tensor replay first.** Set
   `FB_TENSOR_REPLAY_DIR=<mzh9t_microbatch_diagnostics>`,
   `FB_TENSOR_REPLAY_WARMUP_REPEATS=1`, and `FB_TENSOR_REPLAY_REPEAT=1`, then wrap
   `experiments/zorl/standalone/replay_microbatch_tensors.py` with `nsys` or torch
   profiler. This is the cleanest path because it times direct
   `ModelRunner.forward_backward()` on the exact tensors and excludes HTTP,
   request processing, sampler, teacher-cache materialization, and weight sync.
   Focus on MoE/DeepEP dispatch+combine, expert GEMMs, FSDP all-gather or
   reduce-scatter, activation-checkpoint recompute, full-attention/GDN-linear-attn,
   and whether one tiny sample per rank creates launch/collective latency collapse.
2. **Profile the warmed 8k coalesced replay as the comparison point.** Reuse
   `FB_TENSOR_REPLAY_TARGET_INPUT_TOKENS=8192`. This run fits but only reaches
   `0.7060%` input-token MFU, so its profiler split should show whether the extra
   work is going into useful GEMMs or into GDN/Triton/autotune, DeepEP collectives,
   FSDP param motion, checkpoint recompute, or ragged-shape overhead.
3. **Retry 16k only after reducing the OOM pressure.** The 16k/rank target was
   constructed correctly but OOMed in GDN `l2norm_bwd` Triton autotune before timing.
   Try a profiler-informed memory/autotune reduction first; otherwise the 16k result
   will keep failing before it tests the MTP knee.
4. **Run the pure-training-to-OPSD bisection with the replay harnesses.** The
   synchronized phase timer already killed OPD KL/loss as a meaningful target, and
   the bare replay killed a large above-model server-tax theory. A controlled
   bisection from the known-fast pure-training recipe to this exact tiny-shape
   server model path should locate the real cliff.
5. **Fix packing/coalescing only with rank occupancy in mind.** Plain
   `enable_packing: true` can reduce the number of micro-batches below the 8-way
   batch-slice count, leaving some ranks on dummy/no-valid-token work. The useful
   experiment is not just "pack more"; it is "pack enough real tokens per rank while
   avoiding static padding and preserving at least one real packed micro-batch per
   active rank."
6. **Do not promote EP=1 for this workload.** A normal EP=1 smoke
   `opsd-wordle-q36-fullft-8g-6d66x` completed one train event with
   `opd_pipeline_forward_backward_s=608.44` for `263,287` valid tokens
   (`2.31 s/1k`), but it still paid full OPSD machinery and was first-step/warmup
   contaminated. On exact warmed API replay, EP1 was slower than EP8:
   `13.93s` vs `7.17s` for the same `13,986` valid tokens. Direct bare-tensor
   replay confirms the same conclusion without HTTP/server machinery: EP1 was
   `14.06s` / `0.2695%` input-token MFU, versus EP8 `8.03s` / `0.4717%`.
   Keep the config file for reproducibility, but stop treating EP1 as a likely
   throughput fix.
7. **Optional config loose end: run packed to one full `train` event.** First chunks
   are warmup-contaminated; do not compare them to later train averages. If this is
   not clearly better than checkpointed public-reframe first-step metrics, drop stock
   packing too.
8. **Specific bisection seed**: run the measured-winner
   pure-training recipe (`skills/xorl-throughput-tuner/benchmarks/qwen3_6_35b_a3b/
   configs/qwen3_6_35b_a3b_8k_4node_ep8_mbs8_fullrecompute_deepep.yaml`) on synthetic
   8k packed data → confirm ~16% reproduces, then change ONE axis at a time toward our
   setup (packing off → variable-length → small batch → server-mode → full-vocab-KL)
   and watch which collapses MFU. That isolates the culprit definitively.

## Reference (inherited via the OPD-port merge — branch `apanda-dev-prefill-time-compute`)

- Sibling Amdahl doc: `~/xorl-apanda-dev-opd-port/docs/notes/amdahl_allocation_20260613.md`
  (their stack was teacher-bound; capture-fix gave 8-9×; the `ada302bc` FSDP/chunk-cap
  finding is theirs — does NOT transfer to our long-think regime, see above).
- Throughput-tuner skill: `skills/xorl-throughput-tuner/` (measured 35b pure-training
  recipe + `references/xorl-parallelism.md` + `collect_xorl_metrics.py`). Defaults
  `moe_implementation: quack`, `ep_dispatch: deepep` for pure training.
- Canonical Wordle run manifest: `experiments/zorl/k8s/qwen3-6-35b-a3b-opsd-wordle-think-canonical.yaml`
  (RB8+chunk16 - the measured-best tested request shape; keep packing disabled and
  checkpointing enabled). It now accepts `OPD_PROFILE_TIMINGS=1` and
  `OPD_PROFILE_SYNC_CUDA=1` env overrides. It also accepts
  `DUMP_FORWARD_BACKWARD_REPLAY=...`, `SKIP_INITIAL_EVAL=1`, and
  `FB_REPLAY_ARTIFACT=...`. For direct tensor replay it accepts
  `FB_TENSOR_REPLAY_DIR=...`, `FB_TENSOR_REPLAY_ARTIFACT=...`,
  `FB_TENSOR_REPLAY_WARMUP_REPEATS=...`, `FB_TENSOR_REPLAY_REPEAT=...`, and
  `FB_TENSOR_REPLAY_ACTIVE_PARAMS=...`. It also supports
  `FB_TENSOR_REPLAY_COALESCE_REPEAT=...` and
  `FB_TENSOR_REPLAY_TARGET_INPUT_TOKENS=...` for repeated exact-tensor
  coalescing.
- Static API replay client: `experiments/zorl/standalone/replay_forward_backward_artifact.py`.
  It replays captured `/forward_backward` payloads with fresh `seq_id`s and optional
  `lr=0` `optim_step`s, writing `replay_metrics.jsonl`. Existing artifact:
  `experiments/zorl/results/opsd_wordle_native_baseline/20260613T202921Z-opsd-wordle-q36-fullft-8g-r8shd-8tbkw-wordle-r16-asymmetric_opsd/forward_backward_replay.jsonl`.
- Static bare tensor replay client: `experiments/zorl/standalone/replay_microbatch_tensors.py`.
  It loads `microbatch_*_rankNNNNN.pt` payloads captured after
  `_shard_and_slice_batches`, instantiates `ModelRunner`, and calls
  `forward_backward()` directly under `torchrun`. Existing tensor dump:
  `experiments/zorl/results/opsd_wordle_native_baseline/20260613T212614Z-opsd-wordle-q36-fullft-8g-mzh9t-mw6vs-wordle-r16-asymmetric_opsd/microbatch_diagnostics/`.
- Microbatch diagnostic knobs for replay mode:
  `FB_REPLAY_MICROBATCH_DIAGNOSTIC_SUMMARY_ONLY=1`,
  `FB_REPLAY_MICROBATCH_DIAGNOSTIC_DIR=<dir>`, and
  `FB_REPLAY_MICROBATCH_DIAGNOSTIC_TENSORS=1`. Summary-only mode returns before
  model f/b and disables `optim_step`, so it is for shape/rank-occupancy evidence,
  not throughput.
- Request-processor timing fields are now returned in f/b responses:
  `executor_pack_s`, `executor_backend_s`, `executor_build_output_s`,
  `executor_total_s`, `executor_batches`, and `executor_samples`. The replay client
  aggregates arbitrary numeric metrics, so future replay jobs should expose whether
  the server-side tax is in packing/output handling or the rank backend.
- Profiling knobs: `--opd-profile-timings` emits OPD loss-region `opd_profile_*_ms`;
  `--opd-profile-sync-cuda` makes fwd/bwd phase timers synchronize CUDA. In
  `summarize_responses`, `opd_profile_*_{s,ms}` durations are summed across
  responses, not token-weighted.
- GPU-power check: `kubectl exec -n apanda <trainer-pod> -c trainer -- nvidia-smi
  --query-gpu=index,utilization.gpu,power.draw --format=csv`.

## Post-Consolidation Resume

After B1 lands and the run data relocation is complete:

1. Merge the post-B1 `origin/apanda-dev` into `exp/opsd-wordle` and verify
   `git diff origin/apanda-dev -- src/xorl` is empty or contains only intentionally
   retained Wordle glue.
2. Preserve the existing evidence artifacts when relocating the checkout results to
   `/shared`; update the paths in this runbook if the old
   `experiments/zorl/results/opsd_wordle_native_baseline/...` paths disappear.
3. Move Wordle experiment scripts/docs/runbooks according to `MIGRATION-MANIFEST.md`:
   Wordle harness/docs to `xorl-client/experiments/wordle/`, k8s/configs to
   `xorl-infra`.
4. If the throughput track resumes before a full migration, launch new replay jobs
   with an explicit shared result root, e.g.:

```bash
kubectl set env --local \
  -f experiments/zorl/k8s/qwen3-6-35b-a3b-opsd-wordle-think-canonical.yaml \
  -o yaml \
  RESULT_ROOT=/shared/apanda/opsd-wordle-mfu-replays \
  CONFIG_PATH=/workspace/home/xorl-opsd-wordle-apanda-dev-run-20260607/experiments/zorl/configs/qwen3_6_35b_a3b_opsd_wordle_fullft_ep8_warmstart_sft48_nockpt.yaml \
  FB_TENSOR_REPLAY_DIR=/workspace/home/xorl-opsd-wordle-apanda-dev-run-20260607/experiments/zorl/results/opsd_wordle_native_baseline/20260613T212614Z-opsd-wordle-q36-fullft-8g-mzh9t-mw6vs-wordle-r16-asymmetric_opsd/microbatch_diagnostics \
  FB_TENSOR_REPLAY_ARTIFACT=/workspace/home/xorl-opsd-wordle-apanda-dev-run-20260607/experiments/zorl/results/opsd_wordle_native_baseline/20260613T202921Z-opsd-wordle-q36-fullft-8g-r8shd-8tbkw-wordle-r16-asymmetric_opsd/forward_backward_replay.jsonl \
  FB_TENSOR_REPLAY_WARMUP_REPEATS=1 \
  FB_TENSOR_REPLAY_REPEAT=1 \
  FB_TENSOR_REPLAY_COALESCE_REPEAT=1 \
  FB_TENSOR_REPLAY_TARGET_INPUT_TOKENS=0 \
  WANDB_NAME=OPSD-WORDLE-Q36-35B-8G-BARE-TENSOR-NOCKPT-SHARED \
| kubectl create --dry-run=server -f - -o name
```

Only create the job after the dry-run succeeds and the consolidation owner agrees
that running more perf diagnostics will not interfere with relocation.

## Bottom line

RB8+chunk16 with packing disabled and gradient checkpointing enabled remains the
best tested request shape. Checkpointing OFF is neutral on the first full train
step and uses substantially more memory, so do not promote it. Packed first-chunk
evidence did not show a warmup win, but it still lacks a full `train` event. The
synchronized profile makes the next target explicit: OPD KL is only ~0.5% of f/b,
while the transformer train path dominates the step. Warmed API replay removes
sampler, teacher-cache creation, and weight sync and improves to ~0.44% MFU, so cold
replay is not steady-state evidence; however ~0.44% is still far below the MTP
bare-f/b 1.85% tiny-token point and the 12-14% larger-token knee. Bare tensor
replay on the exact post-slice tensors is also slow once warm: 8.03s,
0.3962% valid-token MFU, and 0.4717% input-token MFU. That means the low MFU is
not primarily weight sync, sampler, teacher-cache construction, HTTP, or request
packing. The first captured request's post-slice shape is one real, unpadded,
single-sample microbatch per rank with only 479-3,455 input positions/rank. The
8k/rank coalesced tensor replay fits and helps, but only to `0.7060%`
input-token MFU, far below the sibling MTP synthetic 8k/rank `6.6%` point; 16k/rank
currently OOMs during GDN backward autotune. EP=1 is slower than EP8 in both warmed
API replay and direct bare-tensor replay, so removing EP all-to-all is not the next
fix. Quack MoE is slightly faster than Triton on exact bare replay, but only
`0.5061%` input-token MFU versus `0.4717%`, so a MoE implementation swap does not
explain the cliff either. The next target is the direct
model-runner fwd/bwd path on exact and 8k-coalesced real shapes, followed by a
pure-training-to-OPSD bisection to find what collapses the synthetic curve in this
server model path. Do not accept the current MFU as a floor.
