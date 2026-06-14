# Qwen3.6-35B-A3B OPD Reprogrammable Slots Runbook

Date: 2026-06-02
Last updated: 2026-06-03 15:57 UTC

This is the live-run handoff for the Qwen3.6-35B-A3B OPD follow-up to the
235B Config A/B experiment. The goal is to reuse already-scheduled Kubernetes
pods as programmable GPU slots: keep SGLang, dispatch, teachers, and trainer
pods allocated, then rewrite per-role control scripts under `/shared/opd-control`
instead of resubmitting Kubernetes jobs for every config change.

For the detailed operational slot-control procedure, see:

```bash
experiments/opd_profile/K8S_REPROGRAMMABLE_SLOTS_RUNBOOK_2026_06_03.md
```

## 0. Current Verdict

Config A collapsed quickly on Qwen3.6-35B-A3B.

Config B was stopped at step 80. It was materially better than Config A:
`acc_pause` stayed far above the collapsed Config A baseline. The pause/no-pause
gap was not durable, though; it oscillated around zero, and both pause/no-pause
control accuracies peaked around step 15 before drifting down.

Config C was then launched as a KL-only, answer-masked diagnostic
(`opd_supervise_buffer_only=true`, `opd_hidden_match_coef=0.0`). It failed fast:
by step 10, `acc_pause` had fallen to 0.104 while KL/loss had improved. This
shows hidden-state MSE is not the only cause of the objective-behavior mismatch.

Configs D-F tested long random-token buffers. All were negative:
random ASCII failed immediately, random common words stayed below no-pause and
leaked verbose/filler-like behavior, and a tokenizer-stable random symbol buffer
damaged both pause and no-pause controls by step 5. Random printable text tokens
at the old 100-token length are not a good replacement for the NATO/pause buffer
in this OPD setup.

Config G tested lower LR (`3e-6`) with Config B semantics. It preserved high task
accuracy through step 20 (`acc_pause=0.823`, `acc_nopause=0.854`) but still did
not create a pause-buffer advantage.

Config H tested lower LR (`3e-6`) with Config A/buffer-only semantics. It delayed
the Config A collapse and kept outputs short/clean, but `acc_pause` peaked around
step 10 and fell again by step 15. Current verdict: lower LR mitigates optimizer
damage, but none of A-H demonstrates a durable encoded-reasoning channel on
Qwen3.6-35B-A3B.

Configs I-L tested the short-buffer family: one NATO alphabet pass
(34 tokens + 3-token answer suffix) instead of the previous 100-token buffer.
This is the closest family so far. Config I briefly crossed positive at step 10
and tied by step 20. Configs K/L with a 64-token generation cap avoided the
192-token runaway but still tied or went negative by step 15. Current verdict:
short buffer + lower LR is the best direction, but the recipe still learns the
answer path more than a durable pause-buffer channel.

Configs N-Q revisited the user's "random tokens, not necessarily pause tokens"
hypothesis with a tiny 9-token random-symbol buffer:

```text
 ! | ~ _ * ^ # @
```

The earlier "random tokens are bad" verdict is too broad. Long random ASCII,
random words, and 100-token random symbol buffers were bad, but the tiny
random-symbol buffer produced a modest positive buffer delta in the 32-token
rollout family. The strongest positive result was the serialized-sampler Config
O rerun:

```text
step  loss    eval_acc  acc_buffer  acc_nobuffer  delta   mean_len  sync_s
0     1.0741  0.469     0.708       0.750        -0.042   25.8      2.050
5     0.8635  0.656     0.823       0.781         0.042   30.6      1.914
10    0.8131  0.641     0.833       0.812         0.021   31.3      1.967
```

Clean follow-ups made that verdict weaker. Config R reran Config O with a
larger paired control eval (`eval_num_problems=192`) and did not robustly
confirm the random-symbol edge: step 5 was negative and step 10 was effectively
a tie (`delta=+0.005`). Config S then tested the tiny NATO cue under the same
32-token rollout cap. It started slightly positive, tied by step 5, and was
negative by step 10:

```text
Config R, tiny random symbol, 192-control eval:
step  loss    eval_acc  acc_buffer  acc_nobuffer  delta   mean_len  sync_s
0     1.0868  0.391     0.729       0.740        -0.010   25.5      1.972
5     0.8473  0.719     0.818       0.839        -0.021   31.2      2.012
10    0.7630  0.719     0.844       0.839         0.005   32.0      1.918

Config S, tiny NATO, 96-control eval:
step  loss    eval_acc  acc_buffer  acc_nobuffer  delta   mean_len  sync_s
0     1.0264  0.516     0.760       0.729         0.031   27.0      1.895
5     0.8538  0.625     0.802       0.802         0.000   30.4      1.913
10    0.7883  0.734     0.833       0.865        -0.031   32.0      1.950
```

Current verdict: arbitrary/random printable buffer tokens are viable enough to
test, but neither tiny random symbols nor tiny NATO currently forms a durable
encoded-reasoning channel. Across O/R/S, completions quickly hit the 32-token cap
and samples often continue with verbose reasoning or echo the cue text. The next
hypothesis targeted this output-format/rollout drift directly.

Config T shortened Config O's rollout cap to 16 tokens. This did improve late
task accuracy (`eval_acc=0.750` at steps 9 and 10) and prevented 32-token verbose
runaway, but it did not improve the buffer channel: step 5 tied and step 10 was
negative (`delta=-0.042`). Samples were often clipped immediately after the
answer or after `</think>`. Current verdict after Config T: the problem is not
only post-answer verbosity. The next most direct hypothesis is to reduce
hidden-state match pressure while keeping the 32-token cap.

Config U tested that directly by keeping the tiny random-symbol buffer and
32-token cap but lowering `opd_hidden_match_coef` from `1.0` to `0.25`.
It produced a small positive buffer edge at both control checkpoints
(`delta=+0.010` at step 5, `delta=+0.031` at step 10), but did not solve the
output-channel problem: task eval ended at 0.641, mean completion length hit the
32-token cap, and samples still showed verbose continuations plus occasional
random-symbol leakage. Current verdict after Config U: arbitrary short filler
tokens remain viable, but printable text fillers are not clean latent slots.
The next cheap control is to remove hidden-state matching entirely on the same
random-symbol setup; the stronger follow-up is a true token-ID prefill path.

Config V removed hidden-state matching entirely while keeping generated-CoT
supervision on (`opd_hidden_match_coef=0.0`). It improved raw task accuracy
relative to U (`eval_acc=0.703` at step 10, peaking at 0.781 at step 5), but it
did not improve the buffer channel: step 5 tied and step 10 favored no-buffer
(`delta=-0.042`). Current verdict after Config V: hidden-MSE is not the main
blocker. It may slightly help the buffer-specific effect, while KL-only
distillation mainly teaches the direct answer/no-buffer path. Do not keep
sweeping hidden-match coefficients as the primary line.

Config W tested the true token-ID version of Config U. It kept Config U's lower
hidden-match setting (`opd_hidden_match_coef=0.25`) and 32-token rollout cap,
but replaced the printable tiny random-symbol text buffer with exact random
token IDs sent through SGLang `/generate`:

```text
18437,62981,31709,90553,74216,118927,56344,100731
```

This directly tested the user's point that the buffer need not be `pause`
tokens or even printable text. Result: exact random token IDs did not create a
buffer channel. Step 5 was slightly negative (`delta=-0.010`) and step 10 was
also slightly negative (`delta=-0.010`). Task eval recovered to 0.719 by step
10, but mean completion length saturated at 32 tokens and samples still emitted
verbose post-answer reasoning. Current verdict after Config W: printable-text
contamination is not the main blocker; filler-token variants A-W are exhausted
as-is. The next useful intervention is objective/format-level.

Config X was the first output-format control experiment. It reused Config U's
tiny random-symbol buffer and lower hidden-match weight, but applied
`student_stop_sequences='["\\n"]'` to both on-policy rollouts and paired
control eval so OPD should train on the answer line instead of answer plus
verbose post-answer reasoning. It did not produce a robust buffer channel.
Task eval improved quickly and peaked at 0.781 at steps 5 and 9, but paired
control was negative at steps 0 and 5 and only weakly positive at step 10:

```text
step  loss    eval_acc  acc_buffer  acc_nobuffer  delta   mean_len
0     0.9394  0.516     0.729       0.750        -0.021   26.1
5     0.7347  0.781     0.812       0.833        -0.021   31.3
10    0.7148  0.719     0.875       0.865         0.010   32.0
```

Current verdict after Config X: newline stop did not prevent the model from
hitting the 32-token cap, and the tiny final positive delta is too small to
treat as evidence of a durable latent-buffer effect. The strongest remaining
hypothesis is objective-level: the answer-unmasked recipes are teaching the
direct no-buffer answer path. Config Y is wired as the next cheap test:
tiny random-symbol buffer, `opd_supervise_buffer_only=true`,
`opd_hidden_match_coef=0.25`, `lr=3e-6`, and `max_new_tokens=32`.

Important caveat: Config R exposed a Qwen3.6/SGLang batched decoding failure.
With `--max-running-requests 256`, many batched eval prompts collapsed to
near-identical suffixes and the run reported near-zero accuracy. After changing
student SGLang to `--max-running-requests 1`, a base held-out probe recovered
normal behavior. Treat Config R and any batched rerun showing repeated suffixes
as invalid science, not as an OPD failure.

2026-06-03 update: Config AC made the two-serialized-sampler substrate
operationally viable but remained a science reject. It completed 6 steps with
two fresh serial endpoint syncs per step and balanced sampler routing, but the
final control row still favored no-pause over pause (`0.7500` vs `0.7292`).
Config AD added an answer-logprob causal gate. Its warmup row scored all three
arms cleanly with zero scorer failures, but pause was worse than corrupted pause
on answer logprob (`-0.0092`, z `-1.2480`). AD then failed on the next
two-endpoint P2P sync with a Mooncake RDMA path mismatch while sampler traffic
was still queued/in flight.

2026-06-03 12:50 UTC update: Config AE fixed the autoresearch-loop ordering
problem by disabling RL pipelining before sync and adding a sampler-quiescence
gate. It completed 6 steps with two serialized student samplers, two fresh
serial endpoint syncs per step, W&B/profile parity, and
`sampler_quiesce_success=1.0` with zero outstanding/active/inflight sampler
requests before each sync. AE still failed the mechanism gate: final
`acc_pause=0.7083`, `acc_nopause=0.7500`, `acc_corrupt=0.7083`,
`eval/answer_logprob_margin=-0.0433`, and
`eval/answer_logprob_vs_corrupt_margin=-0.0264`. Current diagnosis: the
operational blocker is fixed enough for science iteration, but the science
blocker is causal credit assignment, not filler-token syntax. "Copy RiM" is not
the next step; RiM should be treated as an ablation template and must pass the
same pause/no-pause/corrupt-pause and answer-logprob causal gates.

2026-06-03 13:15 UTC update: Config AF increased the corrupt-buffer contrast
weight to test whether a stronger negative arm would force separability in the
pause buffer. Treat this run as infrastructure-invalid after step 2: step 3
quiesced the sampler successfully, then the first serial endpoint P2P sync
failed with a Mooncake RDMA `received packet mismatch` to endpoint 0. The
partial science signal is still informative. Hidden-match separation moved in
the intended direction (`neg_raw - pos_raw`: `0.1594` at step 0, `0.1867` at
step 1, `0.2801` at step 2), but the step-0 causal controls were still
negative (`acc_pause=0.6771`, `acc_nopause=0.7708`,
`acc_corrupt=0.7083`, `answer_logprob_margin=-0.0010`,
`answer_logprob_vs_corrupt_margin=-0.0084`). This is exactly the proxy-vs-causal
split the loop was previously missing: the objective can learn a positive vs
corrupt hidden-state distinction without yet making the answer depend on the
pause buffer. Rerun AF only after clearing or restarting endpoint 0/P2P state;
do not promote it without a clean final causal checkpoint.

2026-06-03 13:45 UTC update: A clean AF rerun
(`20260603T132217Z-configAF-er-opd-q36-35b-slots-trainer-head`, W&B
`u9wqf40r`) restarted the student sampler pods first and got farther, but still
failed the repeated P2P-sync reliability gate. Steps 0-3 were operationally
valid: both samplers were synced, quiescence passed, routing was exactly
balanced, and hidden separation strengthened (`neg_raw - pos_raw`: `0.1601`,
`0.1860`, `0.2830`, `0.4963`). Step 4 then failed during the serial endpoint
P2P sync with repeated Mooncake `received packet mismatch` errors to
`10.42.77.65:15151`, followed by
`batch_transfer_sync ... failed ... after 50 attempts`. The failure happened
despite endpoint-scoped groups and pre-registration stale-state cleanup, so the
next science run should not be another blind AF/P2P rerun. Use Config AG
instead: AF objective and controls, but `sync_method=nccl_broadcast` /
`sync_inference_method=nccl_broadcast`, then require the same final
pause/no-pause/corrupt and answer-logprob causal gates.

2026-06-03 14:10 UTC update: Config AG completed that rerun cleanly with NCCL
broadcast sync (`20260603T134618Z-configAG-er-opd-q36-35b-slots-trainer-head`,
W&B `oxo5yw7i`). This is the cleanest AF-family science result: six profile
rows, two freshly synced serialized samplers, exact balanced routing, W&B/profile
parity, and `sampler_quiesce_success=1.0` on every row. NCCL sync avoided the
Mooncake repeated-sync failure (`sync_inference_weights_s` about `20s` per row;
`sync_endpoint_success_count=2`, `sync_endpoint_failure_count=0`). The science
answer is still a reject. Hidden separation rose from `0.1597` to `0.6446`, but
final causal controls worsened: `acc_pause=0.7083`,
`acc_nopause=0.7708`, `acc_corrupt=0.7292`,
`eval/buffer_delta=-0.0625`,
`eval/buffer_vs_corrupt_delta=-0.0208`,
`eval/answer_logprob_margin=-0.0510`, and
`eval/answer_logprob_vs_corrupt_margin=-0.0232`. This rules out stale sync,
sampler routing, and P2P partial-write artifacts as the explanation for the
AF-family failure. The underlying loop is optimizing a hidden-state contrastive
proxy that is not causally load-bearing for the answer.

2026-06-03 14:31 UTC update: Config AH added an answer-level causal contrast on
the same clean AG substrate
(`20260603T141126Z-configAH-er-opd-q36-35b-slots-trainer-head`, W&B
`zfkrmifn`). AH kept NCCL broadcast, two freshly synced serialized samplers,
quiescence, and answer-logprob controls, but changed the objective:
`opd_supervise_buffer_only=false` keeps answer rows and
`opd_contrastive_corrupt_answer_weight=0.125` gives real-buffer answer positions
positive KL and corrupt-buffer answer positions negative KL, with
`opd_loss_max_clamp=5.0`. Operationally it was clean: six profile rows,
`sync_endpoint_success_count=2` on every row, exact sampler balance, and
W&B/profile parity. Scientifically it is the first slots run with a strong
positive causal answer signal: final `acc_pause=0.5000`,
`acc_nopause=0.2292`, `acc_corrupt=0.1354`,
`eval/buffer_delta=+0.2708` (`z=4.06`),
`eval/buffer_vs_corrupt_delta=+0.3646` (`z=5.90`),
`eval/answer_logprob_margin=+0.2234` (`z=5.33`), and
`eval/answer_logprob_vs_corrupt_margin=+0.8454` (`z=23.80`). Analyzer still
rejects promotion because `control_n=96 < 192` and corrupted-pause generation
degenerates (`corrupt_pause_cap_hit_frac=0.9375`,
`corrupt_pause_filler_leak_frac=1.0`). Treat AH as a mechanism hit but not a
promoted recipe. Next: rerun with 192 controls and replace arbitrary corrupt
symbols with a less degenerate in-distribution corrupt arm; after AI, do not
interpret this as token-level shuffling of the fixed slot scaffold.

2026-06-03 15:10 UTC update: Config AI ran the AH follow-up with the strict
192-problem control gate and a less destructive corrupt arm
(`20260603T143837Z-configAI-er-opd-q36-35b-slots-trainer-head`, W&B
`fzittb3d`). AI kept the AH answer-level contrast and clean NCCL/two-sampler
substrate, but set `opd_contrastive_corrupt_buffer_mode=rotate` and
`opd_contrastive_corrupt_buffer_span=memory_only`, leaving the literal
`Answer: ` suffix intact. Operationally it was clean: six rows,
`sync_endpoint_success_count=2` on every row, serial endpoint sync, quiescence,
exact 2-worker sampler balance, and W&B/profile parity passed. Scientifically
it is a partial mechanism hit but still rejects promotion. Final exact-match
controls improved in the
right direction but missed the strict z gate: `acc_pause=0.6719`,
`acc_nopause=0.5781`, `acc_corrupt=0.6094`,
`eval/buffer_delta=+0.0938` (`z=1.91`), and
`eval/buffer_vs_corrupt_delta=+0.0625` (`z=1.28`). Answer-logprob was strongly
positive: `eval/answer_logprob_margin=+0.2163` (`z=16.67`) and
`eval/answer_logprob_vs_corrupt_margin=+0.3917` (`z=17.31`). The corrupt arm is
less degenerate than AH (`filler_leak_frac=0.0`) but still pathological:
`corrupt_pause_cap_hit_frac=0.7083`. Next: keep the answer-causal objective, but
do not add a token-level shuffled-memory config to the current fixed-slot
recipe. The slot tokens are shared scaffolding, so cross-prompt token shuffling
would mostly be a no-op. The next causal control must externalize
prompt-specific memory before shuffling, or mismatch prompt-specific
teacher-memory/cache targets while keeping answer-level validation. Client-side
control latency metrics were added after AI so the next run logs per-arm control
latency and answer-logprob group latency. A corrupt-control no-op guard was also
added so profile rows report changed-token fraction and no-op fraction for the
contrastive corrupt span.

2026-06-03 15:31 UTC update: ran a one-step Config AI instrumentation smoke
(`20260603T151837Z-configAI-er-opd-q36-35b-slots-trainer-head`, W&B
`05o5f8h6`). This is not a promotion run because the only profile row is warmup;
the analyzer correctly reports `VERDICT: incomplete (no non-warmup control
rows)`. The smoke validated the new metrics and live W&B/profile parity:
`opd_contrastive_corrupt_change_frac=1.0`,
`opd_contrastive_corrupt_noop_frac=0.0`,
`eval/control_request_latency_p95_s=331.81`,
`eval/control_request_latency_max_s=348.83`, and
`eval/answer_logprob_group_latency_max_s=18.76`. The latency tail is real:
`nopause` and `corrupt_pause` were the slow arms, with mean request latencies
`218.63s` and `224.61s` respectively. Operational substrate remained clean:
two synced endpoints, serial endpoint sync, exact sampler balance `320/320`,
and W&B/profile parity passed after W&B finished syncing.

2026-06-03 15:57 UTC update: added and smoked Config AJ, an opt-in
teacher-memory pair diagnostic
(`20260603T153913Z-configAJ-er-opd-q36-35b-slots-trainer-head`, W&B
`yazdxxgu`). AJ is not a promotion run because the only profile row is warmup;
the analyzer correctly reports `VERDICT: incomplete (no non-warmup control
rows)`. The new diagnostic was active and clean:
`opd_teacher_memory_pair_diag_active=1`,
`opd_teacher_memory_pair_diag_failure=0`, and
`opd_teacher_memory_pair_sample_count=64`. Cross-prompt teacher memory rows were
farther apart than adjacent rows within the same buffer, but only modestly:
`cross_cosine_distance_mean=0.2501`,
`within_adjacent_distance_mean=0.1959`, and
`cross_minus_within_distance=0.0543`. Interpretation: the teacher slot hiddens
are prompt-specific enough to justify a careful cache-mismatch diagnostic, but
not separated enough to treat cache mismatch as an obviously strong objective.
Do not add a naive negative answer-KL term on an identical prompt paired with
another prompt's teacher memory; that would punish the real answer path. Keep
answer-level validation separate. Operationally AJ stayed clean: two synced
endpoints, serial endpoint sync, exact sampler balance `320/320`, W&B/profile
parity passed, and the measured control tail persisted
(`eval/control_request_latency_p95_s=333.63`, max `344.50`).

## 1. Active Stack

```bash
STACK=er-opd-q36-35b-slots
NS=apanda
CONTROL_ROOT=/shared/opd-control/er-opd-q36-35b-slots
RESULT_ROOT=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots
MANIFEST=experiments/opd_profile/k8s/generated/er-opd-q36-35b-slots.yaml
GENERATOR=experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py
```

Model:

```bash
/shared/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0
```

Current slot layout:

- `sglang-0`: student SGLang, TP=8, `node-group=nccl`, port 30060.
  Currently launched with `--max-running-requests 1` as a correctness
  mitigation for the Qwen3.6 batched decoding failure.
- `dispatch`: SMG dispatch, `node-group=default`, port 8080.
- `trainer-head` plus `trainer-worker-1..7`: 8 trainer nodes, 64 GPUs total, `node-group=nccl`.
- `teacher-sglang-0/1`: teacher SGLang, TP=8 each, `node-group=default`, port 30000.
- `teacher-smg`: teacher SMG, not used by the current client path.

All GPU pods are labeled `team: turbo`.

Status after the Config AJ diagnostic smoke:

- Student dispatch, `sglang-0`, `teacher-sglang-1` as the second student
  sampler, `teacher-sglang-0`, and `teacher-smg` are still running.
- All trainer slots were stopped by 2026-06-03 15:50 UTC after trainer-head
  exited `rc=0`.
- The active two-sampler layout is `--sampler-layout spare-teacher1`; endpoint 0
  is `sglang-0:30060` and endpoint 1 is `teacher-sglang-1:30000`.
- Keep each SGLang sampler serialized with `--max-running-requests 1`. Increase
  throughput with more serialized sampler pods, not per-pod batching, until the
  repeated-suffix issue is fixed and revalidated.
- Keep `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0` on the sampler slots;
  generation-backed `/health` caused restarted Qwen3.6 SGLang workers to appear
  unavailable to SMG even while `/v1/models` worked.
- Future science runs should keep `opd_pipeline_rl=false` plus
  `sampler_quiesce_before_sync=true` until a pipelined variant has its own
  freshness proof.

Known bad nodes excluded in the generator:

```text
research-common-h100-113.cloud.together.ai
research-common-h100-014.cloud.together.ai
research-common-h100-050.cloud.together.ai
research-common-h100-087.cloud.together.ai
```

`dispatch` is SMG.

## 2. Important Runtime Fixes

These fixes were required to get the experiment past step 0.

### 2.1 Server Config Support

The Qwen3.6 YAMLs use:

```yaml
gradient_checkpointing_method: recompute_before_dispatch
```

Server-mode parsing did not previously accept or thread this field. The local
tree now includes support in:

- `src/xorl/server/server_arguments.py`
- `src/xorl/server/runner/model_runner.py`
- `src/xorl/trainers/model_builder.py`
- `tests/server/test_server_arguments.py`

Validation already run:

```bash
uv run ruff check src/xorl/server/server_arguments.py src/xorl/trainers/model_builder.py src/xorl/server/runner/model_runner.py tests/server/test_server_arguments.py experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py
uv run pytest tests/server/test_server_arguments.py -q
```

### 2.2 Filtered CoT Dataset

The original CoT file had seven empty-CoT rows:

```text
4543 4819 5414 5641 5661 5707 7022
```

The live run uses filtered aligned files:

```bash
PROMPTS_JSON=/shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
COT_JSON=/shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
NUM_PROMPTS=8185
```

The OPD client confirmed:

```text
Teacher per-prompt CoT loaded: 8185 entries, min=227 median=2048 max=2048 tokens
```

### 2.3 GPU-Direct P2P Path

The experiment requires the small CUDA GPU-direct sender path in
`src/xorl/server/weight_sync/backends/p2p.py`.

Additional local runtime hardening:

- `XORL_P2P_TRANSFER_RETRIES=50`
- `XORL_P2P_CPU_POOL_MIN_BYTES=65536`
- small CUDA entries are batched/retried via `XORL_P2P_SMALL_TRANSFER_CHUNK`

Validation:

```bash
uv run ruff check src/xorl/server/weight_sync/backends/p2p.py
```

The batching/retry change alone did not fix the transfer failure. The key fix
was disabling PyTorch expandable allocator segments for both sender and receiver.

### 2.4 Disable Expandable Allocator Segments

Mooncake CUDA registration failed when CUDA allocations came from PyTorch
expandable segments. The generator now unsets both variables in student SGLang
and trainer scripts:

```bash
unset PYTORCH_ALLOC_CONF
unset PYTORCH_CUDA_ALLOC_CONF
```

This is mandatory. Do not reintroduce `expandable_segments:True` for the
student sampler or trainer while using GPU-direct P2P sync.

Confirmed working sync after this fix:

```text
step 0: 69.32 GB in 3.46-3.62s
steady steps: about 1.77-2.15s per 69.32 GB
```

### 2.5 Fresh Triton Cache For Sampler Restarts

One student sampler restart failed because an old Triton cache entry pointed to
a missing `.cubin`. The sampler control script now uses a per-revision cache:

```bash
export TRITON_CACHE_DIR=/tmp/triton-cache-${SLOT_ROLE}-<revision>
rm -rf "$TRITON_CACHE_DIR"
```

### 2.6 Endpoint Sync And Stale P2P Cleanup

The trainer registration path now passes `sync_weights=true` when adding the
student SGLang endpoint and greps for `"weights_synced":true`. This ensures the
student sampler starts every OPD run from the trainer base weights before step 0.

The generator also sends a best-effort stale-session cleanup before endpoint
registration:

```bash
curl -m 30 -sS -X POST "http://er-opd-q36-35b-slots-sglang-0:30060/complete_weights_update" \
  -H "Content-Type: application/json" \
  -d '{"group_name":"weight_sync_group","transport":"p2p","run_post_process_weights":false}' || true
```

This was added after Config E first failed before step 0 with:

```text
A P2P weight update for group 'weight_sync_group' is already in progress.
Call complete_weights_update_p2p first.
```

### 2.7 Qwen3.6 Batched SGLang Decoding Hazard

Config R exposed a serious sampler-side correctness problem when the student
SGLang server allowed high request concurrency. With the default
`--max-running-requests 256`, batched Qwen3.6/GDN decoding produced repeated or
near-repeated numeric suffixes across unrelated prompts. The failure reproduced
without training in a held-out base probe:

```text
batched 96-prompt probe, answer-prefill acc: 0.021
batched 96-prompt probe, random-symbol-prefill acc: 0.021
symptom: unrelated prompts shared repeated suffixes such as 31346592/32646888
```

After changing the student sampler to serialize requests:

```bash
--max-running-requests 1
```

the same style of probe recovered normal behavior:

```text
serialized 24-prompt probe, answer-prefill acc: 0.833
serialized 24-prompt probe, random-symbol-prefill acc: 0.708
```

The generator currently carries this mitigation in
`experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py`. Keep it for
science runs unless SGLang/GDN batching is fixed and validated. The cost is
runtime: checkpoint evals with 96 prompts x 2 arms take several minutes through
one serialized TP=8 sampler.

If throughput is needed before the batching bug is fixed, prefer multiple
student sampler pods each with `--max-running-requests 1` and route through SMG,
instead of increasing per-pod request concurrency.

## 3. Control Commands

Status:

```bash
python experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py status
```

Restart only student inference and dispatch:

```bash
python experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py write-student-inference-control
```

Restart only dispatch:

```bash
python experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py write-dispatch-control
```

Stop trainer only:

```bash
python experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py stop-trainer-control
```

Start Config A trainer only:

```bash
python experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py write-trainer-control --config A --num-steps 400 --prompts-per-step 64
```

Start Config B trainer only:

```bash
python experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py write-trainer-control --config B --num-steps 400 --prompts-per-step 64
```

Start the most recent tiny-random-symbol baseline:

```bash
python experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py write-trainer-control --config O --num-steps 11 --prompts-per-step 64
```

Start the exact random-token-ID follow-up:

```bash
python experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py write-trainer-control --config W --num-steps 11 --prompts-per-step 64
```

Start the newline-stop output-format follow-up:

```bash
python experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py write-trainer-control --config X --num-steps 11 --prompts-per-step 64
```

Start the next buffer-only tiny-random-symbol follow-up:

```bash
python experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py write-trainer-control --config Y --num-steps 11 --prompts-per-step 64
```

Available configs in the generator as of this update:

```text
A  NATO, buffer-only, hidden+KL, lr=1e-5
B  NATO, answer-unmasked, hidden+KL, lr=1e-5
C  NATO, buffer-only, KL-only, lr=1e-5
D  random ASCII buffer, answer-unmasked, lr=1e-5
E  random common-word buffer, answer-unmasked, lr=1e-5
F  random ASCII-symbol token buffer, answer-unmasked, lr=1e-5
G  NATO, answer-unmasked, hidden+KL, lr=3e-6
H  NATO, buffer-only, hidden+KL, lr=3e-6
I  short NATO, answer-unmasked, hidden+KL, lr=3e-6
J  short NATO, buffer-only, hidden+KL, lr=3e-6
K  short NATO, answer-unmasked, hidden+KL, lr=3e-6, max_new_tokens=64
L  short NATO, answer-only, hidden+KL, lr=3e-6, max_new_tokens=64
M  tiny NATO, answer-unmasked, hidden+KL, lr=3e-6, max_new_tokens=64
N  tiny random-symbol, answer-unmasked, hidden+KL, lr=3e-6, max_new_tokens=64
O  tiny random-symbol, answer-unmasked, hidden+KL, lr=3e-6, max_new_tokens=32
P  tiny random-symbol, answer-only, hidden+KL, lr=3e-6, max_new_tokens=32
Q  tiny random-symbol, answer-unmasked, hidden+KL, lr=1e-6, max_new_tokens=32
R  O with eval_num_problems=192; prior R attempt invalid under batched sampler,
   clean serialized retry finished
S  tiny NATO, answer-unmasked, hidden+KL, lr=3e-6, max_new_tokens=32
T  tiny random-symbol, answer-unmasked, hidden+KL, lr=3e-6, max_new_tokens=16
U  tiny random-symbol, answer-unmasked, hidden+KL with
   opd_hidden_match_coef=0.25, lr=3e-6, max_new_tokens=32
V  tiny random-symbol, answer-unmasked, KL-only with generated-CoT supervision,
   opd_hidden_match_coef=0.0, lr=3e-6, max_new_tokens=32
W  exact random token-ID buffer, answer-unmasked, generated-CoT supervision,
   opd_hidden_match_coef=0.25, lr=3e-6, max_new_tokens=32
X  tiny random-symbol, answer-unmasked, generated-CoT supervision,
   opd_hidden_match_coef=0.25, lr=3e-6, max_new_tokens=32,
   student_stop_sequences=["\n"]
Y  tiny random-symbol, buffer-only generated-CoT supervision,
   opd_supervise_buffer_only=true, opd_hidden_match_coef=0.25, lr=3e-6,
   max_new_tokens=32
```

The clean trainer restart sequence is important:

```bash
python experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py stop-trainer-control --remove-run

# Wait until all trainer roles are stopped and no trainer torchrun/rank
# processes remain inside the pods.
python experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py status

python experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py write-trainer-control --config B --num-steps 400 --prompts-per-step 64
```

Reason: a direct A-to-B rewrite once produced a split torchrun state where some
worker pods spawned local ranks and others stayed at the torchrun parent. A
clean stop with `--remove-run`, followed by a fresh revision, fixed it.

Verify all trainer pods spawned ranks:

```bash
for p in er-opd-q36-35b-slots-trainer-head \
  er-opd-q36-35b-slots-trainer-worker-1 \
  er-opd-q36-35b-slots-trainer-worker-2 \
  er-opd-q36-35b-slots-trainer-worker-3 \
  er-opd-q36-35b-slots-trainer-worker-4 \
  er-opd-q36-35b-slots-trainer-worker-5 \
  er-opd-q36-35b-slots-trainer-worker-6 \
  er-opd-q36-35b-slots-trainer-worker-7; do
  n=$(kubectl exec -n apanda "$p" -- bash -lc \
    'ps -eo cmd | egrep "torch.distributed.run|runner_dispatcher|xorl.server.launcher" | grep -v egrep | wc -l')
  echo "$p trainer_proc_count=$n"
done
```

Healthy B restart produced:

```text
trainer-head trainer_proc_count=12
each trainer-worker trainer_proc_count=9
```

## 4. Live Run Directories

Config A run used for the step-20 negative result:

```bash
RUN_A=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260602T195119Z-configA-er-opd-q36-35b-slots-trainer-head
PROFILE_A=$RUN_A/opd_profile.jsonl
WANDB_A=yhzxebd3
```

Failed partial Config B restart; ignore for science:

```bash
/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260602T200752Z-configB-er-opd-q36-35b-slots-trainer-head
```

Stopped Config B run:

```bash
RUN_B=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260602T201249Z-configB-er-opd-q36-35b-slots-trainer-head
PROFILE_B=$RUN_B/opd_profile.jsonl
WANDB_B=j5hy662k
```

Stopped Config C run:

```bash
RUN_C=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260602T210909Z-configC-er-opd-q36-35b-slots-trainer-head
PROFILE_C=$RUN_C/opd_profile.jsonl
WANDB_C=g64zkdgm
```

Stopped Config D run:

```bash
RUN_D=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260602T215105Z-configD-er-opd-q36-35b-slots-trainer-head
PROFILE_D=$RUN_D/opd_profile.jsonl
WANDB_D=euh75lgo
```

Stopped Config E run:

```bash
RUN_E=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260602T220758Z-configE-er-opd-q36-35b-slots-trainer-head
PROFILE_E=$RUN_E/opd_profile.jsonl
WANDB_E=2cvqn5vb
```

Stopped Config F run:

```bash
RUN_F=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260602T222123Z-configF-er-opd-q36-35b-slots-trainer-head
PROFILE_F=$RUN_F/opd_profile.jsonl
WANDB_F=z2um6i24
```

Stopped Config G run:

```bash
RUN_G=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260602T222957Z-configG-er-opd-q36-35b-slots-trainer-head
PROFILE_G=$RUN_G/opd_profile.jsonl
WANDB_G=0xbvnuvv
```

Stopped Config H run:

```bash
RUN_H=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260602T224648Z-configH-er-opd-q36-35b-slots-trainer-head
PROFILE_H=$RUN_H/opd_profile.jsonl
```

Stopped Config I run:

```bash
RUN_I=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260602T230036Z-configI-er-opd-q36-35b-slots-trainer-head
PROFILE_I=$RUN_I/opd_profile.jsonl
WANDB_I=jf8hdjpw
```

Stopped Config J run:

```bash
RUN_J=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260602T231707Z-configJ-er-opd-q36-35b-slots-trainer-head
PROFILE_J=$RUN_J/opd_profile.jsonl
WANDB_J=l9ifyx1x
```

Stopped Config K run:

```bash
RUN_K=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260602T232443Z-configK-er-opd-q36-35b-slots-trainer-head
PROFILE_K=$RUN_K/opd_profile.jsonl
WANDB_K=g57r7zn6
```

Stopped Config L run:

```bash
RUN_L=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260602T233635Z-configL-er-opd-q36-35b-slots-trainer-head
PROFILE_L=$RUN_L/opd_profile.jsonl
WANDB_L=tmgnqdom
```

Stopped Config N run:

```bash
RUN_N=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260602T235110Z-configN-er-opd-q36-35b-slots-trainer-head
PROFILE_N=$RUN_N/opd_profile.jsonl
WANDB_N=410u1bao
```

Stopped Config O batched-sampler run; use as suggestive only:

```bash
RUN_O_BATCHED=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T000751Z-configO-er-opd-q36-35b-slots-trainer-head
PROFILE_O_BATCHED=$RUN_O_BATCHED/opd_profile.jsonl
WANDB_O_BATCHED=1voc936c
```

Stopped Config P run:

```bash
RUN_P=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T002502Z-configP-er-opd-q36-35b-slots-trainer-head
PROFILE_P=$RUN_P/opd_profile.jsonl
```

Stopped Config Q run:

```bash
RUN_Q=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T003423Z-configQ-er-opd-q36-35b-slots-trainer-head
PROFILE_Q=$RUN_Q/opd_profile.jsonl
```

Invalid Config R batched-sampler run:

```bash
RUN_R_INVALID=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T012342Z-configR-er-opd-q36-35b-slots-trainer-head
PROFILE_R_INVALID=$RUN_R_INVALID/opd_profile.jsonl
```

Clean serialized Config O rerun:

```bash
RUN_O_SERIAL=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T013938Z-configO-er-opd-q36-35b-slots-trainer-head
PROFILE_O_SERIAL=$RUN_O_SERIAL/opd_profile.jsonl
WANDB_O_SERIAL=5r8vml5x
```

Completed Config X run:

```bash
RUN_X=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T060839Z-configX-er-opd-q36-35b-slots-trainer-head
PROFILE_X=$RUN_X/opd_profile.jsonl
WANDB_X=bhj9dmso
```

## 5. Results So Far

### 5.1 Config A: Negative By Step 20

```text
step  loss    eval_acc  acc_pause  acc_nopause  delta   sync_s
0     0.4157  0.438     0.677      0.781        -0.104  3.462
5     0.3913  0.422     0.531      0.604        -0.073  1.853
10    0.2907  0.094     0.062      0.042         0.021  1.998
15    0.2815  0.078     0.031      0.021         0.010  1.897
20    0.2603  0.047     0.021      0.010         0.010  1.836
```

Interpretation: loss decreases while task behavior collapses. Config A should
not be extended as-is.

### 5.2 Config B: Healthy But Not A Pause-Lift Winner Through Step 80

```text
step  loss    eval_acc  acc_pause  acc_nopause  delta   sync_s
0     0.3389  0.016     0.531      0.740        -0.208  3.618
5     0.3060  0.172     0.573      0.635        -0.062  1.847
10    0.3074  0.578     0.792      0.844        -0.052  1.818
15    0.3304  0.766     0.833      0.854        -0.021  1.864
20    0.2997  0.688     0.802      0.781         0.021  1.894
25    0.3225  0.609     0.708      0.740        -0.031  1.801
30    0.2950  0.609     0.729      0.708         0.021  1.899
35    0.2944  0.844     0.750      0.750         0.000  1.816
40    0.2785  0.719     0.698      0.740        -0.042  1.858
45    0.2900  0.672     0.729      0.750        -0.021  1.953
50    0.2919  0.531     0.677      0.667         0.010  1.836
55    0.2842  0.672     0.729      0.708         0.021  1.925
60    0.3039  0.547     0.698      0.708        -0.010  2.029
65    0.2649  0.734     0.667      0.646         0.021  1.862
70    0.2844  0.672     0.750      0.719         0.031  1.878
75    0.2629  0.688     0.688      0.677         0.010  1.940
80    0.3155  0.672     0.677      0.719        -0.042  2.069
```

Interpretation: unlike Config A, Qwen3.6 Config B is not collapsing early.
Through step 80, `acc_pause` remains above its own step-0 value and far above
Config A at the same horizon. However, both control accuracies peaked around
step 15 and the pause/no-pause delta oscillated around zero. This is not a
durable encoded-reasoning advantage. The run was stopped at the user's request.

### 5.3 Config C: KL-Only, Answer-Masked Diagnostic

Config C used:

```text
opd_supervise_buffer_only=true
opd_hidden_match_coef=0.0
```

It reuses the Config A trainer YAML; the change is on the OPD client command
line generated by `experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py`.

Selected rows:

```text
step  loss    eval_acc  acc_pause  acc_nopause  delta   hidden  kl      sync_s
0     0.3290  0.672     0.656      0.750        -0.094  0.0000  0.3290  3.475
5     0.2684  0.547     0.708      0.750        -0.042  0.0000  0.2684  1.818
10    0.2224  0.078     0.104      0.188        -0.083  0.0000  0.2224  1.949
```

Interpretation: removing hidden-state MSE did not fix the core problem. KL-only
training still drove the objective down while behavior collapsed by step 10.
Config C was stopped early.

### 5.4 Config D: Random ASCII Buffer

Config D used B-style answer-unmasked OPD with a fixed random ASCII/base64-like
buffer. It was stopped after step 1.

```text
step  loss    eval_acc  acc_pause  acc_nopause  delta   mean_len  sync_s
0     0.3515  0.000     0.062      0.792        -0.729  192.0     3.465
1     0.4558  0.000     -          -             -      192.0     1.774
```

Interpretation: catastrophic. Random ASCII is too adversarial for this setup.

### 5.5 Config E: Random Common-Word Buffer

Config E used B-style answer-unmasked OPD with a fixed common-word buffer
(`river copper window ...`) at about the same token budget.

```text
step  loss    eval_acc  acc_pause  acc_nopause  delta   mean_len  sync_s
0     0.5282  0.500     0.688      0.771        -0.083   58.0     1.968
5     0.3300  0.703     0.708      0.792        -0.083  186.0     1.935
10    0.3157  0.578     0.646      0.781        -0.135  189.1     1.940
```

Interpretation: less catastrophic than random ASCII, but still no pause/buffer
advantage. Samples leaked filler-like words and verbose CoT behavior.

### 5.6 Config F: Random Symbol Token Buffer

Config F used B-style answer-unmasked OPD with a fixed pseudo-random sequence of
100 tokenizer-stable ASCII symbol tokens, then the normal answer suffix.

```text
step  loss    eval_acc  acc_pause  acc_nopause  delta   mean_len  sync_s
0     0.4822  0.438     0.594      0.781        -0.188   60.6     2.133
5     0.3121  0.672     0.427      0.615        -0.188  153.6     1.941
```

Interpretation: negative. It did not mostly copy the symbol string, but it
damaged both control accuracies and pushed `/no_think` completions into verbose
CoT/code-like outputs.

### 5.7 Config G: Lower-LR Answer-Unmasked Control

Config G was Config B with `learning_rate=3e-6`.

```text
step  loss    eval_acc  acc_pause  acc_nopause  delta   mean_len  sync_s
0     0.4696  0.484     0.615      0.729        -0.115   84.5     1.898
5     0.3184  0.672     0.750      0.823        -0.073  157.1     1.977
10    0.2879  0.609     0.740      0.833        -0.094  183.8     2.214
15    0.2678  0.719     0.802      0.844        -0.042  188.9     1.883
20    0.2711  0.719     0.823      0.854        -0.031  191.7     2.106
```

Interpretation: lower LR reduces destructive drift and preserves strong task
accuracy, but it still trains the answer path rather than a buffer advantage.
No-pause remains better at every control point.

### 5.8 Config H: Lower-LR Buffer-Only Control

Config H was Config A with `learning_rate=3e-6`.

```text
step  loss    eval_acc  acc_pause  acc_nopause  delta   mean_len  sync_s
0     0.4166  0.516     0.594      0.760        -0.167   62.7     1.972
5     0.3369  0.578     0.635      0.760        -0.125   25.4     1.965
10    0.2825  0.484     0.656      0.729        -0.073   30.1     1.865
15    0.2684  0.500     0.604      0.729        -0.125   33.9     1.869
```

Interpretation: lower LR delays the Config A collapse and keeps generations
short/clean, but the buffer-only objective still does not produce a durable
pause advantage. `acc_pause` peaks around step 10 and falls by step 15.

### 5.9 Config I: Short NATO, Lower-LR Answer-Unmasked

Config I was Config G with only one NATO alphabet pass:
34 filler tokens + 3 suffix tokens, instead of 100 + 3.

```text
step  loss    eval_acc  acc_pause  acc_nopause  delta   mean_len  sync_s
0     0.6428  0.578     0.688      0.729        -0.042   64.4     1.855
5     0.4153  0.688     0.760      0.823        -0.062  164.5     1.881
10    0.3369  0.766     0.854      0.844         0.010  191.9     1.856
15    0.3470  0.719     0.833      0.865        -0.031  184.5     1.930
20    0.3453  0.719     0.875      0.875         0.000  184.4     1.967
```

Interpretation: closest result so far. A small positive delta appeared at step
10, but it did not persist. The run still drifted into long completions.

### 5.10 Config J: Short NATO, Lower-LR Buffer-Only

Config J was Config H with the short NATO buffer.

```text
step  loss    eval_acc  acc_pause  acc_nopause  delta   mean_len  sync_s
0     0.7132  0.531     0.615      0.760        -0.146   56.3     1.880
5     0.6164  0.547     0.635      0.771        -0.135   54.8     1.913
```

Interpretation: negative by step 5. Shortening the buffer did not rescue
buffer-only supervision.

### 5.11 Config K: Short NATO, Lower-LR, 64-Token Rollout Cap

Config K was Config I with `max_new_tokens=64`.

```text
step  loss    eval_acc  acc_pause  acc_nopause  delta   mean_len  sync_s
0     0.8159  0.516     0.677      0.729        -0.052   32.3     2.438
5     0.6107  0.625     0.750      0.812        -0.062   61.9     2.242
10    0.5439  0.719     0.844      0.844         0.000   63.1     2.061
15    0.5375  0.734     0.823      0.875        -0.052   61.9     2.195
```

Interpretation: the cap prevents 192-token runaway but does not create a pause
edge. By step 15 no-pause is again better.

### 5.12 Config L: Short NATO, Lower-LR, Answer-Only Masking

Config L was short NATO + `max_new_tokens=64` + `supervise_student_cot=false`.
The student still samples with the short buffer, but the buffer positions are
masked out of the OPD loss; only answer positions are distilled.

```text
step  loss    eval_acc  acc_pause  acc_nopause  delta   mean_len  sync_s
0     0.5556  0.578     0.688      0.719        -0.031   39.6     1.929
5     0.4166  0.594     0.729      0.750        -0.021   62.1     1.906
10    0.4105  0.625     0.854      0.854         0.000   62.3     1.933
15    0.4294  0.781     0.844      0.865        -0.021   62.0     1.868
```

Interpretation: answer-only masking narrows the gap but still does not produce a
durable positive buffer delta.

### 5.13 Config M: Tiny NATO, Defined But Not Launched

Config M is present in the generator but has no recorded run as of this update.
It is Config K with an 11-token structured cue:

```text
Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel
```

Run it under the serialized sampler before drawing any conclusion about tiny
structured cues.

### 5.14 Config N: Tiny Random-Symbol Buffer, 64-Token Rollout Cap

Config N was Config K with the 9-token random-symbol cue:

```text
 ! | ~ _ * ^ # @
```

It used `max_new_tokens=64`, `learning_rate=3e-6`, generated-CoT supervision,
and answer-unmasked OPD.

```text
step  loss    eval_acc  acc_buffer  acc_nobuffer  delta   mean_len  sync_s
0     1.0287  0.500     0.708       0.750        -0.042   32.8      2.111
5     0.7320  0.531     0.812       0.802         0.010   52.0      1.973
10    0.6396  0.594     0.844       0.823         0.021   62.4      1.962
15    0.6014  0.578     0.823       0.802         0.021   62.1      1.958
20    0.5710  0.672     0.781       0.823        -0.042   64.0      1.927
25    0.5952  0.562     0.771       0.802        -0.031   64.0      1.967
```

Interpretation: the tiny random-symbol buffer can create a short-horizon edge,
so literal pause/NATO semantics are not necessary. The 64-token cap still drifts
to max-length completions and the edge is gone by step 20.

### 5.15 Config O: Tiny Random-Symbol Buffer, 32-Token Rollout Cap

Config O was Config N with `max_new_tokens=32`.

The first Config O run was launched before the batched decoding hazard was
isolated. It is suggestive, not authoritative:

```text
step  loss    eval_acc  acc_buffer  acc_nobuffer  delta   mean_len  sync_s
0     1.0669  0.578     0.677       0.760        -0.083   26.4      1.917
5     0.8670  0.594     0.802       0.771         0.031   30.9      1.982
10    0.7886  0.641     0.802       0.750         0.052   31.6      1.885
15    0.7274  0.672     0.823       0.833        -0.010   32.0      1.990
20    0.6968  0.703     0.823       0.802         0.021   32.0      1.944
25    0.7495  0.609     0.823       0.844        -0.021   31.8      1.947
30    0.6631  0.688     0.823       0.833        -0.010   30.6      1.958
```

After setting student SGLang to `--max-running-requests 1`, Config O was rerun
for 11 steps. This is the clean result to use:

```text
step  loss    eval_acc  acc_buffer  acc_nobuffer  delta   mean_len  sync_s
0     1.0741  0.469     0.708       0.750        -0.042   25.8      2.050
1     1.0656  0.484     -           -             -       25.0      1.910
2     0.9728  0.578     -           -             -       25.5      2.716
3     0.9500  0.672     -           -             -       30.0      3.403
4     0.9096  0.625     -           -             -       29.6      1.947
5     0.8635  0.656     0.823       0.781         0.042   30.6      1.914
6     0.8761  0.641     -           -             -       31.3      1.988
7     0.8750  0.656     -           -             -       32.0      1.930
8     0.8754  0.531     -           -             -       32.0      2.014
9     0.8357  0.656     -           -             -       31.1      1.956
10    0.8131  0.641     0.833       0.812         0.021   31.3      1.967
```

Interpretation: this is the best current evidence that random tokens can serve
as the reprogrammable slot. The edge is modest and decays from step 5 to step
10, so the next hypothesis should focus on stabilizing or early-stopping the
edge, not on longer training at the same settings.

### 5.16 Config P: Config O With Answer-Only Masking

Config P kept the tiny random-symbol prefill but set
`supervise_student_cot=false`.

```text
step  loss    eval_acc  acc_buffer  acc_nobuffer  delta   mean_len  sync_s
0     0.5219  0.453     0.708       0.740        -0.031   26.8      1.929
5     0.3675  0.688     0.792       0.823        -0.031   31.2      1.921
10    0.3824  0.641     0.875       0.927        -0.052   32.0      1.964
```

Interpretation: answer-only masking is not the right objective for this tiny
random-symbol setup. It improves no-buffer accuracy more than buffer accuracy.
Generated-CoT supervision appears necessary for the buffer edge.

### 5.17 Config Q: Config O With Lower LR

Config Q lowered the client learning rate to `1e-6`.

```text
step  loss    eval_acc  acc_buffer  acc_nobuffer  delta   mean_len  sync_s
0     1.0567  0.469     0.708       0.750        -0.042   27.0      1.879
5     0.9985  0.609     0.760       0.771        -0.010   25.0      1.923
10    0.9027  0.562     0.823       0.823         0.000   31.3      1.926
15    0.8541  0.625     0.812       0.812         0.000   30.9      1.877
20    0.8187  0.688     0.833       0.823         0.010   31.6      1.928
25    0.7369  0.594     0.844       0.833         0.010   31.6      1.943
30    0.7229  0.594     0.854       0.865        -0.010   30.2      1.994
```

Interpretation: lower LR is stable but weak. It does not reproduce the cleaner
Config O step-5/10 edge.

### 5.18 Config R: Larger Eval Verification of Config O

Config R was intended to rerun Config O with `eval_num_problems=192`, but it was
first launched before the batched decoding failure was isolated.

```text
Invalid batched run:
step  loss    eval_acc  acc_buffer  acc_nobuffer  delta   mean_len  sync_s
0     1.1072  0.000     0.005       0.000         0.005   22.0      1.928
5     0.7959  0.000     0.005       0.005         0.000   32.0      1.881
```

Manual inspection found repeated numeric suffixes across unrelated prompts. Do
not use Config R as evidence against OPD or against random tokens. It is evidence
against the high-concurrency Qwen3.6 student sampler path.

After setting the student SGLang pod to `--max-running-requests 1`, Config R was
rerun cleanly:

```text
run_dir=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T021553Z-configR-er-opd-q36-35b-slots-trainer-head
wandb=6tbkg906

step  loss    eval_acc  acc_buffer  acc_nobuffer  delta   mean_len  sync_s
0     1.0868  0.391     0.729       0.740        -0.010   25.5      1.972
1     1.0376  0.516     -           -             -       25.8      1.949
2     0.9926  0.562     -           -             -       25.8      1.952
3     0.9271  0.688     -           -             -       28.6      1.988
4     0.8944  0.641     -           -             -       30.0      1.919
5     0.8473  0.719     0.818       0.839        -0.021   31.2      2.012
6     0.8575  0.641     -           -             -       32.0      1.920
7     0.8683  0.562     -           -             -       31.7      2.202
8     0.8229  0.719     -           -             -       32.0      1.948
9     0.7714  0.672     -           -             -       30.8      1.941
10    0.7630  0.719     0.844       0.839         0.005   32.0      1.918
```

Interpretation: larger eval did not confirm Config O's small random-symbol
advantage. Config R is essentially a tie/noise result. The random-symbol cue can
be learned as text too: samples sometimes echoed `| ~ _ * ^ # @ Answer`.

### 5.19 Config S: Tiny NATO, 32-Token Rollout Cap

Config S is Config M with `max_new_tokens=32`, matching the short-cap O/R family
while using the tiny structured NATO cue:

```text
Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel
```

```text
run_dir=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T025923Z-configS-er-opd-q36-35b-slots-trainer-head
wandb=duhyd4e4

step  loss    eval_acc  acc_buffer  acc_nobuffer  delta   mean_len  sync_s
0     1.0264  0.516     0.760       0.729         0.031   27.0      1.895
1     1.0321  0.562     -           -             -       26.7      1.902
2     0.9419  0.578     -           -             -       28.5      1.917
3     0.9173  0.641     -           -             -       30.6      1.933
4     0.8837  0.688     -           -             -       31.6      3.663
5     0.8538  0.625     0.802       0.802         0.000   30.4      1.913
6     0.8700  0.531     -           -             -       31.7      2.236
7     0.8491  0.641     -           -             -       32.0      1.942
8     0.8397  0.656     -           -             -       32.0      1.994
9     0.8004  0.688     -           -             -       32.0      1.958
10    0.7883  0.734     0.833       0.865        -0.031   32.0      1.950
```

Interpretation: tiny structured tokens do not solve the problem. S starts with a
small buffer edge, ties at step 5, and is negative by step 10. Samples show the
same failure mode as random symbols: max-length verbose continuations and
occasional cue echoing, for example `Alpha Bravo Charlie Delta Echo Foxtrot Golf
Hotel Answer: ...`.

### 5.20 Config T: Tiny Random-Symbol, 16-Token Rollout Cap

Config T is Config O with `max_new_tokens=16`. It tests whether the 32-token cap
lets post-answer verbosity erase a real tiny random-symbol edge.

```text
run_dir=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T033519Z-configT-er-opd-q36-35b-slots-trainer-head
wandb=k08076wg

step  loss    eval_acc  acc_buffer  acc_nobuffer  delta   mean_len  sync_s
0     1.1940  0.359     0.698       0.740        -0.042   15.7      3.779
1     1.1549  0.344     -           -             -       15.9      1.988
2     1.1106  0.500     -           -             -       15.7      2.009
3     1.0705  0.562     -           -             -       15.8      1.940
4     1.0463  0.531     -           -             -       15.8      1.931
5     0.9866  0.609     0.823       0.823         0.000   15.7      1.925
6     0.9760  0.656     -           -             -       15.9      2.011
7     0.9726  0.703     -           -             -       15.9      2.201
8     0.9498  0.641     -           -             -       15.6      1.999
9     0.9059  0.750     -           -             -       16.0      2.205
10    0.8764  0.750     0.844       0.885        -0.042   15.9      1.898
```

Interpretation: shorter rollout improves late task accuracy but not the encoded
buffer edge. The cap often clips text immediately after a correct answer or
after `</think>`, and the final paired eval favors no-buffer. Do not pursue
shorter caps as the primary fix. If output-length control is revisited, try an
answer-format/parser fix rather than just lowering `max_new_tokens`.

### 5.21 Config U: Tiny Random-Symbol, Lower Hidden-Match Weight

Config U is Config O with `opd_hidden_match_coef=0.25` instead of `1.0`. It
tests whether hidden-MSE pressure is causing the model to learn the teacher's
verbose/filler artifacts faster than the answer channel.

```text
run_dir=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T040214Z-configU-er-opd-q36-35b-slots-trainer-head
wandb=igmnt3sq

step  loss    eval_acc  acc_buffer  acc_nobuffer  delta   mean_len  sync_s
0     0.9261  0.531     0.719       0.760        -0.042   27.5      1.945
1     0.9356  0.453     -           -             -       24.6      1.939
2     0.9118  0.578     -           -             -       23.5      1.948
3     0.8760  0.625     -           -             -       27.5      1.923
4     0.8287  0.656     -           -             -       29.8      2.169
5     0.7640  0.719     0.812       0.802         0.010   31.4      1.900
6     0.7743  0.609     -           -             -       31.0      1.946
7     0.7963  0.547     -           -             -       31.8      1.917
8     0.7538  0.625     -           -             -       32.0      2.003
9     0.7465  0.672     -           -             -       31.5      1.915
10    0.7404  0.641     0.792       0.760         0.031   32.0      3.324
```

Interpretation: lowering hidden-MSE weight helped the paired buffer metric
relative to Config R/T, but it did not cleanly solve the mechanism. U ended with
a positive step-10 delta, while task eval stayed similar to O and below T. Mean
completion length still saturated at the 32-token cap, and samples showed the
same text-prefill artifact class: verbose post-answer arithmetic and occasional
random-symbol leakage such as `! | ~ _ * ^ # @ Answer`. This supports the
user's random-token hypothesis in a narrow sense: arbitrary short filler can be
competitive with or better than the NATO/pause cue. It also says printable text
tokens are a contaminated proxy for latent slots.

### 5.22 Config V: Tiny Random-Symbol, KL-Only Generated-CoT Supervision

Config V is Config U with `opd_hidden_match_coef=0.0`. It keeps generated-CoT
supervision enabled and leaves the tiny random-symbol text prefill, 32-token cap,
answer-unmasked OPDB server config, and `lr=3e-6` unchanged.

```text
run_dir=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T043740Z-configV-er-opd-q36-35b-slots-trainer-head
wandb=fho1il09

step  loss    eval_acc  acc_buffer  acc_nobuffer  delta   mean_len  sync_s
0     0.9159  0.500     0.708       0.750        -0.042   26.6      1.859
1     0.9008  0.578     -           -             -       25.0      1.917
2     0.8706  0.594     -           -             -       23.3      1.948
3     0.8233  0.734     -           -             -       27.7      1.938
4     0.7868  0.734     -           -             -       29.2      2.005
5     0.7077  0.781     0.823       0.823         0.000   31.1      1.969
6     0.7384  0.672     -           -             -       31.6      1.985
7     0.7258  0.688     -           -             -       32.0      1.971
8     0.7431  0.766     -           -             -       32.0      1.946
9     0.7095  0.719     -           -             -       31.9      1.973
10    0.6903  0.703     0.792       0.833        -0.042   32.0      1.938
```

Interpretation: KL-only generated-CoT supervision is better for raw task
accuracy, but worse for the buffer-specific channel. It ties at step 5 and is
negative by step 10, while no-buffer reaches the best paired-control accuracy of
the recent random-symbol runs. Hidden-state matching is not the primary cause of
the failure mode; if anything, some hidden matching may help preserve the
buffer-conditioned behavior. The persistent issue is that text prefill plus
teacher-CoT imitation trains verbose answer continuations, not a clean latent
slot.

### 5.23 Config W: Exact Random Token-ID Buffer

Config W is the true token-ID version of Config U. It uses OPDB
answer-unmasked generated-CoT supervision, `lr=3e-6`, `max_new_tokens=32`,
`opd_hidden_match_coef=0.25`, and exact random prefill IDs:

```text
18437,62981,31709,90553,74216,118927,56344,100731
```

Client/generator changes:

- `xorl-client-chat-completions/examples/on_policy_distillation.py` now accepts
  `student_prefill_token_ids` as JSON or comma/space-separated IDs.
- When exact IDs are configured, the student sampler uses SGLang `/generate`
  with `ModelInput.from_ints(rendered_chat_prompt + exact_ids + suffix_ids)`.
  The high-level config still uses `inference_api_format=chat_completions` so
  prompt loading and chat-template rendering remain unchanged.
- Control eval uses the same exact-prefix path: pause arm is
  `prompt_ids + exact_ids + suffix_ids`, no-pause arm is
  `prompt_ids + suffix_ids`.

Launch command:

```bash
python experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py write-trainer-control --config W --num-steps 11 --prompts-per-step 64
```

Result:

```text
run_dir=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T051706Z-configW-er-opd-q36-35b-slots-trainer-head
wandb=e4fspgob

step  loss    eval_acc  acc_buffer  acc_nobuffer  delta   lead_delta  mean_len  sync_s
0     0.9564  0.250     0.688       0.719        -0.031   +0.364      27.7      1.995
1     0.9298  0.438     -           -             -       -           27.8      2.019
2     0.8569  0.406     -           -             -       -           30.8      2.021
3     0.7649  0.625     -           -             -       -           30.4      2.137
4     0.7501  0.578     -           -             -       -           31.5      2.029
5     0.7277  0.656     0.792       0.802        -0.010   -0.002      30.6      2.023
6     0.7375  0.438     -           -             -       -           31.5      1.977
7     0.7374  0.562     -           -             -       -           31.8      2.062
8     0.7332  0.688     -           -             -       -           31.7      2.000
9     0.7112  0.766     -           -             -       -           31.6      2.035
10    0.7116  0.719     0.833       0.844        -0.010   -0.005      32.0      2.115
```

Operational notes:

- Exact-token path was active: log line reported `8 buffer tokens + suffix
  'Answer: ' (3 tokens) = 11 filler tokens (K, /generate input_ids)`.
- The client instantiated the student sampler with `api_format=generate`, while
  keeping chat prompt rendering via `inference_api_format=chat_completions`.
- The exact-ID path was much slower than text chat-completions because the
  serialized student sampler served `/generate` requests one at a time; steady
  mean `student_sampling_s` was about 403s.

Interpretation: exact random token IDs did not outperform the printable
random-symbol buffer. They avoided text-token echo/leakage by construction, but
the model still learned verbose post-answer continuations and filled the
32-token cap. Step 5 and step 10 both slightly favored no-buffer. The positive
step-0 leading-digit delta did not survive training. This falsifies the narrow
"printable text contamination is the main blocker" hypothesis.

### 5.24 Config X: Tiny Random-Symbol With Newline Stop

Config X is Config U plus a student-side stop-sequence control:

```text
student_stop_sequences=["\n"]
```

It keeps OPDB answer-unmasked generated-CoT supervision, `lr=3e-6`,
`max_new_tokens=32`, `opd_hidden_match_coef=0.25`, and the tiny random-symbol
prefill:

```text
 ! | ~ _ * ^ # @
```

Client/generator changes:

- `xorl-client-chat-completions/examples/on_policy_distillation.py` now accepts
  `student_stop_sequences` as a JSON list or pipe-separated string.
- `_sample_student_batch` passes the parsed stop sequences into SGLang sampling.
- `_buffer_control_eval` uses the same stop sequences for paired buffer/no-buffer
  control eval.
- Tests were added for stop parsing and for stop propagation to rollout/control
  sampling params.
- The generator adds Config X with the `randsymbol-stopnl` run label.

Validation before launch:

```bash
cd /home/apanda/xorl-client-chat-completions
uv run pytest tests/test_on_policy_distillation_example.py -q
uv run ruff check examples/on_policy_distillation.py tests/test_on_policy_distillation_example.py

cd /home/apanda/xorl-apanda-dev-opd-port
uv run ruff check experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py
python -m py_compile experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py
```

All checks passed.

Launch:

```bash
python experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py write-trainer-control --config X --num-steps 11 --prompts-per-step 64
```

Result:

```text
run_dir=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T060839Z-configX-er-opd-q36-35b-slots-trainer-head
wandb=bhj9dmso

step  loss    eval_acc  acc_buffer  acc_nobuffer  delta   lead_delta  mean_len  sync_s
0     0.9394  0.516     0.729       0.750        -0.021   +0.116      26.1      1.922
1     0.9437  0.547     -           -             -       -           25.4      1.978
2     0.8782  0.609     -           -             -       -           26.6      2.108
3     0.8289  0.656     -           -             -       -           30.0      1.925
4     0.7965  0.672     -           -             -       -           30.8      1.966
5     0.7347  0.781     0.812       0.833        -0.021   -0.010      31.3      1.902
6     0.7618  0.719     -           -             -       -           30.4      1.915
7     0.7689  0.734     -           -             -       -           31.6      1.954
8     0.7583  0.734     -           -             -       -           32.0      2.039
9     0.7271  0.781     -           -             -       -           31.8      2.170
10    0.7148  0.719     0.875       0.865         0.010   +0.004      32.0      2.105
```

Operational notes:

- The first Config X attempt failed before science because the student SGLang
  receiver still had stale P2P state/NIC metadata from an earlier run:
  `Peer nic not found`, `received packet mismatch`, and `batch_transfer`
  failure. Restarting student inference flushed that state.
- The successful launch initially exposed a dispatch race: SMG started before
  the student SGLang backend had registered the model and returned
  `/v1/models` with `unknown`. The generator now waits for each backend
  `/v1/models` response to contain `Qwen/Qwen3.6-35B-A3B` before launching SMG,
  and exposes `write-dispatch-control` to restart only dispatch.
- GPU-direct P2P sync was healthy throughout the successful run. The initial
  registration sync moved 69.32 GB in about 3.0s; steady training syncs moved
  69.32 GB in about 1.9-2.2s.
- Trainer head exited cleanly at the end of step 10. Trainer slots were then
  stopped with:

```bash
python experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py stop-trainer-control --remove-run
```

Interpretation: newline stop did not solve the output-channel failure. Task
accuracy was strong by the standards of this sweep, peaking at 0.781 at steps 5
and 9, but the paired buffer metric stayed negative at steps 0 and 5 and only
barely positive at step 10. Mean completion length reached the 32-token cap by
step 8 despite `student_stop_sequences=["\n"]`, and samples still contained
post-answer reasoning after the answer. Treat Config X as another weak/noisy
result, not a durable encoded-reasoning channel.

### 5.25 Config Y: Tiny Random-Symbol Buffer-Only, Lower Hidden-Match

Config Y is defined as the next objective-level test and is not yet launched as
of this update. It keeps the tiny random-symbol buffer, `lr=3e-6`,
`max_new_tokens=32`, and `opd_hidden_match_coef=0.25`, but changes the OPD
client objective to:

```text
opd_supervise_buffer_only=true
```

Rationale: Configs O/R/S/T/U/V/W/X mostly learn the answer path, so no-buffer
control accuracy rises with buffer accuracy. Config Y masks the answer positions
out of the OPD loss, forcing the teacher-CoT-conditioned signal to land on the
student buffer positions. This is a direct test of whether the recent failures
come from the answer-unmasked objective dominating the buffer-conditioned path.

Launch command:

```bash
python experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py write-trainer-control --config Y --num-steps 11 --prompts-per-step 64
```

## 6. Monitoring Commands

Recent rows:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T060839Z-configX-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
print(f"rows={len(rows)}")
for r in rows[-20:]:
    def f(x, nd=3):
        return "-" if x is None else f"{x:.{nd}f}"
    print(
        f"step={r.get('step')} loss={f(r.get('loss'),4)} acc={f(r.get('eval/accuracy'))} "
        f"pause={f(r.get('eval/acc_pause'))} nopause={f(r.get('eval/acc_nopause'))} "
        f"delta={f(r.get('eval/buffer_delta'))} sync_transfer_s={f(r.get('sync_transfer_time_s'))}"
    )
PY
```

Control rows only:

```bash
jq -r 'select(.["eval/acc_pause"] != null) |
  [.step,
   .loss,
   .["eval/accuracy"],
   .["eval/acc_pause"],
   .["eval/acc_nopause"],
   .["eval/buffer_delta"],
   .sync_transfer_time_s] | @tsv' "$PROFILE_X"
```

Trainer wrapper log:

```bash
tail -120 /shared/opd-control/er-opd-q36-35b-slots/trainer-head/logs/20260603T060839Z-run.log
```

Trainer API health:

```bash
kubectl exec -n apanda er-opd-q36-35b-slots-trainer-head -- \
  bash -lc 'curl -m 5 -fsS http://127.0.0.1:26050/health'
```

Student/dispatch health:

```bash
kubectl exec -n apanda er-opd-q36-35b-slots-sglang-0 -- \
  bash -lc 'curl -m 3 -fsS http://127.0.0.1:30060/health'

kubectl exec -n apanda er-opd-q36-35b-slots-dispatch -- \
  bash -lc 'curl -m 3 -fsS http://127.0.0.1:8080/v1/models'
```

## 7. Decision Criteria

Keep a future run running while:

- `sync_success` remains true.
- steady-state sync transfer time remains around 1.8-2.2s.
- the serialized sampler path is active, or a fixed batched path has been
  separately validated with a held-out base probe.
- `acc_buffer`/`acc_pause` stays above its own step-0 baseline.
- `acc_buffer - acc_nobuffer` turns positive by step 5 or step 10, or the run
  is explicitly measuring a negative-control variant.

Checkpoints to record:

- Step 5
- Step 10
- Step 15
- Step 20
- Step 50
- Step 100
- Step 200
- Step 400 or final step

Stop or reprogram if:

- P2P sync fails.
- Any trainer pod loses rank processes.
- `acc_buffer` collapses below about 0.5 by step 50.
- `acc_buffer - acc_nobuffer` is negative at two consecutive control
  checkpoints after step 5.
- Samples show severe format corruption dominating the batch.
- Batched eval shows repeated or near-repeated suffixes across unrelated
  prompts. That is a sampler bug, not a science result.

Do not rerun A-X as-is. The next useful hypotheses are:

- Objective change: preserve or increase `acc_buffer - acc_nobuffer`. Config V
  shows KL-only generated-CoT supervision teaches the direct answer path more
  than the buffer-conditioned path, Config W shows exact random IDs do not fix
  the channel, and Config X shows newline stop is not enough. Config Y is the
  immediate next test: buffer-only, tiny random-symbol, low hidden-match.
- If Config Y is negative, stop sweeping filler-token variants and hidden-match
  coefficients. The next science recipe likely needs an explicit
  contrastive/eval-aware term, a paired no-buffer penalty, or another mechanism
  that makes the no-buffer path worse while keeping the buffer path trainable.
- Output-format control remains relevant because X still saturated the 32-token
  cap with post-answer reasoning. If revisited, use a stronger format/control
  mechanism than SGLang newline stop, such as a digit-only answer parser in the
  training target or a prompt/schema that naturally terminates after the number.
- Multiple serialized student sampler pods. This restores throughput without
  reintroducing per-pod batched decoding.

## 8. Cleanup

To stop only trainer processes and keep warm student/teacher services:

```bash
python experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py stop-trainer-control
python experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py status
```

Use `--remove-run` when intentionally clearing stale run control state before a
fresh trainer launch. It is not needed just to stop workers after a completed
run.

To stop all slot processes without deleting pods:

```bash
python experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py stop-control --remove-run
```

To delete the pods and services entirely:

```bash
kubectl delete -n apanda -f experiments/opd_profile/k8s/generated/er-opd-q36-35b-slots.yaml --wait=true --timeout=300s
```

Use deletion only when the slots are no longer useful. The whole point of this
stack is to preserve scheduled pods and reprogram their control scripts.
