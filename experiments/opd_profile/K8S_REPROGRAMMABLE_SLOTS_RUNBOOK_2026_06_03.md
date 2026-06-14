# Kubernetes Reprogrammable Slots Runbook

Date: 2026-06-03
Stack: `er-opd-q36-35b-slots`
Namespace: `apanda`

This runbook describes how the OPD Qwen3.6-35B experiments reuse already
scheduled Kubernetes pods as programmable GPU slots. The key point: we are not
submitting a fresh Kubernetes Job for every recipe. We keep pods allocated and
rewrite per-role control scripts on the shared filesystem. Each pod runs a tiny
slot agent that starts, stops, and replaces its child process when the control
script changes.

## 0. Handoff Summary, 2026-06-03 20:21 UTC

This file is the canonical handoff for the Qwen3.6-35B OPD reprogrammable-slot
series. Treat local `opd_profile.jsonl` rows as the source of truth during a
run. W&B usually catches up and is useful for UI review, but it has lagged by a
step or more during long final controls.

Current stack state:

- Stack: `er-opd-q36-35b-slots`, namespace `apanda`.
- Warm inference roles are running: `sglang-0`, `dispatch`,
  `teacher-sglang-0`, `teacher-sglang-1`, and `teacher-smg`.
- The dedicated `sglang-1` slot is stopped. Current two-sampler runs use
  `--sampler-layout spare-teacher1`, where `teacher-sglang-1` is the second
  student sampler and `teacher-sglang-0` remains the teacher hidden-cache server.
- Student dispatch is `round_robin` over:
  `er-opd-q36-35b-slots-sglang-0:30060` and
  `er-opd-q36-35b-slots-teacher-sglang-1:30000`.
- Per-native SGLang pods stay serialized with `--max-running-requests 1`.
  For throughput, add serialized sampler pods and sync all of them freshly. Do
  not increase per-pod batching for Qwen3.6 science until the repeated-suffix
  batched-decoding failure is fixed and revalidated.
- `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0` is required; `/health` must be a
  cheap readiness probe, not a hidden generation request.
- All GPU pods must carry the pod-template label `team: turbo`.

### 0.1 Current Run State

Config AN is the active run at this handoff:

```text
run_dir=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T200723Z-configAN-er-opd-q36-35b-slots-trainer-head
wandb_run=ix09vxoy
control_revision=20260603T200722Z
launch=python experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py write-trainer-control --config AN --num-steps 6 --prompts-per-step 64 --sampler-replicas 2 --sampler-layout spare-teacher1
```

AN is the integrated clean-control rerun of AM. It keeps the final-only
`n=1024` held-out controls, `rotate_preserve_ws` corrupt generation, chunked
answer-logprob scoring, and answer-selection distractor gate. It restores the
AM/AH/AI corrupt-negative answer/buffer training objective so the run answers:
"does the earlier AM signal survive when the fixed corrupt boundary artifact is
removed in-loop?"

Observed AN state as of this snapshot:

- Trainer head and seven trainer workers are running.
- Local profile has rows `0..4`. The final step-5 control row has not landed
  yet.
- Step accuracies so far: `0.46875`, `0.546875`, `0.4375`, `0.40625`,
  `0.203125`.
- Every emitted row reports clean two-endpoint serial sync:
  `sync_endpoint_success_count=2`, `sync_endpoint_failure_count=0`,
  `sync_serial_endpoint_sync=1.0`.
- Sampler routing is balanced with `sampler_worker_success_balance_ratio=1.0`.
- W&B run `ix09vxoy` is live but may lag the local profile during the final
  control.

AN had four failed operational attempts before the live run above:

- `20260603T195017Z`: interrupted manually while a fresh sync was in progress.
- `20260603T195301Z`: hit stale/wedged `sglang-0` after the interrupted sync.
- `20260603T200016Z`: endpoint sync succeeded, but the old 30-second post-sync
  `/generate` probe timed out during sampler warmup. No profile row was emitted.
- `20260603T200547Z`: invalid trainer gang because `trainer-worker-3` was in
  Kubernetes `Error`.

The recovery sequence was:

```bash
python "$GENERATOR" stop-trainer-control --remove-run
kubectl delete pod er-opd-q36-35b-slots-trainer-worker-3 -n apanda --wait=true
python "$GENERATOR" render-manifest --sampler-replicas 2 --sampler-layout spare-teacher1 | kubectl apply -n apanda -f -
python "$GENERATOR" write-trainer-control --config AN --num-steps 6 --prompts-per-step 64 --sampler-replicas 2 --sampler-layout spare-teacher1
```

When applying the full manifest to recreate one raw Pod, immutable-field errors
for already-running Pods are expected. The important line was
`pod/er-opd-q36-35b-slots-trainer-worker-3 created`.

### 0.2 Results Ledger

Decision-critical completed runs:

| Config | W&B | Result | Interpretation |
| --- | --- | --- | --- |
| T | `k69dfbt5` | Two-endpoint infra smoke passed. | Proved `spare-teacher1`, round-robin dispatch, serial sync, and W&B logging substrate. Not science, because `max_new_tokens=16` made cap-hit `1.0`. |
| AG | `oxo5yw7i` | NCCL strong-corrupt contrast passed operationally. | Established the cleaner two-sampler NCCL substrate. Science remained limited by corrupt degeneracy. |
| AH | `zfkrmifn` | `acc_pause=0.5000`, `acc_nopause=0.2292`, `acc_corrupt=0.1354`; `buffer_delta=+0.2708`, `z=4.06`. | First strong mechanism hit. Rejected because control `n=96` was too small and corrupt generation was degenerate (`cap_hit=0.9375`, filler leak `1.0`). |
| AI | `fzittb3d` | `acc_pause=0.6719`, `acc_nopause=0.5781`, `acc_corrupt=0.6094`; `buffer_delta=+0.0938`, `z=1.9063`; answer-logprob `z=16.6732` vs no-pause and `z=17.3105` vs corrupt. | Operationally clean strict `n=192` rerun. Partial science hit. Exact match was underpowered, answer-logprob was strong, corrupt cap-hit remained high (`0.7083`). |
| AL | `uxwype0s` | `buffer_delta=+0.1146`, `z=2.4091`; `buffer_vs_corrupt=+0.1094`, `z=2.3028`; answer-logprob `z=17.0395`. | Real mechanism hit with balanced cache-mismatch plus answer contrast. Rejected only by corrupt cap-hit (`0.5833`). |
| AM | `c7f09nym` | Final-only `n=1024`: `acc_pause=0.6494`, `acc_nopause=0.5557`, `acc_corrupt=0.5811`; `buffer_delta=+0.0938`, `z=4.3548`; `buffer_vs_corrupt=+0.0684`, `z=3.1871`. | Scaled AI/AH to 1k controls and confirmed the pause-vs-no-pause exact-match signal. In-loop answer-logprob failed from oversized batches, but post-hoc chunked scoring was strongly positive. Not promoted because fixed corrupt exact-match had a boundary/cap artifact and answer-selection was negative. |
| AO | `8psasdb4` | `acc_pause=0.7480`, `acc_nopause=0.7803`, `acc_corrupt=0.7559`; `buffer_delta=-0.0322`, `z=-1.719`; answer-logprob margin `-0.00688`, `z=-2.514`; answer-selection `z=-34.61`. | Positive-answer KL plus balanced cache-mismatch, no corrupt-negative training. Operationally clean and scientifically rejected. It did not make pause-causal answer use emerge. |
| AN | `ix09vxoy` | Live, rows `0..4`, final row pending. | Integrated clean rerun of the AM/AH/AI answer-contrast direction with final-only `n=1024`, preserved-boundary corrupt, chunked logprob, and answer-selection gate. |

Earlier substrate and diagnostic runs:

- Z/AA/AB/AC established the fixed-slot recipe, stop-newline substrate,
  contrastive-buffer plumbing, W&B/profile parity checks, and the first
  two-sampler controls.
- AD added the answer-logprob causal gate.
- AE added the sync-barrier pilot and made sampler quiescence observable.
- AF tried the strong-corrupt rerun before NCCL sync was the default; AG is the
  cleaner version to cite.
- AJ proved teacher-memory pair diagnostics: teacher memory rows are
  prompt-specific but only weakly separated.
- AK proved bounded control fanout but showed the unbalanced cache-mismatch
  objective pushed the held-out control in the wrong direction.
- The AL one-step smoke `fl7577vn` showed balanced cache-mismatch removed most
  of AK's exact-match regression before the full AL run.

### 0.3 W&B Run Map

Recent decision-critical W&B runs:

```text
AG  oxo5yw7i
AH  zfkrmifn
AI  fzittb3d
AI-smoke  05o5f8h6
AJ  yazdxxgu
AK  ch41l7u5, 9li9mwbu
AL  fl7577vn, uxwype0s
AM  c7f09nym
AO  8psasdb4
AN  ix09vxoy
```

Historical run map from the earlier sweep:

```text
A   3a7byynt, txvq0ad3, vli4jitt, yhzxebd3
B   j5hy662k
C   g64zkdgm
D   euh75lgo
E   2cvqn5vb
F   z2um6i24
G   0xbvnuvv
H   ee6fr8no
I   jf8hdjpw
J   l9ifyx1x
K   g57r7zn6
L   tmgnqdom
N   410u1bao
O   1voc936c, 5r8vml5x
P   gxp2fv3g
Q   o44wnxnp
R   5tikvk21, ol5wmkfl, 6tbkg906
S   duhyd4e4
T   k08076wg, 4hcj4amx, k69dfbt5
U   igmnt3sq
V   fho1il09
W   e4fspgob
X   bhj9dmso
Y   3hyyw0i6
Z   30ytmgke, 24wzc3g2, n5ngv35w
AA  2wb7s6da
AB  ij8495s3, 21wumhbl, cizpg3ae
AC  r8bc87nb, t8eazs3n
AD  mu7bhmm4, fngf2zxn
AE  r1egk9sc
AF  wx2hfhew, u9wqf40r
```

Known W&B caveats:

- `30ytmgke` and `21wumhbl` are stale or partial after stop/crash boundaries.
- During live controls, W&B can lag local profile rows. Do not decide
  promote/reject from W&B alone until `audit_wandb_profile.py` passes.
- AN failed attempts before `20260603T200723Z` did not produce usable W&B runs.

### 0.4 Code And Instrumentation Changes

Generator changes in `experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py`:

- Added configs AM, AN, and AO.
- Added `--sampler-layout spare-teacher1` so `teacher-sglang-1` can be reused
  as a second student sampler when the dedicated `sglang-1` pod is unavailable.
- Switched multi-sampler student dispatch to `round_robin`. `cache_aware` was
  scientifically bad here because same-prefix traffic could route almost
  entirely to one sampler endpoint.
- Enabled serial endpoint weight sync for two student samplers:
  `XORL_SERIAL_INFERENCE_ENDPOINT_SYNC=1`.
- Added robust post-sync sampler recovery: call `/continue_generation`, retry
  `/generate` probes, and treat healthy `/health` after probe timeouts as a
  warning instead of killing a valid run. This fixed the false fatal abort seen
  in the `20260603T200016Z` AN attempt.
- Kept native SGLang per-pod serialization with `--max-running-requests 1`.

Client changes in
`/home/apanda/xorl-client-chat-completions/examples/on_policy_distillation.py`:

- Added `opd_positive_answer_weight` and metrics
  `opd_positive_answer_examples`.
- Added `eval_control_start_step`; AM/AN/AO use final-only `n=1024` controls at
  step 5 instead of running 1k controls at every eval point.
- Added bounded sampled-control fanout via `eval_control_max_concurrency=16`
  plus client-queue and service-latency metrics.
- Added chunked answer-logprob scoring:
  `eval_answer_logprob_batch_size=64` and
  `eval_answer_logprob_max_concurrency=2` for AM/AN/AO. This avoids the AM
  oversized batch failure.
- Added answer-selection distractor scoring:
  `eval_answer_logprob_distractor_control=true`, with correct-vs-wrong answer
  margins and z scores.
- Added control progress W&B logging during long final controls.
- Added corrupt-control boundary metrics and the `rotate_preserve_ws` mode.

Analysis/monitoring changes:

- `experiments/opd_profile/monitor_live_opd.py` now combines local profile,
  W&B state/history, SMG dispatcher counters, and optional native SGLang logs.
  It distinguishes `live_progress` from `live_progress_native` while a long
  final control is still draining.
- `experiments/opd_profile/analyze_slot_profile.py` now enforces serial sync,
  sampler routing/balance, answer-selection gates, and post-hoc overlay inputs.
- `experiments/opd_profile/audit_wandb_profile.py` audits the expanded metric
  set, including positive answer, sync, sampler, corrupt-boundary, and
  answer-selection fields.

### 0.5 Infra Rules For The Next Agent

Reprogrammable slots:

- `kubectl get pods` only tells you the slot-agent Pod is alive. Use
  `python "$GENERATOR" status` for the child process state.
- Editing or rewriting `$CONTROL_ROOT/$ROLE/run.sh` is a deployment. The
  slot-agent hashes `run.sh`, stops the old child process group, and starts the
  new script.
- Use `stop-trainer-control --remove-run` before relaunching trainer roles.
  Without `--remove-run`, the slot agent can process `stop` and then relaunch a
  stale `run.sh`.
- Keep warm inference roles up between recipe iterations. Use
  `write-trainer-control` for normal science iteration. Use
  `write-student-inference-control` only when student samplers or dispatch need
  recovery.
- If a raw Pod is in Kubernetes `Error`, stop/remove the run, delete that Pod,
  then apply the manifest. Full-manifest `kubectl apply` may error on immutable
  fields for existing Pods; the missing Pod creation line is what matters.

Sync and sampler rules:

- Prefer `sync_method=nccl_broadcast` for this stack.
- With two sampler endpoints, require:
  `sync_endpoint_success_count=2`, `sync_endpoint_failure_count=0`,
  `sync_serial_endpoint_sync=1.0`, `sampler_worker_active_count=2`, and a
  near-balanced `sampler_worker_success_balance_ratio`.
- Keep sampler quiescence enabled before sync:
  `sampler_quiesce_success=1.0` and zero outstanding/connections/inflight.
- A clean two-endpoint sync transfers about `2 * 70GB` and commonly takes
  roughly 14-20 seconds in the current layout.

Monitoring rules:

- During a live final control, watch the local profile first:

```bash
PROFILE=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T200723Z-configAN-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl
python experiments/opd_profile/monitor_live_opd.py "$PROFILE" \
  --wandb-run ix09vxoy --samples 2 --interval-s 15 \
  --native-log-pod er-opd-q36-35b-slots-sglang-0 \
  --native-log-pod er-opd-q36-35b-slots-teacher-sglang-1
```

- `VERDICT: live_progress` means dispatcher traffic is returning and aged
  inflight buckets are not building.
- `VERDICT: live_progress_native` means sampled-control traffic has drained and
  native SGLang scoring is still active.
- Long `eval/control_request_latency_*` is acceptable when it is mostly
  `eval/control_client_queue_latency_*`. Rising service latency, request
  failures, or SMG aged-inflight counters are infra regressions.

Post-run gates:

```bash
python experiments/opd_profile/analyze_slot_profile.py "$PROFILE" \
  --min-control-n 1024 \
  --require-answer-select-control \
  --min-answer-select-z 2 \
  --min-answer-select-paired-n 1024 \
  --expected-sync-endpoints 2 \
  --require-serial-sync \
  --require-sampler-routing \
  --min-sampler-active-workers 2 \
  --min-sampler-balance-ratio 0.95

python experiments/opd_profile/audit_wandb_profile.py "$PROFILE" --wandb-run ix09vxoy
```

### 0.6 Science Diagnosis And Next Steps

The answer is not "copy RiM." RiM is relevant because it isolates a trainable
memory-like mechanism, but the OPD failure mode here is more specific: the loop
can make the pause/filler path increase answer likelihood without proving that
the model selects prompt-specific memory content. Absolute answer logprob is too
easy to improve; AM's answer-selection distractor control was strongly negative.

Current diagnosis:

- AH/AI/AM show a real pause-vs-no-pause mechanism signal. AM confirms that the
  AI exact-match effect was underpowered at `n=192`, not absent.
- The legacy fixed-token corrupt arm is contaminated by a visible boundary/cap
  artifact. `rotate_preserve_ws` fixes that artifact and weakens the
  free-generation corrupt exact-match contrast.
- AO shows that positive-answer KL plus balanced cache-mismatch hidden
  supervision is not sufficient.
- The next causal proof must be prompt-specific. Good candidates are:
  externalized prompt-specific memory tokens that can be shuffled, or
  same-visible-input cache/hidden-target mismatch where the teacher memory rows
  come from another prompt.
- Promotion should require both exact-match pause-vs-no-pause improvement and
  positive answer-selection improvement, not just absolute answer-logprob.

Immediate next steps:

1. Let AN reach the step-5 row unless infra clearly fails.
2. Run the strict analyzer and W&B audit above.
3. If AN only fails the corrupt exact-match gate while answer-selection remains
   negative, stop spending full runs on fixed-token rotation.
4. Implement the next prompt-specific memory control. Keep the reprogrammable
   slot substrate unchanged: two serialized samplers, serial NCCL sync,
   round-robin dispatch, bounded control fanout, and final-only 1k controls.

## 1. Source Of Truth

```bash
GENERATOR=experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py
MANIFEST=experiments/opd_profile/k8s/generated/er-opd-q36-35b-slots.yaml
CONTROL_ROOT=/shared/opd-control/er-opd-q36-35b-slots
RESULT_ROOT=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots
NS=apanda
STACK=er-opd-q36-35b-slots
```

The generator is the only thing future agents should edit for slot layout,
role scripts, and config recipes. The control root is shared by all slot pods
and is how we reprogram them.

## 2. Slot Roles

Current roles:

```text
sglang-0                 student SGLang, TP=8, port 30060
sglang-1                 optional second dedicated student SGLang when rendered with --sampler-replicas 2
dispatch                 SMG dispatch for student sampling, port 8080
teacher-sglang-0         teacher SGLang, TP=8, port 30000
teacher-sglang-1         spare/second teacher SGLang, or spare student sampler in spare-teacher1 layout
teacher-smg              teacher SMG
trainer-head             XORL API/trainer head, 8 local ranks
trainer-worker-1..7      XORL trainer workers, 8 local ranks each
```

The Kubernetes pods may show `Running` even when the role's workload is stopped.
That is expected. The pod is the slot; the child process inside the pod is the
actual workload.

Use the generator status command, not `kubectl get pods`, to know whether a
role is actually running:

```bash
python "$GENERATOR" status
```

## 3. How A Slot Works

Every pod starts the same `slot_agent` loop from the generator. The pod receives:

```text
SLOT_ROLE=<role>
STACK_NAME=er-opd-q36-35b-slots
CONTROL_ROOT=/shared/opd-control/er-opd-q36-35b-slots
```

For each role, the agent watches:

```text
$CONTROL_ROOT/$ROLE/run.sh
$CONTROL_ROOT/$ROLE/stop
$CONTROL_ROOT/$ROLE/pid
$CONTROL_ROOT/$ROLE/status
$CONTROL_ROOT/$ROLE/logs/
```

Loop behavior:

1. Every 2 seconds, hash `$ROLE/run.sh`.
2. If `run.sh` has a new hash, stop the previous child process group.
3. Start the new script with `setsid bash run.sh`.
4. Tee stdout/stderr to `$ROLE/logs/YYYYMMDDTHHMMSSZ-run.log`.
5. Write `$ROLE/pid` and `$ROLE/status`.
6. If `$ROLE/stop` exists, remove it and stop the child process group.

`desired.sha256` is written by the generator for auditability. The agent itself
detects changes by hashing `run.sh`.

Important consequence: rewriting `run.sh` is a deployment. You do not need to
restart the Kubernetes pod.

## 4. Initial Scheduling

Only use Kubernetes scheduling when the slot fleet does not exist or must be
moved to different nodes.

Render the manifest:

```bash
python "$GENERATOR" render-manifest --output "$MANIFEST" --sampler-replicas 1
```

For multi-sampler runs, render/apply with `--sampler-replicas 2` first. Do not
launch a trainer with `--sampler-replicas 2` until both `sglang-0` and
`sglang-1` pods exist and report healthy SGLang children.

Capacity workaround: if the dedicated `sglang-1` pod is Pending, use the
already allocated spare teacher slot as the second student sampler:

```bash
python "$GENERATOR" render-manifest --output "$MANIFEST" \
  --sampler-replicas 2 --sampler-layout spare-teacher1
```

In this layout the two student sampling endpoints are:

```text
er-opd-q36-35b-slots-sglang-0:30060
er-opd-q36-35b-slots-teacher-sglang-1:30000
```

The OPD client must continue using `teacher_base_url=http://...teacher-sglang-0:30000`;
do not route teacher hidden-cache requests through `teacher-sglang-1` while it is
acting as a student sampler. The generator's trainer script already pins the
teacher to `teacher-sglang-0`.

For any multi-sampler science run, the student dispatch script must use
`round_robin`, not `cache_aware`. The 2026-06-03 two-sampler smoke showed that
`cache_aware` sent the same-prefix rollout stream to `sglang-0`, making the
second sampler operationally present but scientifically unused. The generator
now renders `round_robin` whenever more than one student endpoint is configured.

Apply it:

```bash
kubectl apply -n "$NS" -f "$MANIFEST"
```

All GPU pod templates must have `team: turbo`. The generator includes this.
Kyverno will inject the Volcano scheduler and turbo queue labels. Do not
manually set `schedulerName` unless you are deliberately overriding the queue
behavior.

Confirm pods exist:

```bash
kubectl get pods -n "$NS" -o wide | grep "$STACK"
```

This only proves the slots exist. It does not prove the child workloads are
running. Use:

```bash
python "$GENERATOR" status
```

## 5. Full Stack Bring-Up

Use this when all roles should be rewritten: student SGLang, dispatch, teachers,
teacher SMG, and trainers.

```bash
python "$GENERATOR" write-control --config Y --num-steps 11 --prompts-per-step 64 --sampler-replicas 1
```

This writes one `run.sh` per role under `$CONTROL_ROOT`. Each slot agent will
notice the hash change and restart its child process.

Prefer this only for full restarts. For normal recipe iteration, keep warm
student/teacher services up and rewrite only trainer roles.

## 6. Normal Recipe Iteration

This is the standard loop for OPD config sweeps:

```bash
python "$GENERATOR" stop-trainer-control --remove-run

# Wait until trainer-head and trainer-worker-1..7 are stopped.
python "$GENERATOR" status

python "$GENERATOR" write-trainer-control --config Y --num-steps 11 --prompts-per-step 64 --sampler-replicas 1
```

Why `--remove-run` matters:

- It removes stale trainer `run.sh` files while writing stop markers.
- It prevents slot agents from immediately relaunching an old trainer script.
- It avoids split torchrun states where some pods run stale workers and others
  run the new recipe.

After launch, confirm all trainer roles picked up the new revision:

```bash
python "$GENERATOR" status
cat "$CONTROL_ROOT/last_config.txt"
```

Expected `last_config.txt` for a trainer-only launch:

```text
config=Y
num_steps=11
prompts_per_step=64
sampler_replicas=1
written_at=...
roles=trainer
```

## 7. Student Inference And Dispatch Restarts

Restart student SGLang plus dispatch:

```bash
python "$GENERATOR" write-student-inference-control --sampler-replicas 1
```

Restart dispatch only:

```bash
python "$GENERATOR" write-dispatch-control --sampler-replicas 1
```

Use dispatch-only restart when SMG registered `unknown` or otherwise came up
before SGLang had a model id. The current generator's dispatch script waits for
each backend `/v1/models` response to include `Qwen/Qwen3.6-35B-A3B` before
launching SMG, but older dispatch processes may still be stale.

Verify dispatch:

```bash
kubectl exec -n "$NS" "$STACK-dispatch" -- \
  bash -lc 'curl -m 5 -fsS http://127.0.0.1:8080/v1/models'
```

Verify student SGLang:

```bash
kubectl exec -n "$NS" "$STACK-sglang-0" -- \
  bash -lc 'curl -m 5 -fsS http://127.0.0.1:30060/v1/models'
```

Both should report `Qwen/Qwen3.6-35B-A3B`.

## 8. Stopping Workloads

Stop trainers only, keeping warm inference/teacher services:

```bash
python "$GENERATOR" stop-trainer-control --remove-run
python "$GENERATOR" status
```

Stop all child processes while keeping pods allocated:

```bash
python "$GENERATOR" stop-control --remove-run
python "$GENERATOR" status
```

This does not free GPUs because the pods remain scheduled. It only stops child
processes inside the slots.

To free GPUs, delete the Kubernetes pods/controllers from the manifest. Do this
only when you are intentionally giving up the slot allocation.

## 9. Trainer Bring-Up Checks

Head wrapper log:

```bash
ls -td "$CONTROL_ROOT/trainer-head/logs/"*-run.log | head
tail -160 "$(ls -td "$CONTROL_ROOT/trainer-head/logs/"*-run.log | head -1)"
```

Expected sequence:

```text
SGLang 0 healthy after ...
teacher-sglang-0 healthy after ...
SMG dispatch ready
Starting xorl training server config=...
xorl trainer engine ready after ...
Registering SGLang endpoint 0
Recovering any stale P2P/pause state on SGLang endpoint 0
..."weights_synced":true...
Probing SGLang endpoint 0 generation after fresh sync
Running OPD config=...
```

For `--sampler-replicas 2`, the same sequence must include endpoints `0` and
`1`, and profile rows should report `sync_endpoint_count=2`,
`sync_endpoint_success_count=2`, and `sync_serial_endpoint_sync=1.0`.
With `--sampler-layout spare-teacher1`, endpoint `1` should be
`teacher-sglang-1:30000`.

Trainer API health:

```bash
kubectl exec -n "$NS" "$STACK-trainer-head" -- \
  bash -lc 'curl -m 5 -fsS http://127.0.0.1:26050/health'
```

Rank-process count:

```bash
for p in "$STACK-trainer-head" \
  "$STACK-trainer-worker-1" "$STACK-trainer-worker-2" \
  "$STACK-trainer-worker-3" "$STACK-trainer-worker-4" \
  "$STACK-trainer-worker-5" "$STACK-trainer-worker-6" \
  "$STACK-trainer-worker-7"; do
  n=$(kubectl exec -n "$NS" "$p" -- bash -lc \
    'ps -eo cmd | egrep "torch.distributed.run|runner_dispatcher|xorl.server.launcher" | grep -v egrep | wc -l')
  echo "$p trainer_proc_count=$n"
done
```

Healthy shape:

```text
trainer-head: about 12 matching processes
each trainer-worker: about 9 matching processes
```

## 10. OPD Run Monitoring

The run directory is printed in the trainer-head wrapper log:

```text
run_dir=/shared/opd-coord/.../YYYYMMDDTHHMMSSZ-configX-er-opd-q36-35b-slots-trainer-head
```

Profile rows:

```bash
PROFILE=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/<run_id>/opd_profile.jsonl

jq -r '[ "step", "eval_acc", "acc_buffer", "acc_nobuffer", "delta", "mean_len", "loss" ],
  (. | [.step,
        .["eval/accuracy"],
        .["eval/acc_pause"],
        .["eval/acc_nopause"],
        .["eval/buffer_delta"],
        .["eval/mean_completion_tokens"],
        .loss]) | @tsv' "$PROFILE"
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
   .sync_transfer_time_s] | @tsv' "$PROFILE"
```

Artifact/sampler-health rows on newer OPD clients:

```bash
jq -r '[ "step", "delta", "delta_z", "corrupt_delta", "corrupt_z",
          "corrupt_repnum", "pause_repnum", "nopause_repnum",
          "pause_filler_leak", "pause_answer_cue_leak", "pause_cap_hit",
          "nopause_cap_hit", "corrupt_cap_hit",
          "pause_stop", "nopause_stop", "corrupt_stop",
          "pause_request_fail",
          "nopause_request_fail", "corrupt_request_fail",
          "control_max_tokens", "sampler_req", "sampler_workers",
          "sampler_active", "sampler_balance", "diag_requested", "diag_active",
          "diag_unavailable", "health_filler_leak", "health_answer_cue_leak",
          "health_stop" ],
  (select(.["eval/buffer_delta"] != null) |
   [.step,
    .["eval/buffer_delta"],
    .["eval/buffer_delta_z"],
    .["eval/buffer_vs_corrupt_delta"],
    .["eval/buffer_vs_corrupt_delta_z"],
    .["eval/corrupt_pause_repeated_numeric_frac"],
    .["eval/pause_repeated_numeric_frac"],
    .["eval/nopause_repeated_numeric_frac"],
    .["eval/pause_filler_leak_frac"],
    .["eval/pause_answer_cue_leak_frac"],
    .["eval/pause_cap_hit_frac"],
    .["eval/nopause_cap_hit_frac"],
    .["eval/corrupt_pause_cap_hit_frac"],
    .["eval/pause_stop_sequence_seen_frac"],
    .["eval/nopause_stop_sequence_seen_frac"],
    .["eval/corrupt_pause_stop_sequence_seen_frac"],
    .["eval/pause_request_failure_frac"],
    .["eval/nopause_request_failure_frac"],
    .["eval/corrupt_pause_request_failure_frac"],
    .["eval/control_max_completion_tokens"],
    .sampler_router_requests_delta,
    .sampler_worker_count,
    .sampler_worker_active_count,
    .sampler_worker_success_balance_ratio,
    .opd_full_vocab_diag_requested,
    .opd_full_vocab_diag_active_expected,
    .opd_full_vocab_diag_unavailable_expected,
    .["eval/filler_leak_frac"],
    .["eval/answer_cue_leak_frac"],
    .["eval/stop_sequence_seen_frac"]]) | @tsv' "$PROFILE"
```

Answer-logprob causal gate on newer OPD clients:

```bash
jq -r '[ "step", "anslp_margin", "anslp_z",
          "anslp_corrupt_margin", "anslp_corrupt_z",
          "anslp_fail", "anslp_scored_pause", "anslp_scored_nopause",
          "anslp_scored_corrupt", "anslp_clients", "anslp_requests" ],
  (select(.["eval/answer_logprob_control_active"] == 1.0) |
   [.step,
    .["eval/answer_logprob_margin"],
    .["eval/answer_logprob_margin_z"],
    .["eval/answer_logprob_vs_corrupt_margin"],
    .["eval/answer_logprob_vs_corrupt_margin_z"],
    .["eval/answer_logprob_request_failure_frac"],
    .["eval/answer_logprob_scored_pause"],
    .["eval/answer_logprob_scored_nopause"],
    .["eval/answer_logprob_scored_corrupt_pause"],
    .["eval/answer_logprob_client_count"],
    .["eval/answer_logprob_total_requests"]]) | @tsv' "$PROFILE"
```

Strict gate summary:

```bash
python experiments/opd_profile/analyze_slot_profile.py "$PROFILE" \
  --expected-sync-endpoints 1
```

For a verified two-sampler run:

```bash
python experiments/opd_profile/analyze_slot_profile.py "$PROFILE" \
  --expected-sync-endpoints 2 \
  --require-serial-sync \
  --require-sampler-routing \
  --min-sampler-active-workers 2 \
  --min-sampler-balance-ratio 0.75
```

For new rows that enable answer-distractor scoring, use the prompt-specific
answer-selection gate:

```bash
python experiments/opd_profile/analyze_slot_profile.py "$PROFILE" \
  --expected-sync-endpoints 2 \
  --require-serial-sync \
  --require-sampler-routing \
  --min-sampler-active-workers 2 \
  --min-sampler-balance-ratio 0.75 \
  --require-answer-select-control \
  --min-answer-select-z 2 \
  --min-answer-select-paired-n 1024
```

For post-hoc diagnostic JSONs produced from the same run directory, the analyzer
can overlay those scalar metrics onto the latest control row before gating. This
is useful for legacy rows where the in-loop scorer failed but the checkpoint was
still warm enough to recover the intended measurement:

```bash
python experiments/opd_profile/analyze_slot_profile.py "$PROFILE" \
  --posthoc-overlay "$RUN_DIR/posthoc_answer_select_distractor_preserve_ws_1024_20260603T1848Z.json" \
  --expected-sync-endpoints 2 \
  --require-serial-sync \
  --require-sampler-routing \
  --require-answer-select-control \
  --min-answer-select-z 2 \
  --min-answer-select-paired-n 1024
```

The overlay path is guarded: by default the JSON's `posthoc_source_run` or
`source_run` must match the profile directory. Use
`--allow-posthoc-source-mismatch` only for manual forensics, never promotion.
For nested post-hoc files with a top-level `results` object, pass
`--posthoc-result-key <key>`.

Exit code `0` means the latest control row passes the current promotion gates.
Exit code `1` means reject the recipe or continue only as a diagnostic.
The analyzer now gates on sync endpoint count, sync success,
pause/no-pause/corrupted-pause control submission, answer-logprob causal margin
when that control is active, control token budget, request-failure fractions,
cap-hit/artifact fractions, diagnostic availability flags, and optional
sampler-routing proof. With `--require-answer-select-control`, it also requires
the pause arm to improve correct-vs-distractor answer selection relative to both
no-pause and corrupted-pause controls. Use `--no-require-control-artifacts`,
`--no-require-corrupt-control`, or omit `--require-answer-select-control` only
when auditing legacy runs; do not use those relaxations for promotion.

W&B/profile parity check:

```bash
python experiments/opd_profile/audit_wandb_profile.py "$PROFILE" --wandb-run <run_id>
```

This checks whether the W&B history contains the critical local `opd_profile`
rows. If it reports `wandb_stale_or_mismatched`, treat the local JSONL profile
as the source of truth and do not make a promote/reject decision from the W&B UI
alone.

Wrapper log:

```bash
tail -160 "$CONTROL_ROOT/trainer-head/logs/<timestamp>-run.log"
```

Server log:

```bash
tail -160 "$RESULT_ROOT/<run_id>/server.log"
```

Control-script preflight without touching live slots:

```bash
python experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py render-control \
  --config AN \
  --num-steps 6 \
  --prompts-per-step 64 \
  --sampler-replicas 2 \
  --sampler-layout spare-teacher1 \
  --role trainer-head
```

This dry-runs the exact script-generation path used by `write-control`. Use it
before live writes to verify endpoint topology, scoring batch/concurrency
settings, control start step, and science knobs.

## 11. Latest Results

### Config AM Final-Only 1024-Control Confirmation

Run:

```text
/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T171221Z-configAM-er-opd-q36-35b-slots-trainer-head
W&B run: c7f09nym
```

Config AM is the AI/AH answer-causal recipe with memory-only rotated corrupt
buffer, two serially synced sampler endpoints, and a larger held-out control
set. It sets `eval_num_problems=1024` and `eval_control_start_step=5`, so the
`1024 * 3 = 3072` sampled-policy control grid runs only at the final control
point.

Operational result:

- Completed all 6 requested steps; trainer-head exited `rc=0` at
  `2026-06-03T18:12:04Z`; trainer workers were stopped.
- W&B/profile parity passed: `VERDICT: wandb_matches_profile`, W&B state
  `finished`.
- Step 0 validated the final-only gate:
  `eval/control_allowed_by_start_step=0.0` and no `eval/buffer_delta`.
- Final control row submitted all `3072` sampled-policy requests with
  `eval/control_request_failure_frac_max=0.0`.
- Sync and routing were clean:
  `sync_endpoint_success_count=2`, `sync_endpoint_failure_count=0`,
  `sync_serial_endpoint_sync=1.0`, `sampler_worker_active_count=2`, and
  `sampler_worker_success_balance_ratio=1.0`.

Final control row:

| step | n | acc_pause | acc_nopause | acc_corrupt | delta | delta_z | corrupt_delta | corrupt_z | corrupt_cap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 1024 | 0.6494 | 0.5557 | 0.5811 | +0.0938 | 4.3548 | +0.0684 | 3.1871 | 0.5918 |

Analyzer:

```text
rows=6 control_rows=1
VERDICT: reject
- eval/answer_logprob_request_failure_frac=1.0000 > 0.0000
- eval/corrupt_pause_cap_hit_frac=0.5918 > 0.5000
```

Post-hoc chunked answer-logprob salvage:

```text
/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T171221Z-configAM-er-opd-q36-35b-slots-trainer-head/posthoc_answer_logprob_chunked_20260603T1816Z.json
```

This ran after the AM trainer exited, against the still-warm final AM sampler
endpoints, using the patched chunked scorer. It did not retrain or resample
answers.

| prompts | requests | chunks | batch | maxconc | reqfail | margin | z | corrupt_margin | corrupt_z | elapsed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 3072 | 48 | 64 | 2 | 0.0000 | +0.1784 | 36.7621 | +0.3114 | 32.2773 | 194.3s |

Boundary-preserving corrupt-mode probes:

```text
/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T171221Z-configAM-er-opd-q36-35b-slots-trainer-head/posthoc_corrupt_mode_probe_192_20260603T1825Z.json
/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T171221Z-configAM-er-opd-q36-35b-slots-trainer-head/posthoc_answer_logprob_rotate_preserve_ws_1024_20260603T1835Z.json
```

These probes tested the hypothesis that the old text corruptor was changing the
assistant continuation boundary, not just the memory symbols. The legacy text
rotate drops the leading whitespace from the pause prefix (`leading_ws_match=0`,
`len_delta_chars=-1`). The new `rotate_preserve_ws` mode keeps boundary
whitespace intact while rotating the same memory pieces.

Sampled free-generation, 192 held-out prompts:

| mode | acc_pause | acc_corrupt | corrupt_delta | corrupt_z | lead_delta | corrupt_cap | leading_ws |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rotate` | 0.6979 | 0.6094 | +0.0885 | 1.8312 | +0.4843 | 0.1719 | 0.0 |
| `rotate_preserve_ws` | 0.7135 | 0.7083 | +0.0052 | 0.1126 | +0.0141 | 0.0000 | 1.0 |

Chunked answer-logprob with `rotate_preserve_ws`, 1024 held-out prompts:

| requests | chunks | reqfail | margin | z | corrupt_margin | corrupt_z |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3072 | 48 | 0.0000 | +0.1784 | 36.7621 | +0.0568 | 8.1144 |

Answer-selection distractor probe:

```text
/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T171221Z-configAM-er-opd-q36-35b-slots-trainer-head/posthoc_answer_select_distractor_preserve_ws_1024_20260603T1848Z.json
```

This scores both the correct answer and a paired wrong answer from another held
out prompt under the same prompt/prefix. It tests prompt-specific answer
selection, not absolute answer-token likelihood.

| requests | chunks | pairs | reqfail | select_pause | select_nopause | select_corrupt | pause-nopause | z | pause-corrupt | z |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6144 | 96 | 1024 | 0.0000 | +3.2916 | +3.4187 | +4.0431 | -0.1271 | -13.1780 | -0.7515 | -68.7370 |

Decision:

- AM confirms the user's concern: AI's exact-match deltas were underpowered at
  `n=192`. At `n=1024`, the same AI/AH direction passes the exact
  pause-vs-no-pause and pause-vs-corrupt z gates.
- This is not a reason to copy RiM. It is a reason to keep RiM's causal
  invariant: real memory/pause context should help the answer, and corrupted
  memory should hurt it.
- AM's answer-logprob failure is an infra/client artifact, not a negative
  science result. The old scorer grouped all scoring items by endpoint and sent
  one huge prompt-scoring batch per native SGLang endpoint, roughly `1536`
  sequences each. Both native scorers returned transient HTTP 503s, so
  `eval/answer_logprob_request_failure_frac=1.0` and the answer-logprob margins
  in the in-loop profile row are unusable. The post-hoc chunked scorer recovered
  the intended measurement with `0.0` request failures and very strong margins:
  `+0.1784` vs no-pause (`z=36.76`) and `+0.3114` vs corrupt (`z=32.28`).
- After AM, the OPD client was patched to chunk answer-logprob scoring via
  `eval_answer_logprob_batch_size` and
  `eval_answer_logprob_max_concurrency`. Config AM now renders
  `eval_answer_logprob_batch_size=64` and
  `eval_answer_logprob_max_concurrency=2`, producing 48 scoring chunks instead
  of two oversized endpoint batches.
- The free-generation corrupt arm was not just a length artifact; it was partly
  a boundary artifact. When the corrupt text preserves leading/trailing
  whitespace, corrupt cap-hit falls to `0.0` on the 192-prompt probe, but
  sampled pause-vs-corrupt exact-match collapses from `+0.0885` to `+0.0052`.
- The stronger remaining evidence is pause-vs-no-pause:
  exact-match `+0.0938` at `n=1024` and answer-logprob `+0.1784` with
  `z=36.76`. Boundary-preserved pause-vs-corrupt answer-logprob is still
  positive (`+0.0568`, `z=8.11`), but far weaker than the legacy corrupt margin.
  Do not use legacy rotate corrupt exact-match as a promotion gate.
- The answer-selection probe is a serious negative result for the current
  fixed-token pause recipe. Although pause raises absolute true-answer logprob,
  it lowers correct-vs-wrong answer separation relative to no-pause
  (`select_delta=-0.1271`, `z=-13.18`) and relative to boundary-preserved corrupt
  (`select_vs_corrupt_delta=-0.7515`, `z=-68.74`). This means the absolute
  answer-logprob gain is not evidence by itself that the pause buffer improves
  prompt-specific answer discrimination.

Next follow-up:

- Treat AM plus the post-hoc probes as evidence for a real pause-conditioned
  sampling/format/absolute-likelihood effect, not as a complete memory-selection
  mechanism. Rerun AM only if an integrated W&B/profile row is required for
  bookkeeping.
- Config AN is available as AM with `rotate_preserve_ws`, chunked
  answer-logprob, answer-distractor selection metrics, and the same final-only
  1024 controls. Run it only if an integrated W&B/profile row is needed; the
  post-hoc probes already show the likely outcome.
- The next real science step is a prompt-specific corrupt control: either
  externalize prompt-specific memory as tokens before shuffling, or use a
  cache/hidden-target mismatch whose visible prompt/prefix boundary is identical.
  Fixed-token order corruption is too weak once the boundary artifact is removed,
  and the next objective should optimize or at least gate on correct-vs-distractor
  answer selection rather than absolute answer likelihood alone.
- The analyzer now has `--require-answer-select-control` plus
  `--min-answer-select-{delta,z,paired-n}`. New promotion checks should include
  that flag; AM would reject under it because the pause lowered
  correct-vs-distractor selection despite raising absolute answer logprob.
- The analyzer also supports `--posthoc-overlay JSON`. This is now the canonical
  way to include recovered post-hoc scoring metrics in a gate transcript without
  hand-copying them into `opd_profile.jsonl`.
- The slot generator now supports `render-control`, which renders live control
  scripts without writing `/shared/opd-control/.../run.sh`. Preflight AN with
  `--role trainer-head` showed the expected two-sampler logprob endpoints,
  `eval_answer_logprob_batch_size=64`,
  `eval_answer_logprob_max_concurrency=2`,
  `eval_answer_logprob_distractor_control=true`,
  `eval_control_start_step=5`, and
  `opd_contrastive_corrupt_buffer_mode=rotate_preserve_ws`.

### Config AI Rotated Memory-Only Corrupt Control

Run:

```text
/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T143837Z-configAI-er-opd-q36-35b-slots-trainer-head
W&B run: fzittb3d
```

Config AI keeps AH's fixed substrate and answer-level contrast, but changes the
corrupt arm from full-prefix reversal to rotated memory-only corruption:
`opd_contrastive_corrupt_buffer_mode=rotate` and
`opd_contrastive_corrupt_buffer_span=memory_only`. It also uses the strict
`192`-problem control gate. This leaves the literal `Answer: ` suffix intact,
so the corrupt copy tests memory usefulness rather than a broken answer cue.

Operational result:

- Completed all 6 requested steps; trainer-head exited `rc=0`.
- Initial registration freshly synced both sampler endpoints with NCCL and
  passed direct generation probes:
  endpoint 0 in `7.26s`, endpoint 1 in `7.86s`.
- Every profile row synced both endpoints:
  `sync_success=true`, `sync_endpoint_success_count=2`,
  `sync_endpoint_failure_count=0`, and `sync_serial_endpoint_sync=1.0`.
- Sampler routing stayed exact round-robin on every row:
  `sampler_worker_active_count=2`,
  `sampler_worker_success_balance_ratio=1.0`.
- W&B/profile parity passed:
  `VERDICT: wandb_matches_profile`, W&B state `finished`.
- The new objective/control instrumentation was present:
  `opd_contrastive_corrupt_memory_only=1.0`,
  `opd_contrastive_corrupt_span_tokens=9.0`, and
  `eval/control_corrupt_pause_mode=rotate`.

Compact trajectory:

| step | acc | loss | sync | notes |
| ---: | ---: | ---: | --- | --- |
| 0 | 0.4531 | 0.0897 | 2 endpoints, serial | warmup/control |
| 1 | 0.5156 | 0.0657 | 2 endpoints, serial | steady |
| 2 | 0.4062 | 0.0607 | 2 endpoints, serial | steady |
| 3 | 0.3906 | 0.0309 | 2 endpoints, serial | steady |
| 4 | 0.3594 | 0.0069 | 2 endpoints, serial | steady |
| 5 | 0.5000 | -0.0193 | 2 endpoints, serial | final control |

Final control row:

| step | acc_pause | acc_nopause | acc_corrupt | delta | delta_z | corrupt_delta | corrupt_z | anslp_margin | anslp_z | anslp_corrupt_margin | anslp_corrupt_z | corrupt_cap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.6719 | 0.5781 | 0.6094 | +0.0938 | 1.9063 | +0.0625 | 1.2790 | +0.2163 | 16.6732 | +0.3917 | 17.3105 | 0.7083 |

Analyzer:

```text
rows=6 control_rows=2
VERDICT: reject
- eval/buffer_delta_z=1.9063 < 2.0000
- eval/buffer_vs_corrupt_delta_z=1.2790 < 2.0000
- eval/corrupt_pause_cap_hit_frac=0.7083 > 0.5000
```

Decision:

- AI is an operational pass and a partial mechanism signal, but not a promotion.
  The answer-logprob evidence is very strong: pause beats no-pause and corrupt
  pause with `z=16.67` and `z=17.31` respectively. Exact-match also moves in
  the right direction, but misses the strict z gate.
- The corruption artifact is improved relative to AH (`filler_leak_frac=0.0`
  rather than `1.0`), but not solved. The rotated corrupt arm now cap-hits
  `70.8%` of the time at step 5, so the corrupt exact-match comparison still
  partly measures pathological generation length/format, not only broken memory.
- The steady on-policy health is weak: accuracy drops through steps 1-4 before
  recovering to `0.50` at step 5, while loss becomes negative. Treat the
  answer-logprob signal as evidence that the model learned a buffer-conditioned
  likelihood preference, not evidence that the sampled answer policy is
  improving robustly.

Next follow-up:

- Do not copy RiM as a surface recipe. Keep RiM's causal invariant: real memory
  should improve the answer; corrupted memory should hurt it.
- First, run Config AM as the statistical confirmation of AI/AH:
  AI's answer-causal objective, memory-only rotated corrupt buffer, two
  serially synced sampler endpoints, `eval_num_problems=1024`, and
  `eval_control_start_step=5`. This tests whether AI's step-5 exact-match
  deltas were underpowered without paying for a 1024-problem warmup control.
- For AM, interpret exact pause-vs-no-pause at 1024 as the sampled-policy
  confirmation, but treat free-generation corrupt exact-match as conditional on
  cap-hit health. If corrupt cap-hit remains high, use answer-logprob
  pause-vs-corrupt as the primary corrupt-memory evidence and record corrupt
  free-generation as an artifact, not as a reason to throw out the
  answer-causal mechanism.
- Do not add a token-level "shuffle memory buffers" config for the current
  fixed-slot recipe. The slot tokens are shared scaffolding; the prompt-specific
  memory signal lives in the hidden/cache targets. Shuffling the fixed token span
  across prompts would be close to a no-op and would not test RiM's invariant.
- Replace synthetic rotate/reverse corruption with a genuinely in-distribution
  paired control only after the memory is prompt-specific at the corruption
  boundary. Two plausible routes: externalize learned/generated memory as tokens
  and shuffle those buffers across prompts, or add a cache/hidden-target paired
  negative that mismatches a prompt with another prompt's teacher memory while
  preserving answer-causal validation.
- Per-arm latency/age metrics for control eval were added after AI. The next
  client run should log `eval/{arm}_request_latency_{mean,p95,max}_s`,
  aggregate `eval/control_request_latency_{mean,p95,max}_s`, and
  `eval/answer_logprob_group_latency_{mean,p95,max}_s`. Step-5 control
  completed, but 66 requests sat in the 300-600s age bucket before finishing.
- A corrupt-control no-op guard was also added: next profile rows should include
  `opd_contrastive_corrupt_changed_tokens`,
  `opd_contrastive_corrupt_change_frac`, and
  `opd_contrastive_corrupt_noop_frac`. Any proposed corruption with near-zero
  change fraction should be rejected before spending a full Qwen3.6 run.
- Keep the two serialized sampler endpoints behind SMG and fresh NCCL syncs.
  Throughput should scale by adding more serialized endpoints, not by increasing
  per-pod SGLang batching, until the repeated-suffix/tail-latency artifacts are
  separately fixed and revalidated.

### Config AI Instrumentation Smoke

Run:

```text
/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T151837Z-configAI-er-opd-q36-35b-slots-trainer-head
```

Command:

```bash
python experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py write-trainer-control \
  --config AI \
  --num-steps 1 \
  --prompts-per-step 64 \
  --sampler-replicas 2 \
  --sampler-layout spare-teacher1
```

W&B:

```text
05o5f8h6
```

Purpose: validate the new instrumentation on the live Qwen3.6 slots path. This
is not a promotion/science run because the single row is warmup-only.

Outcome:

- Trainer-head exited `rc=0`; workers stopped.
- Analyzer result: `VERDICT: incomplete (no non-warmup control rows)`.
- W&B/profile parity passed after W&B finished syncing:
  `VERDICT: wandb_matches_profile`.
- Direct W&B history query also found the new metrics:
  `eval/control_request_latency_max_s`,
  `eval/control_request_latency_p95_s`,
  `eval/nopause_request_latency_mean_s`,
  `eval/corrupt_pause_request_latency_mean_s`,
  `eval/answer_logprob_group_latency_max_s`,
  `opd_contrastive_corrupt_change_frac`, and
  `opd_contrastive_corrupt_noop_frac`.
- Operational substrate stayed clean:
  `sync_endpoint_count=2`, `sync_endpoint_success_count=2`,
  `sync_endpoint_failure_count=0`, `sync_serial_endpoint_sync=1.0`,
  `sync_transfer_time_s=14.58`, `sampler_worker_active_count=2`,
  `sampler_worker_success_balance_ratio=1.0`, and sampler routing split exactly
  `320/320`.

Instrumentation readout:

| metric | value |
| --- | ---: |
| `opd_contrastive_corrupt_changed_tokens` | 576 |
| `opd_contrastive_corrupt_span_token_total` | 576 |
| `opd_contrastive_corrupt_change_frac` | 1.000 |
| `opd_contrastive_corrupt_noop_frac` | 0.000 |
| `eval/control_request_latency_mean_s` | 176.42 |
| `eval/control_request_latency_p95_s` | 331.81 |
| `eval/control_request_latency_max_s` | 348.83 |
| `eval/pause_request_latency_mean_s` | 86.03 |
| `eval/nopause_request_latency_mean_s` | 218.63 |
| `eval/corrupt_pause_request_latency_mean_s` | 224.61 |
| `eval/answer_logprob_group_latency_mean_s` | 18.46 |
| `eval/answer_logprob_group_latency_max_s` | 18.76 |

Decision:

- The corrupt no-op guard works for the current rotate-memory control:
  all 576 corrupt-span tokens changed and no examples were no-ops.
- The control eval tail is real and now measured directly. During the run, SMG
  showed hundreds of active/inflight control requests aging into the 60-180s
  bucket. Final per-arm latency shows `nopause` and `corrupt_pause` were the
  slow arms, while `pause` was much faster on average.
- The warmup science metrics are not interpretable as learning:
  `acc_pause=0.6771`, `acc_nopause=0.7552`, `acc_corrupt=0.6875`,
  `eval/answer_logprob_margin=+0.0042`, and
  `eval/answer_logprob_vs_corrupt_margin=-0.0171`.

### Config AJ Teacher-Memory Pair Diagnostic Smoke

Run:

```text
/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T153913Z-configAJ-er-opd-q36-35b-slots-trainer-head
```

Command:

```bash
python experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py write-trainer-control \
  --config AJ \
  --num-steps 1 \
  --prompts-per-step 64 \
  --sampler-replicas 2 \
  --sampler-layout spare-teacher1
```

W&B:

```text
yazdxxgu
```

Purpose: validate an opt-in diagnostic for whether teacher cache memory rows are
prompt-specific enough to support a future cache-mismatch causal control. This
does not train a new objective; it compares each prompt's supervised teacher
memory rows against rows donated from a different prompt, then logs cross-prompt
and within-buffer cosine distances.

Outcome:

- Trainer-head exited `rc=0`; workers stopped.
- Analyzer result: `VERDICT: incomplete (no non-warmup control rows)`, expected
  for a one-step smoke.
- W&B/profile parity passed for the standard audit keys:
  `VERDICT: wandb_matches_profile`, W&B state `finished`.
- Direct W&B history query found the new teacher-memory diagnostic keys:
  `opd_teacher_memory_pair_diag_active`,
  `opd_teacher_memory_pair_diag_failure`,
  `opd_teacher_memory_pair_cross_cosine_distance_mean`,
  `opd_teacher_memory_pair_within_adjacent_distance_mean`, and
  `opd_teacher_memory_pair_cross_minus_within_distance`.
- Operational substrate stayed clean:
  `sync_endpoint_count=2`, `sync_endpoint_success_count=2`,
  `sync_endpoint_failure_count=0`, `sync_serial_endpoint_sync=1.0`,
  `sync_transfer_time_s=15.02`, `sampler_worker_active_count=2`,
  `sampler_worker_success_balance_ratio=1.0`, and sampler routing split exactly
  `320/320`.

Diagnostic readout:

| metric | value |
| --- | ---: |
| `opd_teacher_memory_pair_diag_requested` | 1 |
| `opd_teacher_memory_pair_diag_active` | 1 |
| `opd_teacher_memory_pair_diag_failure` | 0 |
| `opd_teacher_memory_pair_sample_count` | 64 |
| `opd_teacher_memory_pair_skipped_samples` | 0 |
| `opd_teacher_memory_pair_cross_token_count` | 576 |
| `opd_teacher_memory_pair_cross_cosine_similarity_mean` | 0.7499 |
| `opd_teacher_memory_pair_cross_cosine_distance_mean` | 0.2501 |
| `opd_teacher_memory_pair_cross_cosine_distance_min` | 0.0108 |
| `opd_teacher_memory_pair_cross_cosine_distance_max` | 1.0582 |
| `opd_teacher_memory_pair_within_adjacent_token_count` | 512 |
| `opd_teacher_memory_pair_within_adjacent_distance_mean` | 0.1959 |
| `opd_teacher_memory_pair_cross_minus_within_distance` | 0.0543 |
| `opd_contrastive_corrupt_change_frac` | 1.000 |
| `opd_contrastive_corrupt_noop_frac` | 0.000 |
| `eval/control_request_latency_p95_s` | 333.63 |
| `eval/control_request_latency_max_s` | 344.50 |

Decision:

- The diagnostic path works: it was requested, active, did not fail, and skipped
  zero samples. W&B also receives the new metrics.
- The teacher slot hiddens are prompt-specific, but only weakly separated at the
  current corruption boundary. Cross-prompt memory rows are farther apart than
  adjacent rows in the same buffer (`0.2501` vs `0.1959` cosine distance), but
  the margin is only `0.0543`. This is enough to justify a carefully designed
  cache-mismatch experiment; it is not strong evidence that cache mismatch will
  be a clean or high-signal negative objective.
- Do not add a naive negative answer-KL term on an identical prompt paired with
  another prompt's teacher memory. If the visible input and answer are unchanged,
  that would directly punish the real answer path. A safer next objective would
  make the mismatch target diagnostic-only first, or apply the mismatch only to
  memory/hidden rows while keeping answer-level validation separate.
- The control latency tail persists under the same serialized sampler setup:
  `eval/control_request_latency_p95_s=333.63` and max `344.50`, with no request
  failures. This is an operational throughput problem, not evidence against the
  mechanism.

### Config AH Answer-Level Causal Contrast

Run:

```text
/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T141126Z-configAH-er-opd-q36-35b-slots-trainer-head
W&B run: zfkrmifn
```

Config AH keeps AG's fixed substrate: NCCL broadcast weight sync, serialized
endpoint sync, sampler quiescence before every sync, two sampler endpoints
behind dispatch, stop-on-newline hygiene, and answer-logprob controls. It
changes the objective: `opd_supervise_buffer_only=false` keeps answer rows in
the teacher cache, and `opd_contrastive_corrupt_answer_weight=0.125` adds a
small signed answer KL contrast. Real-buffer answer positions get positive
teacher KL; corrupted-buffer answer positions get negative teacher KL. The run
also used `opd_loss_max_clamp=5.0`.

Operational result:

- Completed all 6 requested steps; trainer-head exited `rc=0`.
- Initial registration synced both sampler endpoints with NCCL and passed direct
  generation probes: endpoint 0 in `7.55s`, endpoint 1 in `7.43s`.
- Every profile row synced both endpoints:
  `sync_success=true`, `sync_endpoint_success_count=2`,
  `sync_endpoint_failure_count=0`, and `sync_serial_endpoint_sync=1.0`.
- Sampler routing stayed exact round-robin on every row:
  `sampler_worker_active_count=2`,
  `sampler_worker_success_balance_ratio=1.0`.
- W&B/profile parity passed:
  `VERDICT: wandb_matches_profile`.

Compact trajectory:

| step | acc | hidden_margin | loss | sync |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0.5312 | 0.1601 | 0.0733 | 2 endpoints, serial |
| 1 | 0.5000 | 0.1874 | 0.0462 | 2 endpoints, serial |
| 2 | 0.5469 | 0.2952 | 0.0245 | 2 endpoints, serial |
| 3 | 0.6719 | 0.4997 | -0.0299 | 2 endpoints, serial |
| 4 | 0.5469 | 0.6022 | -0.0494 | 2 endpoints, serial |
| 5 | 0.5938 | 0.6544 | -0.0706 | 2 endpoints, serial |

Final control row:

| step | acc_pause | acc_nopause | acc_corrupt | delta | delta_z | corrupt_delta | corrupt_z | anslp_margin | anslp_z | anslp_corrupt_margin | anslp_corrupt_z |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.5000 | 0.2292 | 0.1354 | +0.2708 | 4.0626 | +0.3646 | 5.8959 | +0.2234 | 5.3267 | +0.8454 | 23.8045 |

Analyzer:

```text
rows=6 control_rows=2
VERDICT: reject
- eval/control_n=96 < 192
- eval/corrupt_pause_cap_hit_frac=0.9375 > 0.5000
```

Decision:

- AH is a real mechanism hit, but not yet a promotion. It is the first
  Qwen3.6-35B slots run where the trained intervention improves both exact-match
  pause-vs-no-pause and true-answer logprob, and where corrupted pause is much
  worse than real pause on both metrics.
- The reject reason is an artifact gate, not a negative causal signal. The
  corrupted-pause control almost completely degenerates:
  `corrupt_pause_cap_hit_frac=0.9375` and
  `corrupt_pause_filler_leak_frac=1.0`. That means the enormous corrupt contrast
  is partly measuring a pathological corrupted-prefix failure mode. The
  pause-vs-no-pause answer-logprob margin is cleaner and also positive.
- This supports the deeper diagnosis: AG failed because the autoresearch loop
  optimized a hidden-state proxy that was not answer-causal. AH imported the
  causal intervention invariant we liked in RiM -- real memory must help the
  answer and corrupted memory must hurt it -- without copying RiM's memory
  surface. The next run should strengthen the causal answer objective and clean
  the corrupt control, not start another filler-token sweep.

Next follow-up:

- Add AH to the 192-problem control-eval set so the strict analyzer sample-size
  gate is meaningful.
- Add/try a less degenerate corrupt arm for answer-logprob and generation
  controls. After AI, do not implement this as token-level cross-prompt
  shuffling of the fixed slot scaffold; the memory must be prompt-specific at
  the corruption boundary for the control to be meaningful. The goal is a
  corrupted buffer that remains in-distribution enough not to cap-hit, while
  still breaking the per-prompt reasoning state.
- Consider a lower answer-contrast weight ablation (`0.05` or `0.075`) if
  corruption collapse persists, but keep the answer-level contrast; AG already
  showed hidden-only contrast is insufficient.

### Config AG NCCL-Sync Strong-Corrupt Contrast

Run:

```text
/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T134618Z-configAG-er-opd-q36-35b-slots-trainer-head
W&B run: oxo5yw7i
```

Config AG is AF's objective and controls with `sync_method=nccl_broadcast` /
`sync_inference_method=nccl_broadcast`, added after repeated Mooncake P2P syncs
failed mid-run.

Operational result:

- Completed all 6 requested steps; trainer-head exited `rc=0`.
- Initial registration synced both endpoints with NCCL and passed direct
  generation probes:
  endpoint 0 in `7.09s`, endpoint 1 in `7.23s`.
- Every profile row synced both endpoints:
  `sync_success=true`, `sync_endpoint_success_count=2`,
  `sync_endpoint_failure_count=0`, and `sync_serial_endpoint_sync=1.0`.
- Every profile row passed quiescence:
  `sampler_quiesce_success=1.0`, zero outstanding requests, zero active
  connections, and zero inflight-age counts before sync.
- Sampler routing stayed exactly balanced:
  step-5 `sampler_worker_0_success_delta=176`,
  `sampler_worker_1_success_delta=176`,
  `sampler_worker_success_balance_ratio=1.0`.
- W&B/profile parity passed:
  `VERDICT: wandb_matches_profile`.

Final control row:

| step | hidden_pos_raw | hidden_neg_raw | hidden_margin | acc_pause | acc_nopause | acc_corrupt | delta | corrupt_delta | anslp_margin | anslp_z | anslp_corrupt_margin | anslp_corrupt_z | sync |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 5 | 0.2280 | 0.8726 | 0.6446 | 0.7083 | 0.7708 | 0.7292 | -0.0625 | -0.0208 | -0.0510 | -3.8238 | -0.0232 | -2.0460 | 2 endpoints, serial |

Analyzer:

```text
rows=6 control_rows=2
VERDICT: reject
- eval/buffer_delta=-0.0625 < 0.0300
- eval/buffer_vs_corrupt_delta=-0.0208 < 0.0300
- eval/answer_logprob_margin=-0.0510 < 0.0000
- eval/answer_logprob_vs_corrupt_margin=-0.0232 < 0.0000
```

Decision:

- AG fixes the remaining infrastructure ambiguity for this recipe. Two sampler
  pods, fresh syncs, quiescence, routing, and W&B logging are all clean.
- AG rejects the AF mechanism. The objective strongly separates positive vs
  corrupted hidden states (`hidden_margin: 0.1597 -> 0.6446`), but the answer is
  less likely and less accurate with the pause buffer than without it, and worse
  than corrupted pause on the final exact-match control.
- Stop doing more filler/hidden-coefficient sweeps in this family. The next
  step should change the causal training signal or architecture so the answer
  path must depend on a verified buffer state.

### Config AF Strong-Corrupt Contrast, Clean Rerun

Run:

```text
/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T132217Z-configAF-er-opd-q36-35b-slots-trainer-head
W&B run: u9wqf40r
```

Config AF keeps AE's fixed autoresearch substrate and increases the corrupted
buffer hidden-match contrast weight to `1.0`.

Operational result:

- Initial registration synced both sampler endpoints and passed direct
  post-sync generation probes.
- Steps 0-3 were valid profile rows:
  `sync_success=true`, `sync_endpoint_success_count=2`,
  `sync_serial_endpoint_sync=1.0`, `sampler_quiesce_success=1.0`,
  `sampler_worker_active_count=2`, and
  `sampler_worker_success_balance_ratio=1.0`.
- Step 4 failed during serial P2P sync with repeated Mooncake
  `received packet mismatch` errors and
  `batch_transfer_sync ... failed ... after 50 attempts` to receiver
  `10.42.77.65:15151`. Treat the run as infrastructure-invalid after step 3.

Partial science signal:

| step | accuracy | hidden_pos_raw | hidden_neg_raw | hidden_margin | acc_pause | acc_nopause | acc_corrupt | anslp_margin | anslp_corrupt_margin |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.4531 | 0.2679 | 0.4280 | 0.1601 | 0.7083 | 0.7708 | 0.6979 | 0.0044 | -0.0067 |
| 1 | 0.4844 | 0.2554 | 0.4414 | 0.1860 | | | | | |
| 2 | 0.6250 | 0.2367 | 0.5197 | 0.2830 | | | | | |
| 3 | 0.5469 | 0.2357 | 0.7320 | 0.4963 | | | | | |

Decision:

- AF proves the stronger corrupt-hidden objective can optimize the auxiliary
  hidden separation proxy under a clean sampler/sync substrate.
- AF still has not proven causal buffer use. The only causal row before the
  infra failure was step 0, where pause underperformed no-pause and answer
  logprob versus corrupt was negative.
- Do not run another AF/P2P attempt as the next science step. Add/use Config AG:
  the AF objective and controls with `nccl_broadcast` weight sync, so the final
  causal checkpoint is not gated on the repeated Mooncake packet-mismatch issue.

### Config AE Sync-Barrier Pilot

Run:

```text
/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T122559Z-configAE-er-opd-q36-35b-slots-trainer-head
W&B run: r1egk9sc
```

Config AE is AD with the operational substrate fixed before making another
science claim:

- `opd_pipeline_rl=false`, so step N+1 prepare/sampling is not launched before
  step N sync.
- `sampler_quiesce_before_sync=true`, requiring SMG metrics to show zero new
  outstanding requests, zero active connections, and zero inflight-age counts
  before every sync.
- two serialized student samplers behind SMG:
  `sglang-0:30060` and `teacher-sglang-1:30000`
- serial fresh weight sync to both endpoints
- native SGLang `/generate` scorers for answer-logprob controls

Operational result:

- Completed all 6 requested steps; trainer-head exited `rc=0`.
- Initial endpoint registration synced both endpoints:
  endpoint 0 in `2.90s`, endpoint 1 in `2.36s`.
- Every profile row passed the new quiescence gate:
  `sampler_quiesce_success=1.0`,
  `sampler_quiesce_new_outstanding=0.0`,
  `sampler_quiesce_connections_active=0.0`, and
  `sampler_quiesce_inflight_request_age_count=0.0`.
- Every row synced both inference endpoints:
  `sync_endpoint_success_count=2`,
  `sync_endpoint_failure_count=0`,
  `sync_serial_endpoint_sync=1.0`.
- Sampler routing stayed exactly balanced:
  step-5 `sampler_worker_0_success_delta=176`,
  `sampler_worker_1_success_delta=176`,
  `sampler_worker_success_balance_ratio=1.0`.
- W&B/profile parity passed:
  `VERDICT: wandb_matches_profile`.

Final control row:

| step | loss | hidden | acc_pause | acc_nopause | acc_corrupt | delta | corrupt_delta | anslp_margin | anslp_z | anslp_corrupt_margin | anslp_corrupt_z | q_ok | sync |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 5 | 0.4418 | 0.0097 | 0.7083 | 0.7500 | 0.7083 | -0.0417 | 0.0000 | -0.0433 | -3.6113 | -0.0264 | -2.8177 | 1.0 | 2 endpoints, serial |

Analyzer:

```text
rows=6 control_rows=2
VERDICT: reject
- eval/buffer_delta=-0.0417 < 0.0300
- eval/buffer_vs_corrupt_delta=0.0000 < 0.0300
- eval/answer_logprob_margin=-0.0433 < 0.0000
- eval/answer_logprob_vs_corrupt_margin=-0.0264 < 0.0000
```

Decision:

- AE fixes the autoresearch-loop ordering bug exposed by AD. The two-serialized
  sampler substrate with fresh syncs is now operationally credible.
- AE does not fix the OPD mechanism. Loss and hidden-match loss improved
  monotonically, but the causal controls worsened: the no-pause path remained
  better than pause, and pause tied corrupted pause on exact-match accuracy.
- Do not interpret the falling hidden-match/loss curves as buffer use. They are
  proxy metrics unless pause beats both no-pause and corrupted pause on accuracy
  and answer logprob.
- The next science change should alter the causal objective/control design. RiM
  can supply ablation ideas, but copying RiM is not the answer unless the
  variant passes these OPD causal gates.

### Config AD Answer-Logprob Causal Gate Pilot

Run:

```text
/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T120111Z-configAD-er-opd-q36-35b-slots-trainer-head
W&B run: fngf2zxn
```

Config AD reused AC's training substrate but enabled the answer-logprob causal
gate:

- two serialized student samplers behind SMG:
  `sglang-0:30060` and `teacher-sglang-1:30000`
- serial fresh weight sync to both endpoints
- native SGLang `/generate` scorers for answer logprob:
  `sglang-0:30060` and `teacher-sglang-1:30000`
- same AC objective:
  `opd_supervise_buffer_only=true`,
  `opd_hidden_match_coef=2.0`,
  `opd_contrastive_corrupt_buffer_weight=0.25`,
  `max_new_tokens=64`,
  `student_stop_sequences='["\\n"]'`

Operational result:

- Startup registration succeeded to both endpoints after restarting the sampler
  services with the SGLang P2P receiver-cache invalidation patch.
- The warmup profile row was operationally clean: answer-logprob request failure
  fraction was `0.0`, all three answer-logprob arms scored `96` examples, and
  sampler routing was balanced exactly `208/208` across the two workers.
- The run failed on the next post-step sync to endpoint 1 with Mooncake
  `received packet mismatch` / RDMA path mismatch. The receiver-cache
  invalidation fixed stale startup registration, but it did not fix repeated
  step sync.
- Logs showed the step sync starting while sampler traffic was still queued or
  in flight. That means the current orchestration does not yet provide a hard
  "rollout epoch complete, samplers quiesced, then sync" barrier.

Warmup row:

| step | warmup | loss | eval_acc | acc_pause | acc_nopause | acc_corrupt | delta | corrupt_delta | anslp_margin | anslp_z | anslp_corrupt_margin | anslp_corrupt_z | anslp_fail | sampler_active | balance | sync |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | yes | 0.5770 | 0.4844 | 0.6979 | 0.7604 | 0.6771 | -0.0625 | +0.0208 | +0.0028 | +0.3237 | -0.0092 | -1.2480 | 0.0000 | 2 | 1.0000 | 2 endpoints, serial |

Analyzer:

```text
rows=1 control_rows=1
VERDICT: incomplete (no non-warmup control rows)
```

Warmup-inclusive diagnostic gate:

```text
VERDICT: reject
- eval/buffer_delta=-0.0625 < 0.0300
- eval/buffer_delta_z=-0.9768 < 2.0000
- eval/control_n=96 < 192
- eval/buffer_vs_corrupt_delta=0.0208 < 0.0300
- eval/buffer_vs_corrupt_delta_z=0.3115 < 2.0000
- eval/answer_logprob_vs_corrupt_margin=-0.0092 < 0.0000
- eval/answer_logprob_vs_corrupt_margin_z=-1.2480 < 0.0000
```

Decision:

- AD is not a completed science run. It never reached a non-warmup control
  checkpoint.
- The new answer-logprob gate is doing useful work: even at warmup, it separated
  "accuracy with a pause prompt" from "the pause buffer causally raises answer
  probability." Pause barely beat no-pause on answer logprob and was worse than
  corrupted pause.
- The next blocker is operational, not RiM-vs-OPD: enforce sampler quiescence
  before every weight sync and revalidate two serialized samplers after the P2P
  mismatch is fixed. Do not increase per-pod SGLang batching while this remains
  unresolved.
- Keep RiM as an ablation template, not as the recipe to copy. A RiM-like method
  must pass the same pause/no-pause/corrupt-pause and answer-logprob causal
  gates before it counts as evidence for an OPD memory channel.

### Config AC Contrastive64 Stop-Newline Rerun After P2P Guard

Run:

```text
/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T111539Z-configAC-er-opd-q36-35b-slots-trainer-head
W&B run: t8eazs3n, opd-q36-35b-randsymbol-contrastive64-stopnl-0shot-hm-configac
```

This reran AC after adding sync request timeouts, endpoint-scoped P2P sync group
names, bounded P2P pending-transfer drains, and cleanup for both base and
endpoint-scoped receiver groups. It used the same two serialized student
samplers behind SMG:

```text
er-opd-q36-35b-slots-sglang-0:30060
er-opd-q36-35b-slots-teacher-sglang-1:30000
```

Operational result:

- Trainer completed all 6 requested steps and exited `rc=0`.
- Registration syncs succeeded to both endpoints:
  `sglang-0` moved 69.32 GB in 3.05s and `teacher-sglang-1` moved 69.32 GB in
  2.34s.
- All six training-time syncs succeeded to both endpoints in serial mode,
  including the prior failure point at step 3.
- Server logs showed endpoint-scoped groups
  `weight_sync_group_ep0` and `weight_sync_group_ep1`.
- Post-step transfer time stayed about 5.2-5.4s; total sync wall time per step
  was about 28.6-32.9s.
- W&B/profile parity passed:
  `audit_wandb_profile.py ... --wandb-run t8eazs3n` returned
  `VERDICT: wandb_matches_profile`.
- The only server errors found after completion were normal torch elastic
  `SIGTERM` messages from slot cleanup.

Rows:

| step | warmup | loss | eval_acc | cap | stop | acc_pause | acc_nopause | acc_corrupt | delta | corrupt_delta | sampler_active | balance | sync |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | yes | 0.5760 | 0.4688 | 0.1875 | 0.0156 | 0.6771 | 0.7292 | 0.7292 | -0.0521 | -0.0521 | 2 | 1.0000 | 2 endpoints, serial |
| 1 | no | 0.5510 | 0.5000 | 0.1875 | 0.0000 | - | - | - | - | - | 2 | 1.0000 | 2 endpoints, serial |
| 2 | no | 0.5252 | 0.4531 | 0.0469 | 0.0000 | - | - | - | - | - | 2 | 0.8649 | 2 endpoints, serial |
| 3 | no | 0.5027 | 0.4844 | 0.0625 | 0.0156 | - | - | - | - | - | 2 | 0.7297 | 2 endpoints, serial |
| 4 | no | 0.4688 | 0.4688 | 0.0312 | 0.0000 | - | - | - | - | - | 2 | 0.8378 | 2 endpoints, serial |
| 5 | no | 0.4190 | 0.5000 | 0.0469 | 0.0000 | 0.7292 | 0.7500 | 0.7083 | -0.0208 | +0.0208 | 2 | 0.9931 | 2 endpoints, serial |

Analyzer:

```text
rows=6 control_rows=2
VERDICT: reject
```

Reject reasons:

- `eval/buffer_delta=-0.0208 < 0.0300`
- `eval/buffer_delta_z=-0.3290 < 2.0000`
- `eval/control_n=96 < 192`
- `eval/buffer_lead_delta=-0.0069 <= 0`
- `eval/buffer_vs_corrupt_delta=0.0208 < 0.0300`
- `eval/buffer_vs_corrupt_delta_z=0.3211 < 2.0000`

Decision:

- AC is now a valid operational result: repeated two-endpoint serial sync and
  two-sampler routing are viable for Qwen3.6 science after the P2P guard patch.
- AC is still a science reject. Loss decreased from `0.5760` to `0.4190`, and
  the cap/stop artifacts were low by the final control row, but the pause slots
  did not become causally load-bearing. At step 5, real pause remained below
  no-pause (`0.7292` vs `0.7500`) and beat corrupted pause by only `+0.0208`.
- This supports the current diagnosis: hidden-match plus a corrupted-buffer
  auxiliary can move internal-state/loss metrics without creating an
  answer-level causal margin. The next science recipe should directly optimize
  or structurally enforce that margin instead of continuing filler-token sweeps.

### Config AC Contrastive64 Stop-Newline Pilot

Run:

```text
/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T104851Z-configAC-er-opd-q36-35b-slots-trainer-head
W&B run: r8bc87nb, opd-q36-35b-randsymbol-contrastive64-stopnl-0shot-hm-configac
```

Config AC is AB with a larger rollout/control budget:

- `max_new_tokens=64`
- `eval_max_new_tokens=64`
- `eval_accuracy_every=5`
- stop-sequence leakage metrics for rollout and all control arms
- same contrastive buffer objective as AB:
  `opd_supervise_buffer_only=true`,
  `opd_hidden_match_coef=2.0`,
  `opd_contrastive_corrupt_buffer_weight=0.25`
- two serialized student samplers behind SMG:
  `sglang-0:30060` and `teacher-sglang-1:30000`
- fresh serial P2P sync to both endpoints at registration and after every
  successful train step

Rows before abort:

| step | warmup | loss | eval_acc | cap | stop | acc_pause | acc_nopause | acc_corrupt | delta | corrupt_delta | hm_raw | sampler_active | balance | sync |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | yes | 0.5766 | 0.5000 | 0.2500 | 0.0156 | 0.6979 | 0.7604 | 0.6875 | -0.0625 | +0.0104 | 0.1104 | 2 | 1.0000 | 2 endpoints, serial |
| 1 | no | 0.5504 | 0.3750 | 0.2031 | 0.0312 | - | - | - | - | - | 0.1079 | 2 | 0.9200 | 2 endpoints, serial |
| 2 | no | 0.5267 | 0.4531 | 0.0469 | 0.0000 | - | - | - | - | - | 0.1168 | 2 | 0.8049 | 2 endpoints, serial |
| 3 | no | 0.5039 | 0.5156 | 0.0469 | 0.0000 | - | - | - | - | - | 0.1411 | 2 | 0.7812 | failed |

Step-0 control artifacts were clean: pause/no-pause/corrupt cap-hit fractions,
stop-sequence fractions, and request-failure fractions were all `0.0`. The
on-policy cap-hit artifact was also much lower than AB: `0.25` at step 0 and
`0.0469` by steps 2-3. AC therefore fixed a measurement problem, but did not
reach a non-warmup control point.

Analyzer:

```text
VERDICT: incomplete (no non-warmup control rows)
```

W&B/profile audit:

```text
audit_wandb_profile.py ... --wandb-run r8bc87nb
state=failed
VERDICT: wandb_matches_profile
```

Failure mode:

- Registration sync was clean:
  `sglang-0` synced 69.32 GB in 3.03s and `teacher-sglang-1` synced
  69.32 GB in 2.47s.
- Steps 0-2 synced both endpoints cleanly after training. Each post-step sync
  moved 138.64 GB total across two endpoints, with transfer time about
  5.0-5.5s.
- Step 3 failed while syncing endpoint 0, `sglang-0` (`10.42.52.45`):
  repeated `received packet mismatch` for
  `10.42.60.75:16796@mlx5_6 -> 10.42.52.45:15463@mlx5_2`, then
  `batch_transfer_sync ... endpoint_idx=0 ... after 50 attempts`.
- The client correctly wrote a step-3 profile row with `sync_success=false`,
  `sync_failure`, `sync_endpoint_success_count=0`, and
  `sync_serial_endpoint_sync=0.0`, then aborted.
- After the abort, trainer controls were stopped with `--remove-run`, the two
  student sampler slots were restarted, and dispatch was restarted after both
  SGLang children reported `Qwen/Qwen3.6-35B-A3B`. Dispatch was clean afterward:
  `/v1/models` returned only `Qwen/Qwen3.6-35B-A3B`.

Decision:

- This first AC attempt is not a science result. It did not reach the step-5
  non-warmup control gate.
- This failure was superseded by the post-guard AC rerun above. The immediate
  fix was to use endpoint-scoped P2P groups, bounded sync waits, and receiver
  cleanup before registration.
- Do not "fix" this by using per-pod SGLang batching. Keep serialized sampler
  workers for correctness; increase throughput with multiple workers behind SMG.
- Continue treating any sync endpoint failure as a hard invalidation for
  on-policy conclusions.

### Config AB Contrastive Buffer Fail-Fast Smoke

Run:

```text
/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T103552Z-configAB-er-opd-q36-35b-slots-trainer-head
W&B run: cizpg3ae, opd-q36-35b-randsymbol-contrastive-stopnl-0shot-hm-configab
```

This reran AB for three steps after adding bounded sampler requests, bounded
sync awaits, and failure-row emission. It used two serialized student samplers
behind SMG in `spare-teacher1` layout.

Rows:

| step | warmup | eval_acc | cap | acc_pause | acc_nopause | acc_corrupt | delta | corrupt_delta | hm_raw | datums | contrast_mult | sampler_active | balance | sync |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | yes | 0.4844 | 0.4219 | 0.7396 | 0.7188 | 0.7396 | +0.0208 | 0.0000 | 0.1104 | 128 | 2.0 | 2 | 1.0000 | 2 endpoints, serial |
| 1 | no | 0.4844 | 0.4219 | - | - | - | - | - | 0.1077 | 128 | 2.0 | 2 | 0.9640 | 2 endpoints, serial |
| 2 | no | 0.4688 | 0.3281 | - | - | - | - | - | 0.1162 | 128 | 2.0 | 2 | 0.8000 | 2 endpoints, serial |

Analyzer:

```text
VERDICT: incomplete (no non-warmup control rows)
```

W&B/profile audit:

```text
audit_wandb_profile.py ... --wandb-run cizpg3ae
VERDICT: wandb_matches_profile
```

Diagnosis:

- The fail-fast smoke passed operationally: registration and all three step
  syncs succeeded against both endpoints, W&B matched the local profile, and
  sampler routing used both workers.
- It was intentionally too short for science. The only control row was warmup.
- The warmup control did not show causal separation: real pause barely beat
  no-pause but tied corrupt pause, so the real buffer content was not
  load-bearing.
- The 32-token rollout cap was still a major artifact:
  `eval/cap_hit_frac` was `0.4219`, `0.4219`, then `0.3281`. Samples still
  showed post-answer reasoning, `</think>` format behavior, and occasional
  filler/format leakage.
- AC was introduced to separate the cap artifact from the mechanism question.

Decision:

- Treat AB smoke as an operational pass and a science incomplete.
- Do not run a longer 32-token AB as the next science run. Use AC or a stronger
  objective/control redesign after the P2P stability issue is fixed.

### Config AB Contrastive Buffer Partial

Run:

```text
/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T100713Z-configAB-er-opd-q36-35b-slots-trainer-head
W&B run: 21wumhbl, opd-q36-35b-randsymbol-contrastive-stopnl-0shot-hm-configab
```

This was the first training-time contrastive/counterfactual buffer recipe:

- buffer-only supervision: `opd_supervise_buffer_only=true`
- hidden matching strengthened to `opd_hidden_match_coef=2.0`
- each real datum got a paired corrupted-buffer datum
- KL weights stayed on the real buffer only; corrupted datums used zero KL
  weight and negative hidden-match weight on buffer positions
- `opd_contrastive_corrupt_buffer_weight=0.25`
- student dispatch stayed `round_robin` over `sglang-0:30060` and
  `teacher-sglang-1:30000` in `spare-teacher1` layout

Useful rows before interruption:

| step | eval_acc | acc_pause | acc_nopause | acc_corrupt | delta | z | corrupt_delta | datums | contrast_mult | hm_raw | sampler_active | balance | sync |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.4219 | 0.6771 | 0.7500 | 0.7083 | -0.0729 | -1.1210 | -0.0312 | 128 | 2.0 | 0.1104 | 2 | 1.00 | 2 endpoints, serial |
| 1 | 0.5156 | - | - | - | - | - | - | 128 | 2.0 | 0.1077 | 2 | 0.97 | 2 endpoints, serial |

Analyzer:

```text
VERDICT: incomplete (no non-warmup control rows)
```

W&B/profile audit:

```text
audit_wandb_profile.py ... --wandb-run 21wumhbl
VERDICT: wandb_stale_or_mismatched
```

The W&B run was manually synced after stopping the hung client, and W&B still
contained only local step 0. Treat the local `opd_profile.jsonl` as the source
of truth for this interrupted run.

Failure mode:

- Step 0 and step 1 both had clean two-endpoint serial syncs and showed the
  expected contrastive payload: `num_opd_datums=128`,
  `opd_contrastive_corrupt_examples=64`,
  `opd_contrastive_data_multiplier=2.0`, and hidden-match diagnostics present.
- Step 2 hit a P2P sync failure while syncing `teacher-sglang-1`:
  `Peer nic not found in that server: 10.42.60.75:15532@mlx5_6`,
  repeated `received packet mismatch`, then
  `batch_transfer_sync to 10.42.77.65:16626 failed ... endpoint_idx=0 ...
  after 50 attempts`.
- SMG also showed `544` selected chat-completion requests but only `542`
  upstream responses before the trainer was stopped, so the client was stuck
  behind two stale requests and a failed sync rather than producing a valid
  science row.
- The trainer was stopped with `stop-trainer-control --remove-run`, then the
  student inference controls were restarted. Dispatch, `sglang-0`, and
  `teacher-sglang-1` were healthy again after 12 readiness polls.

Decision:

- Do not promote or reject AB on science yet; this is an infra-interrupted
  partial.
- AB does validate the next direction mechanically: the loss stack can carry
  separate hidden-match weights and contrast real versus corrupted buffers.
- Follow-up implemented after this run: the OPD client now bounds sampler
  gathers by `request_timeout`, bounds post-step sync awaits by
  `weight_sync_timeout + 30s`, writes a profile row with `sync_success=false`
  and `sync_failure`, then aborts instead of continuing with stale samplers.
- The generator now renders AB with `request_timeout=600` and
  `weight_sync_timeout=300`; use a short smoke after any P2P restart before a
  longer AB retry.

### Config AA Buffer-Only Stop-Newline Causal Follow-Up

Run:

```text
/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T092642Z-configAA-er-opd-q36-35b-slots-trainer-head
W&B run: 2wb7s6da, opd-q36-35b-randsymbol-bufferonly-stopnl-0shot-hm-configaa
```

This reran on the corrected two-sampler substrate and changed the recipe, not
the substrate:

- buffer-only supervision: `opd_supervise_buffer_only=true`
- stop-on-newline student sampling: `student_stop_sequences='["\n"]'`
- larger control completion budget: `eval_max_new_tokens=64`
- stronger hidden matching than Config Z: `hidden_match_coef=0.5`
- student dispatch `round_robin` over `sglang-0:30060` and
  `teacher-sglang-1:30000` in `spare-teacher1` layout
- both endpoints freshly synced at registration and after every train step

Control points:

| step | loss | hidden_match | eval_acc | acc_pause | acc_nopause | acc_corrupt | delta | z | causal_margin | corrupt_delta | pause_cap | sampler_active | balance | sync |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 1.1009 | 0.0858 | 0.4531 | 0.7083 | 0.7708 | 0.6979 | -0.0625 | -0.99 | -0.0625 | +0.0104 | 0.0000 | 2 | 1.00 | 2 endpoints, serial |
| 5 | 0.9594 | 0.0902 | 0.6094 | 0.7188 | 0.7500 | 0.6771 | -0.0312 | -0.49 | -0.0312 | +0.0417 | 0.0000 | 2 | 1.00 | 2 endpoints, serial |

Strict analyzer result:

```text
VERDICT: reject
- eval/buffer_delta=-0.0312 < 0.0300
- eval/buffer_delta_z=-0.4905 < 2.0000
- eval/buffer_lead_delta=-0.0112 <= 0
- eval/buffer_vs_corrupt_delta_z=0.6293 < 2.0000
```

W&B parity:

```text
audit_wandb_profile.py ... --wandb-run 2wb7s6da
VERDICT: wandb_matches_profile
```

Diagnosis:

- AA removed Config Z's biggest measurement artifact: all three control arms had
  `*_cap_hit_frac=0.0` and `control_request_failure_frac_max=0.0` at step 5.
- The substrate again passed: `sync_endpoint_count=2`,
  `sync_endpoint_success_count=2`, `sync_endpoint_failure_count=0`,
  `sync_serial_endpoint_sync=1.0`, `sampler_worker_active_count=2`, and
  `sampler_worker_success_balance_ratio=1.0`.
- The pause content still was not load-bearing. The real pause arm lost to the
  no-pause arm by `0.0312` exact-match points while beating the corrupted-pause
  arm by only `0.0417` with weak `z=0.6293`.
- This is a cleaner negative than Config Z. It argues that the current
  autoresearch loop is overfitting to output/answer behavior and weak auxiliary
  losses, then relying on noisy post-hoc controls to notice the miss.

Decision:

- Keep multiple serialized sampler replicas behind one dispatch/SMG with fresh
  weight syncs. That substrate is valuable for convergence and is no longer the
  dominant suspected failure.
- Do not interpret RiM as the thing to copy. Use it as an ablation template:
  contrast real versus corrupted intermediate information during optimization,
  make the intermediate state explicitly useful for the answer, and promote only
  when the causal margin clears the gate.
- The next recipe should add a training-time contrastive or counterfactual
  pressure against corrupted/no-pause buffers, or change the task so the answer
  is hard to recover without the buffer. More filler-surface sweeps are now low
  value.

### Config Z Fixed-Substrate Two-Sampler Pilot

Run:

```text
/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T085833Z-configZ-er-opd-q36-35b-slots-trainer-head
W&B run: n5ngv35w, opd-q36-35b-randsymbol-tiny-0shot-hm-configz
```

This reran Config Z with the corrected two-sampler substrate:

- student dispatch `round_robin` over `sglang-0:30060` and
  `teacher-sglang-1:30000` in `spare-teacher1` layout
- both endpoints freshly synced at registration and after every train step
- post-step sync forced through serial endpoint mode
- profile rows included sampler-routing metrics and sync endpoint summaries
- W&B matched the local profile for all default audit keys

Control points:

| step | loss | hidden_match | eval_acc | acc_pause | acc_nopause | acc_corrupt | delta | z | corrupt_delta | pause_cap | sampler_active | balance | sync |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.9740 | 0.1665 | 0.4219 | 0.7135 | 0.7448 | 0.6979 | -0.0312 | -0.69 | +0.0156 | 0.0000 | 2 | 1.00 | 2 endpoints, serial |
| 5 | 0.7470 | 0.1516 | 0.7969 | 0.8125 | 0.7969 | 0.8177 | +0.0156 | +0.39 | -0.0052 | 1.0000 | 2 | 1.00 | 2 endpoints, serial |

Strict analyzer result:

```text
VERDICT: reject
- eval/buffer_delta=0.0156 < 0.0300
- eval/buffer_delta_z=0.3862 < 2.0000
- eval/buffer_vs_corrupt_delta=-0.0052 < 0.0300
- eval/buffer_vs_corrupt_delta_z=-0.1315 < 2.0000
- eval/buffer_vs_corrupt_lead_delta=-0.0016 <= 0
- eval/pause_cap_hit_frac=1.0000 > 0.5000
- eval/nopause_cap_hit_frac=1.0000 > 0.5000
- eval/corrupt_pause_cap_hit_frac=1.0000 > 0.5000
```

W&B parity:

```text
audit_wandb_profile.py ... --wandb-run n5ngv35w
VERDICT: wandb_matches_profile
```

Diagnosis:

- The infrastructure substrate is now good enough to use for science:
  `sync_endpoint_count=2`, `sync_endpoint_success_count=2`,
  `sync_endpoint_failure_count=0`, `sync_serial_endpoint_sync=1.0`,
  `sampler_worker_active_count=2`, and
  `sampler_worker_success_balance_ratio=1.0` on the decisive control step.
- The recipe did not prove OPD slot use. The step-5 exact-match pause delta was
  only `+0.0156` with `z=0.3862`, and the corrupted-pause arm was slightly
  better than the real-pause arm. That argues against a learned dependence on
  the specific filler content.
- The apparent eval-accuracy jump to `0.7969` is direct task adaptation until a
  control arm shows that the pause slots are load-bearing. It should not be
  treated as a filler-token mechanism win.
- The step-5 control eval hit the `32` token cap in all three control arms. The
  exact-match delta is therefore capped-output evidence, not a clean reasoning
  intervention measurement.
- With `--max-running-requests 1`, the `192 * 3 = 576` control requests took
  about `8.4` minutes after sync to drain through the two serialized samplers.
  This is expected and not a hang, but it makes large control evals expensive.

Decision:

- Keep the two-sampler fresh-sync substrate and analyzer gates.
- Reject Config Z as a mechanism recipe.
- Next science changes should target the causal objective/control design, not
  another filler-surface sweep. The next run must make it hard for the model to
  improve by direct-answer formatting alone.

### Config Z Single-Sampler Historical Run

Run:

```text
/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T072811Z-configZ-er-opd-q36-35b-slots-trainer-head
W&B run: 30ytmgke, opd-q36-35b-randsymbol-tiny-0shot-hm-configz
```

Config Z revalidated Config U's late positive signal with `192` held-out
control prompts and the artifact metrics. It was stopped after step 5 because
the first non-warmup control failed the strict mechanism gate; one already
in-flight non-control step 6 profile row was written after the stop request.

Control points:

| step | loss | hidden_match | eval_acc | acc_pause | acc_nopause | delta | z | lead_delta | cap_hit | sync |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.9464 | 0.1637 | 0.4688 | 0.7188 | 0.7344 | -0.0156 | -0.34 | +0.1828 | 0.5000 | 1 endpoint |
| 5 | 0.7391 | 0.1565 | 0.7500 | 0.8021 | 0.8177 | -0.0156 | -0.39 | -0.0107 | 0.9375 | 1 endpoint |

Diagnosis:

- Direct on-policy accuracy rose to `0.75`, but no-pause remained better than
  pause on the held-out control. This is direct-answer adaptation, not evidence
  that pause slots became load-bearing.
- `eval/buffer_lead_delta` flipped negative at step 5, so the failure is not
  just exact-match noise.
- The health eval saturated the `max_new_tokens=32` cap by step 4/5
  (`eval/cap_hit_frac` `0.9531` then `0.9375`) and completions mostly emitted
  an answer plus `</think>` followed by explanation. That is an artifact warning,
  not a mechanism win.
- The post-stop step 6 row stayed consistent with this diagnosis:
  `eval/accuracy=0.7188`, `eval/cap_hit_frac=0.9844`, and still only
  `"1 endpoint(s)"` synced.
- The control eval in this completed run used the old hard-coded `16` token
  control budget, causing `pause_cap_hit_frac=nopause_cap_hit_frac=1.0`.
  The client was patched afterward so control eval uses `eval_max_new_tokens`
  when set, else `max_new_tokens`, and logs
  `eval/control_max_completion_tokens`.
- The client was also patched afterward so pause and no-pause control arms are
  submitted in one concurrent gather, with `eval/control_total_requests`,
  `eval/control_sampler_clients`, `eval/control_arms_concurrent`, and per-arm
  `eval/*_request_failure_frac` metrics. That removes the step-5
  arm-A-then-arm-B delay observed in this run and makes failed control requests
  a first-class rejection signal.
- The run synced `"31333 params to 1 endpoint(s)"`; it did not test the
  multi-sampler/fresh-sync topology.
- W&B run `30ytmgke` is stale/incomplete after the stop/crash boundary: the API
  returns profile metrics through step 4 and control metrics through step 0,
  while the local profile contains the decisive step-5 control row and the
  post-stop step-6 row. Use
  `audit_wandb_profile.py <run_dir>/opd_profile.jsonl --wandb-run 30ytmgke` to
  reproduce this check.

Decision:

- Reject Config Z and do not spend step-10 control time on this recipe.
- Do not launch another filler-surface sweep until the next run has a sharper
  causal intervention or a multi-sampler topology is explicitly deployed and
  verified.

### Config Y

Run:

```text
/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T065150Z-configY-er-opd-q36-35b-slots-trainer-head
W&B run: 3hyyw0i6, opd-q36-35b-randsymbol-bufferonly-0shot-hm-configy
```

Control points:

| step | loss | hidden_match | eval_acc | acc_pause | acc_nopause | delta | lead_delta | think_close | sync_s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.0812 | 0.0858 | 0.4531 | 0.6979 | 0.7500 | -0.0521 | +0.1068 | 0.8125 | 2.01 |
| 5 | 0.9412 | 0.0882 | 0.4844 | 0.7188 | 0.7604 | -0.0417 | -0.0169 | 0.9219 | 2.01 |
| 10 | 0.7665 | 0.1008 | 0.3906 | 0.7188 | 0.7396 | -0.0208 | -0.0039 | 0.9375 | 2.18 |

Diagnosis:

- Config Y is a clean negative on the causal buffer objective. The training loss
  moved, but `acc_pause` stayed below `acc_nopause` at every control point.
- The buffer-only hidden-match recipe did not force useful computation into the
  slots. It mostly produced answer-cue/`</think>` behavior, while direct
  on-policy accuracy ended at `0.3906`.
- The run synced `"31333 params to 1 endpoint(s)"`, so it was a single serialized
  sampler pod. That is valid for science, but throughput was sampler-bound.
- `opd_teacher_entropy`, `opd_student_entropy`, and `opd_top1_agreement` were all
  `0.0` because this stack requested diagnostics while using
  `opd_kl_backend=streaming`; the current loss path only computes those
  full-vocab diagnostics for `torch_compile`/`compile`/`auto_chunker` backends.
  On newer client rows, require `opd_full_vocab_diag_active_expected=1.0`
  before treating entropy/top1 as mechanism evidence.

Decision:

- Do not promote filler recipes based only on loss/hidden-match movement.
- Do not "copy RiM" as the next action. RiM is evidence that the setup needs a
  sharper credit-assignment/evaluation loop; the next OPD loop should prove that
  pause slots are causally load-bearing under matched no-pause controls.
- A candidate is promotable only if the control delta is positive with enough
  evidence and sampler artifacts are clean. Use `eval/buffer_delta`,
  `eval/buffer_delta_z`, `eval/*_repeated_numeric_frac`,
  `eval/*_filler_leak_frac`, `eval/*_answer_cue_leak_frac`, and
  `eval/*_cap_hit_frac` as first-line gates.

Cross-run diagnosis:

- The autoresearch loop has been too surface-oriented. It changed filler text,
  exact token IDs, output caps, stop sequences, and hidden-match coefficients,
  but most variants left the model with a direct answer/no-buffer bypass.
- RiM should be treated as an ablation template, not a method to paste into OPD.
  Its successful ingredients are dedicated memory tokens, dense per-step
  grounding, and a block-causal mask that structurally forces computation
  through memory. Current OPD/OPSD runs use visible ordinary tokens and do not
  prevent the answer decoder from solving from the prompt.
- The AB/AC contrastive hidden-match variant is a step in the right direction
  mechanically, but it still does not directly punish answer likelihood under
  corrupted/no-pause buffers. A stronger next recipe should optimize an
  answer-level margin such as
  `logp(answer | real_buffer) > logp(answer | corrupt_buffer/no_buffer) + m`.
- Before spending another long Qwen3.6 run, test a smaller structural bottleneck
  variant: the answer path must depend on a real intermediate state, and a
  matched corrupt/no-buffer arm must fail under the same answer target. If that
  does not work cheaply, more filler-surface sweeps are low value.
- Operationally, keep multiple serialized sampler workers behind SMG with fresh
  weight syncs. That is the right throughput/convergence substrate. The
  post-guard AC rerun completed six steps with two freshly synced endpoints, so
  the immediate blocker is no longer "multiple samplers"; it is proving a recipe
  where the answer actually depends on the buffer.

Suggested promotion gates for the next run:

- `eval/buffer_delta >= +0.03` and `eval/buffer_delta_z >= 2.0` on at least
  `192` held-out control prompts.
- `eval/buffer_lead_delta > 0` on multiplication tasks, not just exact-match
  noise.
- `eval/buffer_vs_corrupt_delta >= +0.03` and
  `eval/buffer_vs_corrupt_delta_z >= 2.0`; the trained pause must beat a
  same-prefix corrupted pause, not only a no-pause prompt.
- `eval/buffer_vs_corrupt_lead_delta > 0`.
- `eval/pause_repeated_numeric_frac` and `eval/nopause_repeated_numeric_frac`
  below `0.10`; also require `eval/corrupt_pause_repeated_numeric_frac < 0.10`.
  If any spikes, invalidate the sampler output.
- `eval/pause_filler_leak_frac`, `eval/pause_answer_cue_leak_frac`, and
  `eval/pause_cap_hit_frac` should be low and not rising across control points.
- `eval/pause_request_failure_frac` and `eval/nopause_request_failure_frac`
  should remain `0.0`; `eval/corrupt_pause_request_failure_frac` should also be
  `0.0`. Missing control generations invalidate the gate.
- `sync_success=true`, and the profile row must report the expected endpoint
  count (`sync_endpoint_count=1` for single-sampler diagnostics,
  `sync_endpoint_count=2` for the multi-sampler topology). If
  `sync_endpoint_success_count` or `sync_endpoint_failure_count` is present,
  require all endpoints successful and zero endpoint failures. For the current
  two-sampler Qwen3.6 topology also require `sync_serial_endpoint_sync=1.0`.
  A run that trains against unsynced or partially synced samplers is invalid for
  on-policy conclusions.
- For two-sampler topology, routing metrics must prove traffic hit both sampler
  workers: `sampler_metrics_available=1.0`, `sampler_worker_active_count >= 2`,
  `sampler_worker_success_balance_ratio >= 0.75`, and
  `sampler_policy_round_robin_active=1.0`. A run with two synced samplers but
  one active sampler is invalid for throughput and on-policy diversity claims.
- `eval/control_arms_concurrent=1.0`, `eval/control_corrupt_pause_active=1.0`,
  and
  `eval/control_max_completion_tokens >= 32`; otherwise the control comparison
  may be contaminated by arm ordering or token-budget artifacts.
- At least two consecutive control points should pass before spending a long run.

## 12. Known Hazards

### Dispatch Can Register `unknown`

If dispatch starts before SGLang has exposed its model id, `/v1/models` can
return `unknown`, and the OPD client may wait forever for
`Qwen/Qwen3.6-35B-A3B`.

Current mitigation:

- `dispatch_script` waits for each backend `/v1/models` to include
  `Qwen/Qwen3.6-35B-A3B`.
- `write-dispatch-control` can restart only dispatch.

### Stale SGLang P2P Receiver State

After failed runs, SGLang may retain stale P2P receiver state or NIC metadata.
Symptoms include:

```text
Peer nic not found
received packet mismatch
batch_transfer failed
A P2P weight update for group 'weight_sync_group' is already in progress
```

Current mitigations:

- Trainer registration sends best-effort `continue_generation`,
  `complete_weights_update`, and `continue_generation` calls before adding each
  inference endpoint. This clears stale receiver groups and reopens a sampler
  left paused by a failed P2P sync.
- Multi-endpoint serial sync uses endpoint-scoped P2P groups
  (`weight_sync_group_ep0`, `weight_sync_group_ep1`, ...), so one endpoint's
  receiver state is not reused by the next endpoint in the same sync cycle.
- Cleanup clears the base `weight_sync_group` and the endpoint-scoped groups on
  every registered endpoint.
- Registration failure also retries endpoint recovery before exiting, and a
  successful registration is followed by a one-token direct `/generate` probe.
  If this probe fails, do not start OPD; restart the affected SGLang child.
- The trainer registration `add_inference_endpoint` request has a bounded
  `curl -m <weight_sync_timeout>` timeout, so a failed sync cannot leave the
  wrapper blocked forever before cleanup.
- Client and server sync paths now carry bounded timeouts. A failed P2P transfer
  should produce a failure row instead of hanging the trainer indefinitely.
- The 13:22 AF rerun showed these mitigations are not sufficient for repeated
  mid-run P2P stability: endpoint-scoped groups and stale-state cleanup still
  failed at step 4 with `received packet mismatch`. For science runs that need a
  clean final causal checkpoint, use Config AG's NCCL broadcast sync until the
  Mooncake repeated-sync path is fixed and revalidated.
- If stale NIC/cache errors persist, restart student inference:

```bash
python "$GENERATOR" stop-trainer-control --remove-run
python "$GENERATOR" write-student-inference-control --sampler-replicas 2 --sampler-layout spare-teacher1
```

After both restarted SGLang children report `Qwen/Qwen3.6-35B-A3B`, restart
dispatch only if `/v1/models` contains `unknown`:

```bash
python "$GENERATOR" write-dispatch-control --sampler-replicas 2 --sampler-layout spare-teacher1
```

### GPU-Direct Requires Allocator Discipline

The generator unsets both allocator variables in trainer and sampler scripts:

```bash
unset PYTORCH_ALLOC_CONF
unset PYTORCH_CUDA_ALLOC_CONF
```

Do not reintroduce PyTorch expandable CUDA segments while using GPU-direct P2P
sync. It previously broke Mooncake CUDA registration.

### Qwen3.6 Batched Decoding Hazard

The student SGLang pod must stay serialized for science runs:

```text
--max-running-requests 1
```

High per-pod concurrency produced repeated numeric suffixes across unrelated
prompts and invalid eval accuracy. If throughput is needed, prefer multiple
serialized student SGLang pods behind SMG rather than increasing per-pod
concurrency.

For multi-sampler science runs, every sampler endpoint must be registered with
the trainer and receive each fresh weight sync before it serves the next
on-policy batch. The generator derives dispatch backends and trainer endpoint
registration from the CLI `--sampler-replicas` value; after increasing it,
confirm the wrapper log contains all `Registering SGLang endpoint <i>` lines,
the profile row reports the expected endpoint count, and dispatch logs
`SMG dispatch policy: round_robin`.

The 2026-06-03 two-sampler smoke also exposed a combined P2P sync bug: initial
per-endpoint registration syncs succeeded, but the post-step all-endpoint sync
failed with Mooncake/NIC session mismatch. The generator now sets
`XORL_SERIAL_INFERENCE_ENDPOINT_SYNC=1` for multi-endpoint trainer runs, forcing
the post-step sync to reuse the known-good one-endpoint transfer path. Keep this
on until the combined multi-endpoint P2P backend is fixed and revalidated.

2026-06-03 AB follow-up: serial per-endpoint sync is necessary but not
sufficient. In the interrupted Config AB run, endpoint 0 (`sglang-0`) completed
post-step sync, then endpoint 1 (`teacher-sglang-1` as a student sampler) failed
with `Peer nic not found` / `received packet mismatch` against
`10.42.77.65:16626`. Follow-up mitigation is in place: client-side sampler
requests and post-step sync are now bounded, and sync failures persist a
`sync_success=false` / `sync_failure` profile row before aborting. The AB
trainer command renders `request_timeout=600` and `weight_sync_timeout=300`.

2026-06-03 AC follow-up: serial per-endpoint sync still failed on a later step,
this time while syncing endpoint 0 (`sglang-0`, pod IP `10.42.52.45`). The run
completed registration and steps 0-2, then step 3 emitted repeated
`received packet mismatch` messages for
`10.42.60.75:16796@mlx5_6 -> 10.42.52.45:15463@mlx5_2` and eventually
`batch_transfer_sync ... endpoint_idx=0 ... after 50 attempts`. The client wrote
the expected `sync_success=false` row and aborted, but the server did not return
the rank failure promptly; the client timed out after 300s. Treat prompt server
error propagation and receiver/session cleanup as the next infra fix before any
long multi-sampler science run.

2026-06-03 follow-up: `sglang-1` was rendered/applied with
`--sampler-replicas 2`, including `team=turbo`, but initially remained Pending
because no 8-GPU `node-group=nccl` node was available. Do not run
`write-student-inference-control --sampler-replicas 2` or any trainer with
`--sampler-replicas 2` until `kubectl get pod er-opd-q36-35b-slots-sglang-1`
shows `Ready` and the slot status reports a healthy SGLang child, unless using
`--sampler-layout spare-teacher1`.

`spare-teacher1` layout repurposes the already scheduled `teacher-sglang-1` pod
as the second student sampler on its existing service port `30000`. In that
layout:

- Dispatch should list `sglang-0:30060` and `teacher-sglang-1:30000` and run
  with `--policy round_robin`.
- Trainer endpoint registration should include both endpoints and
  profile rows should report `sync_endpoint_count=2`,
  `sync_endpoint_success_count=2`, and `sync_serial_endpoint_sync=1.0`.
- Teacher hidden-cache calls must use `teacher-sglang-0:30000` directly.
- `teacher-smg` should be restarted by the generator with only
  `teacher-sglang-0` in its worker list, because `teacher-sglang-1` is no
  longer a teacher while acting as a sampler.

### Full-Vocab Diagnostics Are Backend-Gated

The OPD loss currently emits nonzero `opd_teacher_entropy`,
`opd_student_entropy`, and `opd_top1_agreement` only for compile-style KL
backends: `torch_compile`, `compile`, or `auto_chunker`. The Qwen3.6 stack uses
`opd_kl_backend=streaming` for throughput, so those three fields can be `0.0`
even when `opd_emit_full_vocab_diagnostics=true`.

Newer OPD client rows log:

```text
opd_full_vocab_diag_requested
opd_full_vocab_diag_active_expected
opd_full_vocab_diag_unavailable_expected
```

Interpret entropy/top1 only when `opd_full_vocab_diag_active_expected=1.0`.
When `requested=1.0` and `unavailable=1.0`, treat the zero entropy/top1 values
as missing diagnostics, not as evidence about the learned mechanism. If exact
full-vocab diagnostics are needed, run a small diagnostic-only job with
`opd_kl_backend=torch_compile`; do not switch the long 35B science run to that
backend without revalidating memory and throughput.

### Trainer Head Exit Can Leave Stale Workers

Before the 2026-06-03 cleanup patch, the OPD client could exit normally while
worker slot children kept running. That made a later launch look healthy at the
pod level while old `torch.distributed.run` / `runner_dispatcher` processes were
still alive.

Current mitigation:

- `trainer_head_script` cleanup now touches all `trainer-worker-*` stop files on
  head exit.
- For any run written before that generator patch, still run:

```bash
python "$GENERATOR" stop-trainer-control --remove-run
```

- Then verify the rank-process count is zero before writing the next trainer
  control script.

### Kubernetes Pod `Running` Is Not Enough

The pods are slot agents. `kubectl get pods` showing `Running` only means the
slot is alive. It does not mean the workload is active. Always check:

```bash
python "$GENERATOR" status
```

## 13. Snapshot History And Detailed Notes

Current state is summarized in Section 0. The first snapshot below is retained
as history from when Config AO was still live; AO has since completed and was
scientifically rejected, and Config AN is now the active run.

Historical snapshot as of 2026-06-03 19:05 UTC after the completed Config AM
final-only 1024-control confirmation, corrupt-boundary probes,
answer-selection distractor probe, and the then-live Config AO launch:

```text
student SGLang-0, teacher-sglang-1-as-student, dispatch, and teacher SMG are warm
Config AO trainer roles were running at this historical snapshot
dedicated sglang-1 control slot is stopped; use --sampler-layout spare-teacher1
until a dedicated sglang-1 pod is intentionally scheduled and healthy
dispatch is running round_robin over sglang-0:30060 and teacher-sglang-1:30000
dispatch /v1/models returns only Qwen/Qwen3.6-35B-A3B
student sampler /health endpoints are cheap readiness probes
SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 is required for this stack
```

Historical AO follow-up science pilot:

- Config:
  AO,
  `randsymbol-cachemisbal-posanswer-rotmemws-final1k-nccl-stopnl`.
- Run dir:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T185758Z-configAO-er-opd-q36-35b-slots-trainer-head`
- W&B run:
  `8psasdb4`,
  `opd-q36-35b-randsymbol-cachemisbal-posanswer-rotmemws-final1k-nccl-stopnl-0shot-hm-configao`.
- Launch:
  `write-trainer-control --config AO --num-steps 6 --prompts-per-step 64 --sampler-replicas 2 --sampler-layout spare-teacher1`.
- Purpose:
  keep AM's final-only 1024 held-out controls, boundary-preserved corrupt
  generation, chunked answer-logprob scoring, and answer-selection diagnostics,
  but remove corrupt-answer/buffer training negatives. AO uses a positive-only
  answer KL target (`opd_positive_answer_weight=0.125`) plus balanced
  cache-mismatch hidden supervision (`opd_cache_mismatch_memory_weight=0.25`).
- Warmup row:
  step 0 emitted successfully. It is intentionally not a control row
  (`eval/control_allowed_by_start_step=0`, `eval/control_start_step=5`), so the
  analyzer reports `VERDICT: incomplete (no control rows)` until step 5.
- Warmup substrate:
  serial NCCL sync succeeded to both sampler endpoints
  (`sync_endpoint_count=2`, `sync_endpoint_success_count=2`,
  `sync_serial_endpoint_sync=1.0`), dispatch routed exactly `32/32` across the
  two workers, sampler quiescence succeeded, and both sampler health metrics
  were good.
- Warmup objective audit:
  `opd_positive_answer_weight=0.125`,
  `opd_positive_answer_examples=64`,
  `opd_contrastive_corrupt_answer_weight=0.0`,
  `opd_contrastive_corrupt_answer_examples=0`,
  `opd_contrastive_corrupt_buffer_weight=0.0`, and
  `opd_contrastive_corrupt_examples=0`. This confirms AO is testing a
  positive-answer/cache-mismatch direction, not the legacy corrupt-negative
  objective.
- Historical next gate:
  this was to wait for the step-5 control row, then run the analyzer with
  `--require-answer-select-control --min-answer-select-z 2 --min-answer-select-paired-n 1024`
  plus strict two-endpoint sync/routing requirements. That gate has since
  resolved negative for AO; see Section 0.2 for the final AO result.
- Live monitor:
  `monitor_live_opd.py` now gives an external liveness verdict for incomplete
  runs by combining local profile rows, W&B summary/history, and dispatcher SMG
  counters. Use it while a large control row is still pending:

```bash
python experiments/opd_profile/monitor_live_opd.py "$PROFILE" \
  --wandb-run 8psasdb4 --samples 2 --interval-s 15 \
  --native-log-pod er-opd-q36-35b-slots-sglang-0 \
  --native-log-pod er-opd-q36-35b-slots-teacher-sglang-1
```

  Snapshot at `2026-06-03 19:37:55 UTC`: `VERDICT: live_progress`,
  profile rows `0..4`, no step-5 control row yet, dispatcher
  `requests=15929`, `responses=15879`, gap `50`, active connections `16`,
  aged inflight over 30s `0`, response rate about `1.79` requests/s. W&B was
  lagging at summary step `3`, while local profile step `4` was already present;
  treat local profile plus SMG as authoritative until W&B catches up or the run
  finishes.
  Snapshot at `2026-06-03 19:44:01 UTC`: `VERDICT: live_progress_native`.
  Dispatcher sampled-control traffic had drained (`active=0`), and recent
  native SGLang logs showed `/generate` activity on `sglang-0`, consistent with
  answer-logprob scoring after sampled controls.
  Snapshot at `2026-06-03 19:45:40 UTC`: native queue telemetry showed
  `sglang-0:0/126` and `teacher-sglang-1:64/126` as
  `last_queue_req/max_queue_req`, confirming the answer-logprob phase was
  actively draining chunks rather than hung.
  Snapshot at `2026-06-03 19:47:20 UTC`: profile still had rows `0..4`, while
  native telemetry showed `teacher-sglang-1:109/126`; this indicates another
  answer-logprob chunk was in flight. The next action remains: wait for the
  step-5 row, then run the strict analyzer and W&B/profile audit.
- Pre-control W&B/profile read:
  W&B history currently contains steps `0..3`; local profile contains `0..4`.
  Local AO objective metrics are active on every emitted row:
  `opd_positive_answer_examples=64`, `opd_cache_mismatch_examples=64`,
  `sync_endpoint_success_count=2`, and sampler balance `1.0`. Local
  `opd_hidden_match_loss` drifted from `0.01743` to `0.01396`; this is not a
  promotion signal by itself, but it confirms the intended AO training objective
  is live while final controls are pending.

Latest completed science pilot:

- Run dir:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T171221Z-configAM-er-opd-q36-35b-slots-trainer-head`
- W&B run:
  `c7f09nym`,
  `opd-q36-35b-randsymbol-contrastive64-answercontrast-rotmem-final1k-nccl-stopnl-0shot-hm-configam`.
- Status:
  completed all 6 requested steps; trainer-head exited `rc=0` at
  `2026-06-03T18:12:04Z`, trainer workers were stopped, and warm inference
  roles remain running.
- Analyzer:
  `VERDICT: reject` because
  `eval/answer_logprob_request_failure_frac=1.0000 > 0.0000` and
  `eval/corrupt_pause_cap_hit_frac=0.5918 > 0.5000`.
- W&B/profile parity:
  `VERDICT: wandb_matches_profile`, W&B state `finished`.
- Operational substrate passed:
  all rows synced two endpoints with serial NCCL
  (`sync_endpoint_success_count=2`, `sync_endpoint_failure_count=0`,
  `sync_serial_endpoint_sync=1.0`), sampler routing stayed exact round-robin,
  and quiescence passed before every sync. The final sampled-policy control
  submitted `3072` requests with zero per-arm request failures.
- Final science signal:
  step-5 exact controls passed the z gates at `n=1024`:
  `acc_pause=0.6494`, `acc_nopause=0.5557`,
  `acc_corrupt_pause=0.5811`, `buffer_delta=+0.0938` (`z=4.3548`), and
  `buffer_vs_corrupt_delta=+0.0684` (`z=3.1871`). This confirms AI's
  step-5 exact-match deltas were underpowered at `n=192`, not absent.
- Objective diagnostics:
  AM did not enable cache mismatch or teacher-memory pair diagnostics. Raw
  hidden separation was positive at the final row:
  `opd_hidden_match_neg_minus_pos_raw=0.6014`.
- Answer-logprob caveat:
  AM's in-loop answer-logprob row is unusable. The pre-fix client sent one
  oversized prompt-scoring batch per native endpoint, roughly `1536` sequences
  each, and both scorers returned transient HTTP 503s. This produced
  `eval/answer_logprob_request_failure_frac=1.0` with no paired scores.
  A post-hoc chunked scorer against the still-warm final AM sampler weights
  recovered the measurement:
  `answer_logprob_margin=+0.1784` (`z=36.7621`),
  `answer_logprob_vs_corrupt_margin=+0.3114` (`z=32.2773`),
  `paired_n=1024`, `request_failure_frac=0.0`, saved at
  `posthoc_answer_logprob_chunked_20260603T1816Z.json`.
- Corrupt-boundary probe:
  legacy text `rotate` drops the leading whitespace from the assistant prefill.
  On a 192-prompt post-hoc free-generation probe, `rotate_preserve_ws` fixed the
  boundary (`leading_ws_match=1.0`, `len_delta_chars=0.0`) and removed corrupt
  cap hits (`0.0`), but also collapsed sampled pause-vs-corrupt exact-match to
  `+0.0052` (`z=0.1126`). The corresponding 1024-prompt chunked answer-logprob
  contrast remained positive but much smaller:
  `answer_logprob_vs_corrupt_margin=+0.0568` (`z=8.1144`).
- Answer-selection distractor probe:
  a 1024-prompt post-hoc scorer compared each correct answer against a paired
  wrong answer from another prompt under the same prompt/prefix. Pause had lower
  correct-vs-wrong separation than no-pause
  (`select_delta=-0.1271`, `z=-13.1780`) and lower separation than
  boundary-preserved corrupt (`select_vs_corrupt_delta=-0.7515`,
  `z=-68.7370`). Request failure was `0.0` across `6144` scoring requests.
- Interpretation:
  AM is the strongest confirmation that the answer-causal AI/AH branch is worth
  studying, but not a clean memory-selection success. It should not be promoted
  using the legacy corrupt exact-match gate: that gate partly measured an
  assistant-boundary artifact. The cleanest positive result is pause-vs-no-pause
  exact `+0.0938` at `n=1024`; absolute answer-logprob is also positive, but
  the answer-selection probe shows that absolute likelihood is not enough to
  establish prompt-specific answer discrimination.
- Consequence:
  do not discard the AI/AH answer-causal branch. Do not spend another full run
  on fixed-token order corruption unless an integrated W&B row is required
  (Config AN is available for that). Move to a prompt-specific corrupt control
  with identical visible boundary: externalized memory-token shuffle or a
  cache/hidden-target mismatch. Future promotion gates should include
  correct-vs-distractor answer-selection deltas.

Previous completed mechanism pilot:

- Run dir:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T164109Z-configAL-er-opd-q36-35b-slots-trainer-head`
- W&B run:
  `uxwype0s`,
  `opd-q36-35b-randsymbol-contrastive64-answercontrast-rotmem-cachemisbal-nccl-stopnl-0shot-hm-configal`.
- Status:
  completed all 6 requested steps; trainer-head exited `rc=0`, trainer workers
  were stopped with `stop-trainer-control --remove-run`.
- Analyzer:
  `VERDICT: reject`, but only because
  `eval/corrupt_pause_cap_hit_frac=0.5833 > 0.5000`.
- Final science signal:
  step-5 exact controls passed the z gates:
  `acc_pause=0.7292`, `acc_nopause=0.6146`,
  `acc_corrupt_pause=0.6198`, `buffer_delta=+0.1146` (`z=2.4091`), and
  `buffer_vs_corrupt_delta=+0.1094` (`z=2.3028`). Answer-logprob controls were
  also strongly positive:
  `answer_logprob_margin=+0.2225` (`z=17.0395`) and
  `answer_logprob_vs_corrupt_margin=+0.2719` (`z=15.1602`).
- Interpretation:
  AL is a real mechanism hit and an AK cleanup success, but AM's cleaner
  no-cache-mismatch setup is now the better final-only confirmation target.

Previous completed science pilot:

- Run dir:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T143837Z-configAI-er-opd-q36-35b-slots-trainer-head`
- W&B run:
  `fzittb3d`,
  `opd-q36-35b-randsymbol-contrastive64-answercontrast-rotmem-nccl-stopnl-0shot-hm-configai`
- Status:
  completed all 6 requested steps; trainer-head exited `rc=0`, workers stopped.
- Analyzer:
  `VERDICT: reject`.
- W&B/profile parity:
  `VERDICT: wandb_matches_profile`, W&B state `finished`.
- Operational substrate passed:
  `sync_endpoint_count=2`, `sync_endpoint_success_count=2`,
  `sync_endpoint_failure_count=0`, `sync_serial_endpoint_sync=1.0`,
  two active sampler workers at both control points,
  `sampler_worker_success_balance_ratio=1.0`, and `sampler_quiesce_success=1.0`
  before every sync.
- Final science gate failed:
  step-5 `eval/buffer_delta=+0.0938` but
  `eval/buffer_delta_z=1.9063 < 2.0`; step-5
  `eval/buffer_vs_corrupt_delta=+0.0625` but
  `eval/buffer_vs_corrupt_delta_z=1.2790 < 2.0`; and
  `eval/corrupt_pause_cap_hit_frac=0.7083`.
- Positive mechanism signal:
  `eval/answer_logprob_margin=+0.2163` (`z=16.6732`) and
  `eval/answer_logprob_vs_corrupt_margin=+0.3917` (`z=17.3105`).
- Treat AI as an operational pass and a partial mechanism hit: it supports the
  answer-causal direction, but the corrupt control still has a cap-hit artifact
  and exact-match significance is below the promotion gate.
- 2026-06-03 15:16 UTC correction: do not pursue a token-level cross-prompt
  shuffle for the current fixed-slot recipe. Because every prompt receives the
  same slot-token scaffold, that shuffle would mostly be a no-op. The next
  causal-control design must either externalize prompt-specific memory as tokens
  before shuffling, or mismatch prompt-specific teacher memory/cache targets
  while preserving answer-level validation.
- The client now reports control request-tail metrics:
  `eval/{arm}_request_latency_{mean,p95,max}_s`,
  `eval/control_request_latency_{mean,p95,max}_s`,
  `eval/control_client_queue_latency_{mean,p95,max}_s`,
  `eval/control_service_latency_{mean,p95,max}_s`,
  `eval/control_{configured_,}max_concurrency`,
  `eval/control_bounded_concurrency_active`, and
  `eval/answer_logprob_group_latency_{mean,p95,max}_s`. After the
  corrupt-boundary fix, control rows also report
  `eval/control_corrupt_pause_mode_preserve_boundary_ws`,
  `eval/control_corrupt_pause_change_frac`,
  `eval/control_corrupt_pause_len_delta_chars`,
  `eval/control_corrupt_pause_leading_ws_match`, and
  `eval/control_corrupt_pause_trailing_ws_match`. After AM, the
  answer-logprob scorer also reports chunking/backpressure metrics:
  `eval/answer_logprob_{configured_,}batch_size`,
  `eval/answer_logprob_chunk_count`,
  `eval/answer_logprob_{configured_,}max_concurrency`,
  `eval/answer_logprob_bounded_concurrency_active`,
  `eval/answer_logprob_client_queue_latency_{mean,p95,max}_s`, and
  `eval/answer_logprob_service_latency_{mean,p95,max}_s`. With
  `eval_answer_logprob_distractor_control=true`, it also reports
  `eval/answer_logprob_select_margin_{pause,nopause,corrupt_pause}`,
  `eval/answer_logprob_select_delta`,
  `eval/answer_logprob_select_vs_corrupt_delta`, and
  `eval/answer_logprob_select_causal_delta` plus z/paired-n variants.
- It also reports corrupt-control no-op metrics:
  `opd_contrastive_corrupt_changed_tokens`,
  `opd_contrastive_corrupt_change_frac`, and
  `opd_contrastive_corrupt_noop_frac`.
- It can now optionally report teacher-memory pair diagnostics with
  `opd_teacher_memory_pair_diagnostics=true`. Use this to measure whether
  teacher cache rows at the supervised memory span are prompt-specific enough
  for a future cache-mismatch causal control.
- Configs AG-AN now set `eval_control_max_concurrency=16`. This keeps
  SGLang per-pod serialization unchanged, but prevents the OPD client from
  submitting the full control grid to SMG at once. AG-AL use
  `192 * 3 = 576` sampled-policy requests per control point; AM/AN use
  `1024 * 3 = 3072` sampled-policy requests at the final control point only.
- The client now supports `eval_control_start_step`. Config AM uses
  `eval_num_problems=1024` and `eval_control_start_step=5`; Config AN inherits
  the same final-only control schedule. This means the expensive
  `1024 * 3` held-out control grid runs only at the final control point.
- The client now supports chunked answer-logprob controls:
  `eval_answer_logprob_batch_size` and
  `eval_answer_logprob_max_concurrency`. Configs AM/AN now set batch size `64`
  and max scoring concurrency `2`; this is required because the pre-fix AM run
  sent one huge scoring batch per endpoint and received native SGLang HTTP 503s.
- The client now supports answer-distractor logprob controls:
  `eval_answer_logprob_distractor_control` and
  `eval_answer_logprob_distractor_offset`. Config AN enables this diagnostic.

Latest completed balanced cache-mismatch diagnostic smoke:

- Run dir:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T162901Z-configAL-er-opd-q36-35b-slots-trainer-head`
- W&B run:
  `fl7577vn`,
  `opd-q36-35b-randsymbol-contrastive64-answercontrast-rotmem-cachemisbal-nccl-stopnl-0shot-hm-configal`.
- Status:
  one requested step completed; trainer-head exited `rc=0`, workers stopped.
- Analyzer:
  `VERDICT: incomplete (no non-warmup control rows)`, expected for a one-step
  smoke.
- W&B/profile parity:
  `VERDICT: wandb_matches_profile`, W&B state `finished`.
- Infra result:
  bounded control fanout worked again. The row reports
  `eval/control_total_requests=576`,
  `eval/control_configured_max_concurrency=16`,
  `eval/control_max_concurrency=16`,
  `eval/control_bounded_concurrency_active=1`,
  and `eval/control_request_failure_frac_max=0`. SMG active connections stayed
  at the configured cap during control eval and drained cleanly before exit.
- Latency interpretation:
  total request latency was dominated by intentional client-side queueing:
  `eval/control_client_queue_latency_p95_s=313.90`, while actual sampler/SMG
  service latency stayed bounded at
  `eval/control_service_latency_p95_s=16.86`.
- Substrate result:
  quiescence and sync passed:
  `sampler_quiesce_success=1`, zero outstanding/connections/inflight before
  sync, `sync_endpoint_success_count=2`, `sync_endpoint_failure_count=0`,
  `sync_serial_endpoint_sync=1`, `sync_transfer_time_s=20.35`, and exact
  sampler routing balance (`640/640`, balance ratio `1.0`).
- New balance instrumentation worked:
  `opd_cache_mismatch_balance_positive_hidden=1`,
  `opd_cache_mismatch_positive_hidden_boost=0.25`,
  `opd_memory_hidden_weight_balance_per_token=0.0`,
  `opd_hidden_match_weight_mean=0.0`,
  `opd_hidden_match_pos_weight_mean=0.0531`, and
  `opd_hidden_match_neg_weight_mean=0.0531`.
- AL science result:
  still not a promotion, but materially better than AK at the warmup/control
  row. Exact-match controls were near neutral:
  `acc_pause=0.6979`, `acc_nopause=0.7135`,
  `acc_corrupt_pause=0.6771`, `buffer_delta=-0.0156` (`z=-0.336`), and
  `buffer_vs_corrupt_delta=+0.0208` (`z=0.4405`). Answer-logprob controls
  partially recovered versus AK:
  `answer_logprob_margin=+0.0095` (`z=1.53`), but corrupt comparison still
  failed:
  `answer_logprob_vs_corrupt_margin=-0.0111` (`z=-2.78`).
- Interpretation:
  AL supports the diagnosis that AK regressed because cache-mismatch added net
  negative hidden-match pressure on memory tokens without a matching positive
  real-memory hidden target. Balancing that pressure removed most of the exact
  match regression, but it did not establish a robust corrupt-control
  advantage. Treat AL as a diagnostic success and a weak recipe candidate, not
  as proof that cache mismatch is the right science direction.
- Recommended next step:
  run a short multi-step AL pilot only if the goal is to test whether the
  balanced objective improves after learning. Otherwise, prefer a new recipe
  that more directly pressures answer-causal use of memory, or a RiM-inspired
  structural intervention that changes which memory representation is trainable
  instead of adding more negative hidden-match variants.

Latest completed bounded-control smoke:

- Run dir:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T161321Z-configAK-er-opd-q36-35b-slots-trainer-head`
- W&B run:
  `9li9mwbu`.
- Status:
  one requested step completed; trainer-head exited `rc=0`, workers stopped.
- Analyzer:
  `VERDICT: incomplete (no non-warmup control rows)`, expected for a one-step
  smoke.
- W&B/profile parity:
  `VERDICT: wandb_matches_profile`, W&B state `finished`.
- Infra result:
  bounded control fanout worked. The row reports
  `eval/control_total_requests=576`,
  `eval/control_configured_max_concurrency=16`,
  `eval/control_max_concurrency=16`,
  `eval/control_bounded_concurrency_active=1`,
  and `eval/control_request_failure_frac_max=0`.
- Latency interpretation:
  total request latency is still high by design,
  `eval/control_request_latency_p95_s=326.86` and max `336.62`, but that is now
  almost entirely client queue time:
  `eval/control_client_queue_latency_p95_s=316.52` and max `321.10`. Actual
  sampler/SMG service latency was bounded:
  `eval/control_service_latency_p95_s=16.34` and max `16.57`. During the run,
  SMG active connections stayed at the configured cap (`16`) instead of the
  earlier several-hundred-request flood.
- Substrate result:
  quiescence passed immediately before sync
  (`sampler_quiesce_success=1`, `connections=0`, `inflight=0`), serial NCCL
  sync to two endpoints succeeded
  (`sync_endpoint_success_count=2`, `sync_endpoint_failure_count=0`,
  `sync_serial_endpoint_sync=1`, `sync_transfer_time_s=14.35`), and sampler
  routing stayed exactly balanced (`320/320`, balance ratio `1.0`).
- AK science result:
  reject/negative at step 0. Exact-match controls worsened under the real pause:
  `acc_pause=0.6458`, `acc_nopause=0.7448`,
  `acc_corrupt_pause=0.6719`, `buffer_delta=-0.0990` (`z=-2.12`), and
  `buffer_vs_corrupt_delta=-0.0260` (`z=-0.54`). Answer-logprob controls also
  failed the corrupt comparison:
  `answer_logprob_margin=+0.0012` (`z=0.19`) and
  `answer_logprob_vs_corrupt_margin=-0.0146` (`z=-4.12`).
- AK objective/diagnostic instrumentation worked:
  `opd_cache_mismatch_examples=64`,
  `opd_cache_mismatch_change_frac=1.0`,
  `opd_cache_mismatch_negative_answer_kl_weight=0.0`,
  `opd_teacher_memory_pair_diag_active=1`, and
  `opd_teacher_memory_pair_cross_minus_within_distance=0.0542`.
- Interpretation:
  the infra issue is fixable and now fixed at the client fanout layer for
  future science runs. AK's cache-mismatch negative did not create causal pause
  dependence; it appears to regularize the pause path in the wrong direction on
  the held-out exact-match control.

Latest completed diagnostic smoke:

- Run dir:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T153913Z-configAJ-er-opd-q36-35b-slots-trainer-head`
- W&B run:
  `yazdxxgu`.
- Status:
  one requested step completed; trainer-head exited `rc=0`, workers stopped.
- Analyzer:
  `VERDICT: incomplete (no non-warmup control rows)`, expected for a one-step
  smoke.
- W&B/profile parity:
  `VERDICT: wandb_matches_profile`, W&B state `finished`.
- New diagnostic validated:
  `opd_teacher_memory_pair_diag_active=1`,
  `opd_teacher_memory_pair_diag_failure=0`,
  `opd_teacher_memory_pair_sample_count=64`,
  `opd_teacher_memory_pair_cross_cosine_distance_mean=0.2501`,
  `opd_teacher_memory_pair_within_adjacent_distance_mean=0.1959`, and
  `opd_teacher_memory_pair_cross_minus_within_distance=0.0543`.
- Interpretation:
  the teacher slot hiddens are prompt-specific but weakly separated. This is
  enough to design a careful cache-mismatch diagnostic or memory-row objective,
  not enough to assume a strong causal memory identity.
- Operational substrate passed:
  `sync_endpoint_count=2`, `sync_endpoint_success_count=2`,
  `sync_endpoint_failure_count=0`, `sync_serial_endpoint_sync=1.0`,
  exact sampler balance `320/320`,
  `sampler_worker_success_balance_ratio=1.0`,
  `eval/control_request_latency_p95_s=333.63`, and
  `eval/control_request_latency_max_s=344.50`.

Latest completed instrumentation smoke:

- Run dir:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T151837Z-configAI-er-opd-q36-35b-slots-trainer-head`
- W&B run:
  `05o5f8h6`.
- Status:
  one requested step completed; trainer-head exited `rc=0`, workers stopped.
- Analyzer:
  `VERDICT: incomplete (no non-warmup control rows)`, expected for a one-step
  smoke.
- W&B/profile parity:
  `VERDICT: wandb_matches_profile`, W&B state `finished`.
- New instrumentation validated:
  `opd_contrastive_corrupt_change_frac=1.0`,
  `opd_contrastive_corrupt_noop_frac=0.0`,
  `eval/control_request_latency_p95_s=331.81`,
  `eval/control_request_latency_max_s=348.83`, and
  `eval/answer_logprob_group_latency_max_s=18.76`.
- Operational substrate passed:
  `sync_endpoint_count=2`, `sync_endpoint_success_count=2`,
  `sync_endpoint_failure_count=0`, `sync_serial_endpoint_sync=1.0`,
  exact sampler balance `320/320`, and
  `sampler_worker_success_balance_ratio=1.0`.

Infrastructure smoke:

- Run dir:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T084850Z-configT-er-opd-q36-35b-slots-trainer-head`
- W&B run:
  `k69dfbt5`, `opd-q36-35b-randsymbol-tiny-0shot-hm-configt`
- Initial registration synced both endpoints freshly:
  `sglang-0:30060` in `2.98s`, `teacher-sglang-1:30000` in `2.30s`.
- Post-step sync used serial endpoint mode and succeeded:
  `sync_endpoint_count=2`, `sync_endpoint_success_count=2`,
  `sync_endpoint_failure_count=0`, `sync_serial_endpoint_sync=1.0`,
  `sync_transfer_time_s=5.67`, `sync_total_bytes=138642442752`.
- Sampler-routing metrics were present and balanced:
  `sampler_router_requests_delta=296`, `sampler_worker_active_count=2`,
  `sampler_worker_0_success_delta=148`,
  `sampler_worker_1_success_delta=148`,
  `sampler_worker_success_balance_ratio=1.0`,
  `sampler_policy_round_robin_active=1.0`.
- The analyzer passed the strict sync+routing checks with relaxed science
  thresholds:
  `--expected-sync-endpoints 2 --require-serial-sync --require-sampler-routing`.
- W&B/profile parity passed for the default audit keys, including
  `sync_endpoint_*`, `sync_serial_endpoint_sync`, `sampler_router_requests_delta`,
  `sampler_worker_active_count`, `sampler_worker_success_balance_ratio`, and
  `sampler_policy_round_robin_active`.
- This was not a recipe promotion. Config T used `max_new_tokens=16`, so control
  cap hit was `1.0`; `eval/buffer_delta=-0.0521` and
  `eval/buffer_vs_corrupt_delta=-0.0104`.

Use the live status command for the authoritative current state:

```bash
python experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py status
```

## 13.7 Direct Answer: Why AI/AH Was Not Discarded

AH and AI should be treated as the first answer-causal mechanism signal, not as
dead ends. AH was rejected for operational reasons: only `96` controls and a
degenerate corrupt arm. AI cleaned up the substrate with strict two-endpoint
sync/routing and `192` controls, but exact-match was still underpowered:
`buffer_delta=+0.0938` with `z=1.9063`, while answer-logprob was already strong
at `z=16.6732` versus no-pause and `z=17.3105` versus corrupt.

The correct follow-up was to increase the final held-out control size to `1024`.
Config AM did that and confirmed the exact-match signal:
`acc_pause=0.6494`, `acc_nopause=0.5557`, `acc_corrupt_pause=0.5811`,
`buffer_delta=+0.0938` with `z=4.3548`, and
`buffer_vs_corrupt_delta=+0.0684` with `z=3.1871`. That means AI's exact-match
effect was underpowered at `n=192`, not absent.

The reason AM was not promoted directly is the corrupt-generation caveat. The
legacy corrupt arm rotated the fixed memory text but also changed assistant
boundary whitespace, which made the corrupt arm generate unusually long outputs
and hit the cap. `rotate_preserve_ws` fixes the visible boundary and removes
that cap-hit artifact, but it also weakens the free-generation pause-vs-corrupt
exact-match contrast. Therefore fixed-token order corruption is no longer a
sufficient causal gate by itself.

For current promotion decisions, require the pause buffer to improve answer
selection, not just absolute true-answer likelihood. The post-hoc AM
answer-selection distractor probe was negative even though absolute answer
logprob was positive, so the next real direction is prompt-specific corruption:
externalized prompt-specific memory shuffles or same-visible-input cache/hidden
target mismatch. Config AO tested one version of that hypothesis with
positive-answer KL plus balanced cache-mismatch hidden supervision and was
scientifically negative. Config AN is the active integrated rerun of the
AM/AH/AI answer-contrast direction with preserved-boundary corrupt eval,
chunked answer-logprob scoring, and answer-selection diagnostics.

## 14. Minimal Safe Iteration Checklist

For the next OPD recipe:

1. Decide the config letter/recipe in the generator. Prefer mechanism tests over
   more filler-surface sweeps.
2. Validate the generator:

```bash
uv run ruff check "$GENERATOR"
python -m py_compile "$GENERATOR"
```

3. Stop trainer slots:

```bash
python "$GENERATOR" stop-trainer-control --remove-run
```

4. Poll until trainer roles are stopped:

```bash
python "$GENERATOR" status
```

5. Launch trainer-only:

```bash
python "$GENERATOR" write-trainer-control --config <CONFIG> --num-steps 11 --prompts-per-step 64 \
  --sampler-replicas <N> --sampler-layout <dedicated|spare-teacher1>
```

6. Tail the latest trainer-head log until endpoint sync succeeds.
7. Watch `opd_profile.jsonl` at step 0, step 5, and step 10.
   With `--max-running-requests 1`, a `192` prompt, three-arm control eval
   sends `576` requests. Configs AM/AN/AO use `1024` prompts and therefore
   send `3072` sampled-policy requests, but only at `step >= 5` via
   `eval_control_start_step=5`.
   For AG-AN, the client caps control fanout with
   `eval_control_max_concurrency=16`, so long
   `eval/control_request_latency_*` should mostly show up as
   `eval/control_client_queue_latency_*`, while
   `eval/control_service_latency_*` should remain in the seconds-to-tens of
   seconds range. If service latency, request failures, or SMG aged inflight
   buckets grow, treat it as an infra regression. If only client queue latency
   dominates, increase serialized sampler replicas or use a smaller/fewer
   control eval schedule for early recipe triage.
   For live runs with a pending control row, prefer the monitor:

```bash
python experiments/opd_profile/monitor_live_opd.py "$PROFILE" \
  --wandb-run <run_id> --samples 2 --interval-s 15 \
  --native-log-pod <student-native-sglang-pod> \
  --native-log-pod <second-native-sglang-pod-if-used>
```

   `VERDICT: live_progress` means the dispatcher is still returning responses
   and no aged-inflight bucket is accumulating. `VERDICT: live_progress_native`
   means dispatcher traffic has drained but native SGLang scoring is active.
   `live_attention_*` means inspect SMG/SGLang logs before waiting longer.
8. After each control row, run:

```bash
python experiments/opd_profile/analyze_slot_profile.py "$PROFILE" --expected-sync-endpoints <N>
```

For `<N>=2`, add:

```bash
--require-serial-sync --require-sampler-routing \
  --min-sampler-active-workers 2 --min-sampler-balance-ratio 0.75
```

For final 1k answer-causal runs, use the stricter science gate:

```bash
--min-control-n 1024 \
--require-answer-select-control \
--min-answer-select-z 2 \
--min-answer-select-paired-n 1024 \
--min-sampler-balance-ratio 0.95
```

If `sampler_quiesce_enabled=1.0` is present in profile rows, the analyzer also
requires `sampler_quiesce_success=1.0` and zero quiescence outstanding,
connection, and inflight counts by default. Keep those defaults unless you are
debugging the barrier itself.

9. Stop/reprogram if sync fails, ranks die, artifacts spike, diagnostics are
   missing for a diagnostic-only run, or the buffer delta stays negative.

## Prefill-Time-Compute Autoresearch Loop

The current candidate queue and controller live under:

```bash
experiments/opd_profile/autoresearch/
```

Use the candidate-driven path for the next OPSD runs rather than adding another
lettered config by hand:

```bash
python experiments/opd_profile/autoresearch/controller.py next
python experiments/opd_profile/autoresearch/controller.py launch --id PTC-001 --dry-run
python experiments/opd_profile/autoresearch/controller.py launch --id PTC-001 \
  --sampler-replicas 2 --sampler-layout spare-teacher1
python experiments/opd_profile/autoresearch/controller.py monitor --profile latest --json
python experiments/opd_profile/autoresearch/controller.py score --profile latest --idea-id PTC-001
python experiments/opd_profile/autoresearch/controller.py advance --id PTC-001 --profile latest
```

The first queue is `PTC-001` through `PTC-004`: pure positive OPSD KL on
filler+answer, gold-answer replacement, filler hidden matching, then scale-up.
Detailed rationale and gates are in:

- `PREFILL_TIME_COMPUTE_OPSD_RESEARCH_MEMO_2026_06_03.md`
- `PREFILL_TIME_COMPUTE_AUTORESEARCH_SPEC_2026_06_03.md`
- `PREFILL_TIME_COMPUTE_OPSD_RUNBOOK_2026_06_03.md`
