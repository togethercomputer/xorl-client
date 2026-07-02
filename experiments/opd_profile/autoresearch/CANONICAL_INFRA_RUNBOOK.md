# OPD Slots Infra Handoff

Current as of 2026-06-15 23:01Z.

> ## ▶ 2026-07-02 UPDATE — hard-won gotchas from the 35B filler-GRPO A/B/C/D session
> Full record: `experiments/opd_profile/FILLER_HELPS_REASONING_35B_HANDOFF_20260702.md`. New stacks this
> session (parallel to `er-opd-q36-35b-slots`): `er-opd-q36-{megafiller(4-node),nofiller,restate,megaonly}`,
> each a full slots stack (trainer-head + sglang-0 + dispatch [+ trainer-worker-1/2/3 for the 4-node]).
> 1. **`PYTORCH_ALLOC_CONF=expandable_segments:True` BREAKS GPUDirect-RDMA MoE weight-sync.** The direct-EP
>    expert transfer registers GPU buffers for RDMA; expandable-segments hands out VMM (`cuMemMap`) virtual
>    ranges with no pinnable backing → `ibv_reg_mr` EFAULT "Bad address [14]" → sync hangs (dense layers are
>    fine — they stage through the host-pinned CPU pool). **Set the trainer `expandable_segments:False`**
>    (both `PYTORCH_ALLOC_CONF` + `PYTORCH_CUDA_ALLOC_CONF`; Wordle "DeepEP needs expandable OFF"; samplers
>    already `unset` it). This was the multi-node-sync blocker — NOT a topology/venv bug; do NOT disable
>    direct-EP. Memory: `expandable-segments-breaks-gpudirect-rdma-weightsync`; also `p2p.py:2482-2494`.
> 2. **NEVER edit a live slot's `run.sh` with an atomic-rename tool.** The slot agent polls
>    `sha256sum run.sh` every 2s under `set -euo pipefail`; a rename races it → `No such file or directory`
>    → agent exits → bare pod stuck `Error` (restartPolicy=Never, no self-heal). **Hash-bust with `>>`
>    append; content-edit with in-place truncate-write** (python `open('w')`), never rename/`sed -i`.
> 3. **Flaky "free" nodes.** 057/058/080 were 7-GPU or dead-CUDA (`nvidia-smi` OK but `torch.cuda.init()`
>    → "CUDA unknown error"). Add to pod `nodeAffinity` `kubernetes.io/hostname NotIn` blacklist.
> 4. **Recreate a dead bare slot pod** from its pristine spec:
>    `kubectl get pod X -o jsonpath='{.metadata.annotations.kubectl\.kubernetes\.io/last-applied-configuration}'`
>    (no nodeName pin) → delete + apply. The delete needs explicit user authorization (classifier blocks
>    slot-pod deletes).
> 5. **Generator `render-manifest` emits teacher pods** (2×8 GPU) — GRPO needs none. Filter docs whose
>    `metadata.name` contains `teacher` before `kubectl apply` (`--teacher-replicas` min is 1). Stand up a
>    1-node stack: `OPD_STACK=<name> python q36_35b_reprogrammable_slots.py --trainer-nodes 1 --eval-replicas 0
>    --teacher-replicas 1 render-manifest --sampler-replicas 1` → strip teachers + add node blacklist → apply
>    → write control run.sh (clone an existing stack's, adapt hostnames/`--nnodes 1`/config/args).
> 6. **Run watchdog** copied to `experiments/opd_profile/scripts/opd_run_watchdog.sh` (recomputes RUN_DIR
>    each poll; the old version false-hung on 1-node stacks by exec-ing a nonexistent `trainer-worker-1` and
>    by latching an empty RUN_DIR at arm time). Usage: `STACK=<stack> WD_MODE=gate|run bash opd_run_watchdog.sh`.
> 7. **Stop a run without deleting pods:** `touch /shared/opd-control/<stack>/<role>/stop` → slot agent kills
>    the child + writes `stopped_at` (separate file, no `run.sh` sha256sum race).

This file is the operational handoff for the `er-opd-q36-35b-slots` stack. The
old chronological infra ledger was archived at:

`/home/apanda/xorl-opd-prefill/experiments/opd_profile/autoresearch/CANONICAL_INFRA_HISTORY_20260615.md`

The throughput/science conclusions live in:

- `/home/apanda/xorl-opd-prefill/experiments/opd_profile/autoresearch/THROUGHPUT_MICROBENCH_RUNBOOK.md`
- `/shared/apanda/opd_throughput_science_channel.md`

## Current Live State

Kubernetes auth works.

The slots stack is not currently running trainer roles. Last checked state:

- `er-opd-q36-35b-slots-dispatch` is running on
  `research-common-h100-057.cloud.together.ai`.
- `er-opd-q36-35b-slots-teacher-smg` is running on
  `research-common-h100-052.cloud.together.ai`.
- No `er-opd-q36-35b-slots-trainer-*` pods are running.
- The separate science stack `er-opd-q36-35b-sci` was not restamped by this
  throughput/debugging work.

Check live state with:

```bash
kubectl get pods -n apanda -o wide | rg 'er-opd-q36-35b-slots-(trainer|dispatch|teacher)|NAME'
```

## Repos And Roots

Primary client/workflow checkout:

`/home/apanda/xorl-opd-prefill`

Engine diagnostics checkout:

`/home/apanda/xorl-opd-repeat2-diagnostics`

Engine branch:

`codex/opd-repeat2-diagnostics-20260615`

Diagnostics PR:

`https://github.com/togethercomputer/xorl-internal/pull/376` is closed as a
superseded diagnostics branch.

Clean replacement PRs:

- `https://github.com/togethercomputer/xorl-internal/pull/381` -
  lm-head TP checkpoint-load parameter sync.
- `https://github.com/togethercomputer/xorl-internal/pull/380` -
  replay checkpoint/prefetch/defrag knobs.

Merged upstream dependency for the larger-batch packer track:

- `https://github.com/togethercomputer/xorl-internal/pull/383` -
  dp-aware/best-fit server packing plus fail-loud oversized policy, merged into
  `apanda-dev` at `4f9f90cb2fcab48f0c15f14a975a178974e84e84`.

Infra checkout:

`/home/apanda/xorl-infra`

SGLang checkout used for reference/debugging:

`/home/apanda/xorl-sglang-internal`

Control root:

`/shared/opd-control/er-opd-q36-35b-slots`

Result root:

`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots`

## Stack Contract

Namespace:

`apanda`

Generator:

`/home/apanda/xorl-infra/k8s/opd_profile/q36_35b_reprogrammable_slots.py`

The generator supports slot restamping through environment-pointed repos. When
testing engine work from a sibling checkout, set the repo paths explicitly:

```bash
export OPD_XORL_CLIENT_REPO=/home/apanda/xorl-opd-prefill
export OPD_XORL_REPO=/home/apanda/xorl-opd-repeat2-diagnostics
export OPD_XORL_INFRA_REPO=/home/apanda/xorl-infra
export OPD_SGLANG_REPO=/home/apanda/xorl-sglang-internal
```

For the larger-batch packer track, do not point `OPD_XORL_REPO` at a dirty or
stale checkout unless it has been verified to contain PR #383:

```bash
rg -n "sample_packing_strategy|sample_packing_on_oversized|packing_microbench|datum_order" \
  "$OPD_XORL_REPO/src/xorl/server" \
  "$OPD_XORL_REPO/experiments/local_benchmark"
```

The slots stack is for benchmark/replay/debug work. The science agent uses a
separate stack and shared channel. Do not co-grab or restamp science roles from
this stack.

## Reprogrammable Slot Model

Each slot pod runs a small slot-agent that watches a generated `run.sh` under
the control root. Writing new control changes the hash and causes the child
process group to restart.

Useful commands:

```bash
CLIENT=/home/apanda/xorl-opd-prefill
INFRA=/home/apanda/xorl-infra
PY=/home/apanda/xorl-internal/.venv/bin/python
cd "$CLIENT"
"$PY" "$INFRA/k8s/opd_profile/q36_35b_reprogrammable_slots.py" \
  --model q36 --trainer-nodes 4 \
  status
```

```bash
CLIENT=/home/apanda/xorl-opd-prefill
INFRA=/home/apanda/xorl-infra
PY=/home/apanda/xorl-internal/.venv/bin/python
cd "$CLIENT"
"$PY" "$INFRA/k8s/opd_profile/q36_35b_reprogrammable_slots.py" \
  --model q36 --trainer-nodes 4 \
  stop-trainer-control --remove-run
```

After stopping trainer control, delete temporary trainer pods if they were
created for a short replay:

```bash
kubectl delete pod -n apanda \
  er-opd-q36-35b-slots-trainer-head \
  er-opd-q36-35b-slots-trainer-worker-1 \
  er-opd-q36-35b-slots-trainer-worker-2 \
  er-opd-q36-35b-slots-trainer-worker-3 \
  --ignore-not-found
```

## Node Placement

Use explicit NCCL compute selectors for trainer pods:

```yaml
nodeSelector:
  node-group: nccl
  node-pool: compute
```

The repeated scheduling failure mode was accidental default-node placement.
Always dry-run rendered manifests before applying:

```bash
kubectl apply --dry-run=server -n apanda -f /tmp/rendered.yaml
```

Trainer scale is quantized by topology. For this Qwen3.6 OPD stack, use 1, 2,
or 4 node trainer-only replays for screening and the intended 4-node gate. Do
not use 6 trainer nodes; the FSDP/EP mesh is invalid for this model shape.

## Current Throughput Knobs

The current science-facing 64-prompt tiers are summarized in the throughput
handoff. Operationally, the important resolved values are:

- MoE path for speed candidates: `moe_implementation: quack`.
- Dispatch path for speed candidates: `ep_dispatch: deepep`.
- Strict fresh speed candidates use `deepep_num_sms: 36`.
- 64-prompt shape uses `opd_prepare_batch_size: 64` and
  `opd_microbatch_size: 64`.
- AMDAHL-014 pipeline path uses `opd_pipeline_rl: true` and is one-step-stale.
- AMDAHL-018/020 are strict fresh-sample candidates but are not fully K3-gated.
- AMDAHL-008 remains the correctness fallback if science requires a gated
  default before a long run.

No current config is both strict-fresh, faster, and static/K3-passing.

## Larger-Batch Packer Track

The larger-batch path is now the next throughput track, but it is not a current
promotion. The detailed runbook is:

`/home/apanda/xorl-opd-prefill/experiments/opd_profile/autoresearch/LARGER_BATCH_PACKER_RUNBOOK.md`

Operational summary:

- The rejected AMDAHL-041 repeat ladder was fixed-payload repeat data on the
  current 64-prompt shape. It did not test a real larger optimizer-step window.
- The larger-batch track should start at `--prompts-per-step 256`, then test
  `512` only after fit and short throughput smokes.
- Server packing fields come from the trainer config file in
  `/home/apanda/xorl-infra/configs/opd_profile/`, not candidate `client_args`.
- Start from the current 4-node no-CP lm-head-TP config and make a pack16k copy
  with:

```yaml
sample_packing_sequence_len: 16384
enable_packing: true
sample_packing_strategy: balanced_dp
sample_packing_on_oversized: error
```

- Candidate YAML should set matching `default_prompts_per_step`,
  `opd_prepare_batch_size`, and `opd_microbatch_size`; it should not enable
  `opd_packed_row_batch_size` on the first larger-batch pass.
- Run `experiments/local_benchmark/packing_microbench.py` offline before any pod
  spend. Then do a 1-2 step fit smoke, a short throughput smoke, and the K3 gate.
- Required observations before any promotion: `dispatcher_dummy_batches == 0`,
  no oversized/dropped samples, same-workload throughput win, full-coverage K3
  pass, and science-owner acceptance of the changed effective batch.

## Trainer-Only Replay Harness

The benchmark work relied on short trainer-only server replays rather than full
science runs. This is the right harness for speed screening because it isolates
forward/backward, packing, loss, cache, and cleanup costs.

The current audited 4-node replay artifact is:

`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_dtypefix_4node_20260614.json`

The current best raw rowbatch 2-node replay artifact is:

`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-rowbatch2-77-ranklocal-stability12-chunk4-serveronly-20260615T091946Z.jsonl`

The rowbatch provenance artifact is:

`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-rowbatch2-81-provenance-real47-chunk4-serveronly-20260615T163845Z.jsonl`

For larger-batch packer work, the first harness is offline
`packing_microbench.py`, then a short trainer-control smoke. Do not create a
runnable AMDAHL candidate that only adds `sample_packing_strategy` under
`client_args`; that would not change server packing.

## K3 / Correctness Harness

Do not promote speed-only changes that touch packing, routing, model numerics,
loss, topology, or parallelism.

The current failed full static gate is:

`/shared/opd-control/er-opd-q36-35b-slots/k3/amdahl077_quack_deepep_sms36_20260615T1028Z/k3_gated_summary.json`

Summary:

- Status: `fail`.
- Coverage: `32/32` prompts, `2820/2820` tokens.
- Mean K3: `0.001628991`, above threshold `1e-3`.
- p95 K3: `0.003193`, below threshold `1e-2`.
- Max K3: `0.622587456`.

Engine branch `/home/apanda/xorl-opd-repeat2-diagnostics` contains default-off
diagnostic launch plumbing for the follow-up probes, including router policies,
router FP32 layer scopes, DeepEP diagnostic envs, SGLang route refresh, GDN
beta parity, and packed-row provenance.

## Operational Rules

- Treat `/shared/apanda/opd_throughput_science_channel.md` as the only
  worktree-independent science handoff.
- Update the science channel only when a config is promoted or when stack
  ownership changes.
- Record exact knobs in science-facing updates, not only AMDAHL names.
- Keep speed and correctness artifacts separate until a gate joins them.
- Do not spend a 4-node replay on a raw speed candidate after a static K3
  failure unless the model-path issue was fixed and re-gated.
- Do not add trainer nodes to hide low MFU. The current 4-node trainer is
  under-filled; adding nodes makes the utilization problem easier to miss.
- Do increase real `--prompts-per-step` on the dedicated larger-batch track
  after the PR #383 engine/config dependency is verified; that is different
  from repeating the fixed 64-prompt replay payload.
- Do not relaunch exact-R3 route replay, router tie-policy, GDN beta-rounding,
  or no-defrag/no-GC style probes as standalone promotions; they were already
  rejected or diagnostic-only.
- Do not restamp the separate science stack for larger-batch sizing, fit smokes,
  or offline packer checks. Coordinate science-facing batch-size changes through
  `/shared/apanda/opd_throughput_science_channel.md`.

## Cleanup Checklist

Before leaving the stack:

```bash
CLIENT=/home/apanda/xorl-opd-prefill
INFRA=/home/apanda/xorl-infra
PY=/home/apanda/xorl-internal/.venv/bin/python
cd "$CLIENT"
"$PY" "$INFRA/k8s/opd_profile/q36_35b_reprogrammable_slots.py" \
  --model q36 --trainer-nodes 4 \
  stop-trainer-control --remove-run

kubectl delete pod -n apanda \
  er-opd-q36-35b-slots-trainer-head \
  er-opd-q36-35b-slots-trainer-worker-1 \
  er-opd-q36-35b-slots-trainer-worker-2 \
  er-opd-q36-35b-slots-trainer-worker-3 \
  --ignore-not-found

kubectl get pods -n apanda -o wide | rg 'er-opd-q36-35b-slots-(trainer|dispatch|teacher)|NAME'
```

Expected idle state is dispatch + teacher-smg only.

## Historical Detail

For the full chronological record, including AMDAHL-048 through AMDAHL-081 and
intermediate failed launches, use:

`/home/apanda/xorl-opd-prefill/experiments/opd_profile/autoresearch/CANONICAL_INFRA_HISTORY_20260615.md`
