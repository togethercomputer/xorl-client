# Run B OPD Runbook — 2-node teacher + per-prompt CoT + pause student

Date: 2026-05-27

Critical-path instructions to deploy a working Run B OPD: Qwen3.6-35B-A3B self-distillation with per-prompt CoT teacher and pause-prefilled student. Validated 2-node teacher delivers 2.4× throughput vs 1-node (10.5k → 25.4k tok/s aggregate at b128 × concurrency=4) — projected step_total drops ~1540s → ~720s.

## Worktrees and branches

- xorl-internal: `/home/apanda/xorl-opd-mainline-run`, branch `codex/opd-mainline-run-20260526`
- xorl-client: `/home/apanda/xorl-client-chat-completions`, branch `feat/opd-per-prompt-cot` at `ce672fb` (pushed). Contains: per-prompt CoT loader, student pause-prefill splice (sgLang's chat completions doesn't echo the prefill in `input_token_ids` or `sampled.tokens`; we splice it in locally), `save_every` checkpointing.
- sgLang: `/home/apanda/xorl-sglang-internal` at `4ece4197`

## Datasets (already prepared)

- Filtered prompts: `/shared/opd-coord/randnum_4digit_filtered_8185.json` (8185 user-message prompts; 7 entries with empty CoT dropped from the original 8192)
- Per-prompt CoT: `/shared/opd-coord/randnum_4digit_filtered_8185_cot.json` (aligned 1:1, median 2048 tokens, max 2048; some CoTs truncated by the precompute's `max_tokens=2048`)
- DCP starting weights: `/shared/huggingface/Qwen3.6-35B-A3B-xorl-dcp-ep8-20260522`

## Required fixes baked into the manifest generator + configs

These were the gating bugs found during debugging. All are already applied to the files in this worktree; listed here so the next agent knows what NOT to revert.

1. **Teacher master's launcher MUST pass `--server.model_path` and `--server.tokenizer_path`** to the runner subprocess. Without them, the master uses the YAML's literal `"Qwen/Qwen3.6-35B-A3B"` (HF model name, not a path), takes the broadcast-resolve path in `src/xorl/models/module_utils.py:744` (`os.path.isdir` is False), while workers receive the resolved snapshot path via their own `--model_path` arg and skip the broadcast. Mismatched code paths deadlock at `_broadcast_object_list_weight_load` on the world pgroup. The trainer-head invocation has both; the teacher-master invocation MUST too.
2. **Teacher 2-node config requires `engine_connect_host: 127.0.0.1`** (`configs/qwen3_6_35b_a3b_teacher_2node.yaml`). Without it, `xorl.server.launcher._get_worker_address` waits 300s for a `.rank0_address` file that never appears (each pod's per-hostname RUN_DIR diverges). The 5-min delay before the master spawns torchrun lets the worker's 600s NCCL TCPStore timeout fire just after the master finally starts.
3. **Do NOT set `NCCL_IB_GID_INDEX` or `NCCL_IB_HCA=^mlx5_*` in the pod env.** Container `~/.bashrc` already sets the correct cluster defaults (`NCCL_IB_HCA=mlx5`, `NCCL_SOCKET_IFNAME=^lo,docker`, `NCCL_DEBUG=WARN`, `NCCL_NET_GDR_LEVEL=2`, `NCCL_VERSION=2.29...`, `AWS_OFI_NCCL_VERSION=1.17.0`). The `bashrc` overrides any pod env you set EXCEPT for `NCCL_IB_GID_INDEX` — that one variable survives and breaks the world NCCL pgroup. The stale memory note `feedback_nccl_ib_env_vars.md` calling these "REQUIRED" is wrong for this image.
4. **Bad nodes (truly)**: `h100-050` (stuck SM), `h100-113` (chronic IB failure). Everything else is transient (PVC mount race, SM-sweep race) and just needs a force-delete + reschedule. `h100-089` has a broken WEKA CSI driver — if a trainer pod lands there, force-delete it and reapply manifest to reschedule.

## Deploy

### 1. Tear down existing Run B (if any)

```bash
kubectl delete -f experiments/opd_profile/k8s/generated/er-opdb-052701.yaml --grace-period=5
# Wait until empty:
kubectl -n apanda get pods --no-headers | grep er-opdb-052701
```

### 2. Generate the manifest

```bash
python3 experiments/opd_profile/k8s/generate_opd_manifest.py \
  --run-name er-opdb-052701 \
  --num-trainer-nodes 8 \
  --num-sglang-pods 4 \
  --sampling-node research-common-h100-077 \
  --teacher-num-nodes 2 \
  --trainer-config experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node.yaml \
  --teacher-config experiments/opd_profile/configs/qwen3_6_35b_a3b_teacher_2node.yaml \
  --wandb-run-name opdb-cot-pause-052701 \
  --teacher-cot-json-path /shared/opd-coord/randnum_4digit_filtered_8185_cot.json \
  --student-prefill-text " pause" --student-prefill-count 100 \
  --save-every 50 --save-name-prefix opdb \
  --num-steps 400 --num-prompts 8185 \
  --opd-microbatch-size 128 --opd-prepare-batch-size 128 --opd-prepare-concurrency 4 \
  --output experiments/opd_profile/k8s/generated/er-opdb-052701.yaml
```

### 3. Preflight: verify the teacher-master invocation has the model_path args

The generator should already emit these — but if you regenerate after editing `generate_opd_manifest.py`, double-check:

```bash
grep -A1 "server.model_path\|server.tokenizer_path" experiments/opd_profile/k8s/generated/er-opdb-052701.yaml | head -8
```

Expected: at least two `--server.model_path "${MODEL_PATH}"` occurrences (trainer-head + teacher-master), same for `--server.tokenizer_path`.

### 4. Apply

```bash
kubectl apply -f experiments/opd_profile/k8s/generated/er-opdb-052701.yaml
```

### 5. Wait + troubleshoot

Expected pod set (11 total): 4 sgLang + 1 dispatch + 1 teacher-master + 1 teacher-worker-1 + 1 trainer-head + 7 trainer-worker-{1..7}.

```bash
watch -n 5 'kubectl -n apanda get pods -o wide --no-headers | grep er-opdb-052701'
```

If a pod stays `ContainerCreating` more than 2 min with WEKA CSI errors:
```bash
kubectl -n apanda describe pod <stuck-pod> | grep -E "FailedMount|csi.sock"
# If yes → force delete; reapply will recreate on a different node
kubectl -n apanda delete pod <stuck-pod> --grace-period=0 --force
kubectl apply -f experiments/opd_profile/k8s/generated/er-opdb-052701.yaml
```

Expected timeline:
- t=0: apply
- t+30s: all pods Running (or one needs reschedule)
- t+6 min: teacher-master + teacher-worker-1 log `✓ Engine Core fully initialized` (HF safetensors + DCP load across 2 nodes)
- t+10 min: trainer-head + workers all reach engine ready, OPD client starts, `=== OPD step 0 ===` logged
- t+25 min: step 0 (warmup) completes (~900s expected with 2-node teacher)
- Steady state: ~720s/step, 400 steps → ~3.3 days

### 6. Monitor

Profile rows land in:
```
experiments/encoded_reasoning/results/qwen3_30b_a3b_full_weight_real_reward/er-opdb-052701/<RUN_ID>-trainer-head-*/opd_profile.jsonl
```

W&B run: `https://wandb.ai/together-research/xorl-prefill-time-compute/runs/<id>`

Saves at steps 50, 100, ..., 400 to:
```
experiments/encoded_reasoning/results/qwen3_30b_a3b_full_weight_real_reward/er-opdb-052701/server_output/<save-name>
```

## Performance reference

Empirical numbers from the teacher-prefill microbench (`experiments/opd_profile/teacher_prefill_bench.py`), run against the same xorl teacher service at `b128 × c4` (Run B production setting):

| Teacher | Aggregate tok/s | Per-batch wall | step_total | 400 steps |
|---|---:|---:|---:|---:|
| 1-node | 10,546 | 100.6s | ~1540s | ~7.1 days |
| **2-node (validated)** | **25,440** | **41.7s** | **~720s** | **~3.3 days** |
| 4-node (projected, untested) | ~50,000 | ~21s | ~500-600s | ~2.6 days |

Bench artifacts: `experiments/opd_profile/results/teacher_bench/1n_v1/` and `2n_v1/`.

Teacher is the bottleneck up to ~3 nodes; at 4-node the trainer's own student fwd+bwd (~340-500s wall) becomes the new gating factor.

## Open items for further optimization

1. **4-node teacher**: not yet benched. `bench-teacher-4n.yaml` is generated but unapplied. Same `engine_connect_host: 127.0.0.1` + `--server.model_path` fix is in `qwen3_6_35b_a3b_teacher_4node.yaml` and `generate_teacher_bench.py`.
2. **Teacher `sample_packing_sequence_len`**: bumped to 16384 in `teacher_2node.yaml` (was 4096) to amortize per-microbatch overhead — 1n bench was run with the old 4096 packing, so the 25.4k tok/s 2n number may include some of the packing improvement. Worth attributing.
3. **FP8 P2P wire** (~17s → ~5s per sync): a separate agent has already wired this into `er-opdb-052701.yaml` via `XORL_WEIGHT_SYNC_QUANTIZATION` + FP8 sgLang model path. Small relative win (~12s of a 720s step).
4. **Loss recipe**: Run B's loss bounces 0.23-0.28 after 7 steps with the 1-node teacher. Not converging like Run A's smooth decay. May indicate the pause-prefilled student → CoT-conditioned teacher distillation gradient signal is weak. Out of scope for perf work but worth flagging.

## Quick reference: files modified for this recipe

- `experiments/opd_profile/configs/qwen3_6_35b_a3b_teacher_2node.yaml` — added `engine_connect_host: 127.0.0.1`, set `ep_dispatch: alltoall`, bumped `sample_packing_sequence_len` to 16384.
- `experiments/opd_profile/configs/qwen3_6_35b_a3b_teacher_4node.yaml` — new, mirrors 2node with `data_parallel_shard_size: 32`.
- `experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node.yaml` — Run B trainer config (identical to Run A's trainer; per-prompt CoT plumbing is entirely client-side).
- `experiments/opd_profile/k8s/generate_opd_manifest.py` — added CLI args for `--teacher-cot-json-path`, `--student-prefill-{text,count}`, `--save-every`, `--save-name-prefix`. Emission logic injects these into the trainer-head OPD client invocation. **The teacher-master launcher invocation in the underlying template still needs `--server.model_path "${MODEL_PATH}"` and `--server.tokenizer_path "${MODEL_PATH}"` — verify this in the generator's emitted output before applying.**
- `experiments/opd_profile/k8s/generate_teacher_bench.py` — generator for standalone bench-teacher manifests at 1n/2n/4n.
- `experiments/opd_profile/teacher_prefill_bench.py` — synthetic-input bench client.
- `experiments/opd_profile/run_teacher_bench_sweep.sh` — sweep harness.
- xorl-client `feat/opd-per-prompt-cot @ ce672fb`: `examples/on_policy_distillation.py` adds `teacher_cot_json_path` (per-sample CoT), `student_prefill_{text,count}` (pause prefill via `continue_final_message=true`), `save_every`/`save_name_prefix`. `xorl_client/types/sampling_params.py` + `xorl_client/client/sampling_client.py` plumb `chat_continue_final_message` to the chat completions payload.
