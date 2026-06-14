# OPD Profile Trials

This file records OPD throughput/profile trials and follow-up cleanup done on PR #209.
Keep rows for changes that should stay in the PR; mark failed or abandoned attempts as discarded.

## 2026-05-13 - Qwen3 235B Teacher to 30B Student

| Field | Value |
| --- | --- |
| Run ID | `20260513t021721z` |
| Branch head after cleanup | `3358b319` |
| Artifact | `/shared/opd-coord/20260513t021721z/artifacts/qwen3_235b_to_30b_8trainer_epfix_full_profile.jsonl` |
| Teacher prefill | Qwen3-235B-A22B, XORL service, grouped HF load, 16 H100s |
| Student sampler | Qwen3-30B-A3B, SGLang TP=4, 4 H100s |
| Student trainer | Qwen3-30B-A3B, XORL EP=8, 8 H100s |
| Shape | 8 prompts, 128 prompt tokens each, 64 sampled tokens each |
| Trainer options | `--opd-kl-backend streaming`, sharded teacher-head device cache, optimizer/sync skipped |
| Decision | Keep |

### Profile Results

`valid_tokens` is EP-duplicated in this profile. Unique distillation throughput uses
`teacher_prefill_tokens=1528` per step.

| Step | Student Sampling | Teacher Prefill Wall | Teacher Forward Compute | Trainer Fwd/Bwd | OPD KL | Step Total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 warmup | 5.11s, 100.2 output tok/s | 86.39s | 0.44s | 30.20s | 335.8 ms | 121.75s |
| 1 | 5.17s, 99.0 output tok/s | 85.37s | 0.44s | 3.37s | 124.7 ms | 93.94s |
| 2 | 5.18s, 98.9 output tok/s | 0.47s | 0.41s | 3.37s | 124.4 ms | 9.04s |

Steady mean over steps 1-2:

| Metric | Value |
| --- | ---: |
| Step total | 51.49s |
| Student sampling | 5.17s |
| Teacher prefill wall | 42.92s |
| Teacher forward compute | ~0.42s |
| Trainer forward/backward | 3.37s |
| OPD KL compute | ~125 ms |
| Unique throughput, all 28 GPUs | 1.06 tok/s/GPU |
| Unique throughput, trainer 8 GPUs only | 3.71 tok/s/GPU |
| Fast step 2 throughput, all 28 GPUs | 6.04 tok/s/GPU |
| Fast step 2 throughput, trainer 8 GPUs only | 21.14 tok/s/GPU |

### Result

- Kept the XORL teacher-prefill path, batched sampling, JSONL profile output, sharded teacher-head streaming backend, and EP-aware trainer dispatch fix.
- Kept the manifest helper script, but dropped cluster-specific K8s manifests from the repo; operators should provide manifests via `OPD_MANIFEST_DIR`.
- Discarded no code changes from this trial.
- Main inefficiency observed: teacher-prefill request wall time is highly variable even though reported teacher forward compute is sub-second for the same batch.
- OPD KL is not the bottleneck for this shape; the fused TileLang/native KL backend remains a separate follow-up tracked in #273.

### Validation

| Check | Result |
| --- | --- |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src ruff format --check ...` | Pass |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src ruff check ...` | Pass |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src python -m py_compile ...` | Pass |
| `git diff --check` | Pass |
| `SKIP=no-commit-to-branch pre-commit run --all-files --show-diff-on-failure` | Pass |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src pytest -q tests/test_example_assets.py` | 1 passed |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src pytest -q tests/ops/loss/test_opd_loss.py tests/utils/test_distillation_teacher_cache.py tests/models/test_module_utils_broadcast.py tests/models/test_qwen3_5_registry.py tests/server/runner/test_opd_runner.py tests/server/runner/test_runner_dispatcher.py tests/scripts/test_opd_pipeline_payloads.py` | 51 passed |
| GitHub lint | Pass |
| GitHub CPU tests | Pass |
| GitHub GPU tests | Pending at time of log entry |

## 2026-05-13 - Qwen3.5 397B Teacher to 35B-A3B Student With Weight Sync

| Field | Value |
| --- | --- |
| Run ID | `20260513t192405zsync` |
| Teacher prefill | Qwen3.5-397B-A17B, XORL service, 2 nodes x 8 H100 (`049`, `048`) |
| Student sampler | Qwen3.5-35B-A3B, SGLang TP=4, 4 H100 |
| Student trainer | Qwen3.5-35B-A3B, XORL EP=8/Ulysses=8, 8 H100 (`083`) |
| Shape | 2 prompts, 32 prompt tokens each, 8 sampled tokens each |
| Trainer options | Muon optimizer, `--opd-kl-backend streaming`, sharded teacher-head device cache, optimizer/sync enabled |
| Decision | Keep code/config fixes; discard the failed runtime attempts |

### Attempts

| Attempt | Result | Decision |
| --- | --- | --- |
| `20260513t192058zsync` | Teacher worker failed because the checked-in venv `torchrun` wrapper had a host-path shebang. | Keep `scripts/run_multinode_server.sh` change to launch `python -m torch.distributed.run`. |
| `20260513t192219zsync` | Trainer/teacher head failed because `src/xorl/server/launcher.py` also launched the venv `torchrun` wrapper. | Keep launcher change to use `sys.executable -m torch.distributed.run`. |
| `20260513t192405zsync`, P2P on base SGLang checkout | First failed because `mooncake-transfer-engine` was missing; after installing it, SGLang `/prepare_weights_update` returned 400 because the base checkout still used the NCCL prepare path. | Keep the dependency note and switch the student manifest to the SGLang `weight-sync-refactor` worktree. |
| `20260513t192405zsync`, NCCL fallback | Got through student sample, teacher prefill, trainer forward/backward, optimizer, NCCL group init, and several weight-update calls. Full sync failed after 633.6s on a SGLang `/update_weights_from_distributed` HTTP timeout. | Discard as an operational path for this topology; NCCL full sync is too slow here. |
| `20260513t192405zsync`, P2P on SGLang `weight-sync-refactor` worktree | P2P prepare succeeded with `31,860` HF names and `250,000` locators across 4 receivers. Streaming then failed after the root bucket because Qwen3.5 linear-attention tensors had trainer/receiver shape mismatches (`A_log`/`dt_bias`: `[32]` vs `[8]`, `conv1d.weight`: `[8192,1,4]` vs `[8192,4]`) and Mooncake reported bad-address registration/segment descriptor errors. Driver hung waiting on the sync future after partial streaming, so the run was killed. | Keep the manifest path to the matching SGLang worktree; add Qwen3.5 TP/linear-attention shape handling before retrying P2P. |
| `20260513t223815zp2pfix`, P2P after merging `origin/weight-sync-refactor` (`1c31019d`, OPD merge `62fc009a`) | P2P prepare again succeeded with `31,860` HF names and `250,000` locators across 4 receivers, and the previous Qwen3.5 linear-attention shape mismatches did not recur. Streaming failed after it began: receiver Mooncake logged repeated `RdmaTransport: Failed to register memory` for CUDA buffers, and the trainer logged repeated `Failed to get segment descriptor` for receiver segment `192.168.229.87:16868`. No profile row was written. | Keep the merge/fix; discard the failed runtime attempt. The next blocker is Mooncake CUDA memory registration/segment descriptor handling, not Qwen3.5 tensor slicing. |
| `20260513t234457zp2preg`, P2P after requiring receiver memory handles for sender coalescing (`ac22c519`) plus uncommitted SGLang receiver registration patch | P2P prepare returned HTTP 200, but receiver Mooncake still logged `2,196` `RdmaTransport: Failed to register memory` warnings while registering CUDA allocator regions. Sender streaming then failed with `2,935` `Failed to get segment descriptor` messages for receiver segment `192.168.229.87:16037`, including coalesced 32 MiB MoE buckets whose debug entries all shared the same registered-base handle. One profile row was flushed after shutdown. | Keep the sender coalescing guard and its unit test. Discard the runtime attempt. Receiver registration needs another pass, likely either exact locator-sized registrations or a reliable way to detect Mooncake per-region registration failures before returning 200 from prepare. |
| `20260514t001429zlocreg`, P2P with strict locator receiver registration (`XORL_P2P_RECEIVER_STRICT_REGISTER=1`, `XORL_P2P_RECEIVER_REGISTER_MODE=locator`) | SGLang attempted exact locator registration for `31,780` memory regions per TP rank and returned HTTP 400 from `/prepare_weights_update` after Mooncake returned `-202` (`Bad address`). The trainer failed `sync_inference_weights` in 2.1s before streaming any buckets, so there were no partial receiver updates and no segment-descriptor storm. | Keep the trial record; discard the failed runtime attempt. Strict prepare failure is better than partial streaming, but locator-sized CUDA registration still does not work for this SGLang/Mooncake path. |
| `20260514t003133zallocreg`, strict allocator control with incorrectly concatenated manifest | The rendered YAML missed explicit document separators between manifest files, so `kubectl apply` created only teacher node0 and the trainer; teacher node1 and the student job were overwritten by later YAML documents. | Discard the runtime attempt. Keep no code changes; future ad hoc render commands must insert `---` between manifest files. |
| `20260514t003206zallocreg`, P2P with strict allocator receiver registration (`XORL_P2P_RECEIVER_STRICT_REGISTER=1`, default allocator mode) | SGLang attempted allocator-block registration for `906` CUDA allocator regions per TP rank and returned HTTP 400 from `/prepare_weights_update` after Mooncake returned `-202` (`Bad address`). The trainer failed `sync_inference_weights` in 2.2s before streaming any buckets. | Keep the trial record; discard the failed runtime attempt. Since both allocator-block and exact-locator registrations fail, the blocker is not locator granularity. |
| `20260514t010209zallocloc`, P2P with SGLang location-aware Mooncake receiver registration | SGLang attempted strict allocator-block registration for `906` CUDA allocator regions per TP rank and logged `location=cuda:0..3` with the matching `cuda_device`. Mooncake still returned `-202` (`Bad address`), `/prepare_weights_update` returned HTTP 400, and the trainer failed `sync_inference_weights` in 2.2s before streaming any buckets. | Keep the trial record; discard the failed runtime attempt. Passing `cuda:<gpu_id>` and entering the matching CUDA device context did not resolve receiver registration. |
| `20260514t013854zsegreg`, P2P with SGLang full CUDA allocator segment receiver registration | The live student pod was verified to run from `/workspace/xorl-sglang-internal/.claude/worktrees/weight-sync-refactor` with `/workspace/xorl-sglang-internal/.venv/bin/python`, `XORL_P2P_RECEIVER_STRICT_REGISTER=1`, `XORL_P2P_RECEIVER_REGISTER_MODE=allocator`, and the patched `p2p_segment_regions_from_memory_snapshot` helper importable from the mounted worktree. SGLang reduced receiver registration from `906` allocator blocks to `3` CUDA allocator segment regions per TP rank and logged `location=cuda:0..3`, but Mooncake still returned `-202` (`Bad address`) for the first segment base `0x602000000`; `/prepare_weights_update` returned HTTP 400 and the trainer failed `sync_inference_weights` in 2.2s before streaming any buckets. | Keep the trial record; discard the failed runtime attempt. Full CUDA allocator segment registration did change the region selection substantially, but it did not resolve Mooncake receiver registration, so the next check should be a direct receiver-side Mooncake CUDA registration smoke test in the pod or a closer look at Mooncake's expected GPUDirect registration API/semantics. |
| `20260514t015748zmcsmoke` / `20260514t015907zmcsmokelarge` / `20260514t015957zmcthresh` / `20260514t020035zmcsmall`, direct Mooncake receiver registration smoke with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | The smoke used the same SGLang `weight-sync-refactor` worktree, venv, host networking, IB mount, and Mooncake wrapper as the student receiver. Small registrations worked, but larger expandable CUDA allocator ranges failed with Mooncake `-202`: 1/2/4/8 MiB tensor registrations worked, a 20 MiB allocator segment worked, but 16/32/48/64 MiB tensor registrations and larger allocator segments failed. 1 GiB and 8 GiB tensor/segment registrations also failed. | Keep the reusable smoke manifest. This shows Mooncake CUDA registration works in the pod, but PyTorch expandable CUDA allocator ranges become non-registerable once they cross backing allocation boundaries. |
| `20260514t020158zmcnoexpand` / `05140202mcnoexplg`, direct Mooncake receiver registration smoke with `PYTORCH_CUDA_ALLOC_CONF` empty | With expandable CUDA segments disabled in the same pod shape, Mooncake successfully registered 1/2/4/8/16/32/48/64 MiB tensors and allocator segments, and also successfully registered 1 GiB and 8 GiB tensors and allocator segments with both auto location and `location=cuda:0`. | Keep the student manifest change that removes `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` from the SGLang RDMA receiver. |
| `20260514t020646znoexppin`, P2P with SGLang no-expand CUDA allocator and pinned node placement | Student was pinned to 4-GPU node `110`, trainer to 8-GPU node `094`, and teachers to `049`/`048`. SGLang no longer set `PYTORCH_CUDA_ALLOC_CONF`; receiver prepare registered `190` CUDA allocator segment regions per TP rank and returned HTTP 200 with no Mooncake `-202` errors. Sync then failed after streaming began because the receiver tensor map was incomplete/incompatible: `model.layers.0.linear_attn.A_log` sliced to `source=16` while the receiver expected `32`, and `model.layers.0.linear_attn.in_proj_a.weight` had no receiver locator. Direct-EP expert streaming had already begun. | Keep the no-expand receiver manifest fix and the smoke manifest. Discard the failed runtime attempt. The current blocker has moved from Mooncake registration to Qwen3.5 linear-attention tensor-map coverage/slicing and pre-stream validation. |
| `05141951p2pattn`, P2P with SGLang no-expand allocator plus Qwen3.5 full-attention HF tensor-map fix | Student was pinned to node `052`, trainer to `092`, and teachers to `083`/`094`. SGLang ran from the `weight-sync-refactor` worktree with the new full-attention HF name mapping. Two OPD iterations completed, both `sync_inference_weights` calls returned HTTP 200 with `success=True`, and post-sync greedy samples changed on the student. The prior missing receiver locator failures for `model.layers.3.self_attn.*` did not recur. Mooncake still logged repeated RDMA QP handshake errors on some mlx5 endpoints, but the transfer completed successfully. | Keep the runtime evidence and the SGLang receiver tensor-map fix. This is the first successful Qwen3.5 397B teacher to 35B-A3B student OPD run with optimizer and P2P weight sync enabled. |

### Measured Phases

NCCL fallback wrote one JSONL row before failing sync:

| Metric | Value |
| --- | ---: |
| Student sampling | 0.478s for 16 new tokens, 33.5 tok/s |
| Teacher prefill | 0.513s for 78 tokens, 152.1 tok/s |
| Trainer forward/backward | 126.6s roundtrip |
| Trainer forward loss | 74.3s |
| Trainer backward | 53.9s |
| OPD KL compute | 162.6 ms |
| Optimizer step | 177.1s |
| NCCL sync attempt | 633.6s, failed timeout |
| Step total to failure | 938.3s |

P2P with the SGLang weight-sync worktree did not write a profile row because it hung after partial streaming, but the log has these phase timings:

| Metric | Value |
| --- | ---: |
| Student sampling | 3.43s for 16 new tokens, 4.7 tok/s |
| Teacher prefill | 0.520s for 78 tokens, 150.1 tok/s |
| Trainer forward/backward | 127.8s roundtrip |
| Trainer forward loss | 75.5s |
| Trainer backward | 53.9s |
| OPD KL compute | 76.0 ms |
| Optimizer step | 171.2s |
| P2P prepare | ~16s |
| P2P stream | Failed immediately after root bucket on incompatible Qwen3.5 linear-attention receiver locators |

P2P retry after the Qwen3.5 slicing fix also did not write a profile row because streaming failed after partial receiver updates:

| Metric | Value |
| --- | ---: |
| Run placement | teacher nodes `052`/`048`, student node `087`, trainer node `083` |
| Student sampling | 3.459s for 16 new tokens, 4.6 tok/s |
| Teacher prefill | 70.802s for 78 tokens, 1.1 tok/s |
| Trainer forward/backward | 126.233s roundtrip |
| Trainer forward loss | 72.551s |
| Trainer backward | 55.143s |
| OPD KL compute | 132.0 ms |
| Optimizer step | 151.098s |
| P2P prepare | ~16s, succeeded |
| P2P stream | Failed after streaming began on Mooncake receiver CUDA memory registration / segment descriptor lookup |

P2P retry after the sender coalescing guard and SGLang receiver registration patch wrote one JSONL row after shutdown:

| Metric | Value |
| --- | ---: |
| Run placement | teacher nodes `049`/`048`, student node `087`, trainer node `092` |
| Student sampling | 3.576s for 16 new tokens, 4.5 tok/s |
| Teacher prefill | 69.155s for 78 tokens, 1.1 tok/s |
| Teacher forward compute | 69.060s |
| Trainer forward/backward | 126.440s roundtrip |
| Trainer forward loss | 74.164s |
| Trainer backward | 53.825s |
| OPD KL compute | 71.9 ms |
| Optimizer step | 176.346s |
| P2P sync attempt | 221.519s, failed |
| Step total to failure | 597.037s |

P2P retry with strict locator receiver registration wrote one JSONL row and failed before streaming:

| Metric | Value |
| --- | ---: |
| Run placement | teacher nodes `092`/`048`, student node `067`, trainer node `049` |
| Student sampling | 3.436s for 16 new tokens, 4.7 tok/s |
| Teacher prefill | 70.135s for 78 tokens, 1.1 tok/s |
| Teacher forward compute | 70.041s |
| Teacher cache write | 0.019s |
| Trainer forward/backward | 124.765s roundtrip |
| Trainer forward loss | 72.331s |
| Trainer backward | 52.708s |
| OPD KL compute | 77.5 ms |
| Optimizer step | 143.186s |
| P2P sync attempt | 2.127s, failed during receiver prepare |
| Step total to failure | 343.650s |

P2P retry with strict allocator receiver registration wrote one JSONL row and failed before streaming:

| Metric | Value |
| --- | ---: |
| Run placement | teacher nodes `096`/`048`, student node `110`, trainer node `083` |
| Student sampling | 3.456s for 16 new tokens, 4.6 tok/s |
| Teacher prefill | 68.752s for 78 tokens, 1.1 tok/s |
| Teacher forward compute | 68.674s |
| Teacher cache write | 0.013s |
| Trainer forward/backward | 127.222s roundtrip |
| Trainer forward loss | 74.719s |
| Trainer backward | 53.956s |
| OPD KL compute | 150.9 ms |
| Optimizer step | 172.460s |
| P2P sync attempt | 2.165s, failed during receiver prepare |
| Step total to failure | 374.056s |

P2P retry with SGLang location-aware Mooncake registration wrote one JSONL row and failed before streaming:

| Metric | Value |
| --- | ---: |
| Run placement | teacher nodes `096`/`048`, student node `094`, trainer node `083` |
| Student sampling | 3.447s for 16 new tokens, 4.6 tok/s |
| Teacher prefill | 69.896s for 78 tokens, 1.1 tok/s |
| Teacher forward compute | 69.814s |
| Teacher cache write | 0.012s |
| Trainer forward/backward | 129.012s roundtrip |
| Trainer forward loss | 74.625s |
| Trainer backward | 55.908s |
| OPD KL compute | 77.3 ms |
| Optimizer step | 163.717s |
| P2P sync attempt | 2.158s, failed during receiver prepare |
| Step total to failure | 368.230s |
| Receiver registration | `906` allocator regions per TP rank, `location=cuda:0..3`, Mooncake `-202` |

P2P retry with SGLang full CUDA allocator segment registration wrote one JSONL row and failed before streaming:

| Metric | Value |
| --- | ---: |
| Run placement | teacher nodes `049`/`048`, student node `110`, trainer node `094` |
| Student sampling | 3.470s for 16 new tokens, 4.6 tok/s |
| Teacher prefill | 71.160s for 78 tokens, 1.1 tok/s |
| Teacher forward compute | 71.070s |
| Teacher cache write | 0.014s |
| Trainer forward/backward | 129.342s roundtrip |
| Trainer forward loss | 74.960s |
| Trainer backward | 56.181s |
| OPD KL compute | 155.5 ms |
| Optimizer step | 177.121s |
| P2P sync attempt | 2.157s, failed during receiver prepare |
| Step total to failure | 383.251s |
| Receiver registration | `3` CUDA allocator segment regions per TP rank, `location=cuda:0..3`, Mooncake `-202` on segment base `0x602000000` |

Direct Mooncake receiver registration smokes:

| Run | Allocator Env | Result |
| --- | --- | --- |
| `20260514t015748zmcsmoke` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 2 KiB tensor, 2 MiB tensor, and 20 MiB allocator segment registered successfully. |
| `20260514t015907zmcsmokelarge` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 1 MiB registered successfully; 1 GiB and 8 GiB tensor/segment registrations failed with Mooncake `-202`. |
| `20260514t015957zmcthresh` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 1 MiB registered successfully; 64/128/256/512/768/1024 MiB registrations failed with Mooncake `-202`. |
| `20260514t020035zmcsmall` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 1/2/4/8 MiB tensors and the initial 20 MiB allocator segment registered successfully; 16/32/48/64 MiB tensor registrations and expanded allocator segments failed with Mooncake `-202`. |
| `20260514t020158zmcnoexpand` | empty | 1/2/4/8/16/32/48/64 MiB tensors and allocator segments registered successfully. |
| `05140202mcnoexplg` | empty | 1 GiB and 8 GiB tensors and allocator segments registered successfully. |

P2P retry with SGLang no-expand CUDA allocator wrote one JSONL row and failed after receiver prepare succeeded:

| Metric | Value |
| --- | ---: |
| Run placement | teacher nodes `049`/`048`, student node `110`, trainer node `094` |
| Student sampling | 3.450s for 16 new tokens, 4.6 tok/s |
| Teacher prefill | 70.323s for 78 tokens, 1.1 tok/s |
| Teacher forward compute | 70.213s |
| Teacher cache write | 0.015s |
| Trainer forward/backward | 125.493s roundtrip |
| Trainer forward loss | 73.631s |
| Trainer backward | 53.534s |
| OPD KL compute | 156.3 ms |
| Optimizer step | 160.202s |
| P2P sync attempt | 21.611s, failed after receiver prepare and after Direct-EP streaming began |
| Step total to failure | 381.078s |
| Receiver registration | `190` CUDA allocator segment regions per TP rank, `location=cuda:0..3`, `/prepare_weights_update` HTTP 200, no Mooncake `-202` |
| Sync failure | Tensor-map incompatibility: `linear_attn.A_log` source slice `16` vs receiver `32`; missing receiver locator for `linear_attn.in_proj_a.weight` |

P2P retry with the Qwen3.5 full-attention HF tensor-map fix completed two OPD steps:

| Metric | Step 0 Cold | Step 1 Warm |
| --- | ---: | ---: |
| Run ID | `05141951p2pattn` | `05141951p2pattn` |
| Run placement | teacher nodes `083`/`094`, student node `052`, trainer node `092` | same |
| Student sampling | 3.427s for 16 new tokens, 4.7 tok/s | 0.470s for 16 new tokens, 34.0 tok/s |
| Teacher prefill | 73.372s for 78 tokens, 1.1 tok/s | 0.592s for 78 tokens, 131.7 tok/s |
| Teacher forward compute | 73.266s | 0.560s |
| Teacher cache write | 0.022s | 0.008s |
| Trainer forward/backward | 126.303s roundtrip | 5.298s roundtrip |
| Trainer forward loss | 74.398s | 1.094s |
| Trainer backward | 53.734s | 2.707s |
| OPD KL compute | 180.1 ms | 9.7 ms |
| Optimizer step | 179.177s | 3.769s |
| P2P sync | 44.988s total, 26.418s transfer | 23.834s total, 23.498s transfer |
| P2P backend init | 18.352s | 0.292s |
| P2P bytes/buckets | 13.97 GB, 83 buckets | 13.97 GB, 83 buckets |
| Post-sync verify sample | 2.103s, changed output confirmed | 0.476s, changed output confirmed |
| Step total | 429.372s | 34.443s |
| Unique throughput, all 28 GPUs | 0.006 tok/s/GPU | 0.081 tok/s/GPU |
| Unique throughput, trainer 8 GPUs only | 0.023 tok/s/GPU | 0.283 tok/s/GPU |
| Sync result | HTTP 200, `success=True` | HTTP 200, `success=True` |

Notes:

- The cold step is dominated by first-use costs: SGLang and XORL prefill warmup, trainer forward/backward warmup, Muon initialization, and P2P backend initialization.
- The warm step is dominated by full-model P2P sync for this tiny 78-token profile shape; KL compute is still not the limiter.
- Mooncake logged repeated `Failed to modify QP to RTR` handshake errors on some mlx5 endpoints during P2P, but the run completed and verified student outputs changed after both syncs.

### Result

- PR #212 was merged into the OPD branch and the OPD smoke now exercises the optimizer and `sync_inference_weights` path rather than stopping after forward/backward.
- `origin/weight-sync-refactor` commit `1c31019d` was merged into the OPD branch as `62fc009a`; the retry confirmed the Qwen3.5 linear-attention slicing fix reaches P2P prepare and streaming.
- Added the XORL sender-side guard that only coalesces adjacent P2P writes when the receiver `memory_handle` is non-null and identical, with regression coverage for missing handles.
- Kept the launcher fixes required for pods whose mounted worktree path differs from the venv creation path.
- Kept the runtime evidence for the trainer manifest fix removing unsupported `lr` from the server config.
- Kept the runtime evidence for running SGLang from `/workspace/xorl-sglang-internal/.claude/worktrees/weight-sync-refactor` while using `/workspace/xorl-sglang-internal/.venv/bin/python`.
- Kept the runtime evidence for disabling PyTorch expandable CUDA allocator segments on the SGLang RDMA receiver and enabling strict allocator-mode P2P receiver registration.
- Dropped the reusable Mooncake CUDA registration smoke manifest from the repo with the other cluster-specific manifests.
- Cleaned up all K8s resources for `20260513t192405zsync`.
- Cleaned up K8s resources for `20260513t223815zp2pfix`.
- Cleaned up all K8s resources for `20260514t001429zlocreg`.
- Cleaned up all K8s resources for `20260514t003133zallocreg` and `20260514t003206zallocreg`.
- Cleaned up all K8s resources for `20260514t010209zallocloc`.
- Cleaned up all K8s resources for `20260514t013854zsegreg`.
- Cleaned up all K8s resources for Mooncake smoke runs `20260514t015748zmcsmoke`, `20260514t015907zmcsmokelarge`, `20260514t015957zmcthresh`, `20260514t020035zmcsmall`, `20260514t020158zmcnoexpand`, and `05140202mcnoexplg`.
- Cleaned up all K8s resources for `20260514t020416znoexp` and `20260514t020646znoexppin`.

### Follow-Ups

- Treat `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` as incompatible with the SGLang Mooncake RDMA weight receiver. Direct smokes show expandable CUDA ranges fail with Mooncake `-202`, while the same sizes succeed when expandable segments are disabled.
- Keep strict receiver registration enabled for P2P trials so registration failures return from prepare instead of causing partial streaming; keep allocator-segment mode as the default for the no-expand SGLang receiver.
- Fix Qwen3.5 linear-attention P2P tensor-map compatibility after no-expand receiver registration: `A_log` slicing is currently reversed (`source=16`, receiver expects `32`) and `linear_attn.in_proj_a.weight` is missing from the receiver locator map.
- Move tensor-map compatibility validation before any RDMA streaming or Direct-EP expert streaming; the no-expand run failed after Direct-EP had already begun, leaving the receiver potentially partially updated.
- Add a guard or timeout so a failed P2P transfer after partial streaming returns promptly instead of leaving the driver waiting on the sync future.
- Treat NCCL broadcast as a correctness fallback only; for this 35B-A3B MoE setup it did not complete a single full sync within the current per-request timeout.

### Additional 2026-05-14 P2P Retries

These retries used the SGLang `weight-sync-refactor` worktree and the OPD branch after the
Qwen3.5 linear-attention sender slicing and dtype fix.

| Run | Change | Result | Decision |
| --- | --- | --- | --- |
| `05140255p2pfix3` | SGLang receiver with no-expand allocator segments plus Qwen3.5 shape fix before the sender dtype guard. | Receiver prepare succeeded with `190` CUDA allocator segment regions per TP rank and no Mooncake `-202`, but sync failed before streaming on `model.layers.0.linear_attn.A_log`: source `16` bytes after slicing, receiver expected `32`. | Keep the receiver/runtime evidence. Keep the follow-up sender dtype-cast guard that fixed this mismatch. |
| `05140313p2pdtype` | First retry after the sender dtype guard. | Scheduling-only failure because teacher node `112` was `Ready=Unknown`/tainted. | Discard runtime attempt. |
| `05140313p2pdtype2` | Sender dtype guard active. | The `A_log` mismatch was gone. P2P prepare succeeded (`31,800` HF names, `250,000` locators), Direct-EP streamed several buckets, then sync failed on a 32 KiB small CUDA direct transfer for `model.layers.1.linear_attn.in_proj_b.weight` with `ret=-1`. | Keep the dtype fix. Disable the small CUDA direct path in the OPD trainer manifest for the next retry. |
| `05140330p2pcpu0` | Added trainer `XORL_P2P_CPU_POOL_MIN_BYTES=0` to force small entries through the registered CPU scratch pool. | The prior `small-entries transfer` error did not recur. P2P streamed multiple CPU-staged buckets, with 384-entry expert buckets coalescing to 8 entries, then failed after streaming began because receiver tensor-map coverage was incomplete for dense layer 3 attention weights (`model.layers.3.self_attn.{q,k,v,o}_proj.weight`, `q_norm.weight`, etc.). | Keep the manifest override for future OPD retries. The next blocker is SGLang receiver tensor-map coverage and pre-stream validation, not small CUDA buffer registration. |

P2P retry after disabling the small CUDA direct path wrote one JSONL row:

| Metric | Value |
| --- | ---: |
| Run placement | teacher nodes `083`/`094`, student node `052`, trainer node `092` |
| Student sampling | 3.521s for 16 new tokens, 4.5 tok/s |
| Teacher prefill | 71.805s for 78 tokens, 1.1 tok/s |
| Teacher forward compute | 71.693s |
| Teacher cache write | 0.026s |
| Trainer forward/backward | 129.629s roundtrip |
| Trainer forward loss | 76.643s |
| Trainer backward | 54.811s |
| OPD KL compute | 76.7 ms |
| Optimizer step | 179.623s |
| P2P sync attempt | 26.513s, failed after receiver prepare and after RDMA streaming began |
| Step total to failure | 411.092s |
| Receiver prepare | `31,800` HF names, `250,000` locators across 4 receivers |
| Sync failure | Missing receiver locators for dense layer 3 attention weights |

Additional cleanup:

- Cleaned up all K8s resources for `05140255p2pfix3`.
- Cleaned up all K8s resources for `05140313p2pdtype`.
- Cleaned up all K8s resources for `05140313p2pdtype2`.
- Cleaned up all K8s resources for `05140330p2pcpu0`.

## 2026-05-14 - Async OPD Chunked Prefetch A/B

| Field | Value |
| --- | --- |
| Run ID | `0514ab2052` |
| Manifests | `/tmp/opd-ab-0514ab2052.yaml`, `/tmp/opd-ab-0514ab2052-async-trainer.yaml`, `/tmp/opd-ab-0514ab2052-async-ch4-trainer.yaml` |
| Teacher prefill | Qwen3.5-397B-A17B, XORL service, 2 nodes x 8 H100 (`094`, `096`) |
| Student sampler | Qwen3.5-35B-A3B, SGLang TP=4, node `052` |
| Student trainer | Qwen3.5-35B-A3B, XORL EP=8/Ulysses=8, node `092` |
| Shape | 8 prompts, 32 prompt tokens each, 8 sampled tokens each |
| Trainer options | `--opd-kl-backend streaming`, sharded teacher-head device cache, optimizer/sync skipped |
| Decision | Keep the opt-in async chunking implementation and idempotent endpoint registration fix; do not enable chunking by default for this workload. |

### Artifacts

| Profile | Path |
| --- | --- |
| Serial baseline | `/shared/opd-coord/0514ab2052/artifacts/qwen35_big_to_small_opd_serial_profile.jsonl` |
| Async chunk size 2 | `/shared/opd-coord/0514ab2052/artifacts/qwen35_big_to_small_opd_async_chunk_profile.jsonl` |
| Async chunk size 4 | `/shared/opd-coord/0514ab2052/artifacts/qwen35_big_to_small_opd_async_chunk4_profile.jsonl` |

### Steady-State Results

Means exclude one warmup step.

| Mode | Chunking | Step Total | Student Sampling | Teacher Prefill | Trainer Fwd/Bwd | Prepare Wait | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Serial | none | 5.083s | 0.474s | 0.575s | 4.033s | n/a | Baseline |
| Async | chunk=2, prefetch=4, teacher concurrency=1 | 11.550s | 2.194s | 2.116s | 9.938s | 1.609s | Slower |
| Async | chunk=4, prefetch=1, teacher concurrency=1 | 7.140s | 0.949s | 1.067s | 6.136s | 1.003s | Slower |

### Observations

- Chunked prefetch works functionally and successfully overlaps later chunk preparation with earlier chunk `forward_backward` calls.
- For this short 8 prompt x 40 token Qwen3.5 profile, chunking hurts throughput because it fragments both SGLang sampling and trainer fwd/bwd into smaller calls.
- The chunk=2/prefetch=4 shape also launches multiple concurrent student sampling requests, dropping aggregate sampling throughput from ~135 output tok/s serial to ~29 output tok/s steady-state chunked.
- The chunk=4/prefetch=1 shape reduces the damage but still loses to serial because two trainer `forward_backward` calls cost more than one full-batch call.
- The original A/B job's second driver invocation failed before the async pass because `/add_inference_endpoint` returned `success=False` for an already registered endpoint. The driver now treats that exact same-endpoint response as idempotent and continues.

### Cleanup

- Cleaned up all K8s jobs, services, and configmaps labeled `run-id=0514ab2052`.
## 2026-05-15 - Dispatch OPD Profile, Qwen3.5-397B Teacher to Qwen3.6-35B-A3B Student

| Field | Value |
| --- | --- |
| Run ID | `opd-q36-397b-4t4s-20260515` |
| Profile artifact | `/home/apanda/xorl-internal-dispatch-run/experiments/opd_profile/results/qwen36_35b_from_qwen35_397b/opd-q36-397b-4t4s-20260515/trainer-head-opd-q36-397b-4t4s-20260515-trainer-head-gdwgg/opd_profile.jsonl` |
| Teacher prefill | Qwen3.5-397B-A17B, XORL service, 2 nodes x 8 H100 (`069`, `075`) |
| Student sampler | Qwen3.6-35B-A3B, 16 SGLang TP2 pods behind Dispatch, 32 H100 (`036`, `058`, `060`, `065`) |
| Student trainer | Qwen3.6-35B-A3B, XORL FSDP2/EP=8, 4 nodes x 8 H100 (`100`, `106`, `116`, `117`) |
| Shape | 128 chat prompts, `opd_microbatch_size=32`, 4 OPD microbatches, `max_new_tokens=192` |
| Trainer options | `enable_packing=false`, Muon optimizer, `opd_kl_backend=streaming`, sharded teacher-head device cache |
| Decision | Keep trial evidence; keep no-packing as the practical profile setting for this 32-DP / 128-prompt shape |

### Attempts

| Attempt | Result | Decision |
| --- | --- | --- |
| Packing enabled (`sample_packing_sequence_len=4096`) | Functional OPD got through sampling, teacher hidden-cache prefill, four `forward_backward` accepts, and `optim_step` accept, but only trainer-head GPUs 0/1 did real OPD work; the other 30 DP ranks were mostly idle because 32 short samples repacked into about two 4096-token raw batches per request. | Discard as a throughput measurement for this shape. |
| Packing disabled, P2P sync enabled | OPD step 0 completed through optimizer and wrote one warmup row. P2P sync then failed during receiver prepare: SGLang `opd-q36-397b-4t4s-20260515-sglang-9` returned Mooncake registration `-202` for 2 memory regions. | Keep as P2P failure evidence; discard for steady OPD throughput. |
| Packing disabled, P2P sync disabled | OPD completed two steps and wrote warmup + steady rows. Trainer jobs were cleaned up after the run; teacher, Dispatch, and SGLang services were left running for follow-up use. | Keep as the current usable OPD profile. |

### Profile Results

| Metric | Step 0 Warmup | Step 1 Steady |
| --- | ---: | ---: |
| Step total | 280.829s | 51.001s |
| Student sampling | 43.596s | 43.430s |
| Sampling output throughput | 563.7 tok/s | 565.9 tok/s |
| Teacher prefill wall | 3.226s | 3.539s |
| Teacher forward compute | 2.377s | 2.559s |
| Teacher cache write | 0.406s | 0.407s |
| Teacher prefill throughput | 8496.6 tok/s | 7745.7 tok/s |
| Trainer forward/backward | 93.856s | 38.263s |
| Optimizer wait | 175.428s | 1.024s |
| Sync | 0.000s | 0.000s |
| Valid distillation tokens | 27,410 | 27,410 |
| Loss | 0.8028 | 0.5879 |

Steady derived rates:

| Metric | Value |
| --- | ---: |
| Distillation throughput, full step, all 80 GPUs | 6.72 tok/s/GPU |
| Distillation throughput, full step, trainer 32 GPUs only | 16.80 tok/s/GPU |
| Trainer fwd/bwd throughput, trainer 32 GPUs only | 22.38 tok/s/GPU |
| Teacher prefill throughput, teacher 16 GPUs only | 484.11 tok/s/GPU |
| Student sampling output throughput, sampler 32 GPUs only | 17.68 tok/s/GPU |

### Notes

- The xorl-client OPD loop overlaps preparation with trainer work: steady `step_total_s=51.0s` while `prepare_window_s=47.0s` and `forward_backward_s=38.3s`.
- For this short-sequence, 32-DP configuration, 4096-token packing is the wrong measurement mode because packed batch count, not datum count, is what the trainer dispatcher distributes over DP ranks.
- With packing disabled, all 32 trainer ranks receive real batches. GPU utilization is still modest because each rank sees one short sequence per OPD request; increasing prompts per request or reducing DP would be needed for higher per-GPU arithmetic intensity.
- The OPD throughput numbers above intentionally exclude sync. The P2P receiver/Mooncake issue observed in that earlier attempt was resolved in the clean-receiver retry below.

## 2026-05-15 - Dispatch OPD P2P Sync Retry, Clean SGLang Receivers

| Field | Value |
| --- | --- |
| Run ID | `opd-q36-397b-4t4s-20260515` |
| Trainer attempt | `trainer-head-opd-q36-397b-4t4s-20260515-trainer-head-v56hd` |
| Log dir | `/home/apanda/xorl-internal-dispatch-run/experiments/opd_profile/results/qwen36_35b_from_qwen35_397b/opd-q36-397b-4t4s-20260515/trainer-head-opd-q36-397b-4t4s-20260515-trainer-head-v56hd` |
| Student sampler | 16 SGLang TP2 pods behind Dispatch, 32 H100 (`036`, `058`, `060`, `079`) |
| Student trainer | Qwen3.6-35B-A3B, XORL FSDP2/EP=8, 4 nodes x 8 H100 (`061`, `117`, `122`, `125`) |
| P2P receiver mode | SGLang allocator segment registration, strict mode, `location=cuda:<rank>` |
| Decision | Keep as P2P success evidence; OPD loop then hit a separate teacher `/api/v1/forward` 400 before writing a profile row. |

### P2P Sync Results

| Endpoint Range | Result |
| --- | --- |
| SGLang endpoints `0..15` | All 16 `/add_inference_endpoint` registrations completed with `sync_method=p2p` and `weights_synced=true`. |
| First endpoint | `42.52s` for `69.32 GB`; backend pool warmup dominated (`p2p_backend_max_pool_init_s=20.452`, `p2p_backend_max_pool_wait_s=25.154`). |
| Later endpoints | `5.23s` to `7.74s` for `69.32 GB` each; final endpoint handler transfer was `5.227s`, total handler `13.719s`. |
| Receiver evidence | SGLang receiver logs showed `184 CUDA allocator segment regions`, strict registration, `location=cuda:0/1`, and successful `/prepare_weights_update` + `/complete_weights_update` responses. |

### Outcome

- This retry confirms the SGLang receiver/Mooncake registration issue from the earlier no-packing P2P attempt is fixed for endpoint registration sync.
- The initial fallback to NCCL was also fixed by passing the server configured `sync_inference_method` through `/add_inference_endpoint`; this run used `sync_method=p2p`.
- The trainer jobs were cleaned up after the head failed; teacher, Dispatch, and SGLang services were left running for follow-up use.
- Remaining blocker is not P2P: xorl-client OPD sampled through Dispatch and then failed during teacher prefill with `400 Client Error` from `http://opd-q36-397b-4t4s-20260515-teacher-master:30002/api/v1/forward`.

## 2026-05-15 - Dispatch OPD xorl-client Loop With Post-Optimizer P2P Sync

| Field | Value |
| --- | --- |
| Run ID | `opd-q36-397b-4t4s-20260515` |
| Trainer attempt | `trainer-head-opd-q36-397b-4t4s-20260515-trainer-head-94hbs` |
| Profile artifact | `/home/apanda/xorl-internal-dispatch-run/experiments/opd_profile/results/qwen36_35b_from_qwen35_397b/opd-q36-397b-4t4s-20260515/trainer-head-opd-q36-397b-4t4s-20260515-trainer-head-94hbs/opd_profile.jsonl` |
| Teacher prefill | Qwen3.5-397B-A17B, XORL service, 2 nodes x 8 H100 (`069`, `075`) |
| Student sampler | Qwen3.6-35B-A3B, 16 SGLang TP2 pods behind Dispatch, 32 H100 (`053`, `058`, `060`, `079`) |
| Student trainer | Qwen3.6-35B-A3B, XORL FSDP2/EP=8, 4 nodes x 8 H100 (`061`, `117`, `122`, `125`) |
| P2P setup | Endpoint registration uses `sync_weights=false`; xorl-client OPD calls `/sync_inference_weights` after each optimizer step with `sync_method=p2p`. |
| Placement note | SGLang pods `0..3` were moved from node `036` to node `053`; node `036` was slow/stuck during reload and node `052` rejected the manifest due node affinity labels. |
| Decision | Keep. This validates the full xorl-client OPD loop with optimizer and proper post-optimizer P2P weight sync. |

### Profile Results

| Metric | Step 0 Warmup | Step 1 Steady |
| --- | ---: | ---: |
| Step total | 346.156s | 68.284s |
| Student sampling | 88.883s | 48.804s |
| Sampling output throughput | 276.5 tok/s | 500.3 tok/s |
| Teacher prefill wall | 3.357s | 3.521s |
| Teacher forward compute | 2.359s | 2.496s |
| Teacher cache write | 0.402s | 0.412s |
| Teacher prefill throughput | 8165.9 tok/s | 7739.3 tok/s |
| Trainer forward/backward | 94.336s | 44.413s |
| Optimizer wait | 212.362s | 7.270s |
| Optimizer step | 174.116s | 1.024s |
| P2P sync wall | 41.552s | 8.687s |
| P2P transfer time | 10.616s | 8.453s |
| P2P transfer throughput | 6.53 GB/s | 8.20 GB/s |
| Sync payload | 69.32 GB, 107 buckets | 69.32 GB, 107 buckets |
| Valid distillation tokens | 27,410 | 27,249 |
| Loss | 0.7936 | 0.5039 |

### Evidence

- Trainer logs show xorl-client session registration, `SamplingClient` with `api_format=chat_completions`, four `forward_backward` requests, one `optim_step`, then `/sync_inference_weights` per OPD step.
- Server logs show `[WeightSync] sync_method=p2p, endpoints=16` for both post-optimizer syncs.
- Trainer-side Mooncake HCA binding used the explicit safe map: GPU `0..7` mapped to `mlx5_2,mlx5_3,mlx5_1,mlx5_5,mlx5_9,mlx5_9,mlx5_6,mlx5_5`.
- Step 1 used cached receiver prepare: `31,413` HF names, `1,987,616` locators across `32` receivers, `cached_prepare=True`.

## 2026-05-15 - Dispatch Sampling Saturation Benchmark

| Field | Value |
| --- | --- |
| Run ID | `opd-q36-397b-4t4s-20260515` |
| Script | `experiments/opd_profile/scripts/dispatch_sampling_benchmark.py` |
| Result dir | `/home/apanda/xorl-internal-dispatch-run/experiments/opd_profile/results/qwen36_35b_from_qwen35_397b/opd-q36-397b-4t4s-20260515/sampling_bench` |
| Student sampler | Qwen3.6-35B-A3B, 16 SGLang TP2 pods behind Dispatch, 32 H100 (`053`, `058`, `060`, `079`) |
| Payload | Chat completions, synthetic OPD prompt, `max_tokens=192`, `temperature=1.0`, `top_p=1.0`, `top_k=-1`, `n=1`, `logprobs=true` |
| Decision | Keep script and benchmark evidence. The current OPD loop undersaturates sampling because it serializes 128 prompts into four 32-request waves. |

### Results

| Mode | Requests | Concurrency | Success | Errors | Elapsed | Completion Tok/s | Req/s | p50 Latency | p95 Latency | Backend Request Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Dispatch | 512 | 64 | 512 | 0 | 82.88s | 1,184.7 | 6.18 | 10.18s | 10.80s | 21-43 |
| Dispatch | 128 | 128 | 128 | 0 | 10.69s | 2,269.4 | 11.97 | 10.38s | 10.67s | 3-14 |
| Dispatch | 512 | 128 | 512 | 0 | 42.85s | 2,286.9 | 11.95 | 10.33s | 10.91s | 23-41 |
| Dispatch | 512 | 256 | 512 | 0 | 22.02s | 4,443.2 | 23.25 | 10.61s | 11.59s | 25-41 |
| Dispatch | 512 | 512 | 512 | 0 | 16.05s | 6,080.2 | 31.90 | 12.04s | 16.01s | 17-39 |
| Direct round-robin | 512 | 512 | 512 | 0 | 16.41s | 5,955.0 | 31.21 | 12.59s | 16.34s | n/a |
| Dispatch | 4,096 | 512 | 4,087 | 9 | 122.86s | 6,348.8 | 33.26 | 14.51s | 20.64s | 237-269 |
| Dispatch | 4,096 | 1,024 | 4,094 | 2 | 71.52s | 10,922.4 | 57.24 | 17.35s | 23.07s | 233-291 |
| Dispatch | 8,192 | 1,024 | 8,168 | 24 | 143.97s | 10,822.2 | 56.73 | 17.68s | 23.42s | 464-545 |
| Direct round-robin, service DNS | 4,096 | 1,024 | 4,088 | 8 | 65.20s | 11,973.1 | 62.70 | 16.10s | 21.26s | n/a |
| Direct round-robin, pod IP | 4,096 | 1,024 | 4,089 | 7 | 65.12s | 11,952.5 | 62.79 | 16.15s | 21.10s | n/a |
| Direct round-robin, no keep-alive | 4,096 | 1,024 | 4,096 | 0 | 46.53s | 16,784.8 | 88.03 | 11.49s | 12.50s | n/a |
| Dispatch, no client keep-alive | 4,096 | 1,024 | 4,096 | 0 | 55.46s | 14,099.7 | 73.85 | 13.07s | 16.46s | 226-282 |
| Dispatch, backend no keep-alive, no client keep-alive | 4,096 | 1,024 | 4,096 | 0 | 50.11s | 15,608.4 | 81.74 | 12.33s | 13.76s | 223-283 |
| Dispatch, backend no keep-alive, client keep-alive | 4,096 | 1,024 | 4,088 | 8 | 71.20s | 10,959.5 | 57.41 | 17.28s | 22.71s | 238-283 |
| Dispatch, backend no keep-alive, proxy keep-alive 300s, client keep-alive | 4,096 | 1,024 | 4,096 | 0 | 134.49s | 5,810.2 | 30.46 | 31.46s | 50.37s | 235-277 |
| Dispatch, backend no keep-alive, proxy keep-alive 300s, no client keep-alive | 4,096 | 1,024 | 4,096 | 0 | 48.81s | 16,011.5 | 83.92 | 11.94s | 13.26s | 232-300 |

### Notes

- Dispatch and direct round-robin have similar throughput at 512 concurrent requests, so Dispatch itself is not the bottleneck for this shape.
- The active OPD loop's steady sampling step was `48.804s` for `24,415` output tokens (`500.3` tok/s) because `opd_microbatch_size=32` creates four serial-ish sampling waves for 128 prompts.
- A direct 128-request, 128-concurrency run completed in `10.69s` for `24,270` output tokens with no errors, which is the closest synthetic estimate for making the current 128-prompt OPD step a single sampling wave.
- The 16-pod sampler can sustain about `10.8k` completion tok/s with 1,024 concurrent OPD-like requests. For the same `24.4k` output-token OPD step, the lower bound from sampler capacity is about `2.3s`, but one full concurrent 128-request wave is latency-bound at about `10.7s`; that is the practical target before changing generation length or request shape.
- The `httpx.ReadError` failures were not fixed by bypassing Dispatch or by switching from per-pod service DNS to pod IPs. They disappeared when the load generator disabled HTTP keep-alive, and throughput improved from about `12.0k` to `16.8k` completion tok/s on direct round-robin. Dispatch with no client keep-alive also completed `4,096/4,096` requests successfully at `14.1k` completion tok/s; Dispatch internally retried a few backend read errors.
- Backend HTTP connection-pool settings are now configurable in `tore-dispatch`, and the live Dispatch pod was restarted with `max_keepalive_connections: 0` for all 16 SGLang backends. This removed backend read errors entirely in the `4,096` request tests.
- A shared benchmark client with HTTP keep-alive still hit client-to-Dispatch `ReadError`s until the proxy keep-alive timeout was raised from Uvicorn's default `5s` to `300s`. With that setting, the shared-client run completed `4,096/4,096`, but it was much slower than using fresh/no-keepalive client sockets.
- The best measured stable path for this OPD sampling shape is fresh/no-keepalive client connections plus no backend keep-alive: `4,096/4,096`, `48.81s`, `16.0k` completion tok/s, and zero Dispatch/SGLang backend errors.

## 2026-05-16 - Qwen3.5 35B Student OPD Tuning

| Field | Value |
| --- | --- |
| Run ID | `opd-q35-397b-mb128-t1-20260516` |
| Manifest | `experiments/opd_profile/k8s/generated/opd-q35-397b-mb128-t1-20260516.yaml` |
| Trainer config | `experiments/opd_profile/configs/qwen3_5_35b_a3b_opd_4node_nopack_gcbf.yaml` |
| Teacher prefill | Reused live `opd-q36-397b-4t4s-20260515-teacher-master` Qwen3.5-397B service |
| Student sampler | Qwen3.5-35B-A3B, 16 SGLang TP2 pods, planned nodes `014`, `047`, `055`, `067` |
| Student trainer | Qwen3.5-35B-A3B, XORL FSDP2/EP=8, planned nodes `049`, `052`, `063`, `064` |
| OPD batch shape | `num_prompts=128`, `opd_microbatch_size=128`, `max_new_tokens=192` |
| Decision | Discard. Placement failed before startup; no OPD timing data. |

### Failure

- Nodes `063`, `064`, and `067` reported only `4` allocatable GPUs despite `8` GPU capacity, so 8-GPU trainer pods and the third/fourth 2-GPU SGLang pods were rejected by kubelet.
- Node `055` also rejected SGLang pods while a GLM pod was terminating, reporting `Available: 0` for `nvidia.com/gpu`.
- The partial launch was deleted. Next trial should use only nodes with `allocatable nvidia.com/gpu=8`.

### Retry Placement Failure

| Field | Value |
| --- | --- |
| Run ID | `opd-q35-397b-mb128-t1b-20260516` |
| Manifest | `experiments/opd_profile/k8s/generated/opd-q35-397b-mb128-t1b-20260516.yaml` |
| Student sampler | Planned nodes `013`, `014`, `040`, `047` |
| Student trainer | Planned nodes `049`, `052`, `083`, `094` |
| Decision | Discard. Placement lost a race to a GLM run before sampler admission. |

- A GLM run scheduled 8-GPU pods onto the default sampler nodes `013`, `014`, `040`, and `047` just before the SGLang pods admitted, so all sampler pods saw `Available: 0`.
- The partial launch was deleted with `--wait=false`. Next trial should avoid the default node pool and place samplers on unlabelled free nodes with no `nodeSelector`.

### Unlabelled Node Placement Failure

| Field | Value |
| --- | --- |
| Run ID | `opd-q35-397b-mb128-t1c-20260516` |
| Manifest | `experiments/opd_profile/k8s/generated/opd-q35-397b-mb128-t1c-20260516.yaml` |
| Student sampler | Planned unlabelled nodes `050`, `059`, `078`, `113` |
| Student trainer | Planned nodes `049`, `052`, `083`, `094` |
| Decision | Discard. Namespace admission injected `nodeSelector: node-group=default`, so unlabelled nodes failed node affinity. |

- The manifest had no sampler `nodeSelector`, but the admitted pods showed `Node-Selectors: node-group=default`.
- Since nodes `050`, `059`, `078`, and `113` do not carry that label, kubelet rejected the pods with `Predicate NodeAffinity failed`.
- The partial launch was deleted with `--wait=false`.

### Successful 16-Sampler, 32-Trainer-GPU Baseline

| Field | Value |
| --- | --- |
| Run ID | `opd-q35-397b-mb128-t1d-20260516` |
| Manifest | `experiments/opd_profile/k8s/generated/opd-q35-397b-mb128-t1d-20260516.yaml` |
| Trainer config | `experiments/opd_profile/configs/qwen3_5_35b_a3b_opd_4node_nopack_gcbf.yaml` |
| Teacher prefill | Reused live `opd-q36-397b-4t4s-20260515-teacher-master` Qwen3.5-397B service on nodes `069`, `075` |
| Student sampler | Qwen3.5-35B-A3B, 16 SGLang TP2 pods on nodes `013`, `014`, `040`, `055` |
| Student trainer | Qwen3.5-35B-A3B, XORL FSDP2/EP=8, 32 H100 on nodes `049`, `052`, `083`, `094` |
| OPD batch shape | `num_prompts=128`, `opd_microbatch_size=128`, `max_new_tokens=192` |
| Dispatch settings | 16 backends, backend `max_concurrent_requests=256`, backend `max_keepalive_connections=0`, proxy keep-alive `300s` |
| Decision | Keep. This is the first successful Qwen3.5-35B student OPD tuning baseline using the tuned Dispatch/SGLang sampler settings and P2P sync. |

#### Profile Results

| Metric | Step 0 Warmup | Step 1 Steady |
| --- | ---: | ---: |
| Step total | 352.415s | 44.678s |
| Student sampling | 40.370s | 15.006s |
| Sampling output throughput | 606.0 tok/s | 1626.5 tok/s |
| Teacher prefill wall | 4.109s | 3.231s |
| Teacher forward compute | 3.283s | 2.503s |
| Teacher cache write | 0.348s | 0.329s |
| Teacher prefill throughput | 6643.6 tok/s | 8430.1 tok/s |
| Trainer forward/backward | 87.900s | 15.430s |
| Optimizer queued wait | 261.545s | 16.797s |
| Optimizer step | 173.646s | 1.367s |
| P2P sync wall | 46.390s | 9.642s |
| P2P transfer time | 11.385s | 9.386s |
| Sync payload | 69.32 GB, 107 buckets | 69.32 GB, 107 buckets |
| Valid distillation tokens | 27,298 | 27,242 |
| Loss | 0.7068 | 0.7321 |

#### Notes

- The steady bottlenecks are now balanced between student sampling (`15.0s`) and trainer forward/backward (`15.4s`); teacher prefill is only `3.2s`.
- The first warmup step is dominated by compile/initialization effects, especially optimizer wait and trainer forward/backward, so future comparisons should use non-warmup rows.
- P2P weight sync succeeded on both steps; the steady sync wall time was `9.6s` with `9.4s` transfer time.

#### Sampler Saturation Check

The live Qwen3.5-35B-A3B Dispatch/SGLang deployment was benchmarked from inside
the Dispatch pod with no client keep-alive, `logprobs=true`, and the same
synthetic OPD chat payload.

| Requests | Concurrency | Success | Errors | Elapsed | Completion Tok/s | p50 Latency | p95 Latency | Backend Request Range |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 128 | 128 | 128 | 0 | 11.52s | 2,132.8 | 10.76s | 11.50s | 3-13 |
| 256 | 256 | 256 | 0 | 11.91s | 4,107.8 | 11.06s | 11.89s | 11-24 |
| 512 | 512 | 512 | 0 | 11.97s | 8,145.3 | 10.88s | 11.91s | 24-41 |
| 1,024 | 1,024 | 1,024 | 0 | 12.64s | 15,506.2 | 11.90s | 12.43s | 52-88 |
| 4,096 | 1,024 | 4,096 | 0 | 52.23s | 14,972.2 | 11.97s | 14.54s | 234-273 |

- The sampler service reaches ~15k completion tok/s once concurrency is at least 1,024.
- The 128-prompt OPD step is latency-bound; the next useful OPD shape should increase prompts while keeping a known-safe trainer microbatch size.

### 512-Prompt, 128-Train-Microbatch Trial

| Field | Value |
| --- | --- |
| Run ID | `opd-q35-397b-mb128-t2-512p-20260516` |
| Manifest | `experiments/opd_profile/k8s/generated/opd-q35-397b-mb128-t2-512p-20260516-trainer-only.yaml` |
| Trainer config | `experiments/opd_profile/configs/qwen3_5_35b_a3b_opd_4node_nopack_gcbf.yaml` |
| Teacher prefill | Reused live `opd-q36-397b-4t4s-20260515-teacher-master` Qwen3.5-397B service on nodes `069`, `075` |
| Student sampler | Reused live `opd-q35-397b-mb128-t1d-20260516` Qwen3.5-35B-A3B Dispatch/SGLang deployment on nodes `013`, `014`, `040`, `055` |
| Student trainer | Qwen3.5-35B-A3B, XORL FSDP2/EP=8, 32 H100 on nodes `049`, `052`, `083`, `094` |
| OPD batch shape | `num_prompts=512`, `opd_microbatch_size=128`, `max_new_tokens=192` |
| Decision | Keep as the current best measured OPD shape. It improves useful token throughput by increasing concurrency while preserving the known-safe trainer microbatch. |

#### Profile Results

| Metric | Step 0 Warmup | Step 1 Steady |
| --- | ---: | ---: |
| Step total | 349.921s | 88.664s |
| Student sampling | 51.911s | 51.103s |
| Sampling output throughput | 1883.6 tok/s | 1915.9 tok/s |
| Teacher prefill wall | 12.996s | 12.969s |
| Teacher forward compute | 10.069s | 9.984s |
| Teacher cache write | 1.413s | 1.431s |
| Teacher prefill throughput | 8421.2 tok/s | 8448.7 tok/s |
| Trainer forward/backward | 119.246s | 61.472s |
| Optimizer queued wait | 244.751s | 14.599s |
| Optimizer step | 174.312s | 1.033s |
| P2P sync wall | 40.256s | 9.982s |
| P2P transfer time | 11.066s | 9.742s |
| Sync payload | 69.32 GB, 107 buckets | 69.32 GB, 107 buckets |
| Valid distillation tokens | 109,443 | 109,573 |
| Useful valid-token throughput | 312.8 tok/s | 1235.8 tok/s |
| Useful valid-token throughput / trainer GPU | 9.8 tok/s/GPU | 38.6 tok/s/GPU |
| Loss | 0.9733 | 0.6020 |

#### Notes

- This shape is about 2.0x the valid-token throughput of the 128-prompt baseline (`1235.8` vs. `609.7` valid tok/s), using the same 32 trainer GPUs and the same sampler/teacher services.
- Teacher prefill remains fast and scales linearly enough for this range: `109,573` teacher tokens in `12.97s`, or `8448.7` tok/s.
- Sampling is still under-utilizing the 16 SGLang pods relative to the saturation benchmark. The OPD client emits four 128-request prepare waves, so the next trial should test `opd_microbatch_size=512` with `num_prompts=512` to see whether one larger prepare wave reaches the sampler ceiling while relying on xorl-client request chunking for trainer safety.
- The generated trainer-only manifest initially had a service-port mismatch after moving the torchrun master port to `29611`; the live service and manifest were patched to `targetPort: 29611` before the successful run.

### 512-Prompt, 512-Prepare-Microbatch Trial

| Field | Value |
| --- | --- |
| Run ID | `opd-q35-397b-mb512-t3-512p-20260516` |
| Manifest | `experiments/opd_profile/k8s/generated/opd-q35-397b-mb512-t3-512p-20260516-trainer-only.yaml` |
| Trainer config | `experiments/opd_profile/configs/qwen3_5_35b_a3b_opd_4node_nopack_gcbf.yaml` |
| Teacher prefill | Reused live `opd-q36-397b-4t4s-20260515-teacher-master` Qwen3.5-397B service on nodes `069`, `075` |
| Student sampler | Reused live `opd-q35-397b-mb128-t1d-20260516` Qwen3.5-35B-A3B Dispatch/SGLang deployment on nodes `013`, `014`, `040`, `055` |
| Student trainer | Qwen3.5-35B-A3B, XORL FSDP2/EP=8, 32 H100 on nodes `049`, `052`, `083`, `094` |
| OPD batch shape | `num_prompts=512`, `opd_microbatch_size=512`, `max_new_tokens=192` |
| Decision | Discard as the winning shape. It proves large prepare batches better saturate sampling, but total step time regressed because the single OPD microbatch loses pipeline overlap. |

#### Profile Results

| Metric | Step 0 Warmup | Step 1 Steady |
| --- | ---: | ---: |
| Step total | 367.562s | 93.088s |
| Prepare window | 32.965s | 31.897s |
| Student sampling | 20.882s | 19.695s |
| Sampling output throughput | 4670.1 tok/s | 4963.0 tok/s |
| Teacher prefill wall | 12.075s | 12.190s |
| Teacher forward compute | 9.093s | 9.079s |
| Teacher cache write | 1.358s | 1.395s |
| Teacher prefill throughput | 9042.5 tok/s | 8975.8 tok/s |
| Trainer forward/backward | 123.948s | 50.929s |
| Optimizer queued wait | 296.682s | 51.955s |
| Optimizer step | 172.736s | 1.027s |
| P2P sync wall | 37.915s | 9.235s |
| P2P transfer time | 11.056s | 9.004s |
| Sync payload | 69.32 GB, 107 buckets | 69.32 GB, 107 buckets |
| Valid distillation tokens | 109,190 | 109,411 |
| Useful valid-token throughput | 297.1 tok/s | 1175.4 tok/s |
| Useful valid-token throughput / trainer GPU | 9.3 tok/s/GPU | 36.7 tok/s/GPU |
| Loss | 0.9754 | 0.6086 |

#### Notes

- The larger prepare batch cut steady student sampling from `51.10s` to `19.69s` and raised sampling output throughput from `1915.9` to `4963.0` tok/s.
- xorl-client chunked the 512 prepared datums into two trainer `forward_backward` requests due `XORL_CLIENT_MAX_CHUNK_BYTES_COUNT=1250000`, so the trainer did not receive one giant unsafe request.
- Despite better sampling and lower fwd/bwd time, the steady end-to-end step was slower than the 512-prompt/128-microbatch trial (`93.09s` vs. `88.66s`). The 128-microbatch path keeps four fwd/bwd chunks in flight while later prepare batches are still running; the 512-microbatch path has only one OPD batch and less overlap.
- Next trial should decouple prepare batch size from trainer microbatch size: sample/prefill in a 512-prompt wave, then split the prepared datums into 128-datum trainer chunks.

### 512-Prompt, 512-Prepare, 128-Train-Microbatch Trial

| Field | Value |
| --- | --- |
| Run ID | `opd-q35-397b-mb128-pb512-t4-512p-20260516` |
| Manifest | `experiments/opd_profile/k8s/generated/opd-q35-397b-mb128-pb512-t4-512p-20260516-trainer-only.yaml` |
| xorl-client change | Added `opd_prepare_batch_size` so prepare/sampling/teacher prefill waves can be larger than trainer microbatches |
| Teacher prefill | Reused live `opd-q36-397b-4t4s-20260515-teacher-master` Qwen3.5-397B service on nodes `069`, `075` |
| Student sampler | Reused live `opd-q35-397b-mb128-t1d-20260516` Qwen3.5-35B-A3B Dispatch/SGLang deployment on nodes `013`, `014`, `040`, `055` |
| Student trainer | Qwen3.5-35B-A3B, XORL FSDP2/EP=8, 32 H100 on nodes `049`, `052`, `083`, `094` |
| OPD batch shape | `num_prompts=512`, `opd_prepare_batch_size=512`, `opd_microbatch_size=128`, `max_new_tokens=192` |
| Decision | Discard as the winning shape. The feature works, but this particular scheduling shape is slower than the current 512-prompt/128-microbatch winner. |

#### Profile Results

| Metric | Step 0 Warmup | Step 1 Steady |
| --- | ---: | ---: |
| Step total | 362.335s | 99.727s |
| Prepare batches | 1 | 1 |
| Trainer microbatches | 4 | 4 |
| Prepare window | 32.116s | 32.130s |
| Student sampling | 20.302s | 20.429s |
| Sampling output throughput | 4829.9 tok/s | 4792.8 tok/s |
| Teacher prefill wall | 11.808s | 11.692s |
| Teacher forward compute | 9.174s | 9.070s |
| Teacher cache write | 1.290s | 1.360s |
| Teacher prefill throughput | 9292.4 tok/s | 9371.6 tok/s |
| Trainer forward/backward | 118.333s | 56.408s |
| Optimizer queued wait | 291.617s | 57.445s |
| Optimizer step | 173.285s | 1.038s |
| P2P sync wall | 38.602s | 10.152s |
| P2P transfer time | 11.068s | 9.907s |
| Sync payload | 69.32 GB, 107 buckets | 69.32 GB, 107 buckets |
| Valid distillation tokens | 109,721 | 109,576 |
| Useful valid-token throughput | 302.8 tok/s | 1098.8 tok/s |
| Useful valid-token throughput / trainer GPU | 9.5 tok/s/GPU | 34.3 tok/s/GPU |
| Loss | 0.9737 | 0.5979 |

#### Notes

- The new `opd_prepare_batch_size` path is correct mechanically: the profile shows one prepare batch and four trainer microbatches.
- Sampling is much faster than t2 (`20.43s` vs. `51.10s`), but the larger prepare batch delays all four trainer chunks until the full 512-prompt teacher cache is ready.
- Current best remains `opd-q35-397b-mb128-t2-512p-20260516`: `88.66s` steady step and `1235.8` valid tok/s.
- The next tuning direction should preserve the original pipelining but test larger total prompt counts, e.g. `1024` prompts with `128`-prompt OPD microbatches, to see whether the trainer stays saturated over more waves.

### 1024-Prompt, 128-Microbatch Trial

| Field | Value |
| --- | --- |
| Run ID | `opd-q35-397b-mb128-t5-1024p-20260516` |
| Manifest | `experiments/opd_profile/k8s/generated/opd-q35-397b-mb128-t5-1024p-20260516-trainer-only.yaml` |
| Teacher prefill | Reused live `opd-q36-397b-4t4s-20260515-teacher-master` Qwen3.5-397B service on nodes `069`, `075` |
| Student sampler | Reused live `opd-q35-397b-mb128-t1d-20260516` Qwen3.5-35B-A3B Dispatch/SGLang deployment on nodes `013`, `014`, `040`, `055` |
| Student trainer | Qwen3.5-35B-A3B, XORL FSDP2/EP=8, 32 H100 on nodes `049`, `052`, `083`, `094` |
| OPD batch shape | `num_prompts=1024`, `opd_microbatch_size=128`, `max_new_tokens=192` |
| Decision | Keep as the previous measured throughput winner. The wall-clock step is longer than 512-prompt t2, but valid-token throughput improves materially. |

#### Profile Results

| Metric | Step 0 Warmup | Step 1 Steady |
| --- | ---: | ---: |
| Step total | 383.674s | 147.787s |
| Prepare batches | 8 | 8 |
| Trainer microbatches | 8 | 8 |
| Prepare window | 128.356s | 125.988s |
| Student sampling | 103.546s | 101.335s |
| Sampling output throughput | 1888.4 tok/s | 1930.3 tok/s |
| Teacher prefill wall | 24.797s | 24.631s |
| Teacher forward compute | 20.126s | 20.033s |
| Teacher cache write | 2.867s | 2.900s |
| Teacher prefill throughput | 8831.8 tok/s | 8894.3 tok/s |
| Trainer forward/backward | 156.162s | 121.761s |
| Optimizer queued wait | 217.884s | 12.748s |
| Optimizer step | 174.069s | 1.025s |
| P2P sync wall | 37.434s | 9.051s |
| P2P transfer time | 10.881s | 8.819s |
| Sync payload | 69.32 GB, 107 buckets | 69.32 GB, 107 buckets |
| Valid distillation tokens | 219,000 | 219,077 |
| Useful valid-token throughput | 570.8 tok/s | 1482.4 tok/s |
| Useful valid-token throughput / trainer GPU | 17.8 tok/s/GPU | 46.3 tok/s/GPU |
| Loss | 0.9793 | 0.6064 |

#### Notes

- This keeps the t2 scheduling shape, with 128-prompt prepare/trainer microbatches, but doubles total prompts from 512 to 1024.
- The sampler remains well below standalone Dispatch/SGLang throughput because the current OPD loop still issues eight 128-request waves and waits at microbatch boundaries.
- Trainer throughput improves versus t2 because more waves reduce fixed sync and optimizer overhead per token: t2 was `1235.8` valid tok/s total, while this run reached `1482.4` valid tok/s.
- This was superseded by the `opd_prepare_concurrency=4` trial below, which keeps the same 1024 prompt / 128 train-microbatch shape but overlaps four prepare batches at a time.

### 1024-Prompt, 128-Microbatch, Prepare-Concurrency-4 Trial

| Field | Value |
| --- | --- |
| Run ID | `opd-q35-c4-t6-1024p-20260516` |
| Manifest | `experiments/opd_profile/k8s/generated/opd-q35-c4-t6-1024p-20260516-trainer-only.yaml` |
| xorl-client change | Added `opd_prepare_concurrency` so multiple student-sampling plus teacher-prefill prepare batches can run concurrently before feeding 128-datum trainer chunks |
| Teacher prefill | Reused live `opd-q36-397b-4t4s-20260515-teacher-master` Qwen3.5-397B service on nodes `069`, `075` |
| Student sampler | Reused live `opd-q35-397b-mb128-t1d-20260516` Qwen3.5-35B-A3B Dispatch/SGLang deployment on nodes `013`, `014`, `040`, `055` |
| Student trainer | Qwen3.5-35B-A3B, XORL FSDP2/EP=8, 32 H100 on nodes `049`, `052`, `083`, `094` |
| OPD batch shape | `num_prompts=1024`, `opd_microbatch_size=128`, `opd_prepare_concurrency=4`, `max_new_tokens=192` |
| Decision | Keep as the current measured throughput winner. The client-side overlap is useful, and the next tuning direction is to test higher prepare concurrency with the multi-completion bug fix applied. |

#### Profile Results

| Metric | Step 0 Warmup | Step 1 Steady |
| --- | ---: | ---: |
| Step total | 393.576s | 130.107s |
| Prepare batches | 8 | 8 |
| Trainer microbatches | 8 | 8 |
| Prepare concurrency | 4 | 4 |
| Prepare window | 48.637s | 48.274s |
| Sum of prepare batch times | 176.154s | 174.112s |
| Student sampling summed wall | 131.504s | 130.817s |
| Sampling output throughput, summed | 1487.4 tok/s | 1495.0 tok/s |
| Sampling output throughput, prepare-window | 4021.5 tok/s | 4051.3 tok/s |
| Teacher prefill summed wall | 44.635s | 43.087s |
| Teacher forward compute | 20.406s | 20.101s |
| Teacher cache write | 2.907s | 2.857s |
| Teacher prefill throughput, summed | 4907.8 tok/s | 5083.7 tok/s |
| Teacher prefill throughput, prepare-window | 4504.0 tok/s | 4537.4 tok/s |
| Trainer forward/backward | 158.811s | 97.433s |
| Optimizer queued wait | 306.440s | 72.225s |
| Optimizer step | 173.029s | 1.014s |
| P2P sync wall | 38.499s | 9.607s |
| P2P transfer time | 11.343s | 9.362s |
| Sync payload | 69.32 GB, 107 buckets | 69.32 GB, 107 buckets |
| Valid distillation tokens | 219,059 | 219,040 |
| Useful valid-token throughput | 556.6 tok/s | 1683.5 tok/s |
| Useful valid-token throughput / trainer GPU | 17.4 tok/s/GPU | 52.6 tok/s/GPU |
| Loss | 0.9767 | 0.6056 |

#### Notes

- The prepare window dropped from t5's `125.99s` steady window to `48.27s`, proving that student sampling plus teacher prefill can be overlapped cleanly through the xorl-client OPD path.
- The summed per-batch sampling and teacher timings are no longer the right wall-clock interpretation under concurrency. For concurrent trials, compare prepare-window throughput in addition to the summed per-batch rates.
- End-to-end steady throughput improved from t5's `1482.4` valid tok/s to `1683.5` valid tok/s on the same services and nodes.
- Trainer fwd/bwd plus queued optimizer time is now the dominant remaining term. `opd_prepare_concurrency=8` was tested next and regressed, so c4 remains the current best prepare-concurrency point.

### 1024-Prompt, 128-Microbatch, Prepare-Concurrency-8 Trial

| Field | Value |
| --- | --- |
| Run ID | `opd-q35-c8-t7-1024p-20260516` |
| Manifest | Removed after trial because the shape did not win |
| xorl-client change | Exercised the fixed concurrent-prepare loop and the new prepare-window throughput profile fields |
| Teacher prefill | Reused live `opd-q36-397b-4t4s-20260515-teacher-master` Qwen3.5-397B service on nodes `069`, `075` |
| Student sampler | Reused live `opd-q35-397b-mb128-t1d-20260516` Qwen3.5-35B-A3B Dispatch/SGLang deployment on nodes `013`, `014`, `040`, `055` |
| Student trainer | Qwen3.5-35B-A3B, XORL FSDP2/EP=8, 32 H100 on nodes `049`, `052`, `083`, `094` |
| OPD batch shape | `num_prompts=1024`, `opd_microbatch_size=128`, `opd_prepare_concurrency=8`, `max_new_tokens=192` |
| Decision | Discard as a non-winning shape. It overdrives the prepare path and is slower than c4 end to end. |

#### Profile Results

| Metric | Step 0 Warmup | Step 1 Steady |
| --- | ---: | ---: |
| Step total | 413.946s | 149.763s |
| Prepare batches | 8 | 8 |
| Trainer microbatches | 8 | 8 |
| Prepare concurrency | 8 | 8 |
| Prepare window | 54.876s | 51.705s |
| Sum of prepare batch times | 351.073s | 327.710s |
| Student sampling summed wall | 241.738s | 221.741s |
| Sampling output throughput, summed | 807.8 tok/s | 882.5 tok/s |
| Sampling output throughput, prepare-window | 3558.3 tok/s | 3784.7 tok/s |
| Teacher prefill summed wall | 109.321s | 105.952s |
| Teacher forward compute | 20.721s | 20.334s |
| Teacher cache write | 2.837s | 2.903s |
| Teacher prefill throughput, summed | 2000.8 tok/s | 2068.4 tok/s |
| Teacher prefill throughput, prepare-window | 3985.9 tok/s | 4238.5 tok/s |
| Trainer forward/backward | 162.151s | 107.567s |
| Optimizer queued wait | 316.475s | 87.679s |
| Optimizer step | 175.977s | 1.520s |
| P2P sync wall | 42.595s | 10.380s |
| P2P transfer time | 11.621s | 10.132s |
| Sync payload | 69.32 GB, 107 buckets | 69.32 GB, 107 buckets |
| Valid distillation tokens | 218,731 | 219,151 |
| Useful valid-token throughput | 528.4 tok/s | 1463.3 tok/s |
| Useful valid-token throughput / trainer GPU | 16.5 tok/s/GPU | 45.7 tok/s/GPU |
| Loss | 0.9785 | 0.5984 |

#### Notes

- The c8 run completed without HTTP, trainer, teacher, or P2P sync failures.
- Despite launching all eight prepare batches concurrently, steady prepare-window sampling throughput fell below c4 (`3784.7` vs. `4051.3` output tok/s), and teacher prefill also became less efficient under the heavier concurrent request burst.
- End-to-end steady throughput regressed versus c4 (`1463.3` vs. `1683.5` valid tok/s), so c4 remains the current best.
- The generated c8 manifest was removed after recording the result.

### 1024-Prompt, 128-Microbatch, Prepare-Concurrency-2 Trial

| Field | Value |
| --- | --- |
| Run ID | `opd-q35-c2-t8-1024p-20260516` |
| Manifest | Removed after trial because the shape did not win |
| Teacher prefill | Reused live `opd-q36-397b-4t4s-20260515-teacher-master` Qwen3.5-397B service on nodes `069`, `075` |
| Student sampler | Reused live `opd-q35-397b-mb128-t1d-20260516` Qwen3.5-35B-A3B Dispatch/SGLang deployment on nodes `013`, `014`, `040`, `055` |
| Student trainer | Qwen3.5-35B-A3B, XORL FSDP2/EP=8, 32 H100 on nodes `049`, `052`, `083`, `094` |
| OPD batch shape | `num_prompts=1024`, `opd_microbatch_size=128`, `opd_prepare_concurrency=2`, `max_new_tokens=192` |
| Decision | Discard as a non-winning shape. It underdrives the prepare path relative to c4. |

#### Profile Results

| Metric | Step 0 Warmup | Step 1 Steady |
| --- | ---: | ---: |
| Step total | 388.876s | 138.963s |
| Prepare batches | 8 | 8 |
| Trainer microbatches | 8 | 8 |
| Prepare concurrency | 2 | 2 |
| Prepare window | 71.004s | 68.588s |
| Sum of prepare batch times | 138.774s | 134.128s |
| Student sampling summed wall | 109.745s | 105.944s |
| Sampling output throughput, summed | 1785.3 tok/s | 1845.6 tok/s |
| Sampling output throughput, prepare-window | 2759.4 tok/s | 2850.7 tok/s |
| Teacher prefill summed wall | 28.869s | 28.167s |
| Teacher forward compute | 20.397s | 20.227s |
| Teacher cache write | 2.920s | 2.904s |
| Teacher prefill throughput, summed | 7599.6 tok/s | 7774.7 tok/s |
| Teacher prefill throughput, prepare-window | 3089.9 tok/s | 3192.8 tok/s |
| Trainer forward/backward | 155.218s | 110.179s |
| Optimizer queued wait | 276.451s | 60.639s |
| Optimizer step | 173.862s | 1.021s |
| P2P sync wall | 41.421s | 9.736s |
| P2P transfer time | 11.616s | 9.511s |
| Sync payload | 69.32 GB, 107 buckets | 69.32 GB, 107 buckets |
| Valid distillation tokens | 219,392 | 218,991 |
| Useful valid-token throughput | 564.2 tok/s | 1575.9 tok/s |
| Useful valid-token throughput / trainer GPU | 17.6 tok/s/GPU | 49.2 tok/s/GPU |
| Loss | 0.9773 | 0.5943 |

#### Notes

- The c2 run completed without HTTP, trainer, teacher, or P2P sync failures.
- Steady prepare-window sampling throughput was substantially below c4 (`2850.7` vs. `4051.3` output tok/s), and end-to-end useful throughput was lower (`1575.9` vs. `1683.5` valid tok/s).
- This brackets the prepare-concurrency sweep: c2 underdrives the services, c8 overdrives them, and c4 remains the current best client-side overlap point.
- The generated c2 manifest was removed after recording the result.

### 1024-Prompt, 128-Microbatch, No-Recompute Trainer Trial

| Field | Value |
| --- | --- |
| Run ID | `opd-q35-nr-c4-t9-1024p-20260516` |
| Manifest | Removed after trial because the shape did not win |
| Trainer config | Removed trial-only `qwen3_5_35b_a3b_opd_4node_nopack_no_recompute.yaml` after it regressed |
| Teacher prefill | Reused live `opd-q36-397b-4t4s-20260515-teacher-master` Qwen3.5-397B service on nodes `069`, `075` |
| Student sampler | Reused live `opd-q35-397b-mb128-t1d-20260516` Qwen3.5-35B-A3B Dispatch/SGLang deployment on nodes `013`, `014`, `040`, `055` |
| Student trainer | Qwen3.5-35B-A3B, XORL FSDP2/EP=8, 32 H100 on nodes `049`, `052`, `083`, `094`, `gradient_checkpointing_method=no_recompute` |
| OPD batch shape | `num_prompts=1024`, `opd_microbatch_size=128`, `opd_prepare_concurrency=4`, `max_new_tokens=192` |
| Decision | Discard as a non-winning trainer config. It fit, but it was slower than recompute-before-dispatch. |

#### Profile Results

| Metric | Step 0 Warmup | Step 1 Steady |
| --- | ---: | ---: |
| Step total | 391.441s | 152.279s |
| Prepare batches | 8 | 8 |
| Trainer microbatches | 8 | 8 |
| Prepare concurrency | 4 | 4 |
| Prepare window | 48.647s | 47.689s |
| Sum of prepare batch times | 176.110s | 172.372s |
| Student sampling summed wall | 131.894s | 128.717s |
| Sampling output throughput, summed | 1486.9 tok/s | 1521.3 tok/s |
| Sampling output throughput, prepare-window | 4031.4 tok/s | 4106.1 tok/s |
| Teacher prefill summed wall | 44.203s | 43.641s |
| Teacher forward compute | 20.496s | 20.313s |
| Teacher cache write | 2.850s | 2.864s |
| Teacher prefill throughput, summed | 4967.5 tok/s | 5024.7 tok/s |
| Teacher prefill throughput, prepare-window | 4513.8 tok/s | 4598.2 tok/s |
| Trainer forward/backward | 154.786s | 117.285s |
| Optimizer queued wait | 305.034s | 94.426s |
| Optimizer step | 175.702s | 2.354s |
| P2P sync wall | 37.760s | 10.164s |
| P2P transfer time | 10.983s | 9.920s |
| Sync payload | 69.32 GB, 107 buckets | 69.32 GB, 107 buckets |
| Valid distillation tokens | 219,580 | 219,282 |
| Useful valid-token throughput | 561.0 tok/s | 1440.0 tok/s |
| Useful valid-token throughput / trainer GPU | 17.5 tok/s/GPU | 45.0 tok/s/GPU |
| Loss | 0.9765 | 0.6035 |

#### Notes

- `gradient_checkpointing_method=no_recompute` fit this 1024-prompt OPD smoke without OOM, so memory was not the blocker.
- It still regressed versus the c4 recompute-before-dispatch baseline: steady fwd/bwd increased (`117.3s` vs. `97.4s`), optimizer queued wait increased (`94.4s` vs. `72.2s`), and useful throughput fell (`1440.0` vs. `1683.5` valid tok/s).
- This falsifies no-recompute as the next trainer-side improvement for the current 32-GPU Qwen3.5-35B-A3B OPD shape.
- The generated no-recompute manifest and trial-only trainer config were removed after recording the result.

### 1024-Prompt, 64-Train-Microbatch, 128-Prepare-Batch Trial

| Field | Value |
| --- | --- |
| Run ID | `opd-q35-mb64-c4-t10-1024p-20260516` |
| Manifest | Removed after trial because the shape did not win |
| Teacher prefill | Reused live `opd-q36-397b-4t4s-20260515-teacher-master` Qwen3.5-397B service on nodes `069`, `075` |
| Student sampler | Reused live `opd-q35-397b-mb128-t1d-20260516` Qwen3.5-35B-A3B Dispatch/SGLang deployment on nodes `013`, `014`, `040`, `055` |
| Student trainer | Qwen3.5-35B-A3B, XORL FSDP2/EP=8, 32 H100 on nodes `049`, `052`, `083`, `094`, `gradient_checkpointing_method=recompute_before_dispatch` |
| OPD batch shape | `num_prompts=1024`, `opd_prepare_batch_size=128`, `opd_microbatch_size=64`, `opd_prepare_concurrency=4`, `max_new_tokens=192` |
| Decision | Discard as a non-winning trainer shape. The smaller train microbatch improved no service-side bottleneck and doubled trainer chunks. |

#### Profile Results

| Metric | Step 0 Warmup | Step 1 Steady |
| --- | ---: | ---: |
| Step total | 403.539s | 188.405s |
| Prepare batches | 8 | 8 |
| Trainer microbatches | 16 | 16 |
| Prepare concurrency | 4 | 4 |
| Prepare window | 48.343s | 47.465s |
| Sum of prepare batch times | 175.100s | 171.271s |
| Student sampling summed wall | 131.609s | 127.301s |
| Sampling output throughput, summed | 1489.2 tok/s | 1540.4 tok/s |
| Sampling output throughput, prepare-window | 4054.4 tok/s | 4131.3 tok/s |
| Teacher prefill summed wall | 43.478s | 43.957s |
| Teacher forward compute | 20.216s | 20.497s |
| Teacher cache write | 2.904s | 2.866s |
| Teacher prefill throughput, summed | 5047.7 tok/s | 4994.9 tok/s |
| Teacher prefill throughput, prepare-window | 4539.8 tok/s | 4625.7 tok/s |
| Trainer forward/backward | 165.537s | 155.297s |
| Optimizer queued wait | 311.419s | 131.187s |
| Optimizer step | 170.979s | 1.004s |
| P2P sync wall | 43.778s | 9.753s |
| P2P transfer time | 11.669s | 9.519s |
| Sync payload | 69.32 GB, 107 buckets | 69.32 GB, 107 buckets |
| Valid distillation tokens | 219,465 | 219,561 |
| Useful valid-token throughput | 543.9 tok/s | 1165.4 tok/s |
| Useful valid-token throughput / trainer GPU | 17.0 tok/s/GPU | 36.4 tok/s/GPU |
| Loss | 0.9771 | 0.5953 |

#### Notes

- The run completed cleanly with the same teacher and sampler services as the c4 winner.
- Service overlap stayed similar to c4, but trainer work regressed: steady fwd/bwd rose to `155.3s` because `opd_microbatch_size=64` doubled the number of trainer microbatches.
- Useful throughput fell to `1165.4` valid tok/s, so the generated manifest was removed after recording the result.

### 2048-Prompt, 128-Microbatch, c4 Prepare-Concurrency Trial

| Field | Value |
| --- | --- |
| Run ID | `opd-q35-c4-t11-2048p-20260516` |
| Manifest | Removed after trial because the shape did not win |
| Teacher prefill | Reused live `opd-q36-397b-4t4s-20260515-teacher-master` Qwen3.5-397B service on nodes `069`, `075` |
| Student sampler | Reused live `opd-q35-397b-mb128-t1d-20260516` Qwen3.5-35B-A3B Dispatch/SGLang deployment on nodes `013`, `014`, `040`, `055` |
| Student trainer | Qwen3.5-35B-A3B, XORL FSDP2/EP=8, 32 H100 on nodes `049`, `052`, `083`, `094`, `gradient_checkpointing_method=recompute_before_dispatch` |
| OPD batch shape | `num_prompts=2048`, `opd_prepare_batch_size=128`, `opd_microbatch_size=128`, `opd_prepare_concurrency=4`, `max_new_tokens=192` |
| Decision | Discard as a non-winning scale-up. The larger step improved sampler and teacher window utilization, but trainer queueing dominated. |

#### Profile Results

| Metric | Step 0 Warmup | Step 1 Steady |
| --- | ---: | ---: |
| Step total | 460.737s | 270.581s |
| Prepare batches | 16 | 16 |
| Trainer microbatches | 16 | 16 |
| Prepare concurrency | 4 | 4 |
| Prepare window | 83.346s | 81.858s |
| Sum of prepare batch times | 312.334s | 309.638s |
| Student sampling summed wall | 241.796s | 238.010s |
| Sampling output throughput, summed | 1622.9 tok/s | 1645.5 tok/s |
| Sampling output throughput, prepare-window | 4708.1 tok/s | 4784.5 tok/s |
| Teacher prefill summed wall | 70.279s | 71.597s |
| Teacher forward compute | 40.583s | 40.206s |
| Teacher cache write | 5.748s | 5.716s |
| Teacher prefill throughput, summed | 6267.1 tok/s | 6141.2 tok/s |
| Teacher prefill throughput, prepare-window | 5284.5 tok/s | 5371.4 tok/s |
| Trainer forward/backward | 226.525s | 236.701s |
| Optimizer queued wait | 339.022s | 178.666s |
| Optimizer step | 173.090s | 1.117s |
| P2P sync wall | 38.369s | 10.057s |
| P2P transfer time | 11.469s | 9.835s |
| Sync payload | 69.32 GB, 107 buckets | 69.32 GB, 107 buckets |
| Valid distillation tokens | 440,442 | 439,689 |
| Useful valid-token throughput | 956.0 tok/s | 1625.0 tok/s |
| Useful valid-token throughput / trainer GPU | 29.9 tok/s/GPU | 50.8 tok/s/GPU |
| Loss | 0.9958 | 0.5967 |

#### Notes

- The larger OPD step improved service-side saturation: steady sampling reached `4784.5` output tok/s over the prepare window, and teacher prefill reached `5371.4` input tok/s over the same window.
- End-to-end throughput still regressed versus the 1024-prompt c4 best (`1625.0` vs. `1683.5` valid tok/s) because steady trainer fwd/bwd grew to `236.7s` and queued future drain remained `178.7s`.
- The 1024-prompt, 128-microbatch, c4 prepare-concurrency shape remains the current best for this 32-GPU trainer and 16-pod sampler setup.

### 64-GPU Trainer Scale Trial, Failed Node Selection

| Field | Value |
| --- | --- |
| Run ID | `opd-q35-64g-c4-t12-1024p-20260516` |
| Manifest | Removed after scheduling failure |
| Teacher prefill | Reused live `opd-q36-397b-4t4s-20260515-teacher-master` Qwen3.5-397B service on nodes `069`, `075` |
| Student sampler | Reused live `opd-q35-397b-mb128-t1d-20260516` Qwen3.5-35B-A3B Dispatch/SGLang deployment on nodes `013`, `014`, `040`, `055` |
| Intended student trainer | Qwen3.5-35B-A3B, XORL FSDP2/EP=8, 64 H100 on nodes `036`, `048`, `050`, `059`, `065`, `078`, `096`, `112` |
| OPD batch shape | `num_prompts=1024`, `opd_prepare_batch_size=128`, `opd_microbatch_size=128`, `opd_prepare_concurrency=4`, `max_new_tokens=192` |
| Failure | Worker 4 on `065` was rejected by kubelet with `Predicate NodeAffinity failed`; the manifest requires `node-group=nccl` but node `065` is labeled `node-group=default`. Several other pinned pods entered `ContainerStatusUnknown` during the failed launch. |
| Decision | Discard the manifest and relaunch only if eight valid NCCL trainer nodes become available. |

### 48-GPU Trainer Scale Trial, Failed Node Start

| Field | Value |
| --- | --- |
| Run ID | `opd-q35-48g-c4-t13-1024p-20260516` |
| Manifest | Removed after node-start failure |
| Teacher prefill | Reused live `opd-q36-397b-4t4s-20260515-teacher-master` Qwen3.5-397B service on nodes `069`, `075` |
| Student sampler | Reused live `opd-q35-397b-mb128-t1d-20260516` Qwen3.5-35B-A3B Dispatch/SGLang deployment on nodes `013`, `014`, `040`, `055` |
| Intended student trainer | Qwen3.5-35B-A3B, XORL FSDP2/EP=8, 48 H100 on nodes `049`, `052`, `083`, `094`, `096`, `112` |
| OPD batch shape | `num_prompts=1024`, `opd_prepare_batch_size=128`, `opd_microbatch_size=128`, `opd_prepare_concurrency=4`, `max_new_tokens=192` |
| Failure | Five trainer pods started, but worker 5 on `112` stayed `Pending` after volume attach with no container start. The 6-node torchrun could not form. |
| Decision | Discard the 6-node config/manifest and relaunch as a 5-node/40-GPU trainer trial using only nodes that started cleanly. |

### 40-GPU Trainer Scale Trial, Invalid FSDP Topology

| Field | Value |
| --- | --- |
| Run ID | `opd-q35-40g-c4-t14-1024p-20260516` |
| Manifest | Removed after topology failure |
| Teacher prefill | Reused live `opd-q36-397b-4t4s-20260515-teacher-master` Qwen3.5-397B service on nodes `069`, `075` |
| Student sampler | Reused live `opd-q35-397b-mb128-t1d-20260516` Qwen3.5-35B-A3B Dispatch/SGLang deployment on nodes `013`, `014`, `040`, `055` |
| Intended student trainer | Qwen3.5-35B-A3B, XORL FSDP2/EP=8, 40 H100 on nodes `049`, `052`, `083`, `094`, `096` |
| OPD batch shape | `num_prompts=1024`, `opd_prepare_batch_size=128`, `opd_microbatch_size=128`, `opd_prepare_concurrency=4`, `max_new_tokens=192` |
| Failure | Trainer started but failed during FSDP2 initialization: `FSDP does not support uneven sharding on dim 1: torch.Size([32, 2048, 1024]) (world size: 5)`. |
| Decision | Discard the 5-node config/manifest. With `expert_parallel_size=8`, valid expert FSDP group sizes for these tensors need to divide `2048`; the next useful trainer-scale attempt is therefore an 8-node/64-GPU run when eight clean NCCL nodes are available. |

### 1536-Prompt, 128-Microbatch, Prepare-Concurrency-4 Trial

| Field | Value |
| --- | --- |
| Run ID | `opd-q35-c4-t15-1536p-20260516` |
| Manifest | `experiments/opd_profile/k8s/generated/opd-q35-c4-t15-1536p-20260516-trainer-only.yaml` |
| Teacher prefill | Reused live `opd-q36-397b-4t4s-20260515-teacher-master` Qwen3.5-397B service on nodes `069`, `075` |
| Student sampler | Reused live `opd-q35-397b-mb128-t1d-20260516` Qwen3.5-35B-A3B Dispatch/SGLang deployment on nodes `013`, `014`, `040`, `055` |
| Student trainer | Qwen3.5-35B-A3B, XORL FSDP2/EP=8, 32 H100 on nodes `049`, `052`, `083`, `094`, `gradient_checkpointing_method=recompute_before_dispatch` |
| OPD batch shape | `num_prompts=1536`, `opd_prepare_batch_size=128`, `opd_microbatch_size=128`, `opd_prepare_concurrency=4`, `max_new_tokens=192` |
| Decision | Keep as a previous best. It improved useful throughput over the prior 1024-prompt c4 winner, but was superseded by the 1792-prompt t16 trial. |

#### Profile Results

| Metric | Step 0 Warmup | Step 1 Steady |
| --- | ---: | ---: |
| Step total | 430.220s | 190.951s |
| Prepare batches | 12 | 12 |
| Trainer microbatches | 12 | 12 |
| Prepare concurrency | 4 | 4 |
| Prepare window | 67.585s | 66.537s |
| Sum of prepare batch times | 248.197s | 242.874s |
| Student sampling summed wall | 188.654s | 185.483s |
| Sampling output throughput, summed | 1555.1 tok/s | 1583.0 tok/s |
| Sampling output throughput, prepare-window | 4340.7 tok/s | 4413.0 tok/s |
| Teacher prefill summed wall | 59.368s | 57.239s |
| Teacher forward compute | 30.749s | 30.508s |
| Teacher cache write | 4.302s | 4.216s |
| Teacher prefill throughput, summed | 5543.8 tok/s | 5754.4 tok/s |
| Teacher prefill throughput, prepare-window | 4869.7 tok/s | 4950.3 tok/s |
| Trainer forward/backward | 189.951s | 157.005s |
| Optimizer queued wait | 319.702s | 114.225s |
| Optimizer step | 173.721s | 1.027s |
| P2P sync wall | 42.933s | 10.189s |
| P2P transfer time | 11.800s | 9.962s |
| Sync payload | 69.32 GB, 107 buckets | 69.32 GB, 107 buckets |
| Valid distillation tokens | 329,123 | 329,378 |
| Useful valid-token throughput | 765.0 tok/s | 1724.9 tok/s |
| Useful valid-token throughput / trainer GPU | 23.9 tok/s/GPU | 53.9 tok/s/GPU |
| Loss | 0.9889 | 0.5965 |

#### Notes

- The 1536-prompt shape improved the steady useful throughput from the prior 1024-prompt c4 best (`1683.5` valid tok/s) to `1724.9` valid tok/s, but was later superseded by the 1792-prompt t16 trial.
- Service-side utilization improved versus 1024 prompts: steady sampling reached `4413.0` output tok/s over the prepare window and teacher prefill reached `4950.3` input tok/s over the same window.
- The trainer remains the limiting stage: steady fwd/bwd rose to `157.0s`, but this was still a better tradeoff than the 2048-prompt trial, where fwd/bwd reached `236.7s`.

### 1792-Prompt, 128-Microbatch, Prepare-Concurrency-4 Trial

| Field | Value |
| --- | --- |
| Run ID | `opd-q35-c4-t16-1792p-20260516` |
| Manifest | `experiments/opd_profile/k8s/generated/opd-q35-c4-t16-1792p-20260516-trainer-only.yaml` |
| Teacher prefill | Reused live `opd-q36-397b-4t4s-20260515-teacher-master` Qwen3.5-397B service on nodes `069`, `075` |
| Student sampler | Reused live `opd-q35-397b-mb128-t1d-20260516` Qwen3.5-35B-A3B Dispatch/SGLang deployment on nodes `013`, `014`, `040`, `055` |
| Student trainer | Qwen3.5-35B-A3B, XORL FSDP2/EP=8, 32 H100 on nodes `049`, `052`, `083`, `094`, `gradient_checkpointing_method=recompute_before_dispatch` |
| OPD batch shape | `num_prompts=1792`, `opd_prepare_batch_size=128`, `opd_microbatch_size=128`, `opd_prepare_concurrency=4`, `max_new_tokens=192` |
| Decision | Keep as the new current best. It improves steady useful throughput over the 1536-prompt t15 trial while remaining below the 2048-prompt trainer-queue regression. |

#### Profile Results

| Metric | Step 0 Warmup | Step 1 Steady |
| --- | ---: | ---: |
| Step total | 445.433s | 201.879s |
| Prepare batches | 14 | 14 |
| Trainer microbatches | 14 | 14 |
| Prepare concurrency | 4 | 4 |
| Prepare window | 79.709s | 73.947s |
| Sum of prepare batch times | 290.200s | 270.475s |
| Student sampling summed wall | 220.576s | 208.100s |
| Sampling output throughput, summed | 1550.7 tok/s | 1647.9 tok/s |
| Sampling output throughput, prepare-window | 4291.1 tok/s | 4637.4 tok/s |
| Teacher prefill summed wall | 69.432s | 62.351s |
| Teacher forward compute | 35.484s | 35.374s |
| Teacher cache write | 4.926s | 5.066s |
| Teacher prefill throughput, summed | 5529.7 tok/s | 6171.9 tok/s |
| Teacher prefill throughput, prepare-window | 4816.7 tok/s | 5204.0 tok/s |
| Trainer forward/backward | 209.677s | 168.063s |
| Optimizer queued wait | 327.700s | 118.206s |
| Optimizer step | 174.176s | 1.025s |
| P2P sync wall | 38.024s | 9.726s |
| P2P transfer time | 11.247s | 9.509s |
| Sync payload | 69.32 GB, 107 buckets | 69.32 GB, 107 buckets |
| Valid distillation tokens | 383,936 | 384,823 |
| Useful valid-token throughput | 861.9 tok/s | 1906.2 tok/s |
| Useful valid-token throughput / trainer GPU | 26.9 tok/s/GPU | 59.6 tok/s/GPU |
| Loss | 0.9958 | 0.6034 |

#### Notes

- The 1792-prompt shape improved steady useful throughput from the 1536-prompt t15 best (`1724.9` valid tok/s) to `1906.2` valid tok/s.
- Service-side utilization increased again: steady sampling reached `4637.4` output tok/s over the prepare window and teacher prefill reached `5204.0` input tok/s over the same window.
- The trainer is still the limiting stage, but this batch size amortizes the fixed sync and optimizer costs better than 1536 prompts without hitting the 2048-prompt fwd/bwd regression.

### 1920-Prompt, 128-Microbatch, Prepare-Concurrency-4 Trial

| Field | Value |
| --- | --- |
| Run ID | `opd-q35-c4-t17-1920p-20260516` |
| Manifest | Removed after nonwinning trial |
| Teacher prefill | Reused live `opd-q36-397b-4t4s-20260515-teacher-master` Qwen3.5-397B service on nodes `069`, `075` |
| Student sampler | Reused live `opd-q35-397b-mb128-t1d-20260516` Qwen3.5-35B-A3B Dispatch/SGLang deployment on nodes `013`, `014`, `040`, `055` |
| Student trainer | Qwen3.5-35B-A3B, XORL FSDP2/EP=8, 32 H100 on nodes `049`, `052`, `083`, `094`, `gradient_checkpointing_method=recompute_before_dispatch` |
| OPD batch shape | `num_prompts=1920`, `opd_prepare_batch_size=128`, `opd_microbatch_size=128`, `opd_prepare_concurrency=4`, `max_new_tokens=192` |
| Decision | Discard as nonwinning. The larger batch improved prepare-window service throughput but pushed trainer fwd/bwd back into the same regression seen at 2048 prompts. |

#### Profile Results

| Metric | Step 0 Warmup | Step 1 Steady |
| --- | ---: | ---: |
| Step total | 459.033s | 258.255s |
| Prepare batches | 15 | 15 |
| Trainer microbatches | 15 | 15 |
| Prepare concurrency | 4 | 4 |
| Prepare window | 81.434s | 81.651s |
| Sum of prepare batch times | 299.710s | 301.892s |
| Student sampling summed wall | 231.074s | 232.532s |
| Sampling output throughput, summed | 1587.7 tok/s | 1579.6 tok/s |
| Sampling output throughput, prepare-window | 4505.1 tok/s | 4498.4 tok/s |
| Teacher prefill summed wall | 68.549s | 68.912s |
| Teacher forward compute | 37.813s | 38.525s |
| Teacher cache write | 5.400s | 5.407s |
| Teacher prefill throughput, summed | 6008.0 tok/s | 5982.5 tok/s |
| Teacher prefill throughput, prepare-window | 5057.4 tok/s | 5049.2 tok/s |
| Trainer forward/backward | 223.433s | 223.590s |
| Optimizer queued wait | 339.044s | 166.544s |
| Optimizer step | 173.625s | 1.030s |
| P2P sync wall | 38.556s | 10.061s |
| P2P transfer time | 10.984s | 9.834s |
| Sync payload | 69.32 GB, 107 buckets | 69.32 GB, 107 buckets |
| Valid distillation tokens | 411,842 | 412,267 |
| Useful valid-token throughput | 897.2 tok/s | 1596.4 tok/s |
| Useful valid-token throughput / trainer GPU | 28.0 tok/s/GPU | 49.9 tok/s/GPU |
| Loss | 0.9981 | 0.6045 |

#### Notes

- The 1920-prompt shape regressed versus the 1792-prompt t16 best (`1596.4` vs. `1906.2` valid tok/s).
- The sampler and teacher stayed healthy, but the trainer fwd/bwd term grew from `168.1s` at 1792 prompts to `223.6s`.
- This brackets the prompt-count peak for the current 32-GPU trainer and 16-pod sampler setup: 1792 prompts is the best measured point, while 1920 and 2048 prompts overload the trainer queue.

### 64-GPU Mixed-Node Trainer Scale Attempts

| Field | Value |
| --- | --- |
| Runs | `opd-q35-64g-mixed-c4-t18-1792p-20260516`, `opd-q35-64g-16x4-c4-t19-1792p-20260516` through `t23` |
| Teacher prefill | Reused live `opd-q36-397b-4t4s-20260515-teacher-master` Qwen3.5-397B service |
| Student sampler | Reused live `opd-q35-397b-mb128-t1d-20260516` Qwen3.5-35B-A3B Dispatch/SGLang deployment |
| Intended student trainer | Qwen3.5-35B-A3B, 64 FSDP2 ranks, EP=8, `gradient_checkpointing_method=recompute_before_dispatch` |
| OPD batch shape | `num_prompts=1792`, `opd_prepare_batch_size=128`, `opd_microbatch_size=128`, `opd_prepare_concurrency=4`, `max_new_tokens=192` |
| Decision | Discard. No valid profile row was produced. Generated 64-GPU manifests and the trial-only 64-GPU config were removed. |

#### Findings

- The 1920-prompt fwd/bwd jump was real trainer-side time, not teacher or sampler time. The 1792-prompt t16 server chunks were mostly around `11.7-12.9s`; the 1920-prompt t17 chunks were mostly around `14.1-16.0s` at similar per-chunk token counts, plus one extra chunk.
- We are not limited to `node-group=nccl`, but the namespace admission controller injects `node-group=default` when a pod has no node selector. Mixed-node manifests therefore need an explicit per-pod selector. Unlabeled nodes cannot be used this way without relabeling or changing admission behavior.
- There were only five full 8-GPU nodes labeled `default` or `nccl` after cleanup (`036`, `122`, `049`, `052`, `083`), so an 8x8 64-rank run was not currently schedulable with the allowed labels.
- The 16x4 layout can be scheduled on labeled nodes, but partial default nodes exposed all GPUs in privileged containers. The Kubernetes/CDI assignment did not reliably correspond to physically free GPUs; nodes `071` and `105` assigned busy `0..3` while `4..7` were actually free.
- Explicitly pinning free physical GPUs got past scheduling, model load, sampling, and teacher prefill, but fwd/bwd failed in DeepEP with `CUDA invalid resource handle` from `deep_ep.cpp:113`. This makes the 16x4 partial-node path unsuitable for a throughput comparison unless DeepEP/device assignment is fixed or we intentionally test a non-DeepEP dispatch path.

#### Trial Outcomes

| Run | Outcome |
| --- | --- |
| `t18` 8x8 mixed nodes | Failed scheduling: admission injected `node-group=default` into pods without a selector, causing `NodeAffinity` failures on nccl/unlabeled nodes. |
| `t19` 16x4 | Failed CUDA OOM: inherited `CUDA_VISIBLE_DEVICES=0,1,2,3` and used busy physical GPUs on partial nodes. |
| `t20` 16x4 | Failed CUDA OOM: removing `CUDA_VISIBLE_DEVICES` was insufficient because this cluster exposes `NVIDIA_VISIBLE_DEVICES` as a CDI path and PyTorch still saw all physical GPUs. |
| `t21` 16x4 | Failed CUDA OOM: CDI-derived mapping assigned busy GPUs on node `071`. |
| `t22` 16x4 | Failed CUDA OOM: replacing `071` with `105` hit the same busy-GPU assignment pattern. |
| `t23` 16x4 | Reached the OPD loop, but fwd/bwd failed with DeepEP `invalid resource handle`; no profile row was written. |
| `t24` 8x8 labeled full nodes | Failed kubelet admission before user code. A concurrent `er-p2prun-05161945` encoded-reasoning run consumed several selected full nodes between capacity check and apply; admitted pods reported `UnexpectedAdmissionError` such as `Requested: 8, Available: 4` or `Available: 0`. Jobs/service were deleted and the generated t24 manifest plus trial-only 8-node config were discarded. |

#### 2026-05-16 19:46 UTC Retry Note

- Attempted clean 8x8 64-rank OPD run `opd-q35-64g-8x8-c4-t24-1792p-20260516` on labeled `default`/`nccl` nodes `049`, `052`, `096`, `001`, `036`, `065`, `089`, `122`.
- The server dry-run passed and the manifest had explicit per-pod `node-group` selectors, avoiding the earlier unlabeled-node admission problem.
- The live cluster changed during submission: `er-p2prun-05161945` started and occupied `001`, `036`, `065`, and `089`; a separate 4-GPU job also occupied part of `052` at admission time.
- Result: no OPD profile row. This is not evidence about trainer throughput; it is a cluster bin-packing/admission race.
- Current active `dsv4-e2e` pods were not found in the `apanda` namespace during this retry. Pod affinity can improve future 4-GPU pod packing, but it will only affect newly created pods and cannot repack already-running pods.

#### 2026-05-16 20:01 UTC 16x4 Retry Note

- Attempted 16-pod / 64-rank OPD run `opd-q35-64g-16x4-c4-t25-1792p-20260516` on labeled `default` and `nccl` nodes, using two 4-GPU pods on full nodes where possible and physical GPU pins on partial nodes.
- The job got past startup, model load, student sampling, teacher prefill, and forward/backward submission. The profile file stayed empty because the failure happened during the warm trainer step before the client could flush a row.
- Server-side failure was DeepEP, not OPD prefill: `DeepEP error: CPU recv timeout` followed by cross-rank `invalid device context` and `invalid resource handle` errors from `deep_ep.cpp:113`.
- Decision: discard the 16x4 layout for the 64-GPU throughput comparison. It is schedulable but not a valid DeepEP topology in this cluster shape. Use full 8-GPU trainer pods on clean labeled `default`/`nccl` nodes instead.

#### 2026-05-16 20:11 UTC 8x8 Node-Start Retry Note

- Attempted clean 8x8 / 64-rank OPD run `opd-q35-64g-8x8-c4-t26-1792p-20260516` on nodes `049`, `052`, `096`, `112`, `046`, `001`, `089`, and `122`.
- Seven pods started quickly, but worker 3 on node `112` stayed Pending after volume attach and never ran user code.
- Node `112` subsequently showed `unschedulable=true` and lacked `topology.csi.weka.io/accessible=true`, so this was a node-start issue rather than trainer throughput evidence.
- Jobs/service were deleted, the stuck `112` pod was force-deleted, and the generated t26 manifest was discarded. The replacement t27 run avoids `112`.

#### 2026-05-16 20:15 UTC 8x8 Runtime Retry Note

- Attempted clean 8x8 / 64-rank OPD run `opd-q35-64g-8x8-c4-t27-1792p-20260516` on full-node trainer pods, using the older packed sampler deployment on nodes `013`, `014`, `040`, and `055`.
- The run reached the OPD loop and submitted warm-step forward/backward work. The server log shows several successful chunks, including the cold compiled chunk at `84.6s` and subsequent chunks around `6-8s`.
- No profile row was written. The job was terminated while the client was still polling `retrieve_future`, before optimizer/sync completed.
- Decision: discard for throughput comparison. It is useful only as evidence that the 8x8 full-node trainer can start and enter the OPD loop when enough clean full nodes are available.

#### 2026-05-16 20:29 UTC 8x8 Dynamic Retry Note

- Attempted `opd-q35-64g-8x8-dyn-t28-1792p`, another 8x8 / 64-rank trainer run.
- The head launcher spawned torchrun and then waited for the rank-0 address file. The job was terminated after about 30 seconds with no OPD profile row and no useful throughput evidence.
- Decision: discard. This run did not reach trainer readiness.

#### 2026-05-16 21:11 UTC 8x8 One-Pod-Per-Node Sampler Retry Note

- Attempted `opd-q35-64g-dyn-t29-1792p`, using the new one-pod-per-node sampler deployment and the refreshed Qwen3.5-397B teacher service.
- The trainer manifest targeted nodes `049`, `052`, `063`, `064`, `067`, `087`, `088`, and `096`. Head `049`, worker `052`, and worker `096` admitted, but several selected nodes rejected 8-GPU pods at kubelet admission with only 4 GPUs available.
- The head launcher waited for rank-0 rendezvous and was terminated without a profile row.
- Decision: discard. This was a cluster capacity/admission failure, not an OPD implementation failure.

### One-Pod-Per-Node Sampler Rescue Trial

| Field | Value |
| --- | --- |
| Run ID | `opd-q35-c4-t30-1792p-1ppn` |
| Manifest | `experiments/opd_profile/k8s/generated/opd-q35-c4-t30-1792p-1ppn-trainer-only.yaml` |
| Teacher prefill | Reused refreshed `opd-q36-397b-4t4s-20260515-teacher-master` Qwen3.5-397B service on nodes `075`, `077` |
| Student sampler | Reused live `opd-q35-1ppn-s1-20260516` Qwen3.5-35B-A3B Dispatch/SGLang deployment with 16 TP2 pods reserved one per 8-GPU node |
| Student trainer | Qwen3.5-35B-A3B, XORL FSDP2/EP=8, 32 H100 on nodes `049`, `052`, `096`, `069`, `gradient_checkpointing_method=recompute_before_dispatch` |
| OPD batch shape | `num_prompts=1792`, `opd_prepare_batch_size=128`, `opd_microbatch_size=128`, `opd_prepare_concurrency=4`, `max_new_tokens=192` |
| Decision | Keep as successful one-pod-per-node sampler evidence, but not as the current best. It is slower than t16 on the same OPD shape. |

#### Profile Results

| Metric | Step 0 Warmup | Step 1 Steady |
| --- | ---: | ---: |
| Step total | 548.989s | 216.510s |
| Prepare batches | 14 | 14 |
| Trainer microbatches | 14 | 14 |
| Prepare concurrency | 4 | 4 |
| Prepare window | 174.885s | 79.487s |
| Sum of prepare batch times | 671.918s | 291.891s |
| Student sampling summed wall | 315.080s | 216.721s |
| Sampling output throughput, summed | 1091.9 tok/s | 1581.8 tok/s |
| Sampling output throughput, prepare-window | 1967.2 tok/s | 4312.7 tok/s |
| Teacher prefill summed wall | 356.815s | 75.146s |
| Teacher forward compute | 105.339s | 37.247s |
| Teacher cache write | 5.038s | 4.847s |
| Teacher prefill throughput, summed | 1081.6 tok/s | 5119.4 tok/s |
| Teacher prefill throughput, prepare-window | 2206.8 tok/s | 4839.8 tok/s |
| Trainer forward/backward | 203.236s | 180.071s |
| Optimizer queued wait | 323.976s | 126.703s |
| Optimizer step | 175.127s | 1.027s |
| P2P sync wall | 50.128s | 10.320s |
| P2P transfer time | 11.678s | 10.071s |
| Sync payload | 69.32 GB, 107 buckets | 69.32 GB, 107 buckets |
| Valid distillation tokens | 385,936 | 384,704 |
| Useful valid-token throughput | 703.0 tok/s | 1776.8 tok/s |
| Useful valid-token throughput / trainer GPU | 22.0 tok/s/GPU | 55.5 tok/s/GPU |
| Loss | 0.7398 | 0.7371 |

#### Notes

- The one-pod-per-node sampler deployment avoided the TP2 colocation OOM class and completed the OPD loop with P2P sync to all 16 sampler endpoints.
- Steady P2P sync was healthy: `69.32 GB` transferred in `10.07s` transfer time, `10.32s` client wall.
- Throughput regressed versus the t16 packed-sampler/current-best row (`1776.8` vs. `1906.2` valid tok/s). The regression is spread across sampling (`216.7s` vs. `208.1s`), teacher prefill (`75.1s` vs. `62.4s`), and fwd/bwd (`180.1s` vs. `168.1s`).
- The t30 trainer jobs were deleted after recording to free nodes. The refreshed teacher and one-pod-per-node sampler deployments were left running for reuse.

### Dispatch Routing Fix Sampler Benchmark

| Field | Value |
| --- | --- |
| Run ID | `opd-q35-1ppn-s1-20260516-dispatch-routing-fix-bench` |
| Script | `experiments/opd_profile/scripts/dispatch_sampling_benchmark.py` |
| Result artifact | `experiments/opd_profile/results/qwen35_35b_from_qwen35_397b/opd-q35-1ppn-s1-20260516/sampling_bench/dispatch_routing_fix_2048.jsonl` |
| Student sampler | Reused `opd-q35-1ppn-s1-20260516` Qwen3.5-35B-A3B one-pod-per-node SGLang deployment |
| Dispatch change | Restarted only the Dispatch pod after patching new-session routing to count current REASONING/in-flight load instead of the recent-session window alone |
| Shape | 2,048 requests, concurrency 1,024, `max_tokens=192`, `logprobs=true`, `prompt_mode=opd`, no client keepalive |
| Result | 2,048/2,048 success, 0 errors, 392,089 completion tokens in 24.679s, 15,887.6 completion tok/s |
| Backend balance | Exactly 128 requests per backend across all 16 SGLang pods |

#### Notes

- This fixes the skew seen in the live encoded-reasoning run where a slow backend could receive a disproportionate share of new requests after long in-flight calls aged out of the 120s active-session window.
- The sampler service now has enough standalone throughput for the OPD 1,792-prompt shape; the next useful measurement is end-to-end OPD with the same trainer shape as t30, recorded as `opd-q35-c4-t31-1792p-dfix`.

### Dispatch Pending-Load Sampler Benchmark

| Field | Value |
| --- | --- |
| Run ID | `opd-q35-1ppn-s1-20260516-dispatch-pending-retry-fix-bench` |
| Script | `experiments/opd_profile/scripts/dispatch_sampling_benchmark.py` |
| Result artifact | `experiments/opd_profile/results/qwen35_35b_from_qwen35_397b/opd-q35-1ppn-s1-20260516/sampling_bench/dispatch_pending_retry_fix_4096.{csv,jsonl}` |
| Student sampler | Reused `opd-q35-1ppn-s1-20260516` Qwen3.5-35B-A3B one-pod-per-node SGLang deployment |
| Dispatch change | Restarted only the Dispatch pod after adding pending-request accounting to backend admission/routing and using live routing-load counts for retries |
| Shape | 4,096 requests, concurrency 1,024, `max_tokens=192`, `logprobs=true`, `prompt_mode=opd`, no client keepalive |
| Result | 4,096/4,096 success, 0 errors, 782,705 completion tokens in 47.747s, 16,392.7 completion tok/s |
| Backend balance | 255-258 requests per backend across all 16 SGLang pods |

#### Notes

- This is the strongest current sampler preflight for OPD: it is twice the previous benchmark size and remained balanced without `httpx.ReadError`, semaphore timeout, or retry skew.
- The next end-to-end OPD trainer run should use this restarted Dispatch pod, while preserving the existing teacher and sampler SGLang services.

### Dispatch Routing Fix End-To-End OPD Trial

| Field | Value |
| --- | --- |
| Run ID | `opd-q35-c4-t31-1792p-dfix` |
| Manifest | `experiments/opd_profile/k8s/generated/opd-q35-c4-t31-1792p-dfix-trainer-only.yaml` |
| Teacher prefill | Reused `opd-q36-397b-4t4s-20260515-teacher-master` Qwen3.5-397B service on nodes `075`, `077` |
| Student sampler | Reused `opd-q35-1ppn-s1-20260516` Qwen3.5-35B-A3B Dispatch/SGLang deployment after restarting Dispatch with routing-load accounting fix |
| Student trainer | Qwen3.5-35B-A3B, XORL FSDP2/EP=8, 32 H100 on nodes `049`, `052`, `096`, `069`, `gradient_checkpointing_method=recompute_before_dispatch` |
| OPD batch shape | `num_prompts=1792`, `opd_prepare_batch_size=128`, `opd_microbatch_size=128`, `opd_prepare_concurrency=4`, `max_new_tokens=192` |
| Decision | Keep as successful Dispatch-fix end-to-end evidence. It is not a new best because trainer fwd/bwd remained slower than t16. |

#### Profile Results

| Metric | Step 0 Warmup | Step 1 Steady |
| --- | ---: | ---: |
| Step total | 441.299s | 216.812s |
| Prepare batches | 14 | 14 |
| Trainer microbatches | 14 | 14 |
| Prepare concurrency | 4 | 4 |
| Prepare window | 76.858s | 75.067s |
| Student sampling summed wall | 215.357s | 209.059s |
| Sampling output throughput, summed | 1589.3 tok/s | 1636.7 tok/s |
| Sampling output throughput, prepare-window | 4453.2 tok/s | 4558.3 tok/s |
| Teacher prefill summed wall | 67.243s | 64.608s |
| Teacher forward compute | 35.604s | 34.713s |
| Teacher cache write | 4.986s | 5.100s |
| Teacher prefill throughput, summed | 5713.0 tok/s | 5944.7 tok/s |
| Teacher prefill throughput, prepare-window | 4998.3 tok/s | 5116.4 tok/s |
| Trainer forward/backward | 203.250s | 182.614s |
| Optimizer queued wait | 325.333s | 131.938s |
| Optimizer step | 175.164s | 1.586s |
| P2P sync wall | 39.109s | 9.806s |
| P2P transfer time | 11.561s | 9.564s |
| Sync payload | 69.32 GB, 107 buckets | 69.32 GB, 107 buckets |
| Valid distillation tokens | 384,160 | 384,075 |
| Useful valid-token throughput | 870.5 tok/s | 1771.5 tok/s |
| Useful valid-token throughput / trainer GPU | 27.2 tok/s/GPU | 55.4 tok/s/GPU |
| Loss | 0.9955 | 0.6017 |

#### Notes

- The Dispatch routing fix did what it was supposed to do for OPD sampling: the steady sampling wall improved to `209.1s`, close to the t16 current-best `208.1s`.
- Teacher prefill stayed comfortably overlapped and fast at `64.6s`, with `34.7s` of teacher forward compute and `5.1s` of hidden-cache write time.
- P2P sync stayed healthy: `69.32 GB` transferred in `9.56s` transfer time, `9.81s` client wall.
- This did not beat t16 because trainer fwd/bwd was `182.6s` versus t16's `168.1s`; the next OPD throughput work should focus on the trainer path rather than the teacher prefill service.

### Pending-Load Dispatch End-To-End OPD Trial

| Field | Value |
| --- | --- |
| Run ID | `opd-q35-c4-t34-1792p-pfix` |
| Manifest | `experiments/opd_profile/k8s/generated/opd-q35-c4-t34-1792p-pfix-trainer-only.yaml` |
| Teacher prefill | Reused `opd-q36-397b-4t4s-20260515-teacher-master` Qwen3.5-397B service on nodes `075`, `077` |
| Student sampler | Reused `opd-q35-1ppn-s1-20260516` Qwen3.5-35B-A3B Dispatch/SGLang deployment after restarting Dispatch with pending-request and retry-load accounting fixes |
| Student trainer | Qwen3.5-35B-A3B, XORL FSDP2/EP=8, 32 H100 on nodes `049`, `052`, `096`, `098`, `gradient_checkpointing_method=recompute_before_dispatch` |
| OPD batch shape | `num_prompts=1792`, `opd_prepare_batch_size=128`, `opd_microbatch_size=128`, `opd_prepare_concurrency=4`, `max_new_tokens=192` |
| Decision | Discard as non-winning. It confirms the pending-load Dispatch path works end-to-end, but it is slower than t16 and t31 and keeps the overprovisioned 16-pod TP2 sampler fleet. |

#### Profile Results

| Metric | Step 0 Warmup | Step 1 Steady |
| --- | ---: | ---: |
| Step total | 445.971s | 221.654s |
| Prepare batches | 14 | 14 |
| Trainer microbatches | 14 | 14 |
| Prepare concurrency | 4 | 4 |
| Prepare window | 78.766s | 75.243s |
| Student sampling summed wall | 217.640s | 212.285s |
| Sampling output throughput, summed | 1575.1 tok/s | 1611.6 tok/s |
| Sampling output throughput, prepare-window | 4352.3 tok/s | 4546.8 tok/s |
| Teacher prefill summed wall | 68.737s | 66.823s |
| Teacher forward compute | 35.502s | 34.807s |
| Teacher cache write | 4.899s | 5.076s |
| Teacher prefill throughput, summed | 5596.9 tok/s | 5746.7 tok/s |
| Teacher prefill throughput, prepare-window | 4884.3 tok/s | 5103.6 tok/s |
| Trainer forward/backward | 205.412s | 187.519s |
| Optimizer queued wait | 324.451s | 136.573s |
| Optimizer step | 174.277s | 1.023s |
| P2P sync wall | 42.755s | 9.838s |
| P2P transfer time | 11.324s | 9.597s |
| Sync payload | 69.32 GB, 107 buckets | 69.32 GB, 107 buckets |
| Valid distillation tokens | 384,712 | 384,008 |
| Useful valid-token throughput | 862.6 tok/s | 1732.5 tok/s |
| Useful valid-token throughput / trainer GPU | 27.0 tok/s/GPU | 54.1 tok/s/GPU |
| Loss | 0.9939 | 0.5990 |

#### Notes

- The pending-load Dispatch fix stayed correct under the full OPD loop; both OPD steps completed and P2P sync returned `success=true`.
- Warm P2P sync was healthy: `69.32 GB` transferred in `9.60s` transfer time, `9.84s` client wall.
- This was slower than both t16 (`1906.2` valid tok/s) and t31 (`1771.5` valid tok/s), mostly because trainer fwd/bwd rose to `187.5s`.
- The trial also exposed an inefficient sampler deployment: `opd-q35-1ppn-s1` reserves 16 full 8-GPU nodes but each SGLang pod launches `--tp-size 2`. The next sampler experiment should keep `inference_world_size=32` while using 4 one-pod-per-node `--tp-size 8` SGLang pods.
- The trainer jobs were deleted after recording, and the generated t34 manifest was discarded.

### TP8 Sampler Trial

| Field | Value |
| --- | --- |
| Run ID | `opd-q35-c4-t35-1792p-tp8s` |
| Sampler manifest | `experiments/opd_profile/k8s/generated/opd-q35-tp8-s1-20260516-sampler-dispatch.yaml` |
| Trainer manifest | `experiments/opd_profile/k8s/generated/opd-q35-c4-t35-1792p-tp8s-trainer-only.yaml` |
| Teacher prefill | Reused `opd-q36-397b-4t4s-20260515-teacher-master` Qwen3.5-397B service on nodes `075`, `077` |
| Student sampler | Qwen3.5-35B-A3B, 4 SGLang pods x TP8 on nodes `036`, `044`, `048`, `051` |
| Student trainer | Qwen3.5-35B-A3B, XORL FSDP2/EP=8, 32 H100 on nodes `049`, `052`, `096`, `069` |
| OPD batch shape | `num_prompts=1792`, `opd_prepare_batch_size=128`, `opd_microbatch_size=128`, `opd_prepare_concurrency=4`, `max_new_tokens=192` |
| Decision | Discard. It loaded and reached the OPD sync path, but P2P failed during small-entry transfer and sampler preflight throughput was lower than TP2 layouts. |

#### Profile Results

| Metric | Step 0 Warmup |
| --- | ---: |
| Step total | 442.574s |
| Prepare batches | 14 |
| Trainer microbatches | 14 |
| Prepare concurrency | 4 |
| Prepare window | 76.968s |
| Student sampling summed wall | 216.556s |
| Sampling output throughput, prepare-window | 4467.8 tok/s |
| Teacher prefill summed wall | 64.894s |
| Teacher forward compute | 32.886s |
| Teacher cache write | 5.101s |
| Trainer forward/backward | 202.312s |
| Optimizer queued wait | 324.720s |
| Optimizer step | 175.463s |
| P2P sync wall | 40.885s |
| Sync success | false |
| Valid distillation tokens | 385,779 |
| Useful valid-token throughput | 871.7 tok/s |
| Loss | 0.7408 |

#### Notes

- The failure was not in teacher prefill or trainer fwd/bwd. The warm step reached P2P sync, then Mooncake returned `ret=-1` on a small-entry transfer to `10.42.43.47:15589`.
- The shape also lost sampler throughput versus TP2: the TP8 sampler preflight was around `5.3k` completion tok/s, while packed TP2 later measured `7.9k` completion tok/s using the same 32 sampler GPUs.
- The TP8 sampler and trainer Kubernetes resources were deleted, and the generated trial manifests were removed after recording.

### Packed TP2 Sampler Trial

| Field | Value |
| --- | --- |
| Run ID | `opd-q35-c4-t36-1792p-pack4` |
| Sampler manifest | `experiments/opd_profile/k8s/generated/opd-q35-tp2-pack4-s1-20260516-sampler-dispatch.yaml` |
| Trainer manifest | `experiments/opd_profile/k8s/generated/opd-q35-c4-t36-1792p-pack4-trainer-only.yaml` |
| Teacher prefill | Reused `opd-q36-397b-4t4s-20260515-teacher-master` Qwen3.5-397B service on nodes `075`, `077` |
| Student sampler | Qwen3.5-35B-A3B, 16 SGLang pods x TP2 packed 4 per clean node on `036`, `044`, `048`, `051` |
| Student trainer | Qwen3.5-35B-A3B, XORL FSDP2/EP=8, 32 H100 on nodes `040`, `053`, `058`, `060` |
| OPD batch shape | `num_prompts=1792`, `opd_prepare_batch_size=128`, `opd_microbatch_size=128`, `opd_prepare_concurrency=4`, `max_new_tokens=192` |
| Decision | Keep. This fixes the root sampler over-reservation problem: 32 sampler GPUs are requested and used, instead of reserving 128 GPUs for 16 TP2 pods. |

#### Sampler Preflight

| Metric | Value |
| --- | ---: |
| Requests | 2,048 |
| Concurrency | 1,024 |
| Success / errors | 2,048 / 0 |
| Completion tokens | 393,206 |
| Elapsed | 49.960s |
| Completion tok/s | 7,870.5 |
| Total tok/s | 8,873.1 |
| Backend request min/max | 128 / 128 |
| Latency p50 / p95 / p99 | 12.746s / 38.637s / 38.839s |

#### Profile Results

| Metric | Step 0 Warmup | Step 1 Steady |
| --- | ---: | ---: |
| Step total | 438.030s | 203.278s |
| Prepare batches | 14 | 14 |
| Trainer microbatches | 14 | 14 |
| Prepare concurrency | 4 | 4 |
| Prepare window | 75.211s | 76.704s |
| Sum of prepare batch times | 274.697s | 280.431s |
| Student sampling summed wall | 212.726s | 215.805s |
| Sampling output throughput, summed | 1616.8 tok/s | 1585.0 tok/s |
| Sampling output throughput, prepare-window | 4572.8 tok/s | 4459.5 tok/s |
| Teacher prefill summed wall | 61.949s | 64.604s |
| Teacher forward compute | 32.886s | 34.596s |
| Teacher cache write | 5.027s | 5.137s |
| Teacher prefill throughput, summed | 6228.1 tok/s | 5943.3 tok/s |
| Teacher prefill throughput, prepare-window | 5129.9 tok/s | 5005.7 tok/s |
| Trainer forward/backward | 199.464s | 169.322s |
| Optimizer queued wait | 320.868s | 118.340s |
| Optimizer step | 173.458s | 1.021s |
| P2P sync wall | 41.951s | 8.233s |
| P2P transfer time | 10.363s | 7.993s |
| Sync payload | 69.32 GB, 107 buckets | 69.32 GB, 107 buckets |
| Valid distillation tokens | 385,825 | 383,959 |
| Useful valid-token throughput | 880.8 tok/s | 1888.8 tok/s |
| Useful valid-token throughput / trainer GPU | 27.5 tok/s/GPU | 59.0 tok/s/GPU |
| Loss | 0.7479 | 0.7383 |

#### Notes

- This validates the root fix for the inefficient `opd-q35-1ppn-s1` workaround. Four TP2 SGLang pods can be colocated per clean 8-GPU node when each pod is pinned to a disjoint physical GPU pair with `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- The working physical GPU pairs are `0,1`, `2,3`, `4,5`, and `6,7`. The matching Mooncake HCA maps are `{"0":"mlx5_2","1":"mlx5_3"}`, `{"0":"mlx5_1","1":"mlx5_5"}`, `{"0":"mlx5_9","1":"mlx5_9"}`, and `{"0":"mlx5_6","1":"mlx5_5"}`.
- P2P sync succeeded to all 16 packed TP2 endpoints. The steady row synced `69.32 GB` in `7.99s` transfer time and `8.23s` client wall.
- Throughput is slightly below t16's historical best (`1888.8` vs `1906.2` valid tok/s), but it is materially better than the one-pod-per-node rescue runs in GPU efficiency and near parity in useful distillation throughput.
- Keep the packed TP2 sampler manifest and use it as the practical default for OPD. Avoid packed TP2 on partial nodes; partial-node failures were caused by privileged/CDI GPU visibility and unreliable physical GPU assignment.

### Qwen3.6 FP8 Student Smoke With Qwen3.5 397B Teacher

| Field | Value |
| --- | --- |
| Run ID | `opd-q36fp8-q35t-05170523` |
| Manifest | `experiments/opd_profile/k8s/generated/opd-q36fp8-q35t-05170523.yaml` |
| Teacher prefill | Reused `opd-q36-397b-4t4s-20260515-teacher-master` Qwen3.5-397B-A17B service on nodes `075`, `077` |
| Student sampler | Qwen3.6-35B-A3B SGLang FP8 receivers, 16 TP2 pods behind Dispatch on nodes `064`, `067`, `087`, `088`, `098`, `099`, `110` |
| Student trainer | Qwen3.6-35B-A3B BF16 XORL FSDP2/EP=8, 32 H100 on nodes `061`, `089`, `122`, `125` |
| OPD batch shape | `num_prompts=128`, `opd_prepare_batch_size=32`, `opd_microbatch_size=32`, `opd_prepare_concurrency=1`, `max_new_tokens=192` |
| Decision | Keep as a correctness smoke for Qwen3.6 BF16 training plus FP8 sampler sync. Do not compare throughput against the tuned Qwen3.5 1792-prompt packed-TP2 runs because this shape intentionally under-saturates the sampler fleet. |

#### Profile Results

| Metric | Step 0 Warmup | Step 1 Steady |
| --- | ---: | ---: |
| Step total | 387.634s | 181.692s |
| Prepare batches | 4 | 4 |
| Trainer microbatches | 4 | 4 |
| Prepare concurrency | 1 | 1 |
| Prepare window | 90.759s | 63.734s |
| Sum of prepare batch times | 90.758s | 63.732s |
| Student sampling summed wall | 87.737s | 60.764s |
| Sampling output throughput, summed | 280.1 tok/s | 399.9 tok/s |
| Sampling output throughput, prepare-window | 270.8 tok/s | 381.3 tok/s |
| Teacher prefill summed wall | 3.020s | 2.967s |
| Teacher forward compute | 2.460s | 2.433s |
| Teacher cache write | 0.271s | 0.235s |
| Teacher prefill throughput, summed | 9074.9 tok/s | 9143.5 tok/s |
| Teacher prefill throughput, prepare-window | 302.0 tok/s | 425.7 tok/s |
| Trainer forward/backward | 86.160s | 59.671s |
| Optimizer queued wait | 203.057s | 13.134s |
| Optimizer step | 163.701s | 1.914s |
| P2P sync wall | 93.819s | 104.824s |
| P2P transfer time | 39.941s | 27.799s |
| Sync payload | 67.91 GB, 107 buckets | 67.91 GB, 107 buckets |
| Valid distillation tokens | 27,410 | 27,133 |
| Useful valid-token throughput | 70.7 tok/s | 149.3 tok/s |
| Loss | 0.6011 | 0.4014 |

#### Notes

- This validates the requested model pairing: Qwen3.5-397B-A17B teacher, Qwen3.6-35B-A3B BF16 trainer, and Qwen3.6-35B-A3B FP8 SGLang receivers synced from the BF16 trainer with `{"quant_method":"fp8","fmt":"e4m3","weight_block_size":[16,128]}`.
- FP8 P2P sync succeeded to all 16 SGLang endpoints. The steady row transferred `67.91 GB` in `27.80s`; the full sync handler still took `104.82s` because backend init/setup dominated this fresh two-step smoke.
- Teacher prefill is not the bottleneck here: steady teacher prefill was `2.97s` for `27,133` valid tokens. The small `128`-prompt shape and `opd_prepare_concurrency=1` make sampler throughput artificially low.
- The smoke resources were deleted after recording. The result files remain under `experiments/opd_profile/results/qwen36_35b_fp8_from_qwen35_397b/opd-q36fp8-q35t-05170523/`.

### Qwen3.6 FP8 Student 1792-Prompt Packed-TP2 Trial

| Field | Value |
| --- | --- |
| Run ID | `opd-q36fp8-q35t-c4-1792p-05170545` |
| Manifest | `experiments/opd_profile/k8s/generated/opd-q36fp8-q35t-c4-1792p-05170545.yaml` |
| Teacher prefill | Reused `opd-q36-397b-4t4s-20260515-teacher-master` Qwen3.5-397B-A17B service on nodes `075`, `077` |
| Student sampler | Qwen3.6-35B-A3B SGLang FP8 receivers, 16 TP2 pods packed four per node on `001`, `046`, `061`, `089` with disjoint physical GPU pairs |
| Student trainer | Qwen3.6-35B-A3B BF16 XORL FSDP2/EP=8, 32 H100 on nodes `100`, `122`, `125`, `096` |
| OPD batch shape | `num_prompts=1792`, `opd_prepare_batch_size=128`, `opd_microbatch_size=128`, `opd_prepare_concurrency=4`, `max_new_tokens=192` |
| Decision | Keep as the successful larger-shape Qwen3.6 BF16-to-FP8 OPD profile. It validates the requested model pair, but it is not the current throughput winner versus the Qwen3.5 t36 packed-TP2 run. |

#### Profile Results

| Metric | Step 0 Warmup | Step 1 Steady |
| --- | ---: | ---: |
| Step total | 519.921s | 358.447s |
| Prepare batches | 14 | 14 |
| Trainer microbatches | 14 | 14 |
| Prepare concurrency | 4 | 4 |
| Prepare window | 118.013s | 92.402s |
| Sum of prepare batch times | 437.781s | 335.318s |
| Student sampling summed wall | 369.521s | 268.446s |
| Sampling output throughput, summed | 929.2 tok/s | 1273.3 tok/s |
| Sampling output throughput, prepare-window | 2909.4 tok/s | 3699.3 tok/s |
| Teacher prefill summed wall | 68.238s | 66.825s |
| Teacher forward compute | 34.864s | 34.742s |
| Teacher cache write | 5.328s | 5.367s |
| Teacher prefill throughput, summed | 5645.6 tok/s | 5742.1 tok/s |
| Teacher prefill throughput, prepare-window | 3264.4 tok/s | 4152.7 tok/s |
| Trainer forward/backward | 205.190s | 226.571s |
| Optimizer queued wait | 303.010s | 163.275s |
| Optimizer step | 164.924s | 1.019s |
| P2P sync wall | 98.899s | 102.771s |
| P2P transfer time | 41.280s | 28.449s |
| Sync payload | 67.91 GB, 107 buckets | 67.91 GB, 107 buckets |
| Valid distillation tokens | 385,246 | 383,715 |
| Useful valid-token throughput | 741.0 tok/s | 1070.5 tok/s |
| Useful valid-token throughput / trainer GPU | 23.2 tok/s/GPU | 33.5 tok/s/GPU |
| Loss | 0.6166 | 0.3938 |

#### Notes

- This validates the requested larger OPD pairing: Qwen3.5-397B-A17B teacher, Qwen3.6-35B-A3B BF16 trainer, and Qwen3.6-35B-A3B FP8 SGLang receivers synced from the BF16 trainer.
- FP8 P2P sync succeeded for both steps. The steady row transferred `67.91 GB` to 16 endpoints in `28.45s`, but full sync wall stayed high at `102.77s`.
- Teacher prefill is not the bottleneck: steady teacher prefill was `66.83s` summed across 14 prepare batches and `4152.7 tok/s` over the concurrent prepare window.
- The main bottlenecks are trainer forward/backward (`226.57s` steady), optimizer future/queue wait (`163.27s` steady), and sync handler setup. Server logs showed `cached_prepare=False` and about `70.03s` of backend init/setup on the steady sync despite `XORL_P2P_BACKEND_CACHE=1`.
- Resources were deleted after recording. The result files remain under `experiments/opd_profile/results/qwen36_35b_fp8_from_qwen35_397b/opd-q36fp8-q35t-c4-1792p-05170545/`.

### Qwen3.6 FP8 Student 200-Step Continuation

| Field | Value |
| --- | --- |
| Run ID | `opd-q36fp8-q35t-c4-1792p-long-05170618` |
| Manifest | `experiments/opd_profile/k8s/generated/opd-q36fp8-q35t-c4-1792p-long-05170618.yaml` |
| Teacher prefill | Reused `opd-q36-397b-4t4s-20260515-teacher-master` Qwen3.5-397B-A17B service on nodes `075`, `077` |
| Student sampler | Qwen3.6-35B-A3B SGLang FP8 receivers, 16 TP2 pods packed four per node on `001`, `046`, `061`, `089` |
| Student trainer | Qwen3.6-35B-A3B BF16 XORL FSDP2/EP=8, 32 H100 on nodes `100`, `122`, `125`, `096` |
| OPD batch shape | `num_steps=200`, `num_prompts=1792`, `opd_prepare_batch_size=128`, `opd_microbatch_size=128`, `opd_prepare_concurrency=4`, `max_new_tokens=192` |
| Decision | Keep running. This is the live long continuation of the validated Qwen3.5-teacher/Qwen3.6-FP8-student setup. |

#### Profile Results

| Metric | Step 0 Warmup | Step 1 Steady |
| --- | ---: | ---: |
| Step total | 512.745s | 338.254s |
| Prepare batches | 14 | 14 |
| Trainer microbatches | 14 | 14 |
| Prepare concurrency | 4 | 4 |
| Prepare window | 118.478s | 99.639s |
| Sum of prepare batch times | 439.590s | 361.805s |
| Student sampling summed wall | 371.578s | 293.146s |
| Sampling output throughput, summed | 925.6 tok/s | 1166.9 tok/s |
| Sampling output throughput, prepare-window | 2903.0 tok/s | 3433.1 tok/s |
| Teacher prefill summed wall | 67.992s | 68.515s |
| Teacher forward compute | 34.888s | 35.220s |
| Teacher cache write | 5.280s | 5.216s |
| Teacher prefill throughput, summed | 5674.7 tok/s | 5604.2 tok/s |
| Teacher prefill throughput, prepare-window | 3256.6 tok/s | 3853.6 tok/s |
| Trainer forward/backward | 203.548s | 207.924s |
| Optimizer queued wait | 301.082s | 138.661s |
| Optimizer step | 162.993s | 0.582s |
| P2P sync wall | 93.185s | 99.954s |
| P2P transfer time | 39.356s | 28.332s |
| Sync payload | 67.91 GB, 107 buckets | 67.91 GB, 107 buckets |
| Valid distillation tokens | 385,835 | 383,972 |
| Useful valid-token throughput | 752.5 tok/s | 1135.2 tok/s |
| Useful valid-token throughput / trainer GPU | 23.5 tok/s/GPU | 35.5 tok/s/GPU |
| Loss | 0.6157 | 0.4091 |

#### Notes

- This run was launched after the earlier two-step profile was mistakenly cleaned up. It uses the same validated placement and settings, but `num_steps=200`.
- Warmup step 0 and steady step 1 completed successfully. All 16 sampler pods, Dispatch, and all four trainer pods remained Running at the last check.
- FP8 P2P sync succeeded on both completed steps. The steady row transferred `67.91 GB` in `28.33s`; full sync wall remained about `99.95s`.
- This run should remain active unless intentionally stopped. Result files are under `experiments/opd_profile/results/qwen36_35b_fp8_from_qwen35_397b/opd-q36fp8-q35t-c4-1792p-long-05170618/`.

### Standalone Qwen3.6 FP8 SGLang Debug

| Field | Value |
| --- | --- |
| Timestamp | `2026-05-17T09:06:30Z` |
| Real-weight pod | `q36fp8-real-standalone-05170848` |
| Dummy pod | `q36fp8-dummy-standalone-05170853` |
| Manifests | `experiments/opd_profile/k8s/generated/q36fp8-real-standalone-05170848.yaml`, `experiments/opd_profile/k8s/generated/q36fp8-dummy-standalone-05170853.yaml` |
| SGLang patch | `/home/apanda/xorl-sglang-internal/.claude/worktrees/weight-sync-refactor/python/sglang/srt/layers/quantization/fp8_utils.py` |
| Decision | Keep the CUTLASS block-size fallback guard. It fixes a real Qwen3.6 FP8 dummy `[16,128]` serving crash. |

#### Findings

- A standalone real-weight Qwen3.6-35B-A3B FP8 TP2 SGLang pod, without XORL, dummy load, or RDMA updates, produced coherent arithmetic and sanity responses. The strongest prompts were parseable on `10/10` generations and correct on `6/10` to `8/10`, and sanity `2+2` / digit-copy probes were correct.
- A standalone dummy Qwen3.6-35B-A3B FP8 TP2 pod using `--load-format dummy` and `fp8_dummy_weight_block_size=[16,128]` reproduced a SGLang backend crash on the first chat request: `RuntimeError: size of scales_b is not matched` in `cutlass_w8a8_block_fp8_linear_with_fallback` while serving the Qwen3.6 linear-attention `in_proj_qkvz` projection.
- The SGLang fix adds a block-size guard so CUTLASS is only used for the canonical `[128,128]` FP8 block-scale layout; non-canonical layouts such as `[16,128]` fall back to Triton, including unpacking UE8M0 int32 scales for that Triton path.
- After restarting the dummy pod with the patched source, the same chat request returned HTTP 200 and the server remained healthy. The output was newline-only, which is expected for unsynced synthetic weights and should not be used as a semantic pass/fail signal.

#### Result Files

- Real-weight prompt probe: `experiments/opd_profile/results/standalone_sglang/q36fp8-real-standalone-05170848/prompt_probe/standalone_fp8_real_prompt_summary.md`
- Dummy SGLang log: `experiments/opd_profile/results/standalone_sglang/q36fp8-dummy-standalone-05170853/sglang-q36fp8-dummy-standalone-05170853/sglang_pod.log`
- Focused debug note: `experiments/opd_profile/results/standalone_sglang/qwen36_fp8_dummy_blocksize_debug_0517.md`

### Qwen3.6 FP8 Direct P2P Sync Probe

| Field | Value |
| --- | --- |
| Timestamp | `2026-05-17T09:48:07Z` |
| Run ID | `opd-q36fp8-syncprobe-05170944` |
| Manifest | `experiments/opd_profile/k8s/generated/opd-q36fp8-syncprobe-05170944.yaml` |
| Student sampler | One Qwen3.6-35B-A3B SGLang FP8 TP2 dummy receiver, `fp8_dummy_weight_block_size=[16,128]`, patched Triton fallback |
| Student trainer | Qwen3.6-35B-A3B BF16 XORL trainer, 4 nodes, no OPD client/teacher/Dispatch |
| Probe | Register one SGLang endpoint, call `/sync_inference_weights`, then prompt the receiver directly through `/v1/chat/completions` |
| Decision | Keep the evidence and continue debugging FP8 sync semantics. The transport path succeeds, but the synced receiver is not semantically correct. |

#### Findings

- P2P sync returned success: `Synced 31643 params to 1 endpoint(s)`.
- Sync transferred `67.91 GB` in `25.37s`; full handler wall was `34.04s` with `8.23s` backend init and `0.41s` complete/update time.
- The receiver served requests after sync and tagged responses with `metadata.weight_version=opd-q36fp8-syncprobe-05170944`, so the request path and update completion are working.
- Post-sync probes were token soup, including `2+2`, digit-copy, and 4-digit multiplication prompts. This rules out Dispatch, prompt wording, OPD first-sample ordering, and optimizer update as the immediate cause of incoherent Qwen3.6 FP8 dummy-sampler outputs.
- The most likely remaining issue is a block-FP8 scale/layout mismatch between XORL sender-side quantization and SGLang receiver/kernel expectations for the non-canonical `[16,128]` dummy receiver format.

#### Result Files

- Sync response: `experiments/opd_profile/results/qwen36_35b_fp8_from_qwen35_397b/opd-q36fp8-syncprobe-05170944/trainer-head-opd-q36fp8-syncprobe-05170944-trainer-head-vfzvn/sync_response.json`
- Direct prompt probe: `experiments/opd_profile/results/qwen36_35b_fp8_from_qwen35_397b/opd-q36fp8-syncprobe-05170944/trainer-head-opd-q36fp8-syncprobe-05170944-trainer-head-vfzvn/direct_sync_probe.jsonl`

### Qwen3.6 FP8 Kernel Smoke After Sync Semantics Failure

| Field | Value |
| --- | --- |
| Timestamp | `2026-05-17T10:01:49Z` |
| Probe pod | `fp8-kernel-smoke-05171005` |
| Code path | Patched SGLang worktree `weight-sync-refactor` |
| Probe | Direct H100 kernel checks for block-FP8 dense Triton linear and Triton fused MoE with non-canonical `[16,128]` block scales |
| Decision | Keep the current diagnosis focused on full-model receiver format / sync semantics, not the isolated dense or MoE Triton kernels. |

#### Findings

- Dense Triton block-FP8 linear with `[16,128]` produced plausible numerical error against dequantized BF16 references across representative Qwen3.6-like shapes. Mean absolute error was about `0.0066` to `0.0194`, with max absolute error below `0.11`.
- Triton fused MoE with `[16,128]`, `[32,128]`, `[64,128]`, and `[128,128]` block scales was also numerically stable against dequantized-weight/dequantized-activation references. For `[16,128]`, relative mean error was `0.012849` and cosine similarity was `0.99989`.
- This makes the remaining semantic failure more likely to be a full-model dummy receiver format or still-unchecked sync/post-load layout issue, rather than an obvious isolated dense or MoE kernel correctness bug.
