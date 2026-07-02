# CANONICAL INFRA RUNBOOK — OPD reprogrammable-slots stack

How to cleanly operate and run the `er-opd-q36-35b-slots` OPD training stack. Operational only — the science is in `CANONICAL_SCIENCE_RUNBOOK.md`. Current as of **2026-06-15 05:13Z**.

## 0. Where things stand & what to do next (orientation for a new agent)

**Throughput runbook location:** the focused next-agent throughput entrypoint is
`THROUGHPUT_MICROBENCH_RUNBOOK.md`. The complete failed/superseded throughput
history is in this section, §7d, §7e, and the `AMDAHL-*` candidate YAMLs under
`experiments/opd_profile/autoresearch/candidates/`. The cheap fwd/bwd
microbenchmark path is §7e; the next agent should iterate on one node first and
use 4-node replay only as a fidelity check after a 1-node candidate has a reason
to survive.
After consolidation, launch orchestration lives in `/home/apanda/xorl-infra`, but
the generator still has repo-path constants; before any launch, verify it will
run the intended client/engine checkouts rather than the deprecated
`/home/apanda/xorl-apanda-dev-opd-port` or shared `/home/apanda/xorl-client`
checkout.

**Throughput ownership is active now; do not wait for science runs or add trainer nodes to hide low utilization.** The AMDAHL audit took the step from ~293 s to **~32 s (~8–9×)** at the 512-prompt probe point and ended trainer-fb-bound. For the actual 64-prompts/step OPRD science window, the same 4-node trainer was still under-filled: AMDAHL-008 coalesced four 16-sample fb calls into one 64-sample call, AMDAHL-009 switched the trainer to quack+DeepEP24, AMDAHL-014 fixed the xorl-client OPRD pipeline guard so `opd_pipeline_rl` actually overlaps the next prepare with current forward/backward + optimizer + inference-weight sync, AMDAHL-016 swept DeepEP reserved SMs to 36 for the strict fresh-sample path, AMDAHL-018 then split strict same-step prepare into four concurrent prompt chunks, and AMDAHL-020 tested the same strict recipe with a single sampler endpoint to remove one weight-sync receiver. AMDAHL-014 is the current 64-prompt wall-time winner on the same 4-node trainer **if one-step-stale pipeline samples are accepted**: non-warmup `step_total_s=13.22`, `valid_tokens_per_step_s=335.9`, `pipeline_rl_active=1.0`, and 2/2 sync health. AMDAHL-020 is the current strict one-endpoint wall-time winner: non-warmup `step_total_s=13.41`, `valid_tokens_per_step_s=305.8`, `prepare_s=6.15`, `forward_backward_s=4.21`, `sync_inference_weights_s=2.35`, 1/1 sync health. It is now resource-aligned because the unused second sampler pod was released; restoring the strict two-sampler fallback AMDAHL-018 (`step_total_s=14.03`, `valid_tokens_per_step_s=292.5`, `prepare_s=5.27`, `sync_inference_weights_s=3.37`, 2/2 sync health) requires recreating `sglang-1`. The trainer-only replay harness is now available for cheap fwd/bwd screening (§7e), but AMDAHL-025/026 proved that a replay-only win is not enough to promote: pack2304 improved the static captured replay (`server_forward_backward_s=4.16` vs 4.46) and then regressed the real full strict run (`step_total_s=19.80`, `forward_backward_s=11.03`, `valid_tokens_per_step_s=206.9`). The denominator audit now explains the apparent MFU conflict: the AMDAHL-021 replay is ~1.37% reconstructed logical MFU over executed packed/dummy student tokens, but only ~0.039% if scaled by valid target tokens; OPD server logs do not emit native MFU, so quote the denominator. **Do not treat 1.37% executed-MFU as acceptable**: the updated OPSD microbench note puts the same ~3.4k tokens/rank shape at a small-batch executed-MFU curve of roughly a few percent and a healthy large-batch knee near 10–14%, so OPD still needs a real current-node utilization fix. The corrected replay breakdown also shows that `opd_profile_forward_compute_s` is not just student forward: the captured 4096 shape dispatches 107,648 student tokens but the trainer-side OPRD teacher forward adds 350,870 teacher tokens; sorted replay metrics measured `server_forward_backward_s=4.80` with `model_forward_s=0.81`, `oprd_teacher_forward_s=0.85`, `loss_compute_s=0.91`, `backward_compute_s=1.59`, and `clear_gradients_s=0.59`. AMDAHL-028 tested two server-feed hypotheses: minimal zero-loss dummy rows were neutral (`server_forward_backward_s=4.461`), while repeating the captured data to 2×/4× larger fwd/bwd calls improved executed-token throughput only from 521→584→776 tok/s/GPU. AMDAHL-041 later rejected the simple larger-prepare/0-dummy story on 4 nodes: repeat 1/2/4/8 reconstructed logical MFU was only `1.424% / 1.625% / 1.718% / 1.831%`. AMDAHL-042/043 moved the lm-head materialization fix forward with PR #375 and proved full64 fit in a CP-sourced lm-head-TP VP-KL topology, but warmed 4-node replay was still `15.4878s` and backward-dominated. AMDAHL-044/045 then tested a local no-CP DP-sourced lm-head-TP follow-up and reached raw warmed 4-node replay mean `server_forward_backward_s=4.1209`, but the reported loss (`1.8479248`, and `1.8495717` on the 1-node no-CP run) differs from the AMDAHL-042/043 VP-KL loss (`1.8686199`) on the same replay. A focused two-group DP-sourced VP-OPD unit test passed, and a production-path FSDP-sharded lm-head OPD unit test now also passes, so the mismatch is not explained by the standalone VP loss-wrapper or local DTensor row-slicing path. The same-topology gate then fixed the dtype drift, the real all-layer SGLang cache one-node replay now fits, the latest chunk sweep leaves the default `opd_oprd_layer_chunk_size=4` as the best stable one-node setting (`server_forward_backward_s=4.6591`; chunk `8` slower; chunk `16` OOM), and the AMDAHL-048 2-node retry completed at `server_forward_backward_s=4.4441`. Do not chase pause-token trimming for MFU: pause/think tokens still ride the same GEMMs; masking them only changes the tiny loss term. Current live slot status at 01:27Z: the AMDAHL-048 retry ran only `er-opd-q36-35b-slots-trainer-head` on `research-common-h100-077` plus `trainer-worker-1` on `research-common-h100-110`, then cleanup completed (`stop-trainer-control --remove-run`, trainer pods deleted, local port-forward killed, no slots trainer pods remain). The stale throughput-owned `er-opd-q36-35b-slots-sglang-0` was deleted during the earlier scheduling attempt and was not restored; slots dispatch and teacher-smg remain. Restamp roles only while actively benchmarking or training.

**Authoritative live slot status is the latest timestamped addendum below.** As
of 2026-06-15 05:13Z, AMDAHL-065 has measured and cleanup is complete. No slots
trainer, student-sampler, or teacher-sglang pods remain; only slots dispatch +
teacher-smg are running. The separate `er-opd-q36-35b-sci` stack was not
touched.

**2026-06-15 01:38Z orientation addendum:** AMDAHL-049 screened the pack2304
zero-dummy row-shape hypothesis on the same real all-layer cache workload and is
rejected (`server_forward_backward_s=5.5459`, loss `2.3666090`) versus the
pack4096/chunk4 one-node baseline (`4.6591s`). Current live slot status: no
slots trainer pods remain; only slots dispatch + teacher-smg are running.
Pack2304 is not a science/default candidate unless a later full-OPD prepare path
changes the generated sample distribution.

**2026-06-15 01:51Z orientation addendum:** AMDAHL-050 screened explicit
runner allocator-flush overrides on the same pack4096/chunk4 real-cache
workload. The partial replay completed two measured rows
(`server_forward_backward_s=4.8437` mean, loss `2.3550419`) and reduced replay
`clear_gradients_s` to about `0.004s`, but the third measured request OOMed in
FSDP pre-backward all-gather while allocating `970 MiB` on near-full ranks.
Conclusion: keep the default `forward_backward` defrag/cache-emptying behavior
enabled for this memory shape; do not promote no-defrag replay knobs. Current
live slot status: no slots trainer pods remain; only slots dispatch + teacher-smg
are running.

**2026-06-15 02:03Z orientation addendum:** AMDAHL-051 screened
`enable_forward_prefetch: true` on the same one-node pack4096/chunk4 no-CP
lm-head-TP VP-KL real-cache workload. It was neutral (`server_forward_backward_s=4.6506`
vs `4.6591` baseline), slightly regressed backward compute, and shifted the
reported loss from `2.3550419` to `2.3562496`. Conclusion: do not promote
forward-prefetch for this no-CP real-cache path without a later static/K3 pass
and a stronger same-workload 4-node speed result. Current live slot status: no
slots trainer pods remain; only slots dispatch + teacher-smg are running.

**2026-06-15 02:12Z orientation addendum:** AMDAHL-052 screened simple
`--repeat-data 2` fatter trainer-server calls on the same real-cache workload.
The server packed 128 repeated samples into 43 batches (80.7% utilization,
142080 tokens) but OOMed before the warmup row completed: rank 5 failed in FSDP
pre-backward all-gather while allocating `970 MiB` with about `798 MiB` free.
Conclusion: do not promote repeat-data on the current one-node memory envelope
and do not spend the 4-node/static/K3 gates on AMDAHL-052. Current live slot
status: no slots trainer pods remain; only slots dispatch + teacher-smg are
running.

**2026-06-15 02:22Z orientation addendum:** AMDAHL-053 split CPU GC from CUDA
cache release and skipped only CPU `gc.collect()` on the same real-cache
trainer-server replay. It lowered explicit `clear_gradients_s` to `0.1783s`
from the `0.4339s` baseline, but total wall regressed to
`server_forward_backward_s=5.4222` / `api_wall_s=5.6626` versus
`4.6591` / `4.9797`; model forward/backward absorbed the time. Conclusion: keep
the default CPU GC path enabled and do not spend the 4-node/static/K3 gates on
AMDAHL-053. Current live slot status: no slots trainer pods remain; only slots
dispatch + teacher-smg are running.

**2026-06-15 02:34Z orientation addendum:** AMDAHL-054 screened
`deepep_num_sms: 24` on the same one-node pack4096/chunk4 no-CP lm-head-TP VP-KL
real-cache workload, versus the current SMS36 baseline. It regressed total wall
(`server_forward_backward_s=4.7599`, `api_wall_s=4.9905`) versus SMS36
(`4.6591` / `4.9797`), worsened model forward/backward, and shifted the reported
loss from `2.3550419` to `2.3569047`. Conclusion: keep `deepep_num_sms=36` for
this real-cache path and do not spend the 4-node/static/K3 gates on AMDAHL-054.
Current live slot status: no slots trainer pods remain; only slots dispatch +
teacher-smg are running.

**2026-06-15 02:42Z orientation addendum:** AMDAHL-055 screened the existing
custom MoE expert-gradient reduce hook (`moe_grad_reduce_mode=bf16_a2a_fp32_sum`)
as a one-node real-cache replay candidate. It failed before replay during engine
init: the server/run logs report `ValueError:
moe_grad_reduce_mode='bf16_a2a_fp32_sum' requires FSDP mp_policy.reduce_dtype=torch.float32,
got torch.bfloat16`. Interpretation: the current one-node topology has singleton
expert FSDP behavior for this hook and already uses BF16 reduce policy, so this
is not a valid one-node screen without an engine/topology fix. Current live slot
status: no slots trainer pods remain; only slots dispatch + teacher-smg are
running.

**2026-06-15 02:55Z orientation addendum:** AMDAHL-056 screened plain
`fsdp_reduce_dtype: bf16` on the same one-node pack4096/chunk4 no-CP lm-head-TP
VP-KL real-cache workload. It preserved the loss (`2.3550419`) but regressed
total wall (`server_forward_backward_s=4.8815`, `api_wall_s=5.1059`) versus the
chunk4/SMS36 baseline (`4.6591` / `4.9797`). Conclusion: do not promote BF16
FSDP reduce-scatter for this real-cache path and do not spend the
4-node/static/K3 gates on AMDAHL-056. Current live slot status: no slots trainer
pods remain; only slots dispatch + teacher-smg are running.

**2026-06-15 03:10Z orientation addendum:** AMDAHL-057 screened
`ep_dispatch: alltoall` on the same one-node pack4096/chunk4 no-CP lm-head-TP
VP-KL real-cache workload, versus the current DeepEP/SMS36 baseline. It reached
warmup forward/backward but OOMed before writing a replay row: the server packed
64 samples into 22 batches (78.8% utilization, 71040 tokens), then ranks 2/6
failed in FSDP pre-backward all-gather while allocating `970 MiB` with only
about `326-446 MiB` free. Conclusion: do not promote alltoall dispatch for this
real-cache path and do not spend the 4-node/static/K3 gates on AMDAHL-057.
Current live slot status: no slots trainer pods remain; only slots dispatch +
teacher-smg are running.

**2026-06-15 03:22Z orientation addendum:** AMDAHL-058 screened a local
engine-only reporting-path change on the same one-node pack4096/chunk4 no-CP
lm-head-TP VP-KL real-cache workload. It deferred detached scalar loss-report
all-reduce from once per packed microbatch to once per `forward_backward` call.
The replay preserved loss (`2.355041742`) but regressed wall time
(`server_forward_backward_s=4.803969`, `api_wall_s=5.006521`) versus the
chunk4/SMS36 baseline (`4.6591` / `4.9797`). Conclusion: repeated detached loss
reporting all-reduces are not the live bottleneck; the local reporting patch was
reverted/not promoted, and do not spend the 4-node/static/K3 gates on
AMDAHL-058. Current live slot status: no slots trainer pods remain; only slots
dispatch + teacher-smg are running. Capacity refresh: Kubernetes auth works,
aggregate free is `56` GPUs (`22` on cordoned nodes), and only two ready
uncordoned full 8-GPU NCCL nodes are free (`research-common-h100-089` and
`research-common-h100-077`), so the same-workload 4-node gate remains
capacity-blocked.

**2026-06-15 03:39Z orientation addendum:** AMDAHL-059 screened EP4 x ep_fsdp2
DeepEP topology on the same one-node pack4096/chunk4 no-CP lm-head-TP VP-KL
real-cache workload, versus the current EP8 x ep_fsdp1/SMS36 baseline. The first
launch proved the mesh formed but failed on EP8-shaped optimizer state in the DCP
checkpoint; a default-true `load_checkpoint_optimizer` knob was added and the EP4
retry alone set it false for server-only replay. The replay was slightly faster
(`server_forward_backward_s=4.598297`, `api_wall_s=4.849234`) than baseline
(`4.659086` / `4.979698`), but shifted same-capture loss/KL to `2.373135030` /
`2.351824299` from baseline `2.355041921` / `2.333743375`. Conclusion: do not
promote EP4 or spend the 4-node/static/K3 gates on AMDAHL-059 without first
explaining and eliminating the loss drift. Current live slot status: no slots
trainer pods remain; only slots dispatch + teacher-smg are running.

**2026-06-15 03:55Z orientation addendum:** AMDAHL-060 screened
`reshard_after_forward:false` on the two-node pack4096/chunk4 no-CP lm-head-TP
VP-KL real-cache workload, versus the AMDAHL-048 2-node baseline. Because
unspecific trainer pods were admission-defaulted to `node-group=default`, the
launch used a filtered trainer-only manifest with explicit
`nodeSelector: {node-group: nccl, node-pool: compute}`; head ran on
`research-common-h100-110` and worker-1 on `research-common-h100-092`. The
corrected replay measured `server_forward_backward_s=4.443982`,
`api_wall_s=4.643382`, loss `2.355553985`, and `opd_kl=2.334234441`, versus
AMDAHL-048 `4.444137` / `4.726916`, loss `2.355041921`, and
`opd_kl=2.333743553`. Backward improved, but model forward and clear-gradients
regressed enough that total server wall was flat. Conclusion: do not promote
`reshard_after_forward:false` or spend the 4-node/static/K3 gates on AMDAHL-060.
Current live slot status: no slots trainer pods remain; only slots dispatch +
teacher-smg are running. Capacity refresh: Kubernetes auth works and three ready
uncordoned full 8-GPU NCCL nodes are free (`research-common-h100-089`,
`research-common-h100-092`, `research-common-h100-110`), so 4-node remains
capacity-blocked but two-node screens are available.

**2026-06-15 04:06Z orientation addendum:** AMDAHL-061 screened
`enable_forward_prefetch:true` on the two-node pack4096/chunk4 no-CP lm-head-TP
VP-KL real-cache workload, versus the AMDAHL-048 2-node baseline. It used the
same filtered trainer-only manifest pattern as AMDAHL-060 with explicit
`nodeSelector: {node-group: nccl, node-pool: compute}`; head ran on
`research-common-h100-077` and worker-1 on `research-common-h100-089`. The
replay measured `server_forward_backward_s=3.677469`, `api_wall_s=3.863640`,
loss `2.359620273`, and `opd_kl=2.338320834`, versus AMDAHL-048 `4.444137` /
`4.726916`, loss `2.355041921`, and `opd_kl=2.333743553`. This is a real speed
win but shifts same-capture KL/loss by about `+0.00458`, so do not promote
forward-prefetch or spend the 4-node/static/K3 gates on AMDAHL-061 until that
drift is explained and fixed. Current live slot status: no slots trainer pods
remain; only slots dispatch + teacher-smg are running. Capacity refresh:
Kubernetes auth works, but a parallel `er-opd-q36-mtp-perf-replay-4n-*` job now
occupies `research-common-h100-077/089/092/110`, so the same-workload 4-node OPD
gate is capacity-blocked again.

**2026-06-15 04:13Z orientation addendum:** AMDAHL-062 is queued as the
correctness-preserving follow-up to AMDAHL-061 rather than a measured result.
The local engine now exposes default-preserving `enable_backward_prefetch` so
forward and backward manual FSDP prefetch can be controlled separately; AMDAHL-062
sets `enable_forward_prefetch:false` and `enable_backward_prefetch:true` on the
same two-node real all-layer cache workload. Candidate/config:
`experiments/opd_profile/autoresearch/candidates/AMDAHL-062-OPRD-PREP64-2NODE-LMHEADTP-NOCP-VPKL-BACKPREFETCH.yaml`
and
`/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_2node_warm009_deepep36_backprefetch_lmheadtp_nocp.yaml`.
Preflight passed locally (`py_compile`, targeted server-argument test,
candidate/config YAML parse, render-control repo-root/`--nnodes 2` check, and
server-side dry-run for only trainer-head service/pod plus trainer-worker-1 pod
with explicit `nodeSelector: node-group=nccl,node-pool=compute`). A follow-up
local hardening pass factored the manual prefetch schedule into
`_configure_manual_fsdp_prefetch()` and added CPU-only coverage for backward-only,
forward-only, both-directions, and no-op scheduling (`18` policy tests passed;
the server-argument threading test still passed). Launch was deferred because
`full_schedulable_8gpu_nodes=NONE`: nodes `077/089/092/110` are held by
`er-opd-q36-mtp-perf-replay-4n-*`, `086/088/099` by another 8-GPU serving job,
`087` is split, and `096` is cordoned with only 7 allocatable GPUs. Current live
slot status: no slots trainer pods remain; only slots dispatch + teacher-smg are
running; the science stack was not touched.

**2026-06-15 04:33Z orientation addendum:** AMDAHL-062 has now run as the
two-node backward-only manual FSDP prefetch screen. Before launch, the local
engine hardening was extended with `_coerce_optional_bool_config()` so string
values such as `"false"`/`"off"` cannot accidentally enable a disabled prefetch
direction; `test_torch_parallelize_policies.py` passed `28` tests and the
server-argument threading test still passed. The server-only replay artifact is
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-backprefetch-chunk4-warmed-serveronly-4x-20260615T043109Z.jsonl`;
server log is
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T042714Z-serveronly-configAMDAHL-062-OPRD-PREP64-2NODE-LMHEADTP-NOCP-VPKL-BACKPREFETCH-er-opd-q36-35b-slots-trainer-head/server.log`.
Mean over 3 measured rows after 1 warmup: `server_forward_backward_s=4.280844`,
`api_wall_s=4.477853`, `model_forward_s=1.097722`,
`backward_compute_s=1.684526`, `clear_gradients_s=0.533312`, loss
`2.355041921`, `opd_kl=2.333743553`, and hidden loss `0.021298385`.
This matches AMDAHL-048 loss/KL/hidden parity and is `3.67%` faster on server
wall / `5.27%` faster on API wall than AMDAHL-048, but it is still `16.41%`
slower than the rejected AMDAHL-061 server wall. Conclusion: backward-only
prefetch is correctness-clean but too small to be the 10% MFU lever; do not
promote it to science/defaults or spend a standalone 4-node/static/K3 gate on it.
Cleanup complete: `stop-control --remove-run`, temporary slots `sglang-0`,
`sglang-1`, `teacher-sglang-0`, `trainer-head`, and `trainer-worker-1` pods
deleted; only dispatch + teacher-smg remain.

**2026-06-15 04:52Z orientation addendum:** AMDAHL-063 has now run as the
two-node forward-only manual FSDP prefetch isolation screen. Candidate:
`experiments/opd_profile/autoresearch/candidates/AMDAHL-063-OPRD-PREP64-2NODE-LMHEADTP-NOCP-VPKL-FWDPREFETCH.yaml`;
infra config:
`/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_2node_warm009_deepep36_forwardprefetch_lmheadtp_nocp.yaml`.
It keeps the AMDAHL-048/061/062 workload and sets
`enable_forward_prefetch:true`, `enable_backward_prefetch:false` to decide
whether AMDAHL-061's KL/loss drift is forward-prefetch-specific. Preflight
passed: YAML parse, render-control repo-root/`--nnodes 2` check, filtered
trainer-only server-side dry-run with explicit NCCL compute selector, and repo
`diff --check`. The refreshed kube auth works (`get pods` and `create pods`
both returned `yes`). The first retry reached API admission but lost the
apparent `110/089` capacity window to `wordle-rpl2n-sms20-0440`, so those
pending pods were deleted; the follow-up retry ran on
`research-common-h100-110/077`. Artifact:
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-chunk4-warmed-serveronly-4x-20260615T044953Z.jsonl`;
server log:
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T044707Z-serveronly-configAMDAHL-063-OPRD-PREP64-2NODE-LMHEADTP-NOCP-VPKL-FWDPREFETCH-er-opd-q36-35b-slots-trainer-head/server.log`.
Mean over 3 measured rows after 1 warmup: `server_forward_backward_s=4.140186`,
`api_wall_s=4.335127`, `forward_compute_s=0.863331`,
`backward_compute_s=2.069476`, `clear_gradients_s=0.633195`, loss
`2.355041921`, `opd_kl=2.333743553`, and hidden loss `0.021298385`. This
matches AMDAHL-048 loss/KL/hidden exactly and is `6.84%` faster on server wall,
but remains much slower than the rejected AMDAHL-061 coupled-prefetch row.
Conclusion: forward-only prefetch is loss-clean for this replay; AMDAHL-061's
drift is not forward-only prefetch by itself. It is still only a diagnostic
small speed lever, not a science/default promotion. Cleanup complete: trainer
controls stopped, temporary trainer-head/worker-1 pods deleted, and only slots
dispatch + teacher-smg remain.

**2026-06-15 05:02Z orientation addendum:** AMDAHL-064 has now run as the
two-node fatter-call fit/throughput screen. It reuses the AMDAHL-063
forward-only prefetch trainer config and replays the same AMDAHL-047 full64 real
all-layer SGLang-cache capture with `--repeat-data 2`, producing a 128-datum
static f/b call. Candidate:
`experiments/opd_profile/autoresearch/candidates/AMDAHL-064-OPRD-PREP128-2NODE-LMHEADTP-NOCP-VPKL-FWDPREFETCH-REPEAT2.yaml`;
infra config:
`/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_2node_warm009_deepep36_forwardprefetch_lmheadtp_nocp.yaml`.
It launched trainer-head on `research-common-h100-089` and trainer-worker-1 on
`research-common-h100-092`, with the same filtered trainer-only manifest and
explicit NCCL compute selector used by the prior 2-node screens. Artifact:
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-repeat2-chunk4-warmed-serveronly-3x-20260615T045836Z.jsonl`;
server log:
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T045625Z-serveronly-configAMDAHL-064-OPRD-PREP128-2NODE-LMHEADTP-NOCP-VPKL-FWDPREFETCH-REPEAT2-er-opd-q36-35b-slots-trainer-head/server.log`.
Mean over 2 measured rows after 1 warmup: `server_forward_backward_s=5.503175`,
`api_wall_s=5.994858`, `forward_compute_s=1.221080`,
`backward_compute_s=3.036948`, `loss_compute_s=0.076166`,
`clear_gradients_s=0.495572`, valid tokens `1030`, loss `2.355820715`,
`opd_kl=2.334518359`, and hidden loss `0.021302296`. Throughput improved:
valid/server-token rate is `187.16` tokens/s, versus AMDAHL-048 `115.88`
tokens/s and AMDAHL-063 `124.39` tokens/s. But same-capture loss/KL/hidden
shifted versus the AMDAHL-048/063 tuple by about `+0.000779`, `+0.000775`, and
`+0.0000039`. Conclusion: AMDAHL-064 is fit/throughput-positive evidence for
fatter calls, but it is not a promotion; the next engine target should audit
packing or token-weighted metric/loss aggregation before using repeat2 or
prepare128 as a science/default path. Cleanup complete: trainer controls
stopped, temporary trainer-head/worker-1 pods deleted, and only slots dispatch +
teacher-smg remain. The separate science stack was not touched.

**2026-06-15 05:13Z orientation addendum:** AMDAHL-065 has now run as the
repeat2 separator diagnostic. It used the same AMDAHL-063/064 two-node
forward-only prefetch trainer config, but a custom replay payload: original
AMDAHL-047 full64 capture, one artificial no-loss 4096-token separator datum,
then the same 64 samples again. Candidate:
`experiments/opd_profile/autoresearch/candidates/AMDAHL-065-OPRD-PREP128-2NODE-LMHEADTP-NOCP-VPKL-FWDPREFETCH-REPEAT2-SEPARATOR.yaml`;
infra config:
`/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_2node_warm009_deepep36_forwardprefetch_lmheadtp_nocp.yaml`.
It launched trainer-head on `research-common-h100-077` and trainer-worker-1 on
`research-common-h100-110`, with the same filtered trainer-only manifest and
explicit NCCL compute selector used by the prior two-node screens. Artifact:
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-repeat2-separator-chunk4-warmed-serveronly-3x-20260615T050943Z.jsonl`;
server log:
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T050707Z-serveronly-configAMDAHL-065-OPRD-PREP128-2NODE-LMHEADTP-NOCP-VPKL-FWDPREFETCH-REPEAT2-SEPARATOR-er-opd-q36-35b-slots-trainer-head/server.log`.
Mean over 2 measured rows after 1 warmup: `server_forward_backward_s=5.892664`,
`api_wall_s=6.389314`, `forward_compute_s=1.262510`,
`backward_compute_s=3.069160`, `loss_compute_s=0.062265`,
`clear_gradients_s=0.883305`, valid tokens `1030`, loss `2.356476724`,
`opd_kl=2.335177020`, and hidden loss `0.021299774`. This did not restore the
AMDAHL-048/063 loss/KL/hidden tuple and moved loss/KL further than AMDAHL-064,
so the simple boundary co-pack explanation is insufficient. Conclusion: do not
promote repeat2, separator-repeat2, or prepare128; the next target is a direct
token-weighted aggregation plus packed-layout invariance audit. Cleanup
complete: trainer controls stopped, temporary trainer-head/worker-1 pods
deleted, and only slots dispatch + teacher-smg remain. The separate science
stack was not touched.

**2026-06-14 23:24Z correction to the 22:32Z loss diagnosis:** the no-CP
lm-head-TP VP-KL drift is now understood and fixed locally. The VP DTensor
lm-head shard loaded as fp32, while the normal gathered FSDP/mixed-precision path
used bf16-rounded lm-head weights. The local fix casts the effective VP student
shard to the hidden/model compute dtype before loss use. Same-server hidden0
limit8 parity now passes (`0.6181625724` streaming vs `0.6181625128` VP-KL,
diff about `6e-8`). The 4-node all-layer OPRD replay still only reconstructs
about `1.04%` logical MFU, so this is not a throughput promotion and §1 stays
unchanged.

**2026-06-14 23:43Z layer-cache follow-up:** AMDAHL-031's reduced-layer
SGLang OPRD layer-cache payload now fits on the current no-CP lm-head-TP VP
engine, proving the old 1-node full-lm-head-gradient OOM is gone for that path.
A local opt-in `TeacherActivationCache` device-residency patch reduced measured
`oprd_layer_fetch_s` from `0.2847` to `0.00138` and improved one-node
`server_forward_backward_s` from `4.9560` to `4.3206` on the same replay. This
is not a promoted config: it is every4 reduced-layer OPRD with only `515` valid
tokens, and §1 remains unchanged.

**2026-06-15 00:13Z valid-only/chunked follow-up:** the normal kube token works
again (`kubectl auth can-i get pods -n apanda` = yes). A local engine follow-up
now gathers only valid SGLang layer-cache rows and computes OPRD hidden MSE in
layer chunks instead of materializing all selected layers as fp32 at once.
Focused validation passed (`py_compile` on `opd_loss.py`, `model_runner.py`,
`teacher_cache.py`; pytest `31 passed` across OPD loss, layer-cache runner, and
teacher-cache tests). The every4 replay is stable with the new path but slower
than the 23:43 device-cache-only row: artifact
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-sglangcache-every4-1node-validonly-chunkedmse-layerdevcache-warmed-serveronly-4x-20260615T0009.jsonl`,
mean `server_forward_backward_s=4.7859`, `oprd_layer_fetch_s=0.000733`, valid
tokens `515`, loss `1.8533004`. The derived all-40-layer synthetic cache stress
replay now completes: artifact
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-synthlayercache-all40-1node-validonly-chunkedmse-layerdevcache-warmed-serveronly-4x-20260615T0013.jsonl`,
mean `server_forward_backward_s=4.8691`, `oprd_layer_fetch_s=0.000758`,
`oprd_teacher_forward_s=0.0`, valid tokens `3049`. Caveat: AMDAHL-047 is a
performance-path stress artifact only; its layer cache repeats final hidden for
every layer, so it is not loss/science validation and not a §1 promotion.

**2026-06-15 00:23Z SGLang cache-source follow-up:** the next real all-layer
cache blocker is the SGLang endpoint materialization path, not trainer-side
cache consumption. The local `/home/apanda/xorl-sglang-internal` patch adds an
internal tensor-output path for `teacher_hidden_cache`: the scheduler returns
kept hidden rows as CPU tensors instead of nested Python float lists, and
`_write_teacher_hidden_cache` accepts tensor chunks and splits all-layer payloads
per sample before concatenation. Local validation passed `py_compile` across the
modified SGLang request/scheduler/cache files and
`PYTHONPATH=/home/apanda/xorl-sglang-internal/python /home/apanda/xorl-sglang-internal/.venv/bin/python -m pytest test/srt/zorl/test_teacher_hidden_cache_writer.py -q`
(`2 passed`, warnings only). A teacher-only slots manifest rendered with
`OPD_SGLANG_REPO=/home/apanda/xorl-sglang-internal` passed server-side dry-run
(`kubectl -n apanda apply --dry-run=server -f /tmp/er-opd-slots-teacher-sglang-0.yaml`),
but no pod was applied: an all-namespace GPU accounting pass found no schedulable
node with 8 free GPUs, and the idle `er-opd-q36-35b-sci` teacher belongs to the
separate science stack.

**2026-06-15 00:30Z PR packaging / capacity update:** the xorl engine follow-up
is now pushed to draft PR #375 at `77cdf112`. The pushed branch includes the
DP-sourced no-CP lm-head-TP path, the bf16/materialization cast that fixed the
same-topology VP-KL drift, opt-in lm-head fingerprint dumps, teacher
activation/layer cache device residency, valid-only layer-cache gather, and
chunked all-layer OPRD hidden MSE. Focused validation passed: xorl
`py_compile`, `git diff --check`, OPD/cache pytest (`31 passed`), and
distributed/loss pytest (`6 passed`). The SGLang tensor-output endpoint patch is
now draft PR #48 at `c8552f4fa`, with the same focused pytest still `2 passed`.
These PRs are visible to the other throughput agents, but neither changes the
promoted science config. Rechecked capacity at 00:30Z: `gpu_report.py --user
apanda` reported cluster free `50` GPUs, only `28` on schedulable nodes, and no
ready node had 8 free GPUs. No pods were applied, no live controls were changed,
and `er-opd-q36-35b-sci` was untouched.

**2026-06-15 00:38Z streaming layer-cache follow-up:** PR #375 is now pushed to
`e8ef7e98`. The new follow-up adds `TeacherActivationCache.get_layer_slice()`
and threads a loss-level teacher-layer fetcher through both OPD loss paths, so
rank-3 SGLang layer-cache OPRD fetches/compares layer chunks instead of
materializing the entire `[valid, layers, hidden]` teacher tensor before chunked
MSE. Local validation passed: xorl `py_compile`, `git diff --check`,
OPD/cache/model-runner pytest (`35 passed`), and distributed/loss pytest
(`6 passed`). This is still not a promotion: no real all-layer cache capture,
static/K3 gate, or same-workload 4-node replay has cleared.

**2026-06-15 00:41Z chunk-size knob follow-up:** PR #375 is now pushed to
`706256f4`. The latest commit exposes `opd_oprd_layer_chunk_size` for both full
and vocab-parallel OPD loss paths, so replay configs can tune the streamed
layer-cache chunk width without another engine edit. Default remains `4` layers
per chunk. Validation was repeated after the param threading: xorl `py_compile`,
`git diff --check`, OPD/cache/model-runner pytest (`35 passed`), and
distributed/loss pytest (`6 passed`). Still no §1 promotion.

**2026-06-15 00:56Z real all-layer SGLang cache follow-up:** SGLang PR #48
(`c8552f4fa`) and xorl PR #375 (`706256f4`) cleared the one-node real-cache
fit gate. A temporary slots `teacher-sglang-0` pod on
`research-common-h100-110.cloud.together.ai` captured full64 real all-layer
teacher cache:
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/amdahl-047-realalllayer-sglangcache-full64-20260615T004904Z.json`
with rank-2 hidden `[3542,2048]` and rank-3 all-layer `[40,3542,2048]`.
Tensor-output SGLang reported `teacher_prefill_forward_compute_s=8.9454`,
`teacher_hidden_cache_write_s=0.7722`, wall `10.001s`, and cache indices matched
the reconstructed AMDAHL-031 row offsets. A temporary slots `trainer-head`
replay of that real cache then completed:
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-warmed-serveronly-4x-20260615T005428Z.jsonl`.
Mean over 3 measured iterations after 1 warmup:
`server_forward_backward_s=4.9408`, `model_forward_s=1.2082`,
`backward_compute_s=2.5217`, `kl_compute_s=0.0414`,
`oprd_layer_fetch_s=0.000436`, `oprd_teacher_forward_s=0.0`, valid tokens
`515`, loss `2.3550419`. Successful VP replay required
`opd_emit_full_vocab_diagnostics=false` and `teacher_heads[0]` pointing to the
prepared teacher-store manifest
`/shared/apanda/opd_teacher_stores/qwen3_6_35b_a3b_lm_head_shard32768/manifest.json`.
This is still not a §1 promotion: static/K3 correctness and same-workload
4-node real-cache throughput gates remain pending.

**2026-06-15 01:27Z AMDAHL-048/chunk sweep update:** AMDAHL-048 added a
2-node tooling candidate plus infra config
(`data_parallel_shard_size=16`, no-CP, `lm_head_tensor_parallel_size=8`) for the
same real all-layer cache workload, but it did not start: worker-1 scheduled on
`research-common-h100-077`, while head stayed Pending after
`research-common-h100-110` was claimed by
`er-opd-q36-mtp-perf-replay-trainer-head`. Cleanup stopped trainer control,
deleted the temporary head/worker pods, and killed the local wait. The one-node
fallback chunk-size sweep then ran only slots `trainer-head` on h100-077. Chunk
`4` remained fastest/stable (`server_forward_backward_s=4.6591`, loss
`2.3550419`), chunk `8` was slower (`4.9491`), and chunk `16` OOMed in FSDP
pre-backward all-gather after one measured row. The AMDAHL-048 retry then
completed on head h100-077 + worker h100-110 after an h100-092 admission failure
(`Requested: 8, Available: 4`): artifact
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-chunk4-warmed-serveronly-4x-20260615T012513Z.jsonl`,
mean `server_forward_backward_s=4.4441`, `model_forward_s=1.1004`,
`backward_compute_s=2.0665`, `kl_compute_s=0.0256`,
`oprd_layer_fetch_s=0.000236`, valid `515`, loss `2.3550419`. §1 is unchanged:
static/K3 and same-workload 4-node real-cache gates remain pending.

**2026-06-15 01:38Z AMDAHL-049 pack2304 update:** AMDAHL-049 added a one-node
pack2304 tooling candidate plus infra config
(`sample_packing_sequence_len=2304`, no-CP, `lm_head_tensor_parallel_size=8`) to
screen the zero-dummy row-shape hypothesis on the same real all-layer cache
workload. The row-shape audit showed pack2304/3072 would give `32` packed rows
and `0` dummy rows at the 32-GPU gate, but the measured one-node replay regressed:
artifact
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-pack2304-chunk4-warmed-serveronly-4x-20260615T013525Z.jsonl`,
mean `server_forward_backward_s=5.5459`, `model_forward_s=1.4871`,
`backward_compute_s=2.8861`, `kl_compute_s=0.0449`,
`oprd_layer_fetch_s=0.000436`, valid `515`, loss `2.3666090`. This is slower
than the pack4096/chunk4 baseline (`4.6591s`), so pack2304 is rejected for this
real-cache path and is not a science/default promotion.

**In flight right now:** no `er-opd-q36-35b-slots` trainer, teacher-sglang, or
trainer-only replay run is active. The latest throughput screen is AMDAHL-065:
the repeat2 separator diagnostic preserved a 64-sample boundary but shifted
same-capture loss/KL/hidden further, so it is not promoted. Trainer control was
stopped, the
temporary trainer-head/worker-1 pods were deleted, and only slots dispatch +
teacher-smg remain. Restart trainer-only replay via
`write-trainer-server-control` only when actively benchmarking (§7e); restart
sampler/teacher roles only for a full OPD capture/run. Capacity check at
2026-06-15 05:13Z: Kubernetes auth works; the temporary AMDAHL-065 nodes
`research-common-h100-077/110` were released after cleanup.
Latest 64-prompt throughput artifact: `candidates/AMDAHL-014-OPRD-PREP64-DEEPEP24-PIPE.yaml`
(faster but one-step stale). Latest strict one-endpoint fresh-sample winner:
`candidates/AMDAHL-020-OPRD-PREP64-DEEPEP36-STRICTCHUNK4-1SAMPLER.yaml`. Latest
raw 4-node lm-head-TP VP-KL speed artifact remains
`candidates/AMDAHL-045-OPRD-PREP64-4NODE-LMHEADTP-NOCP-VPKL.yaml`, now with the
dtype-fix follow-up in `THROUGHPUT_MICROBENCH_RUNBOOK.md`; it is still not
promoted because 4-node MFU is about `1.04%` and static/K3 gates are pending.
AMDAHL-048 is a 2-node tooling candidate with a completed real-cache replay metric
(`server_forward_backward_s=4.4441`) but no promotion status. AMDAHL-049 is a
pack2304 tooling candidate rejected on one-node real-cache replay
(`server_forward_backward_s=5.5459`). AMDAHL-050 is a no-defrag tooling candidate
rejected because it OOMed on the third measured one-node real-cache request.
AMDAHL-051 is a forward-prefetch tooling candidate rejected as neutral/noisy
(`server_forward_backward_s=4.6506` vs `4.6591` baseline). AMDAHL-052 is a
repeat-data=2 tooling candidate rejected because it OOMed before producing a
warmup replay row. AMDAHL-053 is a GC-split tooling candidate rejected because
skipping only CPU GC regressed total wall (`server_forward_backward_s=5.4222`).
AMDAHL-054 is a DeepEP SMS24 tooling candidate rejected because it regressed the
same real-cache workload (`server_forward_backward_s=4.7599` vs the SMS36
baseline `4.6591`).
AMDAHL-055 is a MoE BF16-a2a reduce tooling candidate rejected at one-node launch
gate because engine init saw `mp_policy.reduce_dtype=torch.bfloat16` rather than
the hook-required FP32 reduce buffer.
AMDAHL-056 is a BF16 FSDP reduce tooling candidate rejected because it regressed
the same real-cache workload (`server_forward_backward_s=4.8815` vs baseline
`4.6591`) with no loss change.
AMDAHL-057 is an alltoall dispatch tooling candidate rejected because it OOMed
before writing a warmup row on the same real-cache workload.
AMDAHL-058 is a deferred loss-report reduce engine-screen candidate rejected
because it regressed server/API wall on the same real-cache workload; its local
engine reporting patch was reverted/not promoted.
AMDAHL-064 is a 2-node repeat-data=2 fatter-call screen that fit and improved
valid-token throughput, but it shifted same-capture loss/KL/hidden, so it is a
packing/aggregation diagnostic and not a science/default promotion.
Latest layer-cache artifacts are the real all-layer SGLang cache plus the
one-node chunk sweep; these are not science/default promotions until the
same-workload 4-node and static/K3 gates clear.

**2026-06-14 19:53Z update:** AMDAHL-041 used the now-working kube auth and an explicit `nodeSelector: node-group=nccl` to run a short 4-node slots trainer-only replay on idle nccl nodes `099/110/092/088`, then stopped controls and deleted the trainer pods. It did **not** touch slots samplers/teachers/dispatch or the separate science stack. The repeat-data ladder rejects the simple larger-prepare/0-dummy promotion: repeat 1/2/4/8 reconstructed logical MFU was only `1.424% / 1.625% / 1.718% / 1.831%`; best repeat8 was `20.33s` fb, `37.49` valid tok/s/GPU, `1008.96` dispatcher-executed tok/s/GPU. Full table and artifacts are in `THROUGHPUT_MICROBENCH_RUNBOOK.md` AMDAHL-041. No larger `opd_prepare_batch_size` is promoted.

**2026-06-14 20:16Z update:** draft PR #375 (`throughput/opd-vocab-parallel-kl`) is pushed through `1d7620c7` with opt-in model-runner wiring for `opd_kl_backend: vocab_parallel`. Offline tests cover uneven zero-token ranks, weighted OPD helper gradients/reporting, teacher-store row-range loading, and DTensor lm-head shard offsets. A real slots trainer-server smoke was attempted with a prepared teacher store at `/shared/apanda/opd_teacher_stores/qwen3_6_35b_a3b_lm_head_shard32768/manifest.json`; it failed the promotion gate before backward because the current OPD 1-node replay config uses `tensor_parallel_size: 1` / FSDP gathered `lm_head.weight`, while this implementation requires a DTensor vocab-sharded student `lm_head.weight`. The temporary trainer-head pod was stopped/deleted and no slots trainer pods remain. This is not a §1 promotion and science should not enable VP-KL by default. See `THROUGHPUT_MICROBENCH_RUNBOOK.md` "Engine Branch State — Vocab-Parallel KL" for commands and evidence.

**2026-06-14 20:49Z update:** AMDAHL-042 follow-up used a 1-node Ulysses8 + lm-head-only-TP topology (`lm_head_tensor_parallel_size=8`, `fsdp_sharded_lm_head_loss=true`) on PR #375 commit `5c17f409`. The server now accepts the lm-head TP config, threads the sharded-lm-head flag through `build_training_model`, and skips unnecessary size-1 lm-head replica grad all-reduces. A trainer-head-only replay with `opd_kl_backend=vocab_parallel` succeeded for `limit-data=8`:
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-limit8-serveronly-1x-20260614T2045-size1skip.jsonl`
(`server_forward_backward_s=9.7336`, `kl_compute_s=0.1389`, `valid_tokens=68`). The first same-server full64 replay after that limit8 request failed before KL in DeepEP forward dispatch metadata (`torch.argsort(sort_keys)` CUDA OOM with GPUs around 78-81 GiB); this OOM was later superseded by the fresh-server full64-first AMDAHL-042 diagnostic below. Server log is under run dir
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260614T204321Z-serveronly-configAMDAHL-042-OPRD-PREP64-1NODE-LMHEADTP-VPKL-er-opd-q36-35b-slots-trainer-head`. Cleanup completed: stopped trainer control, killed local replay/port-forward/tail helpers, and deleted `er-opd-q36-35b-slots-trainer-head`; the science stack was not touched. Still not a §1 promotion: prep64 memory fit, same-workload throughput comparison, and K3/static gate remain pending.

**2026-06-14 20:59Z update:** fresh-server AMDAHL-042 full64-first diagnostic changed the fit read. Full64 fit on one node cold (`server_forward_backward_s=138.7807`, `model_forward_s=85.0501`, `backward_compute_s=51.1890`, `kl_compute_s=0.4527`, `valid_tokens=515`) and stabilized after warmup at mean `server_forward_backward_s=24.8447`, `model_forward_s=9.3195`, `backward_compute_s=13.9033`, `kl_compute_s=0.1587`. Artifact:
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-full64-warmed-serveronly-3x-20260614T2058.jsonl`.

**2026-06-14 21:05Z update:** AMDAHL-043 4-node DP4 x Ulysses8 x lm-head-TP fidelity replay also fit full64. Cold replay was `server_forward_backward_s=135.5214`, `model_forward_s=84.5115`, `backward_compute_s=47.5739`, `kl_compute_s=0.2324`. Warmed 3x replay stabilized at mean `server_forward_backward_s=15.4878`, `model_forward_s=4.6513`, `backward_compute_s=9.4123`, `kl_compute_s=0.0672`, `valid_tokens=515`. Artifact:
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-4node-full64-warmed-serveronly-3x-20260614T2110.jsonl`.
Cleanup completed: killed the local port-forward, stopped trainer control with `--remove-run`, deleted trainer-head/workers, and verified no slots trainer pods remain. This is not a §1 promotion: it is a fit/memory milestone, but it does not beat the current strict 4-node path and has no K3/static gate.

**2026-06-14 21:30Z update:** AMDAHL-044/045 local no-CP lm-head-TP follow-up removed Ulysses by sourcing `lm_head_tensor_parallel_size=8` from the DP axis. AMDAHL-044 1-node warmed full64 mean was `server_forward_backward_s=4.7197`, `model_forward_s=1.2169`, `backward_compute_s=2.3254`, `kl_compute_s=0.0289`. AMDAHL-045 4-node topology log confirmed `source_axis=dp`, `source_replica=4`, `mesh=(4,8)` with `data_parallel_shard_size=32` and `ulysses_parallel_size=1`; warmed 3x full64 mean was `server_forward_backward_s=4.1209`, `model_forward_s=0.8699`, `backward_compute_s=1.9017`, `kl_compute_s=0.0121`, `valid_tokens=515`. Artifact:
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-4node-full64-warmed-serveronly-3x-20260614T2127.jsonl`.
This is still not a §1 promotion: the no-CP path reports loss `1.8479248` on AMDAHL-045 and `1.8495717` on AMDAHL-044, while AMDAHL-042/043 VP-KL reported `1.8686199` on the same replay. A new two-group DP-sourced VP-OPD unit test passed (reported-loss abs err `1.110e-06`), so the remaining suspect is full-model topology/checkpoint/packing parity rather than a simple VP loss-wrapper reduction duplicate. Cleanup completed and no slots trainer pods remain.

**2026-06-14 21:40Z update:** kube auth is confirmed working on the normal context (`apanda-admin@research-common-h100`, `kubectl auth can-i get pods -n apanda` = yes). A short AMDAHL-044 no-CP `limit-data=8` replay was attempted but could not schedule a one-node trainer-head (`37 Insufficient nvidia.com/gpu`). Deleting the stale idle throughput-owned `er-opd-q36-35b-slots-sglang-0` did not free a full node because another running job occupies `research-common-h100-001`; I did not touch unrelated users or `er-opd-q36-35b-sci`. Cleanup completed: pending trainer-head deleted, trainer control stopped with `--remove-run`, and no slots trainer pods remain. Added AMDAHL-046 (`candidates/AMDAHL-046-OPRD-PREP64-1NODE-NOCP-STREAMING-VS-VPKL.yaml`) as the next full-node diagnostic: run pure streaming-KL `--limit-data 8` on the same no-CP body topology, then compare against no-CP VP-KL `--limit-data 8`. §1 remains unchanged.

**2026-06-14 21:59Z update:** AMDAHL-046/044 same-topology no-CP limit8 parity ran after an nccl node opened. I launched only `er-opd-q36-35b-slots-trainer-head` on `research-common-h100-089` with explicit `nodeSelector: node-group=nccl`; no slots sampler/teacher/dispatch roles were restarted, and `er-opd-q36-35b-sci` was untouched. AMDAHL-046 streaming full-lm-head no-CP limit8 artifact:
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-streaming-nocp-limit8-serveronly-1x-20260614T2155.jsonl`
(`loss=0.6243904829`, `valid_tokens=397`, `server_forward_backward_s=80.2944`). AMDAHL-044 no-CP lm-head-TP VP-KL limit8 artifact:
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-limit8-serveronly-1x-20260614T2158.jsonl`
(`loss=0.6256709099`, `valid_tokens=397`, `server_forward_backward_s=8.0331`). VP minus streaming loss is `+0.001280427` (`0.205%` relative), so the live loss-equivalence gate failed. Cleanup completed: trainer control stopped with `--remove-run`, trainer-head pod deleted, and no slots trainer pods remain. This is not a §1 promotion; do not enable no-CP VP-KL in science/defaults.

**2026-06-14 22:32Z update:** AMDAHL-046/044 enriched triage reran on only `er-opd-q36-35b-slots-trainer-head` on `research-common-h100-077` after app-level server readiness. Streaming no-CP full-lm-head with hidden-match disabled + OPRD disabled reported KL-only loss `0.6243904829` (`opd_kl=0.6243904652`, hidden loss `0`, OPRD timings `0`) with both the captured HF teacher head and the sharded teacher-store manifest; artifacts:
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-streaming-nocp-hf-hidden0-enriched-serveronly-1x-20260614T221914Z.jsonl`
and
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-streaming-nocp-teacherstore-hidden0-enriched-serveronly-1x-20260614T222041Z.jsonl`.
Hidden-match-on streaming was `4.0513830` with hidden loss `3.4269925`, so hidden-match/config drift is not the small VP mismatch. VP hidden-match-off with the same teacher store still reported `0.6256709099` (`opd_kl=0.6256708486`, hidden loss `0`); artifact:
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-hidden0-enriched-serveronly-1x-20260614T222357Z.jsonl`.
The same-server streaming-on-lm-head-TP probe failed with mixed Tensor/DTensor and then poisoned the trainer for later traffic, so restart after any such failure. Cleanup completed: closed port-forward, stopped trainer control with `--remove-run`, deleted trainer-head, and verified no slots trainer pod remains. This is not a §1 promotion; the remaining suspect is live lm-head-TP student-weight materialization/loading or topology, not teacher store, hidden-match config, or standalone VP arithmetic.

**2026-06-14 23:24Z update:** AMDAHL-046/044 lm-head fingerprint replay found and
fixed the no-CP VP-KL dtype drift. Streaming full-lm-head was consuming bf16
student lm-head weights, while the no-CP lm-head-TP VP path was consuming fp32
local DTensor shards; the local model-runner patch casts the effective VP shard
to the hidden/model compute dtype for the loss. Same-server hidden0 limit8
parity is now `0.6181625724` streaming vs `0.6181625128` fixed VP. A 1-node
full64 all-layer OPRD replay still fits after the fix
(`server_forward_backward_s=7.6398`, valid tokens `3049`). The 4-node dtype-fix
trainer-only replay also completed:
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-dtypefix-4node-full64-warmed-serveronly-4x-20260614T2321.jsonl`,
mean `server_forward_backward_s=5.8661`, `model_forward_s=0.9778`,
`backward_compute_s=1.7877`, `kl_compute_s=0.0173`,
`oprd_teacher_forward_s=1.1042`, valid tokens `3049`, loss `0.5791455`.
Denominator audit:
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_dtypefix_4node_20260614.json`;
reconstructed logical MFU is `0.010405` (`~1.04%`). Cleanup completed: slots
trainer-head/workers deleted, no slots trainer pods remain, and
`er-opd-q36-35b-sci` was untouched. This fixes the dtype correctness drift but
is still not a §1 promotion: it misses the 10% MFU target by roughly `10x` and
has no static/K3 gate.

**2026-06-14 23:43Z update:** AMDAHL-031 SGLang-layer-cache replay was rerun on
the AMDAHL-044 no-CP lm-head-TP VP engine. The reduced every4 layer-cache payload
now fits and removes trainer-side OPRD teacher forward (`oprd_teacher_forward_s=0`),
but the CPU layer-cache fetch cost was visible without a device cache:
`server_forward_backward_s=4.9560`, `oprd_layer_fetch_s=0.2847`. A local opt-in
activation-cache device-residency patch reduced the same replay to
`server_forward_backward_s=4.3206`, `oprd_layer_fetch_s=0.00138`, with identical
loss `1.8533004`. Artifacts:
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-sglangcache-every4-1node-warmed-serveronly-4x-20260614T2340.jsonl`
and
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-sglangcache-every4-1node-layerdevcache-warmed-serveronly-4x-20260614T2343.jsonl`.
Cleanup completed: stopped trainer control with `--remove-run`, deleted only
`er-opd-q36-35b-slots-trainer-head`, and verified only slots dispatch +
teacher-smg remain. This is not a §1 promotion because the payload is reduced
every4 OPRD with only `515` valid tokens and has no 4-node/K3 gate.

**Performance queue (do this before waiting on science):**
1. Decide whether AMDAHL-014's one-step-stale pipeline semantics are acceptable for the next long science run. If yes, AMDAHL-014 is the current 64-prompt wall-time default candidate. If no, use AMDAHL-018 as the strict fresh-sample fallback.
2. Gate any quack+DeepEP long default with static/K3 coverage. This applies to AMDAHL-009, AMDAHL-012, AMDAHL-014, AMDAHL-016, and AMDAHL-018 because the dispatch swap is the correctness-sensitive part.
3. Attack tokens-per-rank-per-forward and current-node serial overhead, not
   node count. Strict AMDAHL-018 spends `prepare_s=5.27`,
   `forward_backward_s=4.68`, and `sync_inference_weights_s=3.37` per step;
   AMDAHL-020 trades one sampler for lower sync
   (`sync_inference_weights_s=2.35`) and improves strict wall time to `13.41 s`,
   but its prepare worsens to `6.15 s`. If adopting AMDAHL-020, release or
   restamp `sglang-1` so the improvement is also a GPU-efficiency improvement.
   For fwd/bwd-only hypotheses, use the microbench ladder in §7e instead of the
   full OPD stack. The default inner loop is the 4-node trainer-only replay: it
   replays the captured 64-row OPRD batch against the trainer API without
   samplers, teacher prefill, endpoint registration, optimizer, or weight sync.
   The validated replay baseline mean was `server_forward_backward_s=4.46`
   (`forward_compute_s=1.60`, `backward_compute_s=1.58`,
   `clear_gradients_s=0.63`) on the AMDAHL-021 capture. New sorted breakdown
   metrics split that forward bucket into student `model_forward_s=0.81`,
   trainer-side OPRD `teacher_forward_s=0.85`, and loss wrapper
   `loss_compute_s=0.91` on a `4.80 s` profiled replay; KL remains only
   `0.046 s`. Its rejected A/Bs include: 2-node is slower wall-clock
   (`5.63 s`), steady no-recompute is neutral/slower (`4.61 s`), pack2304 is a
   replay-only false positive (`4.16 s` replay, but `11.03 s` fwd/bwd in the
   real AMDAHL-026 strict run), EP=1 is slower (`5.87 s`), and minimal dummy
   rows are neutral (`4.46 s`). Historical 1-node OPRD attempts
   AMDAHL-029/030/031/032/033 OOMed before no-CP lm-head-TP VP-KL; the current
   no-CP VP path now fits every4 SGLang layer-cache replay, the AMDAHL-047
   synthetic all-layer cache stress replay, and a real all-layer SGLang cache
   one-node replay. The remaining gate is same-workload 4-node
   correctness/throughput plus static/K3. The capture packs 64 samples into 22
   real rows at 4096 with only ~1.8% row-padding overhead, then dispatcher
   dummy-fill adds 10 zero-loss rows so all 32 ranks execute one batch; the same
   batch also does 350,870 trainer-side teacher-forward tokens for OPRD.
   AMDAHL-028's static repeat ladder shows fatter real calls improve throughput
   but not enough: repeat 1/2/4 = 521/584/776 executed tok/s/GPU with minimal
   dummy rows. The immediate issue is not static pad waste or a large fixed
   server/API tax; it is low useful-token density plus an inefficient real OPRD
   fwd/bwd communication shape. Next priority is a same-workload 4-node
   real-cache replay/K3 gate; rank-occupancy-aware ragged packing/coalescing
   remains a separate lever once the 1-node path is scientifically valid.
4. Compile/shape-bucketing: AMDAHL-013 proved compile can now run through EP DCP load after the `_orig_mod` checkpoint alias fix, but it was not the winner (`step_total_s=14.67`, `valid_tokens_per_step_s=279.5`) and `forward_backward_s` did not improve. Bucket/pad server fb shapes before treating compile as a default path.
5. Fat-call validation of `53ef29a3` (dynamo-disabled checkpoint segments) at ≥5 rows/rank — green lifts the 128-sample OPRD sizing cap (§7c) for larger prompt-window throughput probes.
6. Science-side validations (014D ledger, ARITH-012 muon twin) remain important, but they are not blockers for this infra/performance track.

**New-work menu:** performance → decide AMDAHL-014 staleness boundary, then current-node OPRD prepare/fb/sync reduction and shape-bucketing; science → bootstrap-realization arms (science runbook §7.1) once the performance owner explicitly hands back capacity. Eval-pool Mooncake root-cause remains separate (§5, with Gleb).

**The day in one line:** OPD engine throughput 74→~1020 tok/s/GPU (~13.8×: engine flags + eval concurrency + 64 MB chunk cap + EP-dedup), trainer 64→32 GPUs (32 released), one corruption incident bisected and documented (§9c is the gate it produced), faithful-OPRD unblocked. Full handoffs: `docs/notes/engineering_handoff_20260612.md` + `docs/notes/handoff_throughput_session_20260612.md`.
---

## 1. Stack & topology

Stack `er-opd-q36-35b-slots`, namespace `apanda`. Model is **switchable via the generator's global `--model q35|q36` flag** (Qwen3.5-35B-A3B / Qwen3.6-35B-A3B — both MoE ~3B-active GatedDeltaNet+full-attn hybrids); as of 2026-06-10 the stack serves **q36** for the 4×4/arithmetic self-distillation work.

| Role | Pods | GPUs | Parallelism |
|---|---|---|---|
| Trainer | trainer-head + worker-1..3 | **32 (4×8)** — scaled down 2026-06-12, workers 4-7 RELEASED (`--trainer-nodes 4`; 8-node configs auto-remap to `_4node`) | FSDP=32, EP=8, quack (server path; triton equally fine) |
| Student samplers | sglang-0, sglang-1 | 4 (2×2) | **TP=2 each**, SGLang; p2p weight-sync receivers |
| Teachers | teacher-sglang-0 | **8 (1×8)** — AMDAHL-optimal after the capture fix: ONE TP=8 teacher hides under fb (§7d). Scale to 2–3 only if reverting the capture fix or growing prompts/step a lot | TP=8, SGLang prefill (HTTP, no weight sync) |
| eval-sglang-0/1 | 2 | 4 (2×2) | TP=2, deployed + idle — EXPERIMENTAL eval pool, do not register unattended (§4); released (`--eval-replicas 0`) in the AMDAHL-optimal config |
| dispatch | 1 | 0 | **SMG** router over the 2 samplers (round_robin) |
| teacher-smg | 1 | 0 | SMG router over the teacher(s); auto `power_of_two` (≥2 teachers) / `round_robin` (1). With 1 teacher prefer `--teacher-route direct` and skip this hop (§7d) |

Node groups: trainer + samplers on `nccl` (RDMA); teachers + SMG on `default`. All `nccl` nodes advertise `rdma/infiniband`.

## 2. Code / repos

The pods run **this worktree directly** via `XORL_REPO=/home/apanda/xorl-apanda-dev-opd-port` + `PYTHONPATH=$XORL_REPO/src` — uncommitted edits deploy on the next `write-*-control`. Branch **`apanda-dev-prefill-time-compute`** (= `apanda-dev` + the non-upstreamed research; `codex/opd-port-20260602` is superseded, left intact). Client SDK: `xorl_client` from **`/home/apanda/xorl-client`** — the ONLY client checkout (consolidated 2026-06-12 ~19:45Z: the old main repo + opd/issue-4/internal-work clones were snapshot-committed (`wip/*-era-snapshot-20260612` branches on `internal`) and deleted; the `apanda-dev` worktree was promoted to a standalone repo at this path; `/home/apanda/xorl-client-chat-completions` is now a compat SYMLINK for paths baked into running pods — don't delete it while pre-19:45Z pods live). Branch **`apanda-dev`**, remote `internal` (togethercomputer/xorl-client-internal); no PR flow on the client repo — push straight to `apanda-dev`. NB: `origin` there is the PUBLIC xorl-client repo; always push to `internal`. (`examples/on_policy_distillation.py`; chat sampling pins `chat_template_kwargs={"enable_thinking": false}` + `logprob_start_len=0` and FAILS LOUD on empty `input_token_ids` — never disable these, see §9b). SGLang: `/home/apanda/xorl-sglang-internal` (scheduler KV-desync fixed 2026-06-10 in `schedule_batch.py`; samplers run the UNPINNED default attention backend since 2026-06-10 ~21:48 — NB: SGLang's auto-selection resolves it to **fa3** for this model/hardware, so the explicit `--attention-backend fa3` pin was a no-op all along; teachers pin fa3 explicitly). Same image as the MTP stack: `nvcr.io/nvidia/pytorch:26.02-py3`.

```
GENERATOR=experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py
CONFIG=experiments/opd_profile/configs/qwen3_5_35b_a3b_opd_opdb_8node.yaml
CANDIDATES=experiments/opd_profile/autoresearch/candidates/
CONTROL_ROOT=/shared/opd-control/er-opd-q36-35b-slots
RESULT_ROOT=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots
PY=.venv/bin/python
```

## 3. Control model (reprogrammable slots)

Bare pods run a slot-agent that polls its `run.sh`; on a hash change it stops the old child process group and re-execs the new `run.sh`. Every `write-*-control` stamps a fresh revision, so it always forces a re-exec. `run_opd_pipeline.py` / the OPD client runs on trainer-head; all config is baked into `run.sh` by the generator (no CLI args at the pod). `python $GENERATOR status` shows child-process state (`kubectl get pods` only shows the slot pod is alive).

## 4. Generator subcommands

`$PY $GENERATOR [--model q35|q36] [--trainer-nodes 4] <cmd> [--candidate <path>] [--num-steps N] [--prompts-per-step P] [--sampler-replicas 2] [--sampler-layout dedicated]`

**`--model` (added 2026-06-09):** selects the served model for ALL roles (`q35` = Qwen3.5-35B-A3B default, `q36` = Qwen3.6-35B-A3B). It is a GLOBAL flag — place it BEFORE the subcommand, and pass the SAME value to every `write-*-control` when switching: teachers, samplers+dispatch, and trainer must all serve the same variant or sampling/teacher prefill silently mismatch the trainer weights.

- `write-trainer-control` — (re)launch the trainer pods for a candidate (count = `--trainer-nodes`, default **4**; workers beyond the active topology are auto-stopped).
- `write-trainer-server-control` — launch only the XORL trainer API/engine for trainer-side microbenchmarks. It does **not** wait on SGLang/teacher, register endpoints, run OPD, or sync weights. Use it with the fwd/bwd replay harness in §7e, then `stop-trainer-control --remove-run`.
- `write-student-inference-control` — (re)launch dispatch + sglang-0/1 (clears stale sampler/Mooncake state).
- `write-eval-inference-control` — (re)launch the dedicated eval-pool replicas (`--eval-replicas N`, global flag; 0 = none). Eval pods register with the trainer under `pool="eval"`, are EXCLUDED from the per-step weight sync, get refreshed by a `pools=["eval"]` sync at eval steps (~11 s), and serve the greedy/logprob evals via the client's `eval_inference_base_urls` (+ `eval_async=true` runs the periodic eval bundle in the background against their frozen weights; results land as late `row_kind="eval_async"` profile rows + wandb `eval_async/*`; the terminal control eval stays blocking for the autopilot). Pass the SAME `--eval-replicas` to render-manifest and write-trainer-control. **STATUS 2026-06-12: EXPERIMENTAL — do NOT use unattended.** The smoke test passed end-to-end, but in production the eval-pool refresh sync wedged twice in ~40 steps (ARITH-005 launches 1 & 4), and a wedged P2P sync is NOT survivable by skip-and-continue: the server-side handler can poison the trainer's collectives (observed: eval sync timeout at step 30 → fb 504 at 1800 s → run dead). Until the Mooncake multi-group / alternating-endpoint-set behavior is root-caused (talk to Gleb), run WITHOUT `--eval-replicas` — post-§11 blocking evals are only ~45–65 s anyway. The client code is hardened (180 s cap, auto-unpause on failure) but the trainer-side wedge is the unsolved part. eval-sglang-0/1 pods are deployed and idle (harmless when unregistered).
- `write-dispatch-control` — (re)launch just dispatch.
- `write-teacher-control --teacher-index 0 --teacher-index 1` — (re)launch the teacher sglang pods. NB: under the `dedicated` layout this does NOT rewrite teacher-smg, which caches `/v1/models` at startup — after a model switch, force its re-exec (restamp its run.sh or it will advertise the OLD model).
- `stop-trainer-control --remove-run` / `stop-control --remove-run` — stop trainer / all roles (always use `--remove-run`).
- `render-manifest --output <f>` — emit the full k8s manifest (pods+services).
- `render-control --role <r> --candidate <c> --output <f>` — dry-render a role's run.sh (no deploy).
- `status` — role child-process state.

Sampler topology is set by `--sampler-replicas 2 --sampler-layout dedicated` (= sglang-0 + sglang-1, each TP=2, non-privileged + `rdma/infiniband` + `IPC_LOCK`, GPUs picked at runtime by `select_assigned_cuda_devices`). `dedicated` keeps both teachers as pure teachers.

## 5. Trainer config (`qwen3_5_35b_a3b_opd_opdb_8node.yaml`)

```yaml
model_path: Qwen/Qwen3.5-35B-A3B
attn_implementation: flash_attention_3
moe_implementation: quack           # quack EP validated 2026-06-10 (PRs #353+#355: registration + half-concat numerics; the 0.905 reproduction trained on it); triton equally fine
ep_dispatch: deepep                 # DeepEP expert all-to-all
deepep_num_sms: 24                  # SMs for DeepEP comms (leaves the rest for GEMMs)
rmsnorm_mode: native                # required for OPRD (avoids the inductor compile path)
ce_mode: eager                      # ditto
enable_compile: false               # ditto
data_parallel_mode: fsdp2
expert_parallel_size: 8
ulysses_parallel_size: 1
gradient_checkpointing_method: recompute_before_dispatch
load_weights_mode: all_ranks        # loads HF safetensors directly (no DCP checkpoint exists for this model)
train_router: false                 # required with ep_dispatch=deepep
```

### 5b. Config inventory + optimizer warning (2026-06-12)

| config | shard | optimizer | status |
|---|---|---|---|
| `qwen3_6_35b_a3b_opd_opdb_8node.yaml` | 64 | adamw | the proven science ladder substrate |
| `qwen3_6_35b_a3b_opd_opdb_4node_adamw.yaml` | 32 | adamw | **proven** (ARITH-012A clean ramp) — the 4-node science default |
| `qwen3_6_35b_a3b_opd_opdb_4node.yaml` | 32 | muon 3e-5 (+`lr: 3e-6` anchor)/mom 0.95 | **FIXED 06-12 evening**: ARITH-012's 0.000 was a server bug (client `learning_rate` stomped ALL param groups → muon matrices at ~10× too-low lr), NOT the hypers. Server fix `e97d77e5` scales muon groups by `muon_lr/lr`; keep the config `lr` equal to the candidate's `learning_rate`. PENDING one live re-validation (012 muon twin, expect ≈0.218 — §0 queue) before the muon directive is unblocked; until then 4-node science default = `opdb_4node_adamw.yaml` |
| `..._warm009.yaml` / `..._4node_warm009.yaml` | 64/32 | adamw (matches 009 ckpt state) | warm-start substrate, proven |
| `..._warm009_oprd.yaml` (+4node) | 64/32 | adamw, rmsnorm native + ce eager | exonerated by the bisect (its "divergence" was the client bug) but UNVALIDATED clean — re-test before use; OPRD also runs on the standard substrate (014D step-0 healthy) |
| `..._eff*_4node_muon.yaml`, EFF-PROBE-* | 32 | muon | throughput probes only, NOT science |

**The `_8node`→`_{N}node` auto-remap under `--trainer-nodes N` is substring-based** — a candidate pinning an `_8node` config at `--trainer-nodes 4` silently lands on the `_4node` twin. Twins MUST keep optimizer compatibility with any warm-start checkpoint (DCP load includes optimizer state).

## 6. Candidate schema (`candidates/PTC-*.yaml`)

A candidate is the full recipe. Key fields: `id`, `slug`, `trainer_config`, `student_prefill_text/count/suffix`, `student_stop_sequences`, `teacher_cot_mode` (`replace` | `match_cot`), `teacher_cot_json_path`, `prompts_json_path`, `num_prompts`, `default_num_steps`, `default_prompts_per_step`, `learning_rate` (3e-6 proven), `eval_*`, `sync_method: p2p`, `serial_endpoint_sync: true` (NB: a NO-OP at HEAD — commit `1b9bc6fe` removed the serial loop from `inference_endpoints.py`; sync has been parallel fanout all along, so extra endpoints do NOT add serial sync cost), `trainer_nccl_*` for trainer-only transport probes, `trainer_minimal_dummy_batch_tokens` for replay-only dummy-row probes, `opd_*` (loss knobs), `client_args` (incl. `opd_oprd_layers`, `opd_oprd_num_layers`, `opd_buffer_equals_cot`, `opd_oprd_last_k` for OPRD).

Trainer batching knobs (candidate-settable since 2026-06-12): for large enough prompt windows, **recommended `opd_prepare_batch_size: 256` + `opd_microbatch_size: 256`** (~1020 tok/s/GPU engine; loss-parity validated). Do NOT exceed the measured OPRD 128-sample point until the `53ef29a3` fat-call fix passes its ≥5-rows/rank validation (§0 queue) — beyond that boundary the CheckpointError hang costs an 1800 s fb timeout. For the current 64-prompts/step OPRD science window, **use `opd_prepare_batch_size: 64` + `opd_microbatch_size: 64`** to keep one fb call per step without changing the prompt window. AMDAHL-014 is the measured wall-time winner on that shape (`step_total_s=13.22`, `valid_tokens_per_step_s=335.9`) because the fixed pipeline path overlaps the next prepare; its science boundary is one-step-stale samples. If stale samples are not accepted, AMDAHL-018 is the strict fresh-sample fallback on the same prep64+quack+DeepEP36 stack plus strict prepare chunk4 (`step_total_s=14.03`, `valid_tokens_per_step_s=292.5`, `prepare_s=5.27`). If the science owner accepts a 128-prompts/step optimizer window, AMDAHL-012 (`opd_prepare_batch_size: 128`, `opd_microbatch_size: 128`) remains the larger-window throughput artifact at 1.19× AMDAHL-018 valid-token/s / prompt/s. AMDAHL-011's trainer NCCL socket/buffer probe was healthy but slower, AMDAHL-013 compile was not the winner, and AMDAHL-017 no-profile-sync did not improve wall time, so do not promote them. If strict science correctness gating is required before a long run, use AMDAHL-008 until quack+DeepEP passes static/K3.

Dummy-row probe knob (candidate-settable since 2026-06-13): `trainer_minimal_dummy_batch_tokens: N` exports `XORL_SERVER_MINIMAL_DUMMY_BATCH_TOKENS=N` on the trainer. It makes dispatcher padding rows short and removes OPRD teacher/cache fields from zero-sample dummy batches while preserving collective participation. Default is off; AMDAHL-028 with `N=128` was neutral on the captured replay, so treat it as a safe cleanup probe only, not a throughput promotion.

Eval-throughput knobs (raised 2026-06-11 — the old values were the client-side eval bottleneck): `eval_control_max_concurrency: 128` (was 16 baked / candidate-set 128) and `eval_answer_logprob_max_concurrency: 16` (was 2 — at 6144 logprob requests/cycle that serialized ~96 chunks × ~32 s ≈ 780 s, measured live on ARITH-007F's terminal control). All candidate YAMLs updated; generator defaults match.

New knobs (2026-06-10):
- `eval_answer_logprob_every: 5` (+ `eval_answer_logprob_distractor_control: true`) — periodic gold-answer logprob/margin on the eval set, independent of the control-eval gate.
- `client_args.eval_prompts_json_path` — **REQUIRED**: a DISJOINT eval file (e.g. `/shared/opd-coord/randnum_4digit_eval_fresh_1024.json`). The default (pool tail) is inside the training rotation and inflates controls by ~+0.08 once windows wrap.
- `client_args.sft_mode: true` (+ `opd_teacher_answer_source: gold`) — SFT ablation: plain CE on the gold-spliced answer tokens, NO teacher prefill; everything else identical.
- Client defaults that apply to every run: `eval_heldout_num_problems: 256` (the per-step REAL `eval/accuracy`, greedy held-out), `chat_template_kwargs: '{"enable_thinking": false}'`.

**Control-eval rule (must hold):** `eval_control_start_step = num_steps − 1` **and** `eval_control_start_step % eval_accuracy_every == 0`. With `eval_accuracy_every=5`, pick `num_steps ≡ 1 (mod 5)` → e.g. 81/ctrl-80, 101/ctrl-100, 321/ctrl-320.

### 6b. Loss-objective knobs (2026-06-11)

- **`opd_mask_prompt_kl` — client default is now `true`.** The historical recipe computed KL on prompt positions too (~70% of supervised tokens on 4×4, ~84% on arithmetic) — an anchor-to-base term that deviates from reference OPD (verl GKD masks KL to response positions) and diluted the per-answer-token gradient ~3–8×. With the mask, OPD supervises exactly the positions `sft_mode` does. **Every run before 2026-06-11 (incl. the 4×4 0.905) effectively ran `false` — pin `opd_mask_prompt_kl: false` to reproduce them.** PTC-122 measured the effect: corrected 4×4 OPD ramps 0.68→0.90 by step 15 (= SFT's ramp; the anchored recipe needed ~100 steps), peaks 0.922@25, then drifts — **early-stop corrected-objective runs (~step 25 on 4×4)**.
- **`opd_loss_mode: forward_kl_full` requires `opd_kl_backend: torch_compile`** (candidate key, generator default `streaming`; the streaming kernel is reverse-KL-only — a `streaming`+forward run dies at step 0 in `server.log`, the head log just stalls). Forward KL is the right direction when supervising positions where the student's distribution is far from the teacher's (e.g. pause-buffer↔CoT matching: reverse KL's gradient vanishes when q is a delta off-target AND pins the loss clamp; measured `clamp_frac` 0.99 reverse vs 0.004 forward at clamp 30).
- **`save_every`** — candidate-driven (default 0). `save_every: 101` on a 101-step run = one DCP checkpoint at the end (fires at `(step+1) % save_every == 0`); path lands in `save_checkpoint_path` in the profile row. Pair with `load_checkpoint_path` + `load_weights_mode: skip` in a trainer-config variant to warm-start a follow-up run (SFT→OPD).
- **`opd_loss_max_clamp`** matters more under the prompt mask: with only answer/buffer tokens in the loss, the clamp's reach is no longer hidden by near-zero prompt terms (ARITH-004 had 33% of answer tokens clamp-dead at 5.0; raising to 10 was worth +0.03 control). Check `opd_loss_clamp_frac` at step 0 of any new objective.

Data: `/shared/opd-coord/randnum_5digit_q35_le6000_{prompts,cot}.json` (5179 ex, prompts+CoT index-aligned, non-empty).

## 7. Launch a run (standard recipe)

With the inference stack already warm (samplers + teachers + dispatch `1/1 Running`):

```bash
cd /home/apanda/xorl-apanda-dev-opd-port
$PY $GENERATOR write-trainer-control \
  --candidate experiments/opd_profile/autoresearch/candidates/<PTC>.yaml \
  --num-steps <N> --prompts-per-step <P> \
  --sampler-replicas 2 --sampler-layout dedicated
```

Bring-up ≈ 5–8 min (64-rank rendezvous + HF load + fresh 2-endpoint sync), then training. Healthy step-0 sequence in the head log: `engine ready after N polls` → `Registering SGLang endpoint 0/1` → `Probing SGLang endpoint 0/1 after fresh sync` → `Running OPD config=…` → `=== OPD step 0 ===` → step profile with `sync_endpoint_success_count: 2`.

**Budget guidance:** no-filler ≈ 34–47 s/step at 128 prompts/step (SFT ≈ 42 s); buffer arms (K=1024) ≈ 47 s/step at 64 prompts/step **plus ~14 min per eval cycle — see §7b**. OPRD K=C ≈ 70 s/step at 16 prompts/step (8192-tok sequences; keep prompts/step ≤16 for OPRD memory).

## 7b. Throughput reality & the Amdahl rule (measured 2026-06-11; LARGELY RESOLVED same day)

**The stack WAS eval/sampling-bound.** Original measurements (ARITH-007F, old engine flags, 64 prompts/step ×G=8, 1024-token pause buffers, evals every 5) vs after the §11 engine-flag + §6 eval-concurrency fixes (ARITH-009, same 1024-pause workload, 128 prompts/step):

| component | old flags (007F) | new flags (009, measured) |
|---|---|---|
| plain step (wall) | 47 s (fb 16 s, sync 3.5 s) | ~38 s (fb ~34 s — SFT arm) |
| eval step (wall) | **844 s** | **45 s** (held-out greedy adds ~6 s); step-0 full logprob battery 116 s total |
| run wall (101 steps) | ~6 h | **~70 min** |
| trainer MFU | ≈1.1% | fb-bound again, but still low utilization; keep trainer at 4 nodes and improve per-node work density before any scale-out |

(Engine-side numbers above predate the EP-dedup un-bake — OPD engine throughput is now ~935–1020 tok/s/GPU at 128–256-sample fb calls (§7c), vs ~310 under the duplicated dispatch these rows were measured on.)

The 844 s decomposed as ~780 s logprob battery at the old `eval_answer_logprob_max_concurrency=2` (96 chunks × ~32 s, fully serial — measured live) + greedy evals strangled by the old 16k-token KV pool and 512/1024 prefill-chunk caps. §9b regression gate at the new effective concurrency: PASSED 2026-06-11 on all three conditions (old-flags/trained 0.609@8 vs 0.594@128; new-flags/base 0.656 vs 0.664; new-flags/trained 0.758 vs 0.766; cap-hit 0 throughout).

**Known residual (fix staged):** the conc-16 logprob battery puts 1024 requests in flight per client vs 256-running+512-queued servers → queue-overflow rejections (`eval/answer_logprob_request_failure_frac` 0.4167, scored n 1024→~598/arm — diagnostics only; greedy evals show 0 failures). `--max-queued-requests 2048` is staged in the generator and deploys on the next sampler restamp; until then read logprob-battery stats as n≈600/arm.

Buffer arms remain the worst case structurally: the pause block sits *after* the distinct prompt, so nothing shares (and `--disable-radix-cache` is set anyway).

**Sizing rule — be Amdahl-optimal without hiding low MFU behind more trainer nodes.** Per step-cycle, aggregate sampler/teacher work, trainer fb, and weight sync should be balanced. After AMDAHL-018 at 64 prompts/step, the current allocation is still dominated by current-node work rather than a lack of trainer nodes: strict fresh-sample `prepare_s=5.27`, `forward_backward_s=4.68`, and P2P `sync_inference_weights_s=3.37` are the measured critical-path pieces. If AMDAHL-014's one-step-stale pipeline semantics are accepted, pipeline overlap removes prepare from the critical path (`prepare_window_s≈0.0004`), leaving true OPRD `forward_backward_s=9.36` and P2P `sync_inference_weights_s=3.39` as the next measured targets. Levers, in order:

1. **Keep the trainer at 4 nodes** (`--trainer-nodes 4`, now the generator default). FSDP=32/EP=8 via the auto-remapped `_4node` trainer config. Workers beyond the active topology are auto-stopped by `write-trainer-control`; their PODS keep their GPUs until you `kubectl delete` them. Do **not** recommend trainer-8 as an MFU fix: it reduces wall time by spending more GPUs while the current 32 GPUs are still under-filled.
2. **Reduce current-node work first.** At prep64, adding sampler or trainer nodes only masks the problem. The measured strict targets after AMDAHL-018 are same-step prepare, OPRD forward/backward, and the ~69 GB/step P2P sync to two endpoints. The prepare path is already overlapped by pipeline RL only when AMDAHL-014's one-step-stale science boundary is acceptable.
3. **Eval cadence** (`eval_accuracy_every`, `eval_answer_logprob_every`): only after the performance target is clear, and never below the resolution the science needs — the held-out trajectory is the gate metric (science runbook §5); a peak-then-decay missed between sparse evals costs a rerun, which dwarfs the savings.
4. Eval concurrency (`eval_control_max_concurrency`, logprob batch/concurrency) — usually already at the samplers' capacity; raising past it just queues.

Corollary: **don't read `RUN COMPLETE` ETAs off step time alone** — a 101-step buffer-arm run is ~6 h at cadence 5, ~3 h at cadence 10. And the end-of-run control legitimately holds the profile silent for 15–25 min (1024×2-arm decodes + 6k logprob requests); staleness watchdogs need ≥30 min thresholds on buffer arms (a 20-min alarm false-fired 2026-06-11).

## 7c. EP-group batch duplication REMOVED (2026-06-12, commit `5d42978f`)

The server fb dispatch used to give all `ep_size` ranks of an EP group the
same packed batch (4 distinct slices on a 32-rank EP=8 trainer → 8×-redundant
compute; root-caused in the throughput probes,
`skills/xorl-throughput-tuner/benchmarks/qwen3_6_35b_a3b/results/opd_server_path_probes_20260612.md`).
As of `5d42978f` every rank gets a **distinct** slice (CP/SP ranks still
share). Gradients are provably identical (loss is normalized by global valid
tokens, so the 8× numerator/denominator cancelled; OPD full-vocab KL +
OPRD teacher forward are rank-local; dummy-padding already guarantees uniform
collective counts). Implications:

- **Trainer restarts pick this up automatically** (pods run the worktree
  live). First at-scale exposure: ARITH-014 (2026-06-12 08:00Z).
- Fatten fb calls: there are now 32 data slices on a 4-node trainer, not 4.
  **Measured sweet spot (probes E2/E3, 09:20Z): `opd_prepare_batch_size: 256`
  + `opd_microbatch_size: 256`** → ~1020 tok/s/GPU engine (4.4× old dispatch),
  loss parity. Do NOT exceed ~2 packed rows/rank (≈256–384 samples/call):
  ≥5 rows/rank trips a pre-existing `recompute_before_dispatch`
  CheckpointError (router tie flips on recompute → collective desync → 1800 s
  fb timeout; same root cause as probe D4's fat-call hangs, see the probes
  doc item 4 — it predates the EP-dedup change).
- For the 64-prompts/step OPRD science regime, the largest safe same-objective
  call is `opd_prepare_batch_size=64` / `opd_microbatch_size=64`. AMDAHL-008
  collapsed four 16-sample fb calls into one 64-sample call on the same 4-node
  trainer and measured non-warmup `forward_backward_s=4.91` mean with intact
  OPRD health (`opd_frac_answer=1`, `opd_oprd_num_layers=40`, sync 2/2).
  AMDAHL-009 kept that shape and switched the trainer to quack+DeepEP24,
  measuring `forward_backward_s=4.78`, `step_total_s=15.11`, and
  `valid_tokens_per_step_s=267.9`; it is now the DeepEP24 baseline, pending
  static/K3 before long science default. AMDAHL-016 swept the same strict
  fresh-sample path to `deepep_num_sms=36`, measuring
  `forward_backward_s=4.60`, `step_total_s=14.30`, and
  `valid_tokens_per_step_s=285.4`. AMDAHL-018 kept DeepEP36 and added strict
  same-step prepare chunk4, measuring `prepare_s=5.27`,
  `teacher_prefill_s=2.59`, `forward_backward_s=4.68`, `step_total_s=14.03`,
  and `valid_tokens_per_step_s=292.5`; this is the current strict
  fresh-sample fallback. AMDAHL-014 fixed the client OPRD pipeline guard so pipeline RL
  actually overlaps the next prepare, measuring `step_total_s=13.22`,
  `valid_tokens_per_step_s=335.9`, `prepare_window_s≈0.0004`, and
  `pipeline_rl_active=1.0`; this is the 64-prompt wall-time winner if
  one-step-stale samples are acceptable.
  AMDAHL-012 kept DeepEP24 and doubled the prompt window to 128, measuring
  `step_total_s=23.16` for 128 prompts and `valid_tokens_per_step_s=348.9`
  (1.30× AMDAHL-009 and 1.19× AMDAHL-018 prompt/s on the same nodes); promote
  only if the larger optimizer-step window is acceptable. AMDAHL-010
  (`no_recompute`) fit but regressed `step_total_s` to 15.42 from
  optimizer/sync variance. AMDAHL-011
  added trainer-only NCCL socket/buffer tuning (`nsocks=8`, `socket_nthreads=4`,
  `buffsize=8388608`) and also regressed (`step_total_s=15.35`,
  `sync_inference_weights_s=3.35`). AMDAHL-013 compile ran after the EP DCP
  alias fix but was weaker than pipeline (`step_total_s=14.67`,
  `valid_tokens_per_step_s=279.5`). AMDAHL-017 disabled CUDA sync profiling
  barriers and fit, but was slower than AMDAHL-016 and AMDAHL-018
  (`step_total_s=14.48`, `valid_tokens_per_step_s=283.3`). Do not promote
  010/011/013/017.
  AMDAHL-018 was measured before the profile-row aggregation patch that retains
  `strict_prepare_overlap_*`; future rows have those diagnostics, but use the
  AMDAHL-018 profile/log metrics above for this run.
- Rollback switch if a run misbehaves at step 0 (collective desync/hang):
  `export XORL_SERVER_EP_DUPLICATE_BATCHES=1` in the trainer run.sh env
  (restamp via write-trainer-control) restores the legacy duplicated dispatch.
- Loss METRIC values are unchanged in expectation; `global_valid_tokens` now
  reports the true (deduplicated) token count — historical comparisons of
  that metric across the boundary shift by ~8×.

**Metric boundary:** `global_valid_tokens` (and anything derived from it) was 8×-inflated under EP with the old duplicated dispatch — do not compare it raw across the 2026-06-12 dedup boundary.

### 7c-addendum (UPDATED 19:15Z): rollback UN-BAKED — distinct dispatch live for OPD

- **The bake is GONE** (was: generator forcing `XORL_SERVER_EP_DUPLICATE_BATCHES=1`, the emergency rollback from the 06-12 corruption incident; the bisect exonerated the dedup — the corruptor was the client-side remap change, §9c). Un-bake gates both GREEN on warm009 (2026-06-12 ~19:00Z, distinct dispatch active): §9c step-0 KL = **0.536** (≈0.5 healthy), 6-step trajectory held-out **0.191→0.188**, opd_kl 0.55→0.50, empty_frac 0, no cap-rambling, clean exits. OPD runs now inherit the 3–4.4× engine win (~935–1020 tok/s/GPU at 128–256-sample calls).
- Residual vigilance (small): the gates covered the KL/teacher-cache path; the **OPRD gather under distinct dispatch** is ✅ VALIDATED (014D step-0 @19:18Z: kl 0.575, oprd_layers 40, oprd_raw 0.0107, frac_answer 1.0, seed-intact 0.180, no OOB — the dual-view fix + distinct dispatch compose), and the **first FULL-length science run** on distinct dispatch should get its trajectory eyeballed against the known curves (005: 0.19→0.10–0.12 band; SFT: monotone ramp) before treating the dispatch as fully proven.
- The dedup is upstream: apanda-dev PR #364 (`ff262108`). Per-run rollback remains available by exporting `XORL_SERVER_EP_DUPLICATE_BATCHES=1` on the trainer (code-level switch; no longer generator-baked). Record: `docs/notes/engineering_handoff_20260612.md` §2.

## 7d. AMDAHL verdict — current allocation + run-time estimate (2026-06-13)

The Amdahl audit (`docs/notes/amdahl_allocation_20260613.md`) walked the bottleneck through **three stages** and ended trainer-bound. The headline: **~293 s/step → ~32 s/step (~8–9×), end-to-end, correctness-neutral.** The bottleneck arc:

1. **eval/sampling-bound** → resolved §7b (engine flags + eval concurrency).
2. **teacher-prefill-bound** → the teacher prefilled the full sequence on ONE TP=8 endpoint while a sibling sat idle (client routed `direct` to `teacher-sglang-0`, bypassing the smg). Reshape to 3 teachers + smg `power_of_two` routing → ~2.2× (~293→~130 s).
3. **THE win — teacher capture fix** → the teacher was at ~0.5 % MFU (~690 tok/s/GPU) because it did `seq_hidden.cpu().clone().tolist()` over `CaptureHiddenMode.FULL` (~1 GB of Python floats/forward) then discarded ~97 % of it (kept only the ~32 answer rows of ~1086). Fix threads per-sample `teacher_hidden_keep_indices` through the request and indexes `seq_hidden[keep]` **on-GPU** before `.cpu()`. **Bit-exact** (max|new−old|=0), **14.6×** teacher throughput (690→~9000 tok/s/GPU). Committed+pushed: `xorl-sglang-internal` `apanda-dev` `89cc195c6..458a89a68` (6 files; transparent to the client — deploy = restart teacher pods on the new SGLang). This obviated the radix prefix-reuse lever (#2b — which FAILS GDN/mamba continuation parity anyway, ~6 % layer-0 drift; do not enable `--teacher-radix` for distillation).

After the capture fix the teacher (~18–21 s wall, even on **one** endpoint) and the samplers (~18 s) both hide **under** the trainer fb (~27 s) -> the step is now **trainer-fb-bound** (`step ≈ fb`). The remaining infra work is to raise current-node trainer MFU by coalescing fb calls and fixing shape/compile/kernel efficiency, not by adding trainer nodes.

### Current allocation (Amdahl-balanced without scale-out)

| role | pods | GPUs | why this count |
|---|---|---|---|
| Trainer | head + worker-1..3 | **32 (4×8)** | fixed trainer budget for infra tuning. At the 512-prompt AMDAHL point `step ≈ fb ≈ 25–32 s`; at the actual 64-prompt OPRD science window, AMDAHL-018 strict fresh-sample measures `step_total_s≈14.0 s`, while AMDAHL-014 measures `step_total_s≈13.2 s` only by accepting one-step-stale pipeline samples. Optimize current-node call density/shape/sync before considering any explicit scale-out. Node count is **quantized to {1,2,4,8}** — FSDP shard dim = nodes (EP=8 => shard = nodes*8/8), must evenly divide param dim 2048; **6 is FSDP-INVALID** (`uneven sharding on dim 1`). |
| Teacher | teacher-sglang-0 | **8 (1×8)** | post-capture-fix, ONE TP=8 teacher (~18–21 s) hides under fb. The 2nd/3rd teachers from the reshape are now **waste** at trainer-4. |
| Samplers | sglang-0/1 | 4 (2×2) | ~18 s, hidden under fb. |
| dispatch / teacher-smg | 2 | 0 | routers. |
| **TOTAL** | | **~44** | vs 64 for the 3-teacher reshape — **~20 GPU freed for the same step time.** |

Launch it: `--trainer-nodes 4 --teacher-replicas 1 --teacher-route direct --eval-replicas 0` (global flags; pass to render-manifest + write-trainer-control). **With one teacher use `--teacher-route direct`** (client → `teacher-sglang-0:30000`, no smg hop) — *not* `smg`: the smg's `power_of_two` policy raises `IncompatibleConfig: requires at least 2 workers`, the gateway never comes up, and the trainer **hangs forever** on "Waiting for teacher" (a *hang*, not a crash — crash-only monitors miss it). The launcher now auto-falls-back to `round_robin` for a single smg worker, but `direct` is cleaner for 1 teacher.

### Throughput estimate (so you can predict run wall-clock)

- **Steady step ≈ 32 s** (range 31–36 s) at **512 prompts/step**, trainer-fb-bound. fb ≈ 27 s = 8 microbatches × ~3.4 s: **backward 13.3 s (49 %), forward 5.8 s (21 %), MoE-EP-alltoall+FSDP-comms gap ~6.5 s (24 %), full-vocab KL only 0.37 s (1 %)**. fb is NOT KL-bound here (that's the sibling Wordle 58k-think regime) — it's the model fwd+bwd over the **1024-token pause-padded** sequences.
- **Cold start ≈ ~6 min**: trainer model reload at 4 nodes ~4 min + step-0 ~126 s (Triton compile + first weight sync).
- **Run wall ≈ ~6 min + (num_steps × ~32 s) + eval overhead.** Eval steps add ~45 s (held-out greedy ~6 s; the logprob battery dominates at eval cadence). Worked example: a **101-step science run at eval cadence 5 ≈ ~75 min** (matches §7b's ARITH-009 ~70 min). Double cadence ⇒ roughly double eval overhead.
- **Token math:** the trainer forwards **~556 k tok/step to supervise only ~22 k (~4 %)** — 96 % is the 1024 pause tokens. Engine-side fb throughput is ~935–1020 tok/s/GPU at 256-sample calls (§7c); the *whole-stack* per-GPU number is lower because samplers/teacher idle ~⅓ of each step under fb (that idle headroom is what lets 1 teacher + 2 samplers hide).
- **Caveat:** buffer/OPRD arms hold the profile silent **15–25 min** at end-of-run (1024×2-arm decodes + ~6 k logprob requests) — staleness watchdogs need ≥30 min thresholds on those arms.

### To go faster without wasting GPUs: raise current-node MFU

Do **not** chase MFU by adding trainer nodes. The current 4-node trainer is under-filled; trainer-8 would make the utilization problem easier to miss while increasing GPU spend.

Current infra levers, in order:
1. **Use larger same-node fb calls and overlap before adding GPUs.** ARITH-020 at 64 prompts/step used `opd_prepare_batch_size=16`, producing four fb requests per OPD step. AMDAHL-007 (`32`) and AMDAHL-008 (`64`) proved that coalescing the same 64 prompts is the immediate MFU lever; AMDAHL-009 added quack+DeepEP24, AMDAHL-016 improved the strict fresh-sample path with `deepep_num_sms=36`, and AMDAHL-018 improved strict same-step prepare on the same nodes with chunk4. AMDAHL-014 fixes pipeline+OPRD composition and is the current 64-prompt throughput winner if one-step-stale samples are acceptable. AMDAHL-012 doubled the prompt window to 128 and improved prompt/s + valid-token/s by ~1.19× over AMDAHL-018 on the same 4-node trainer, but changes the optimizer-step window. If the dispatch swap is not K3-gated yet, AMDAHL-008 is the conservative all-to-all fallback for the 64-prompt shape.
2. **Do not rerun no-recompute or NCCL socket/buffer tuning as the next idea.** AMDAHL-010 (`gradient_checkpointing_method: no_recompute`) fit but did not improve whole-step throughput: fb improved slightly, but optimizer/sync variance made `step_total_s` worse than the AMDAHL-009 DeepEP24 baseline and far behind AMDAHL-018. AMDAHL-011 (`NCCL_NSOCKS_PERTHREAD=8`, `NCCL_SOCKET_NTHREADS=4`, `NCCL_BUFFSIZE=8388608`) was healthy but did not reduce sync or whole-step time.
3. **Attack current-node OPRD prepare/fb/sync.** With strict AMDAHL-018, same-step prepare (`5.27 s`), forward/backward (`4.68 s`), and P2P weight sync (`3.37 s`) dominate the step. With AMDAHL-014, prepare is overlapped away on the critical path (`prepare_window_s≈0.0004`), while true OPRD fb is bimodal around `forward_backward_s=9.36` and P2P weight sync is ~3.4 s. More trainer nodes hide the issue; more samplers only attacks half of prepare and can increase sync fanout.
4. **Compile/shape-bucketing is still a model-kernel path, but not yet the winner.** AMDAHL-013 (`enable_compile=true`, decoder `torch.compile(dynamic=True)`) now runs through EP DCP load after the `_orig_mod` alias fix, but it measured `step_total_s=14.67` and did not improve fb. AMDAHL-017 proved that simply disabling CUDA sync profiling barriers is not the missing win (`step_total_s=14.48`). Bucket/pad server fb shapes before treating compile as a default path.
5. **Do not jump past 128 prompts/step until the fat-call gate is green.** §7c measured ~935–1020 tok/s/GPU engine-side at 128–256 sample calls on the server path, but the OPRD 128-sample smoke is the largest clean measured point here. A 256-sample OPRD probe needs the `53ef29a3` ≥5-rows/rank validation first.
6. **Fewer pause tokens is science, not infra.** The 1024 pause tokens are the majority of compute in sampler, teacher, and trainer fb. Cutting them changes the recipe and belongs in `CANONICAL_SCIENCE_RUNBOOK.md`.

Full arc + per-lever evidence: `docs/notes/amdahl_allocation_20260613.md`; memory `project_amdahl_opd_teacher_bottleneck_correction`.

### 7d-addendum — MEASURED throughput at 64/128 prompts per step (the buffer-arm science regime, 2026-06-13)

§7d's ~32 s/step is the **512-prompt AMDAHL-optimal** point. The actual OPD/OPRD science wave (ARITH-015→020, warm009, 1024-pause, `ce_mode: quack_linear`) ran at **64 prompts/step** (memory-bound by the 1024-pause sequences) — a smaller, less GPU-efficient call. Baseline measured on 6 runs (101 steps each, trainer-4 / 1 teacher / 2 samplers):

| quantity | measured |
|---|---|
| **plain (non-eval) step** | **~25–27 s** (fb ~17.7 s ∥ prepare ~17.4 s pipelined + sync ~3.2 s) |
| forward_backward | ~14–19 s, **4 fb calls** from `opd_prepare_batch_size=16` (vs larger 128/256-sample calls in §7c) |
| prepare (on-policy sample + teacher prefill) | ~16–20 s, **fully hidden under fb** (so step ≈ fb + sync, not fb + prepare) |
| weight sync (P2P, 2 endpoints) | ~3.2 s |
| eval step (held-out greedy only) | ~31 s |
| eval step + full logprob battery | **~100 s** (battery dominates) |
| **run-average step** | **~36–38 s** at eval cadence 5 + battery every 5 (R1–R4); **~34.5 s** at cadence 10 / battery 20 (R019) |
| **101-step run wall** | **~61–64 min** |

- **Baseline token/math shape at 64/step:** forwards **~69 k tok/step** (64 × ~1085: ~55 prompt + 1024 pause + ~5 answer) to supervise only **~3–4 k** answer tokens (post-mask, ~5 %). ARITH-020 split that work into **four 16-sample fb calls**. On a 32-rank trainer this was below the natural one-real-row-per-rank floor, so dummy padding and per-call distributed overhead dominated. The baseline whole-step rate was **~3,900 forwarded tok/s aggregate ≈ ~120 forwarded tok/s/GPU**, far below §7c's ~935–1020 tok/s/GPU at 128–256-sample calls.
- **Throughput lever for the science regime:** eval overhead is a big slice of the run wall here — the logprob-battery eval step is ~100 s vs a ~26 s plain step, so **eval cadence + `eval_answer_logprob_every` dominate the run-average** once plain steps are this cheap. R1–R4 (cadence 5 / battery 5) averaged ~38 s/step; R019 (cadence 10 / battery 20) averaged ~34.5 s. For throughput-sensitive science runs, widen the logprob-battery cadence first (it's diagnostic-only) before touching fb.
- **OPSD-Wordle cross-check (updated 2026-06-13):** the sibling OPSD pass in `~/xorl-opsd-wordle-apanda-dev-run-20260607/experiments/zorl/THROUGHPUT_DEBUGGING_HANDOFF.md` and the revised note at `/home/apanda/xorl-mtp-singleshot-port-20260602/docs/notes/opsd_low_mfu_microbench_20260613.md` are useful as a decomposition checklist, not as a transferable numeric prescription. Their CUDA-synchronized profile found OPD KL/top-k was only ~0.5% of f/b and checkpointing-off was neutral; our AMDAHL-018 profile agrees in kind (`opd_profile_kl_compute_s≈0.046`, `opd_profile_loss_total_s≈0.105` on a `14.03 s` step), and AMDAHL-010 already rejected `no_recompute`. Do not spend the next OPD pass on KL/top-k or no-checkpoint. The updated OPSD note explicitly corrects the earlier overclaim: OPD's visible low MFU is not "just accounting", but OPD also does **not** show a large above-model server tax because 4-node trainer-only replay (`4.46 s`) is close to full strict fwd/bwd (`4.21 s`). Their request-shape conclusions do **not** directly transfer: OPSD Wordle has dense valid tokens and a different RB/chunk path, while this OPD regime has sparse answer-valid tokens, 64 prompt-pause rows, and already benefited from one 64-sample fb call. The shared next-principle is the same: quote executed-vs-valid denominators, then use captured replay/bare-tensor replay to separate server feed from in-model shape/comm.
- **Current-node packing + dispatch promotion (2026-06-13):**

  | candidate | changed lever | non-warmup `step_total_s` | non-warmup `forward_backward_s` | `valid_tokens_per_step_s` | decision |
  |---|---|---:|---:|---:|---|
  | ARITH-020 | prep16, 4 fb calls/step | 29.21 s | 18.05 s | 122.9 | interrupted; baseline for perf ownership |
  | AMDAHL-007 | prep32, 2 fb calls/step | 17.34 s | 8.68 s | 234.5 | superseded |
  | AMDAHL-008 | prep64, 1 fb call/step, all-to-all | 15.34 s | 4.91 s | 264.1 | conservative fallback |
  | AMDAHL-009 | prep64 + quack+DeepEP24 | 15.11 s | 4.78 s | 267.9 | DeepEP24 baseline; K3-gate before long science default |
  | AMDAHL-010 | AMDAHL-009 + `no_recompute` | 15.42 s | 4.70 s | 261.5 | fit but reject |
  | AMDAHL-011 | AMDAHL-009 + trainer NCCL socket/buffer env | 15.35 s | 4.76 s | 259.8 | healthy but reject; sync 3.35 s |
  | AMDAHL-012 | prep128 + quack+DeepEP24 | 23.16 s for 128 prompts | 8.10 s | 348.9 | larger-window artifact; 1.19× AMDAHL-018 prompt/s, science-boundary caveat |
  | AMDAHL-013 | AMDAHL-009 + compile dynamic | 14.67 s | 4.84 s | 279.5 | fit after EP DCP alias fix, but reject; weaker than pipeline |
  | AMDAHL-014 | AMDAHL-009 + fixed OPRD pipeline RL | **13.22 s** | 9.36 s | **335.9** | **current 64-prompt stack winner if one-step-stale samples accepted** |
  | AMDAHL-015 | AMDAHL-009 + DeepEP12 | 14.75 s | 4.69 s | 276.7 | positive strict result, superseded by AMDAHL-018 |
  | AMDAHL-016 | AMDAHL-009 + DeepEP36 | 14.30 s | **4.60 s** | 285.4 | positive strict result, superseded by AMDAHL-018 |
  | AMDAHL-017 | AMDAHL-016 + `profile_sync_cuda=false` | 14.48 s | 4.71 s | 283.3 | fit but reject; lower sync did not beat prepare/fb variance |
  | AMDAHL-018 | AMDAHL-016 + strict same-step prepare chunk4 | **14.03 s** | 4.68 s | **292.5** | **current strict fresh-sample 64-prompt winner; prepare 5.27 s; teacher 2.59 s; K3-gate before long science default** |
  | AMDAHL-019 | AMDAHL-018 chunk4 → chunk8 | invalid | 4.46 s on invalid row | invalid | reject/pending infra: first launch synced 1/2 endpoints; corrected 2/2 launch wedged in initial Mooncake P2P sync; invalid 1-sync row had prepare 5.28 s, so no evidence chunk8 beats chunk4 |
  | AMDAHL-020 | AMDAHL-018 + one sampler/one sync receiver | **13.41 s** | **4.21 s** | **305.8** | **current strict one-endpoint wall-time winner; sync 3.37→2.35 s vs AMDAHL-018, prepare worsens 5.27→6.15 s; unused `sglang-1` released after validation** |
  | AMDAHL-021 replay | trainer-only fwd/bwd capture + server-only replay | n/a | **4.46 s server f/b** | n/a | microbenchmark path validated; 6x replay, 2 warmups, no optimizer/sync/sampling/teacher; `forward_compute_s=1.60`, `backward_compute_s=1.58`, `clear_gradients_s=0.63` |
  | AMDAHL-022 replay | AMDAHL-021 capture on 2-node trainer | n/a | 5.63 s server f/b | n/a | reject for wall time; fewer GPU-seconds (16×5.63 vs 32×4.46) only matters if resource efficiency outranks wall throughput |
  | AMDAHL-024 replay | AMDAHL-021 capture + steady no-recompute | n/a | 4.61 s server f/b | n/a | reject; warmed no-recompute replay confirms no checkpointing win on this shape |
  | AMDAHL-025 replay | AMDAHL-021 capture + `sample_packing_sequence_len=2304` | n/a | 4.16 s server f/b | n/a | replay-only win; required full validation because it changes packed-row/rank shape |
  | AMDAHL-026 | AMDAHL-020 + pack2304 | 19.80 s | 11.03 s | 206.9 | reject; full strict run regressed badly despite AMDAHL-025 replay win (`opd_profile_forward_compute_s=8.24` vs AMDAHL-020's 1.57) |
  | AMDAHL-027 replay | AMDAHL-021 capture + EP=1 / local experts | n/a | 5.87 s server f/b | n/a | reject; EP=1 loaded and ran but was slower than EP=8 DeepEP36 replay (`forward_compute_s=2.22`, `backward_compute_s=1.79`, `clear_gradients_s=0.99`) |
  | AMDAHL-028 replay | AMDAHL-021 capture + minimal dummy rows + repeat-data ladder | n/a | 4.46 s at repeat1; 7.97 s at repeat2; 11.84 s at repeat4 | n/a | reject as a promoted run; minimal dummy is neutral, static fatter calls improve throughput only 521→584→776 executed tok/s/GPU, so the gap is in real OPRD model/comm shape rather than fixed API overhead |
  | AMDAHL-029 replay | AMDAHL-021 capture on 1-node trainer | n/a | OOM before replay row | n/a | reject as a usable 1-node loop; all-layer trainer-side OPRD teacher recompute OOMed in `_trainer_teacher_kept_layers` |
  | AMDAHL-030 capture | 1-node trainer + SGLang rank-3 OPRD layer cache, `sync_weights=false` | n/a | no capture written | n/a | reject full all-40-layer cache-at-once path; teacher saturated memory and hit FlashInfer workspace OOM, so use layer-reduced/chunked cache if pursuing 1-node OPD |
  | AMDAHL-031 capture/replay | 1-node trainer + SGLang rank-3 OPRD cache every 4th layer | n/a | OOM before replay row | n/a | reduced 10/40-layer capture works, but prep64 one-node f/b still OOMs on a 1.89 GiB allocation |
  | AMDAHL-032 replay | 1-node all-layer selected-hook capture, no SGLang layer cache | n/a | OOM before replay row | n/a | selected hooks remove full hidden retention but all-layer prep64 still OOMs in the model forward/final-norm path on a 972 MiB allocation |
  | AMDAHL-033 replay | 1-node selected hooks + SGLang rank-3 cache every 4th layer | n/a | OOM before replay row | n/a | reduced-layer cache + hooks reaches streaming-KL backward, then OOMs on the full lm-head grad buffer; experimental dtype patch did not help with `lm_head_fp32=true` |

  AMDAHL-008/009/010/011/013/014/015/016/017/018/019/020/026 kept `opd_microbatch_size=64`; AMDAHL-012
  used `opd_microbatch_size=128`. All kept `opd_frac_answer=1`,
  `opd_oprd_num_layers=40`. Valid promoted/rejected timing rows through AMDAHL-018
  kept balanced 2-sampler routing and 2/2 sync success; AMDAHL-020 intentionally
  used one registered sampler/sync endpoint; AMDAHL-026 intentionally used the
  same one registered sampler/sync endpoint and stayed 1/1 healthy, so its
  regression is the pack2304 trainer shape, not endpoint health. AMDAHL-019 is
  explicitly excluded because its only completed row synced 1/2 endpoints by accident.
  AMDAHL-018 is **2.08× faster step-time**, **2.38× faster valid-token/s**,
  and **3.86× faster fb** than the ARITH-020 full-run non-warmup mean at the
  64-prompt window. Relative to AMDAHL-009, AMDAHL-018 is **7.1 %** lower wall
  step and **9.2 %** higher valid-token/s without stale samples. Relative to
  AMDAHL-016, AMDAHL-018 is **1.9 %** lower wall step and **2.5 %** higher
  valid-token/s, mainly by cutting same-step prepare (`5.87→5.27 s`) and teacher
  prefill (`3.35→2.59 s`). AMDAHL-014 is still the 64-prompt wall-time and
  valid-token/s winner (**5.8 %** faster step, **14.8 %** higher valid-token/s
  than AMDAHL-018) because it overlaps prepare
  (`prepare_window_s≈0.0004`); it is not a pure fb improvement and carries the
  one-step-stale pipeline semantics. AMDAHL-012 remains the first measured
  larger-window win: **1.19× AMDAHL-018 prompt/s and valid-token/s** on the same
  4-node trainer, but it changes the prompts-per-optimizer-step window and
  therefore needs an explicit science/default decision. Do not jump beyond 128
  prompts/step until the `53ef29a3` fat-call fix validates the ≥5-rows/rank
  boundary (§0 queue).
  AMDAHL-022/024/025/028 show the correct use of the replay harness: reject bad
  fwd/bwd hypotheses cheaply, then validate a replay winner once in the full
  path before promotion. AMDAHL-025's pack2304 replay looked good because the
  static capture is one fixed step-0 payload; AMDAHL-026's generated full run
  proved the actual strict path regresses, so do not use pack2304 as a default.
  AMDAHL-028's repeat-data ladder proves that coalescing more real OPD payload
  into one API call helps but is not a fixed-cost 10× lever. AMDAHL-029/030/031/032/033
  prove that the obvious 1-node variant is blocked by memory inside the real OPRD
  trainer f/b path, not by sampler/sync mechanics. The OOM moved from
  trainer-side all-layer hidden capture to final model/FSDP allocation and then
  to the full-vocab KL/lm-head gradient buffer as the cache/hook plumbing improved.
  The next fwd/bwd benchmark should be rank-occupancy-aware OPRD packing/coalescing,
  sharded/vocab-parallel OPD KL gradient handling, or a clearly gated `lm_head_fp32=false`
  fit probe, not another HTTP/sync/sampler probe.

## 7e. Trainer-only forward/backward replay microbenchmark

Yes: for fwd/bwd hypotheses, do not run the full OPD/SGLang/teacher/weight-sync stack. The replay path captures one representative OPD `forward_backward` payload once, copies any teacher hidden-cache assets before the client deletes them, then replays only the trainer API call against a trainer-only server. This is the right tool for EP/DeepEP/no-recompute/compile/shape-bucketing A/Bs where the claim is about trainer fwd/bwd time.

Important MFU guardrail: if a microbench does not reproduce the low-MFU denominator being discussed, it is a ceiling/knee probe, not the bottleneck reproducer. Track these rates separately:

- forwarded/padded model tokens per GPU, including server dispatcher dummy rows (what drives actual GEMMs and collectives);
- trainer-side OPRD teacher-forward tokens per GPU (a real no-grad model-forward bucket inside `forward_backward`, not SGLang teacher prefill);
- real student tokens per GPU before dispatcher dummy fill;
- valid/supervised answer tokens per GPU (what makes OPD look extremely low-utilization because only ~3k of ~108k executed/dummy-filled tokens carry answer loss in the AMDAHL-021 capture).

A low useful-token MFU can coexist with a server replay that removes SGLang/teacher/sync and still lands near the full fwd/bwd latency. Server OPD does **not** emit native XORL `efficiency/mfu`; any OPD MFU number must say whether it is reconstructed over executed student tokens, student+trainer-teacher forward tokens, valid answer tokens, or whole-step wall time.

The sibling OPSD note `/home/apanda/xorl-mtp-singleshot-port-20260602/docs/notes/opsd_low_mfu_microbench_20260613.md` now frames the problem correctly: synthetic local fwd/bwd is a **ceiling/knee probe**, not a proof that any given stack has a 14× server tax. For OPD, AMDAHL-021 4-node trainer-only replay (`4.46 s`) is already close to the full strict AMDAHL-020 fwd/bwd (`4.21 s`), and AMDAHL-028 repeat-data scaling rejected a large fixed API cost. The remaining target is still serious: 1.37% executed-MFU is below the small-batch curve and far below the ~16k-token/rank knee, but the evidence points at real OPRD model/communication shape, rank occupancy, and bare-real-tensor replay rather than hidden SGLang/teacher/weight-sync machinery. If a 1-node synthetic microbench reports ~1-2% MFU at the ~1k-token/rank shape, that is not enough; it must be compared to a captured real-tensor replay and the OPD valid/executed denominator.

The current OPD replay has one extra hidden bucket that the first profile did not split: trainer-side OPRD teacher hidden-state forward. This is not sampler/HTTP/teacher-prefill overhead; it is a no-grad forward inside the trainer `forward_backward` call. On the sorted replay breakdown, `server_forward_backward_s=4.80` with `opd_profile_model_forward_s=0.81`, `opd_profile_oprd_teacher_forward_s=0.85`, `opd_profile_loss_compute_s=0.91`, `opd_profile_backward_compute_s=1.59`, `opd_profile_clear_gradients_s=0.59`, and `opd_profile_kl_compute_s=0.046`. That means moving OPRD hiddens back to a SGLang-side rank-3 cache could save a real bucket, but its Amdahl ceiling is about 1.2-1.3× unless it also changes the broader model/comm shape.

Denominator audit artifact: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_20260613.json`. Re-run with:

```bash
cd /home/apanda/xorl-opd-prefill
PYTHONPATH=/home/apanda/xorl-opd-prefill \
python experiments/opd_profile/scripts/audit_forward_backward_denominator.py \
  --capture /shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/amdahl-021-oprd-prep64-deepep36-strictchunk4.json \
  --sample-packing-sequence-len 4096 \
  --sample-packing-sequence-len 2304 \
  --sample-packing-sequence-len 8192 \
  --sample-packing-sequence-len 16384 \
  --replay-jsonl amdahl021=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-amdahl021-serveronly-6x.jsonl \
  --replay-jsonl amdahl021_breakdown_sorted=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-amdahl021-breakdown-sortedmetrics-serveronly-5x.jsonl \
  --output-json /shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_20260613.json
```

Key audit numbers for AMDAHL-021:

| basis | value |
|---|---:|
| raw student tokens in capture | 71,804 |
| valid target tokens | 3,049 |
| 4096 packed rows | 22 rows, 73,088 row-padded tokens |
| dispatcher dummy fill | 10 zero-loss rows, +34,560 executed tokens |
| dispatcher-executed student tokens | 107,648 |
| trainer-side OPRD teacher-forward tokens | 350,870 |
| student + trainer-teacher forward tokens | 458,518 |
| valid fraction of dispatcher-executed tokens | 2.83% |
| valid fraction of student + trainer-teacher tokens | 0.665% |
| replay rate over dispatcher-executed tokens | 754 tok/s/GPU |
| replay rate over trainer-teacher tokens | 2,459 tok/s/GPU |
| replay rate over student + trainer-teacher tokens | 3,214 tok/s/GPU |
| replay rate over real student tokens | 503 tok/s/GPU |
| replay rate over valid target tokens | 21.4 tok/s/GPU |
| reconstructed logical MFU over executed tokens | 1.37% |
| valid-token-scaled logical MFU diagnostic | 0.039% |

Use this ladder:

- **Tier 0: 1-node synthetic `xorl.cli.train` sweep.** Static dummy data, balanced synthetic MoE routing, no server, no HTTP, no teacher, no sampler, no weight sync. Use it to answer kernel/GEMM-size questions cheaply: tokens-per-rank knee, quack/triton local behavior, recompute-family sanity, and local EP mechanics. The MTP note's 1k→16k tokens/rank result is a useful prior. Port its configs/launcher as a template only; they are hardcoded to the MTP checkout/stack. Treat this as a ceiling/knee probe unless it deliberately includes the OPD dispatcher shape: one batch per rank, dummy zero-loss rows for under-filled ranks, OPRD valid-token sparsity, and the trainer-side teacher-forward/cache behavior.
- **Tier 0b: 1-node OPD topology replay/capture.** This is the desired cheap target for the updated note's "shrink G" lever because 64-prompt OPD on 8 GPUs would put the AMDAHL-021 executed-token shape near the ~16k tokens/rank knee. Current status: still blocked by memory, but the failure has been localized further. AMDAHL-029 OOMed inside trainer-side `_trainer_teacher_kept_layers`; AMDAHL-030 moved OPRD layers to SGLang rank-3 cache but full all-layer capture saturated teacher memory and wrote no capture; AMDAHL-031 reduced the SGLang layer cache to every 4th decoder layer and successfully captured the replay payload, but the one-node trainer-only f/b replay still OOMed before a measured row; AMDAHL-032 added selected hidden hooks and moved the all-layer OOM to the model forward/final-norm path; AMDAHL-033 combined selected hooks with the every-4-layer SGLang cache and reached streaming-KL backward, then OOMed allocating full lm-head gradient state. Next attempts must reduce trainer f/b memory footprint via sharded/vocab-parallel OPD KL weight-gradient handling, smaller row chunks/activation streaming, or a science-gated `lm_head_fp32=false` fit probe before treating 1-node replay as available.
- **Tier 1: 4-node trainer-only captured replay.** This is the fidelity check, not the desired inner loop. It preserves the actual 32-rank FSDP/EP topology, real OPRD payload, real teacher-hidden-cache metadata, and server dispatch path while removing samplers, teacher prefill, endpoint registration, optimizer, and inference-weight sync. Use it after a 1-node candidate has a reason to survive, or when the hypothesis is specifically about 4-node communication behavior.
- **Tier 2: short full OPD promotion gate.** Run this only after a replay winner, or when the hypothesis changes generated-batch shape, prepare overlap, sampler pressure, or weight-sync behavior. AMDAHL-025/026 is the warning: a static replay win can regress the real generated path.

Validated artifacts (2026-06-13):

- Capture candidate: `experiments/opd_profile/autoresearch/candidates/AMDAHL-021-OPRD-PREP64-DEEPEP36-FB-CAPTURE.yaml`.
- Capture payload: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/amdahl-021-oprd-prep64-deepep36-strictchunk4.json`.
- Copied teacher-cache asset dir: same path plus `.assets/`.
- Replay output: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-amdahl021-serveronly-6x.jsonl`.
- Sorted breakdown replay output: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-amdahl021-breakdown-sortedmetrics-serveronly-5x.jsonl`.
- Minimal-dummy candidate: `experiments/opd_profile/autoresearch/candidates/AMDAHL-028-OPRD-PREP64-DEEPEP36-MINDUMMY128-FB-REPLAY.yaml`.
- Minimal-dummy/repeat replay outputs: `fb_replay/replay-amdahl028-mindummy128-serveronly-6x.jsonl`, `fb_replay/replay-amdahl028-repeat2-mindummy128-serveronly-5x.jsonl`, and `fb_replay/replay-amdahl028-repeat4-mindummy128-serveronly-4x.jsonl`.
- One-node trainer-side OPRD replay candidate: `experiments/opd_profile/autoresearch/candidates/AMDAHL-029-OPRD-PREP64-DEEPEP36-1NODE-FB-REPLAY.yaml`; server log: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260613T213025Z-serveronly-configAMDAHL-029-OPRD-PREP64-DEEPEP36-1NODE-FB-REPLAY-er-opd-q36-35b-slots-trainer-head/server.log`. Result: OOM in `_trainer_teacher_kept_layers` before any replay row.
- One-node SGLang rank-3 OPRD cache capture candidate: `experiments/opd_profile/autoresearch/candidates/AMDAHL-030-OPRD-PREP64-DEEPEP36-1NODE-SGLANGCACHE-FB-CAPTURE.yaml`; head log: `/shared/opd-control/er-opd-q36-35b-slots/trainer-head/logs/20260613T214537Z-run.log`; run dir: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260613T214538Z-configAMDAHL-030-OPRD-PREP64-DEEPEP36-1NODE-SGLANGCACHE-FB-CAPTURE-er-opd-q36-35b-slots-trainer-head`; W&B: `54byr2j4`. Result: SGLang teacher saturated memory during all-40-layer hidden capture, emitted FlashInfer workspace OOM warnings, and wrote no `amdahl-030` capture.
- One-node layer-reduced cache fit probe: `experiments/opd_profile/autoresearch/candidates/AMDAHL-031-OPRD-PREP64-DEEPEP36-1NODE-SGLANGCACHE-EVERY4-FB-CAPTURE.yaml`. It captures every 4th decoder layer (10/40) only to test the 8-GPU path; it is not an all-layer OPRD science candidate. First launch showed `/teacher_hidden_cache` returning only the base hidden width for this model wrapper; fixed by appending the OPRD layer stack in `/home/apanda/xorl-sglang-internal/python/sglang/srt/models/qwen3_vl.py`. Final capture log: `/shared/opd-control/er-opd-q36-35b-slots/trainer-head/logs/20260613T220751Z-run.log`; run dir: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260613T220752Z-configAMDAHL-031-OPRD-PREP64-DEEPEP36-1NODE-SGLANGCACHE-EVERY4-FB-CAPTURE-er-opd-q36-35b-slots-trainer-head`; W&B: `4px8r8wr`. Capture payload: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/amdahl-031-oprd-prep64-deepep36-1node-sglangcache-every4.json`; assets include the base hidden cache and `step0-prep0-train0-oprd-layers-teacher_hidden_layers_step0_mb0.safetensors`. Trainer-only replay output path exists but is empty after immediate OOM: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-amdahl031-1node-sglangcache-every4-serveronly-6x.jsonl`; server-only log: `/shared/opd-control/er-opd-q36-35b-slots/trainer-head/logs/20260613T221208Z-run.log`. Result: reduced-cache capture works, but one-node prep64 f/b still OOMs on a 1.89 GiB allocation before any replay row.
- One-node all-layer selected-hook replay candidate: `experiments/opd_profile/autoresearch/candidates/AMDAHL-032-OPRD-PREP64-DEEPEP36-1NODE-HOOKCAP-FB-REPLAY.yaml`; replay output is empty: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-amdahl032-1node-hookcap-serveronly-6x.jsonl`; server log: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260613T221949Z-serveronly-configAMDAHL-032-OPRD-PREP64-DEEPEP36-1NODE-HOOKCAP-FB-REPLAY-er-opd-q36-35b-slots-trainer-head/server.log`. Result: selected OPRD hooks avoid retaining full hidden-state tensors, but all-layer prep64 still OOMs before any replay row on a 972 MiB allocation in the model forward/final-norm path.
- One-node reduced-layer selected-hook replay candidate: `experiments/opd_profile/autoresearch/candidates/AMDAHL-033-OPRD-PREP64-DEEPEP36-1NODE-SGLANGCACHE-EVERY4-HOOKCAP-FB-REPLAY.yaml`; replay output is empty: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-amdahl033-1node-sglangcache-every4-hookcap-serveronly-4x.jsonl`; server log: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260613T222720Z-serveronly-configAMDAHL-033-OPRD-PREP64-DEEPEP36-1NODE-SGLANGCACHE-EVERY4-HOOKCAP-FB-REPLAY-er-opd-q36-35b-slots-trainer-head/server.log`. Result: every-4-layer SGLang cache plus selected hooks reaches streaming-KL backward, then OOMs on a 1.89 GiB full lm-head gradient allocation in `opd_streaming_kl.py`.
- Experimental KL-gradient staging patch retry: output remains empty at `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-amdahl033-1node-sglangcache-every4-hookcap-klgradfix-serveronly-4x.jsonl`; server log: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260613T223148Z-serveronly-configAMDAHL-033-OPRD-PREP64-DEEPEP36-1NODE-SGLANGCACHE-EVERY4-HOOKCAP-FB-REPLAY-er-opd-q36-35b-slots-trainer-head/server.log`. Result: still OOMs at 1.89 GiB because current OPD uses `lm_head_fp32=true`, so `student_weight` is already fp32 before the streaming-KL backward allocates `grad_weight`. This code patch is not a promoted throughput fix for the current recipe.
- Denominator audit script: `experiments/opd_profile/scripts/audit_forward_backward_denominator.py`.
- Denominator audit output: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_20260613.json`.
- Measured replay mean after 2 warmups over 4 rows: `server_forward_backward_s=4.4588`, `api_wall_s=4.8159`, `opd_profile_forward_compute_s=1.6026`, `opd_profile_backward_compute_s=1.5802`, `opd_profile_loss_total_s=0.0596`, `opd_profile_kl_compute_s=0.0465`, `opd_profile_hidden_fetch_s=0.0038`, `opd_profile_clear_gradients_s=0.6289`. The sorted breakdown replay mean after 1 warmup over 4 rows was `server_forward_backward_s=4.8014`, `opd_profile_model_forward_s=0.8056`, `opd_profile_oprd_teacher_forward_s=0.8498`, `opd_profile_loss_compute_s=0.9071`, `opd_profile_backward_compute_s=1.5860`, and `opd_profile_clear_gradients_s=0.5882`. It performs **no** optimizer step and **no** inference weight sync unless `--optim-step` is explicitly passed.

Measured replay A/Bs:

| candidate | replay output | measured mean | decision |
|---|---|---:|---|
| AMDAHL-021 | `fb_replay/replay-amdahl021-serveronly-6x.jsonl` | `server_forward_backward_s=4.4588` | baseline |
| AMDAHL-021 breakdown | `fb_replay/replay-amdahl021-breakdown-sortedmetrics-serveronly-5x.jsonl` | `server_forward_backward_s=4.8014` | diagnostic split only; shows trainer-side OPRD teacher forward is ~0.85 s |
| AMDAHL-022 | `fb_replay/replay-amdahl022-2node-serveronly-6x.jsonl` | `server_forward_backward_s=5.6311` | reject for wall throughput; fewer GPU-seconds only |
| AMDAHL-024 | `fb_replay/replay-amdahl024-norec-serveronly-6x.jsonl` | `server_forward_backward_s=4.6051` | reject; steady no-recompute still neutral/slower |
| AMDAHL-025 | `fb_replay/replay-amdahl025-pack2304-serveronly-6x.jsonl` | `server_forward_backward_s=4.1632` | replay win, but full AMDAHL-026 rejects |
| AMDAHL-027 | `fb_replay/replay-amdahl027-ep1-serveronly-6x.jsonl` | `server_forward_backward_s=5.8706` | reject; EP=1/full local experts slower than EP=8 DeepEP36 |
| AMDAHL-028 | `fb_replay/replay-amdahl028-mindummy128-serveronly-6x.jsonl` | `server_forward_backward_s=4.4612` | minimal dummy rows neutral vs AMDAHL-021; useful cleanup only |
| AMDAHL-029 | no replay output | OOM before measured row | reject as current 1-node loop; all-layer trainer-side OPRD recompute does not fit |
| AMDAHL-030 | no capture output | no capture written | reject all-40-layer SGLang cache-at-once path; teacher memory saturated before capture completed |
| AMDAHL-031 | empty `fb_replay/replay-amdahl031-1node-sglangcache-every4-serveronly-6x.jsonl` | OOM before measured row | reject as current 1-node loop; 10/40-layer SGLang cache capture works, but prep64 trainer f/b still does not fit |
| AMDAHL-032 | empty `fb_replay/replay-amdahl032-1node-hookcap-serveronly-6x.jsonl` | OOM before measured row | reject as current 1-node loop; selected hooks remove full hidden retention but all-layer prep64 still does not fit |
| AMDAHL-033 | empty `fb_replay/replay-amdahl033-1node-sglangcache-every4-hookcap-serveronly-4x.jsonl` | OOM before measured row | reject as current 1-node loop; reduced SGLang cache + hooks reaches streaming-KL backward, then OOMs on full lm-head grad allocation |
| AMDAHL-033 KL staging retry | empty `fb_replay/replay-amdahl033-1node-sglangcache-every4-hookcap-klgradfix-serveronly-4x.jsonl` | OOM before measured row | reject for current `lm_head_fp32=true`; dtype-staging tweak cannot reduce a buffer whose input weight has already been cast fp32 |

AMDAHL-028 static repeat-data ladder (`XORL_SERVER_MINIMAL_DUMMY_BATCH_TOKENS=128`, no optimizer/sync/sampler/teacher):

| repeat | datums | packed rows | rows/rank | dummy rows | executed student tokens | valid tokens | `server_forward_backward_s` | executed tok/s/GPU | valid tok/s/GPU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 22 | 1 | 10 | 74,368 | 3,049 | 4.4612 | 520.9 | 21.4 |
| 2 | 128 | 43 | 2 | 21 | 148,992 | 6,098 | 7.9669 | 584.4 | 23.9 |
| 4 | 256 | 86 | 3 | 10 | 293,888 | 12,196 | 11.8421 | 775.5 | 32.2 |

Interpretation: fatter captured OPD calls help, but only ~1.5× in executed-token rate at 4× data. That rejects a dominant fixed API/collation cost and says the remaining low executed-MFU is in the real OPRD model/communication shape. Do not extrapolate this static repeated payload into a science recipe without a full generated-batch gate; it deliberately repeats the same capture for throughput attribution only.

Packing shape of the AMDAHL-021 capture:

- 64 samples; per-sample sequence length min 1076, median 1138, mean 1121.9, p95 1140, max 1141; answer-region tokens sum 3049.
- `sample_packing_sequence_len=4096` packs into 22 rows, 73,088 row-padded tokens, ~79.7% capacity utilization, and only ~1.8% row-pad overhead over real tokens. Because there are 32 dispatch ranks, the dispatcher adds 10 dummy zero-loss rows cloned from the first packed row; actual executed student-token denominator is 107,648. The same trainer f/b call also forwards 350,870 trainer-side teacher tokens for OPRD hidden matching, so student-token-only MFU undercounts real model work while valid-token MFU still reflects the sparse answer-loss density.
- `sample_packing_sequence_len=2304` packs into 32 rows, 73,216 row-padded tokens, ~97.4% capacity utilization, and no dummy rows; this looked better in replay but regressed the real generated strict run (`AMDAHL-026`).
- Larger packing capacities reduce real row count and increase dummy-fill waste on this 32-rank dispatch path: `8192` gives 10 real rows + 22 dummy rows (246,784 executed tokens), and `16384` gives 5 real rows + 27 dummy rows (497,280 executed tokens).
- Conclusion: do not frame the current 64-prompt strict bottleneck as static padding waste. The 4096 shape leaves some dummy ranks but is faster in the real OPRD path; changing packed row count changes the collective-uniform model-forward behavior and must be full-run validated.

One-time capture run:

```bash
CLIENT=/home/apanda/xorl-opd-prefill
INFRA=/home/apanda/xorl-infra
PY=/home/apanda/xorl-internal/.venv/bin/python
cd "$CLIENT"
"$PY" "$INFRA/k8s/opd_profile/q36_35b_reprogrammable_slots.py" \
  --model q36 --trainer-nodes 4 --teacher-replicas 1 --teacher-route direct --eval-replicas 0 \
  write-trainer-control \
  --candidate "$CLIENT/experiments/opd_profile/autoresearch/candidates/AMDAHL-021-OPRD-PREP64-DEEPEP36-FB-CAPTURE.yaml" \
  --num-steps 1 --prompts-per-step 64 \
  --sampler-replicas 1 --sampler-layout dedicated
```

Trainer-only server, no SGLang/teacher/register/sync:

```bash
CLIENT=/home/apanda/xorl-opd-prefill
INFRA=/home/apanda/xorl-infra
PY=/home/apanda/xorl-internal/.venv/bin/python
cd "$CLIENT"
"$PY" "$INFRA/k8s/opd_profile/q36_35b_reprogrammable_slots.py" \
  --model q36 --trainer-nodes 4 --teacher-replicas 1 --teacher-route direct --eval-replicas 0 \
  write-trainer-server-control \
  --candidate "$CLIENT/experiments/opd_profile/autoresearch/candidates/AMDAHL-021-OPRD-PREP64-DEEPEP36-FB-CAPTURE.yaml"
```

Replay from trainer-head:

```bash
kubectl -n apanda exec er-opd-q36-35b-slots-trainer-head -- bash -lc '
  CLIENT=/home/apanda/xorl-opd-prefill
  ENGINE=${XORL_REPO:-/home/apanda/xorl-internal}
  cd "$CLIENT" &&
  PYTHONPATH="$CLIENT:$ENGINE/src:${PYTHONPATH:-}" \
  /home/apanda/xorl-internal/.venv/bin/python \
  "$CLIENT/experiments/opd_profile/scripts/replay_forward_backward_capture.py" \
  --capture /shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/amdahl-021-oprd-prep64-deepep36-strictchunk4.json \
  --base-url http://127.0.0.1:26050 \
  --iterations 6 --warmup 2 \
  --output-jsonl /shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-amdahl021-serveronly-6x.jsonl
'
```

Stop when done:

```bash
CLIENT=/home/apanda/xorl-opd-prefill
INFRA=/home/apanda/xorl-infra
PY=/home/apanda/xorl-internal/.venv/bin/python
cd "$CLIENT"
"$PY" "$INFRA/k8s/opd_profile/q36_35b_reprogrammable_slots.py" \
  --model q36 --trainer-nodes 4 --teacher-replicas 1 --teacher-route direct --eval-replicas 0 \
  stop-trainer-control --remove-run
```

Limits: this is an exact trainer fwd/bwd replay for the captured topology and static data, but it does not measure prepare overlap, sampler pressure, teacher prefill, optimizer, or inference-weight sync. It also does not prove that a generated-batch shape change is good: AMDAHL-025 improved the static replay, then AMDAHL-026 regressed the real strict path (`forward_backward_s=11.03` vs AMDAHL-020's `4.21`). Treat replay as the first filter, not the promotion gate. A 1-node synthetic bench is useful for local-GEMM/kernel screening, but it is not a replacement for the 4-node replay because it removes the cross-node FSDP/EP communication topology.

## 8. Cold bring-up (no pods yet)

```bash
$PY $GENERATOR render-manifest --output /tmp/m.yaml --sampler-replicas 2 --sampler-layout dedicated
kubectl apply -n apanda -f /tmp/m.yaml
$PY $GENERATOR write-teacher-control --sampler-replicas 2 --sampler-layout dedicated   # teachers
$PY $GENERATOR write-student-inference-control --sampler-replicas 2 --sampler-layout dedicated  # samplers+dispatch
# (add --model q36 to EVERY command above when serving Qwen3.6)
# wait for both samplers ready + dispatch /v1/models lists the chosen model
$PY $GENERATOR write-trainer-control --candidate <…> --num-steps <N> --prompts-per-step <P> --sampler-replicas 2 --sampler-layout dedicated
```

## 9. Recovery (the few you'll actually need)

- **Trainer crash / clean restart / init hang.** Stop → wait for GPUs to clear → relaunch unhurried (do not rapid-fire):
  ```bash
  $PY $GENERATOR stop-trainer-control --remove-run
  # wait ~60s; confirm GPU mem = 0 on head + workers, then:
  $PY $GENERATOR write-trainer-control --candidate <…> --num-steps <N> --prompts-per-step <P> --sampler-replicas 2 --sampler-layout dedicated
  ```
  Init hangs ("Waiting for rank 0 ready", GPU ≈ 0–4 MiB) almost always = stale rendezvous from a too-fast relaunch, **or** a dead bare pod (next item).
- **ABNORMAL TRAINER END ⇒ recreate ALL weight-sync receivers before relaunching** (rule hardened 2026-06-12 after three stacked aborts): any trainer that ends via crash, sync-timeout abort, or mid-run stop can leave half-open Mooncake sessions on its receivers; the NEXT trainer's syncs then hang to the full timeout at an unpredictable step (observed: startup + step-0 syncs fine, step-1 train sync 900 s hang). `write-student-inference-control` AND (if an eval pool exists) `write-eval-inference-control`, wait for all `/v1/models`, then `write-trainer-control`. Normal `RUN COMPLETE` endings do NOT need this.
- **P2P sync wedge** (`[P2P] batch_transfer_sync … ret=-1 … after 50 attempts`). The sampler's Mooncake state is stale. Recreate samplers, then relaunch trainer:
  ```bash
  $PY $GENERATOR stop-trainer-control --remove-run
  $PY $GENERATOR write-student-inference-control --sampler-replicas 2 --sampler-layout dedicated   # ~6 min reload
  # wait for both samplers ready, then write-trainer-control
  ```
- **Dead bare pod** (a worker in `Error`/`Failed` → 64-rank rendezvous never forms → head stuck at "Starting xorl training server"). Bare pods don't self-heal:
  ```bash
  $PY $GENERATOR render-manifest --output /tmp/m.yaml --sampler-replicas 2 --sampler-layout dedicated
  # extract that pod's doc → /tmp/p.yaml
  kubectl delete pod -n apanda <pod> --force --grace-period=0
  kubectl apply -n apanda -f /tmp/p.yaml      # reschedules (usually a different, healthy node)
  ```
- **Wedged slot agent** (desired.sha256 updated but the role keeps the OLD pid/log for >10 min after a `write-*-control`): re-write the control once to nudge; if still stuck, use the dead-bare-pod recipe (delete + re-apply that one pod). Seen on sglang-0 2026-06-10.
- **Eval-pool sync hang (RESOLVED 2026-06-11 ~23:50).** Symptom: per-step train sync fine, but the `pools=["eval"]` refresh hangs to its timeout (receiver logs `prepare_weights_update` 200 + Pause ok, then silence) — and `/complete_weights_update` on the half-armed group CRASHES the eval pods. Root cause: the eval refresh used a CUSTOM group name (`weight_sync_group_eval`); receiver-side session state for that group from a PREVIOUS trainer instance wedged the new trainer's transfer (the default `weight_sync_group` path survives trainer hand-offs via its cold-prepare/invalidate handling — the training samplers crossed the same trainer boundary fine). Fix in the client: the eval-pool sync now uses the DEFAULT group (pools filter alone selects endpoints) + a 180 s timeout cap (failure = skip evals + keep training; only a control-step sync failure aborts). If eval pods are ever left paused/half-armed: do NOT `/complete_weights_update` the custom group (crash) — just restamp them (`write-eval-inference-control`, ~6 min).
- **Launch sequencing after a sampler restamp:** verify the slot RE-EXEC actually happened (`status` shows `running_since` NEWER than your restamp) before probing `/v1/models` readiness — the slot agent's kill can lag the restamp by 1–2 min, so a fast readiness probe can pass against the OLD engine and the trainer then registers against a dying server ("Couldn't connect" mid-registration, seen 2026-06-12 11:20Z).
- Teachers rarely need recreation (HTTP prefill, no weight sync) — keep them warm.

## 9b. RESOLVED 2026-06-10: the two serving bugs that silently poison runs (keep the guards)

Both are FIXED but their guards are load-bearing — do not remove them.

1. **Batched-decode corruption** (was: greedy 0.805 serial → 0.367 @ conc-128, weight-dependent, backend-agnostic). Root cause: `prepare_for_decode` in the sglang worktree rebuilt `seq_lens` from `req.seqlen` every step, desyncing from KV allocation whenever a prefill interleaved under the overlap scheduler → garbage KV slots. **Fixed in `xorl-sglang-internal` `schedule_batch.py`** (rebuild now MTP-only); samplers validated at full capacity (`--max-running-requests 256 --cuda-graph-max-bs 64`; all validation runs effectively fa3 — the unpinned default ALSO resolves to fa3 on this build; flashinfer remains untested at the gate, see §11). **Regression gate** after ANY sglang change touching scheduler/attention/cuda-graph: run `_buffer_control_eval` at concurrency 8 vs 128 on TRAINED weights (base weights can mask it!) — pass = |Δacc| ≤ 0.05, cap-hit < 0.05. Full post-mortem: `docs/notes/sglang_batched_decode_corruption_handoff.md`.
2. **Chat-rendering / silent context mismatch.** The server renders assistant prefills inside an OPEN `<think>` block by default, and without `logprob_start_len=0` it returns EMPTY `input_token_ids`, which the old client silently replaced with a LOCAL re-render → the trainer supervised a different context than the model sampled in (loss converges, capability rots). The client now pins `chat_template_kwargs={"enable_thinking": false}` + `chat_logprob_start_len=0` on every chat request and RAISES on empty `input_token_ids`. If you see that error, fix the request — never re-enable the fallback.

## 9c. The step-0 KL gate (cheap OPD-correctness regression check — added 2026-06-12)

**Background:** a client-side change to the masked teacher-cache remap (08:15Z) silently mistrained EVERY subsequent OPD run for ~5 hours — no crashes, no NaNs, loss "fine", model degrades to cap-length rambling within 10 steps. It also produced two FALSE convictions (the EP-dedup commit and the rmsnorm-native/ce-eager "OPRD substrate") because A/B comparisons spanning the change were confounded. Full record: `docs/notes/ep_dedup_opd_training_corruption_20260612.md`.

**The gate:** on the warm009 seed (ARITH-005 candidate), launch `--num-steps 2` and read `opd_kl` from the FIRST profile row (~8 min total):
- **≈ 0.5** → teacher-cache alignment healthy.
- **≥ 3** → the student is being distilled against MISALIGNED teacher rows. Stop; bisect the datum path.

The reference holds under BOTH dispatch modes (duplicated: 0.526–0.548 measured; distinct: 0.536 measured at the 06-12 un-bake gate) — a high reading indicates datum-path misalignment, not the dispatch.

Run this after ANY change touching: the client's `_opd_loss_data`/remap branches, `packing.py`'s cache-index rebase, `model_runner`'s hidden fetch/teacher-cache gather, or the fb dispatch. The held-out eval can look plausible at step 0 (weights are still the seed's) — the KL is the early signal. Contract reminder (DUAL-VIEW since `2471f5cf`, 2026-06-12): the client emits **GLOBAL** cache rows (masked branches 0-fill masked positions) plus a per-sample `teacher_cache_base`; the packer passes `teacher_cache_indices` through UNCHANGED (the KL hidden-fetch consumes globals) and derives a separate `teacher_cache_local_indices` (base-rebased + cum-kept) that ONLY the trainer-forward OPRD gather consumes. Do not localize the client rows and do not re-point the KL fetch at the local view — that recreates the 06-12 corruption. Engineering handoff §1 is RESOLVED by this; gate passed 2026-06-12 19:00Z (warm009 2-step, step-0 `opd_kl=0.536`).

## 10. Monitoring

- **Head client log** (`$CONTROL_ROOT/trainer-head/logs/<rev>-run.log`): step markers, sync status, eval lines.
- **Server log** (`$RESULT_ROOT/<run_id>/server.log`): per-module fb internals, weight-sync timing, NCCL/DeepEP init. To debug a sync/fb stall you read this, not the head log.
- **Profile** (`$RESULT_ROOT/<run_id>/opd_profile.jsonl`): one row/step. Headline `eval/accuracy` — **since 2026-06-10 this is a REAL eval**: greedy decode on `eval_heldout_num_problems` (256) held-out prompts through the live samplers, every `eval_accuracy_every` steps. The scored TRAINING samples are `eval/train_window_accuracy` (health only; reads ~1.0 under sft_mode/gold replacement). Sync `sync_endpoint_success_count`; OPRD `opd_oprd_loss`/`opd_oprd_num_layers`; collapse check `eval/empty_frac` + `eval/mean_completion_tokens`. **Gate on `eval/accuracy`, not loss.** Watch trainer-head **+ all workers + server.log**. End-of-run: probe final weights with `autoresearch/fresh_pair_probe.py` BEFORE launching anything else (weights are not checkpointed unless `save_every` is set).
- **Diagnostic metrics (added 2026-06-09).** All in the profile row + wandb:
  - `opd_student_entropy` / `opd_teacher_entropy` / `opd_top1_agreement` — now live on the streaming KL backend (no-grad streaming pass; `opd_emit_full_vocab_diagnostics=true`, already baked by the generator).
  - `opd_loss_clamp_frac` — fraction of supervised tokens at the ±`opd_loss_max_clamp` bound (clamped tokens pass ZERO gradient).
  - Region-split KL: `opd_kl_{prompt,buffer,answer}_mean` (+ exact raw `*_per_valid` / `opd_frac_*` fields). Separates the prompt-position anchor-to-base KL from the actual answer signal (~70% of supervised tokens are prompt positions in the no-filler recipe).
  - Correctness-split: `opd_kl_answer_{correct,wrong}_mean` + `opd_{student,teacher}_entropy_answer_{correct,wrong}_mean` — KL/entropy at answer positions of correct vs wrong sampled answers (the wrong-prefix-supervision diagnostic).
  - `eval/answer_logprob_mean_*` + `eval/answer_logprob_select_margin_*` every `eval_answer_logprob_every` steps (prefill-only gold-answer scoring on the held-out tail, independent of the control-eval gate).
- **Eval samples** (`$RESULT_ROOT/<run_id>/eval_samples.jsonl`): EVERY decoded health-eval sample per step (step/prompt/completion/correct) — full-resolution digit-level error decomposition offline (the head log only shows `eval_log_samples=8`/step).

## 11. Settings that must hold (the generator bakes these — preserve on edits)

- **Trainer Mooncake env:** `XORL_P2P_CPU_POOL_MIN_BYTES=0`, `XORL_WEIGHT_SYNC_BATCH_DENSE=1` + `_MOE=1`, `XORL_P2P_FP8_QUANTIZE_DEVICE=gpu`, `XORL_P2P_HANDSHAKE_BASE_PORT` (pins `MC_HANDSHAKE_PORT`), `P2P_TRAINER_HOSTNAME=$(POD_IP)`. Cold `/prepare` sends `p2p_invalidate_cache=True` (re-arms the receiver) — load-bearing.
- **`expandable_segments` OFF on the trainer** (`PYTORCH_ALLOC_CONF`/`PYTORCH_CUDA_ALLOC_CONF` unset) — it breaks Mooncake registration >~20 MiB.
- **No `NCCL_IB_GID_INDEX`/`NCCL_IB_HCA` on Mooncake-init pods** (trainer); they force a GID path that fails.
- **GPU pods:** non-privileged + `rdma/infiniband: 1` + `IPC_LOCK`, never `privileged`, never hardcode `CUDA_VISIBLE_DEVICES`; `team: turbo` on the pod template.
- **TP=2 sampler flags (resized 2026-06-11, deploys on next `write-student-inference-control`):** `--mem-fraction-static 0.85 --max-running-requests 256 --max-total-tokens 262144 --cuda-graph-max-bs 128 --chunked-prefill-size 8192 --max-prefill-tokens 16384` + `--sampling-backend flashinfer`. The old `16384`-token KV pool + `512/1024` prefill throttles were the dominant eval bottleneck (an eval cycle is ~10M prefill tokens; KV is cheap on this GDN hybrid — only 10/40 full-attn layers ≈ 10 KB/token/rank). After deploying, the §9b trained-weights regression gate at the new effective concurrency is MANDATORY before consuming science (reference pair on the old flags + ARITH-007F weights: 0.609@conc8 / 0.594@conc128). The old "35B at TP=2 OOMs at the TP=8 footprint" note motivated mem-fraction 0.85, which stays. Attention backend = UNPINNED default since 2026-06-10 ~21:48, which SGLang resolves to **fa3** on this build/model (verified in ServerArgs) — identical to all validated runs. To genuinely test flashinfer you must pin `--attention-backend flashinfer` explicitly, and then run the §9b trained-weights regression gate (in-loop conc-128 control vs serial probe of the same weights).
- **SMG dispatch:** `round_robin` for >1 endpoint (auto); wait for `/v1/models` to list the model, not `/health`.
- **Never hand-launch the autopilot** while a manual candidate runs (it would launch a queued idea and clobber it).
