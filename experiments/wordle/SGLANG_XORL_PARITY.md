# SGLang ↔ xorl logprob parity — knobs, findings, recipe (2026-06-26)

## Why this matters (the k3 symptom)

The GRPO Wordle runs are **on-policy** (rollouts come from the policy the trainer just synced to
the samplers; no `--pipeline-rl`). For a true on-policy run the importance ratio
`π_xorl(token)/π_sglang(token)` should be **≈1** and the k3 KL estimator
(`loss/is_kl_sample_train_k3`) should be **≈0**. Instead we measured:

- **k3 ≈ 0.05–0.07** and **`is_ratio_max` ≈ 20–45** throughout training.

That gap is a **train/inference numerical mismatch**: SGLang *samples* and reports logprobs; xorl
*recomputes* logprobs in its training forward; the two disagree per-token. Consequences:
1. Inflates k3 (the "too high" symptom).
2. **Destabilizes the unclipped `importance_sampling` loss** — the large `ratio_max` tail keeps
   yanking the over-confident policy, which is what drove the late **over-training collapse**
   (IS run peaked ~0.48 @ s76 → 0.16 @ s128). So closing the gap should *both* lower k3 *and*
   stabilize/extend the climb.

Goal (per user): turn on every available parity knob, drive k3 as low as possible (ideally ~0).

---

## Root causes of the mismatch (ranked)

1. **SGLang Triton packed GDN decode beta precision bug** — for the current worst raw-K3 token
   (`train.parquet:rg0:row1` pos26), `fused_recurrent_gated_delta_rule_packed_decode` cast
   `sigmoid(b)` through the bf16 `b` dtype before re-upcasting to fp32. Non-packed Triton decode
   keeps beta in fp32. Removing that bf16 round-trip makes packed and non-packed same-call GDN
   outputs/states bit-identical on the row1 probe and flips the bad decode argmax back from `5454`
   to `26003`. Full fixed-SGLang refreshed-trace K3 improves but does not reach zero: mean K3
   drops from `0.0394778060` to `0.0146352545`, with the remaining tail dominated by other
   SGLang generation-vs-prefill logprob residuals.
2. **SGLang generation-vs-prefill logprob surface mismatch** — SGLang's generation-time
   `output_token_logprobs` can disagree sharply with SGLang's own fixed-sequence prefill/top-k
   scoring for the same completed sequence. xorl tracks the fixed-sequence scoring surface much
   more closely than the generation-time logprobs. The packed GDN bug above is the proven cause of
   the dominant row1/pos26 generation-vs-prefill split on the refreshed current-sampler worst4 R3
   replay below. A second diagnostic showed SGLang's prefill-side `fused_gdn_gating` beta
   bf16-rounding is also a real decode/prefill lever, but it is not a standalone K3 fix: xorl's
   `_sglang_compatible_beta_gate` deliberately rounds beta through `b_input.dtype`, and changing the
   SGLang prefill contract without solving the downstream row2 tail makes raw K3 much worse.
3. **MoE router divergence** — SGLang and xorl pick *different experts* for the same token →
   different outputs. (Drives the `ratio_max` tail.)
4. **Batch-variance** — SGLang computes logprobs in large decode batches; standard kernels are
   *not* batch-invariant, so a token's logprob depends on what else is in the batch. xorl
   recomputes in different batches → systematic gap. (Drives the average k3.)
5. **Kernel/numerics differences** — fused vs unfused RoPE, fused vs native SiLU, FA3 vs other
   attention, RMSNorm mode, Q/K dtype after RoPE, fp32-vs-bf16 logits.
6. **Non-deterministic reductions** — NCCL all-reduce order, cuBLAS GEMM, FA3 split-K, DeepEP
   all-to-all combine.

---

## The knobs (grouped) + status

Legend: ✅ applied · 🟡 applied to config/manifest, pending restart · 🔜 pending/follow-up · ⚪ considered, not changed

### A. SGLang sampler side — the biggest lever (`wordle-sci-opsd-sampler-b`)
- 🟡 **`--rl-on-policy-target xorl-batch-invariant`** — the headline knob. Our SGLang
  (`xorl-sglang-internal`) ships `RL_ON_POLICY_TARGET_CHOICES = ["fsdp","xorl","xorl-batch-invariant"]`
  + a `batch_invariant_ops` module. Setting it:
  - auto-enables **deterministic inference** (`enable_deterministic_inference=True`);
  - makes the sampler compute logprobs via a **batch-invariant `log_softmax`** kernel
    (`sampler.py: use_log_softmax_logprob = rl_on_policy_target is not None`) — i.e. *the same
    `log_softmax` math xorl uses* (`new_logprob = −CE`) → ratios collapse toward 1;
  - uses batch-invariant matmul / log_softmax / rmsnorm kernels (batch-size-independent);
  - on Hopper (H100/SM90) forces **`attention_backend=fa3`** (its deterministic variant — *no*
    FA3 uninstall needed, unlike the slime doc which targeted other arch), **`sampling_backend=pytorch`**,
    disables aiter-allreduce-fusion, and disables radix cache if the backend can't support it.
  - **Trade-off: slower sampling** (pytorch sampling, no radix cache, batch-invariant kernels).
    Parity over throughput — acceptable, the run is already rollout-bound.
  - Patched into `launch/wordle-sci-opsd-sampler-b.yaml`; applies on the next sampler restart.

### B. xorl trainer numerics (server config YAML — `configs/grpo-ep8x2node-muon-lowlr-isr3.yaml`)
These are documented `ServerArguments` (`server_arguments.py`), read by `model_runner` from
`model_config`. Three say **"for SGLang alignment"** in their docstrings:
- ✅ **`activation_native: true`** — unfused SiLU (matches SGLang), vs fused Triton kernel.
- ✅ **`rope_native: true`** — naive RoPE (matches SGLang), vs flash_attn fused kernel.
- ✅ **`attention_cast_bf16: true`** — cast Q/K to bf16 *after* RoPE (matches SGLang).
- ✅ **`flash_attention_deterministic: true`** — deterministic FA kernels.
- ⚪ **`rmsnorm_mode`** — `eager`|`native`|`compile`, default/kept **`native`** (`torch.nn.functional.rms_norm`).
  No evidence a different mode bit-matches SGLang; revisit if k3 is still high after the above.
- ✅ **`lm_head_fp32: true`** — fp32 logits (already the default; SGLang also upcasts).

### C. Determinism env vars (trainer head + worker — builder IBENV)
From slime's deterministic recipe (Megatron side ≈ xorl side):
- ✅ **`NCCL_ALGO=Ring`** — deterministic all-reduce order.
- ✅ **`CUBLAS_WORKSPACE_CONFIG=:4096:8`** — deterministic cuBLAS GEMM workspace.
- ⚪ `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` — TransformerEngine only; xorl doesn't use TE → skipped.

### D. MoE routing parity (R3 / routing replay)
- ✅ **`--return-routed-experts`** (client) — capture SGLang's per-token expert routing and force
  the xorl forward to use it (`RoutingReplayHandler`, model-agnostic, auto-enabled when present;
  engine consumes `p.routed_experts`). **Was only half-wired in `train_grpo_wordle.py`** (requested
  from SGLang but never fed to the Datum) — completed the wiring (extract `meta_info["routed_experts"]`
  → `TurnSample.routed_experts` → `Datum(routed_experts=…)`, truncated to match output_ids).
  **Measured effect: `ratio_max` halved (12 → ~5–6); average k3 unchanged.** So routing fixes the
  *tail*; the average gap is from B/C above. (slime calls this `--use-routing-replay`; arXiv:2507.18071.)
- 🔜 **`routed_expert_logits`** (the exact gate *softmax weights*, not just expert ids) — the engine
  *uses* them ("MoE layers use these exact softmax weights") but the installed
  `xorl_client.types.Datum` doesn't expose the field. **Would push k3 lower; needs an xorl_client
  upgrade** (user OK'd doing this). Follow-up.
- ⚪ **DeepEP** (`ep_dispatch: deepep`) — its all-to-all combine reduction can add non-determinism
  (user flagged "deepep can increase k3"). Routing-replay pins the routing; the combine order is the
  residual. A deterministic dispatch / disabling deepep is a possible lever but has EP-stability
  trade-offs. EP4 Triton/alltoall now passes a 4-GPU static replay as a diagnostic path, but
  EP8+DeepEP is still the proven Wordle training topology. Not changed yet.

---

## What's measured vs expected

| change | effect on k3 | effect on ratio_max |
|---|---|---|
| routed-experts (R3) alone | ~unchanged (0.055) | **halved (12→5–6)** |
| + batch-invariant sampler + numerics + env (this round) | **expected: large drop toward ~0** | expected: further down |

The batch-invariant sampler (A) + the numerics knobs (B) attack the *pervasive* average gap that
routing-replay didn't touch. If k3 → ~0, the unclipped IS should stay stable past s76 (no collapse)
and climb higher; that's the test.

---

## 2026-06-26 evidence refresh

This section is the current evidence trail from the existing K3 harnesses and artifacts. Keep it
separate from the recipe above: the recipe is the Wordle run to test; the items below are what is
already proven.

### Live Wordle parity run state

- `wordle-sci-opsd-sampler-b` is running with `--rl-on-policy-target xorl-batch-invariant` and
  `team: turbo`.
- Runtime logs from `wordle-sci-opsd-sampler-b-0` confirm the effective SGLang settings:
  `enable_deterministic_inference=True`, `sampling_backend='pytorch'`, `attention_backend='fa3'`,
  and `rl_on_policy_target='xorl-batch-invariant'`.
- The command still includes `--sampling-backend flashinfer`, but SGLang overrides it to pytorch
  during deterministic-inference handling.
- The live StatefulSet template now includes `--enable-return-routed-experts`. During the rollout,
  old revision `wordle-sci-opsd-sampler-b-785c5557f4` logged
  `enable_return_routed_experts=False`, while new revision `wordle-sci-opsd-sampler-b-98dd4bbb`
  logged `enable_return_routed_experts=True`. As of the later check, all eight pods were on
  `98dd4bbb`.
- Direct probe of old-revision `wordle-sci-opsd-sampler-b-0` proved the failure mode: a `/generate`
  request with `return_routed_experts=True` returned status 200 and logprobs, but no
  `meta_info["routed_experts"]`. That means the client silently trains without R3 when pointed at
  old-revision samplers.
- Direct probe of new-revision `wordle-sci-opsd-sampler-b-7` returned routed experts as a base64
  string in `meta_info["routed_experts"]`, so the server-side capture path is now live. The next
  Wordle launch should still verify all sampler endpoints are actually listening, because the
  StatefulSet readiness bit became true before SGLang finished loading.
- `grpo-wq36-2n-isr3` did not produce parity metrics:
  `/shared/apanda/wordle-sft-runs/20260626T173022Z-grpo-wq36-2n-grpo-wq36-2n-isr3-head-lpxcn-wordle-r16-grpo2node`.
  It created the training session, registered seven sampler load endpoints, completed the step-0
  full-weight sync in 31.5s, then the xorl server received SIGTERM at `2026-06-26 17:42:58 UTC`
  before generation/training-step metrics were written. `metrics.jsonl` contains only `init` and
  `create_model`; `sampler_exports.jsonl` contains only the step-0 full-weight sync.
- Treat the live StatefulSet command/logs as the verified sampler source until the local manifest
  generator path is located.

### Harness status

- In `/home/apanda/xorl-apanda-dev`, core R3 server plumbing is green:

  ```bash
  cd /home/apanda/xorl-apanda-dev
  uv run pytest tests/server/runner/test_routing_replay_handler.py \
    tests/server/orchestrator/test_request_processor.py -q
  # 26 passed
  ```

- In the same checkout, the experiment-level K3 test module currently skips because
  `experiments/k3_tests/` is incomplete relative to `tests/experiments/test_k3_static_traces.py`:

  ```bash
  cd /home/apanda/xorl-apanda-dev
  uv run pytest tests/experiments/test_k3_static_traces.py -q \
    -k 'routing or threshold_failed or selection_flip or compare_logprobs_writes_static_trace_bundle'
  # collected 0 items / 1 skipped
  ```

- The complete static-trace K3 harness is in
  `/home/apanda/xorl-slime-parity-low-precision`, and its focused routing/gating slice is green:

  ```bash
  cd /home/apanda/xorl-slime-parity-low-precision
  PYTHONPATH=src pytest tests/experiments/test_k3_static_traces.py -q \
    -k 'routing or threshold_failed or selection_flip or compare_logprobs_writes_static_trace_bundle'
  # 11 passed, 170 deselected
  ```

### What can and cannot be driven to K3 ~= 0 today

- **Qwen3.6-35B-A3B, SGLang static traces vs xorl FLA/quack+DeepEP baseline:** bulk tokens are
  already near zero by median, but raw mean is tail-dominated. The full32 FLA baseline artifact
  has mean k3 `0.0016289913`, median `1.02e-9`, p95 `0.0031932317`, max `0.6225874560`
  over 2820 tokens.
- **Qwen3.6-35B-A3B row2 router isolation:** forcing the SGLang router choice at layer 15 drove
  the row2 artifact from mean k3 `0.0058902821` to `0.0004941657`, p95 `0.0020817355`,
  max `0.0168205437`. This points to MoE selection/routing boundary effects, not a broad
  log-softmax or label-alignment bug.
- **FlashQLA GDN path:** not cleared. The real FlashQLA layout-fix full32 artifact has mean k3
  `0.0053439915`, median `9.99e-10`, p95 `0.0032027320`, max `6.7085201718`.
  Offline diagnosis confirms `best_shift=0`; prefill reference mode still has mean k3
  `0.0038357443`, so this is not an off-by-one replay issue.
- **Qwen3.6-35B-A3B worst4, simpler EP4 Triton/alltoall xorl path:** not zero, but it passes the
  current selection-robust gate. Replaying the four worst FlashQLA traces against
  `qwen3_6_35b_ep4_triton_alltoall.yaml` produced aggregate mean k3 `0.0110965137`,
  median `1.08e-8`, p95 `0.0060728402`, p99 `0.2561502904`, max `1.6984313229`,
  `flip_count=8`, `flip_rate=0.0207792208`, and non-flip mean k3 `0.0015176162`
  over 385 tokens. The run passed `--max-median-k3 1e-6 --max-flip-rate 0.05`, so bulk parity is
  good, but the raw tail remains.
- **Qwen3.6-35B-A3B current batch-invariant sampler + R3 routed replay, worst4 subset:** raw
  generation-reference K3 is still not zero, but the failure is now mostly SGLang's generation
  logprob surface. The durable refreshed trace bundle is
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/refreshed_traces.json`.
  All four old worst4 outputs changed during regeneration against
  `wordle-sci-opsd-sampler-b-7`; all four traces captured `routed_experts` as base64 strings.
  Replaying xorl EP4 Triton/alltoall with `--xorl-replay-routing` wrote
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/k3_result_r3_ep4_triton_alltoall.json`:
  total tokens `283`, mean k3 `0.0394778060`, median `4.02e-10`, p95 `0.0290207502`,
  p99 `0.5087232566`, max `6.5168260056`, `flip_count=9`, `flip_rate=0.0318021201`,
  and non-flip mean k3 `0.0018007774`. The worst token is
  `train.parquet:rg0:row1` pos `26`, token `5454` (`"command"`), with SGLang generation logprob
  `-0.6789726615` and xorl logprob `-2.9612650871`.
- **Same current batch-invariant R3 replay, but using SGLang prefill/fixed-sequence scoring as
  reference:** the offline diagnosis at
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/diagnosis_r3_ep4_triton_alltoall.json`
  drops to mean k3 `0.0018772817`, median `1.25e-11`, p95 `0.0017593794`, and max
  `0.1875218157`. SGLang's own generation-vs-prefill self-consistency on the same 283 tokens has
  mean absolute logprob delta `0.0563636811`, p95 `0.2101510704`, and max `2.2813790441`.
  The top raw-K3 token above is exactly the top SGLang self-consistency failure:
  generation logprob `-0.6789726615`, prefill logprob `-2.9603517056`, and xorl logprob
  `-2.9612650871`. So this run does not implicate xorl's forward pass for the dominant raw-K3
  spike; it shows that the training old-logprob source must be fixed or switched to the same
  fixed-sequence scoring surface as xorl.
- **Qwen3-Coder-30B-A3B non-GDN control:** a two-prompt live K3 smoke passed on
  `Qwen/Qwen3-Coder-30B-A3B-Instruct` using SGLang from
  `/home/apanda/xorl-sglang-k3-beta-fp32/python`, SGLang Python
  `/home/apanda/xorl-sglang-internal/.venv/bin/python`, and the xorl FSDP config
  `experiments/k3_tests/configs/qwen3_coder_30b_fsdp.yaml`. The result artifact
  `/shared/opd-control/er-opd-q36-35b-slots/k3/qwen3_coder_30b_gdn_control_20260627T014911Z/k3_result_q30b-coder-live-k3-20260627T014911Z.json`
  has total tokens `256`, mean k3 `0.0003034366`, median `2.57e-12`, p95
  `0.0009302507`, p99 `0.0084238473`, max `0.0161128602`, and `flip_count=0`.
  The reusable trace bundle is
  `/shared/opd-control/er-opd-q36-35b-slots/k3/qwen3_coder_30b_gdn_control_20260627T014911Z/static_traces_q30b-coder-live-k3-20260627T014911Z.json`.
  Those traces do not include routed-expert payloads, but the independent-routing live comparison
  already had zero K3 flips. Offline re-scoring of the saved per-token logprobs in
  `/shared/opd-control/er-opd-q36-35b-slots/k3/qwen3_coder_30b_gdn_control_20260627T014911Z/q30b_prefill_vs_generation_reference_summary.json`
  was also low against SGLang generation-time logprobs: mean k3 `0.0002546412`, p95
  `0.0007118383`, and max `0.0188480212`. This supports the current hypothesis that the remaining
  Qwen3.6 tail is GDN-specific rather than a broad Qwen3 MoE/logprob path failure.
  A follow-up routed-capture trace bundle at
  `/shared/opd-control/er-opd-q36-35b-slots/k3/qwen3_coder_30b_r3_control_20260627T165222Z/static_traces_q30b-coder-r3-live-20260627T165222Z.json`
  includes SGLang `routed_experts` payloads for both traces. Replaying it with
  `--xorl-replay-routing` wrote
  `/shared/opd-control/er-opd-q36-35b-slots/k3/qwen3_coder_30b_r3_control_20260627T165222Z/k3_result_q30b-coder-r3-static-20260627T170019Z.json`
  and reduced K3 to mean `0.0000889317`, median `2.17e-13`, p95 `0.0004173926`,
  p99 `0.0014055080`, max `0.0084859795`, and `flip_count=0`. The matched static no-R3
  control at
  `/shared/opd-control/er-opd-q36-35b-slots/k3/qwen3_coder_30b_r3_control_20260627T165222Z/k3_result_q30b-coder-static-nor3-20260627T170418Z.json`
  stayed at the live independent-routing result: mean `0.0003034366`, p95
  `0.0009302507`, max `0.0161128602`, and `flip_count=0`. The replay pod still emitted
  `R3: No valid routing data after decoding` warnings on some ranks, so treat this as useful
  evidence that routing replay affects the 30B smoke, not yet as a clean R3 certification harness.
- **Narrowed source inside SGLang:** the live sampler logs for `wordle-sci-opsd-sampler-b-7` show
  `Linear attention kernel backend: decode=triton, prefill=triton` and
  `GDN kernel dispatcher: decode=TritonGDNKernel, extend=TritonGDNKernel, verify=TritonGDNKernel
  packed_decode=True`. Generation requests use the Qwen3.5/3.6 GDN `forward_decode()` branch
  (`gdn_backend.py`) with `causal_conv1d_update` plus `TritonGDNKernel.packed_decode`, which calls
  `fused_recurrent_gated_delta_rule_packed_decode`; fixed-sequence scoring uses `forward_extend()`
  with `causal_conv1d_fn` plus `TritonGDNKernel.extend`, which calls `chunk_gated_delta_rule`.
  Non-packed decode/target-verify call `fused_sigmoid_gating_delta_rule_update`. Since the failing
  generation was greedy (`temperature=0`) and token5454 was rank 3 under prefill but chosen by
  decode, the mismatch is a real decode-state/logit divergence, not just a trace logprob extraction
  bug or post-hoc normalization difference.
- **Direct SGLang-only confirmation:** `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_row1_pos26_decode_prefill_probe.json`
  replays the same row1 prompt against `wordle-sci-opsd-sampler-b-7` for 33 greedy decode tokens and
  exactly matches the saved first 33 generated tokens. At pos26, decode top-5 is
  `[5454:-0.6789726615, 26003:-1.6789727211, 9951:-2.1789727211, 11411:-3.4289727211,
  22319:-3.9289727211]`; fixed-sequence prefill top-5 for the same token position is
  `[26003:-0.3353516161, 9951:-2.3353517056, 5454:-2.9603517056, 11411:-3.3353517056,
  22319:-4.0853514671]`. So the decode path really chose a different top token than the prefill
  path for the same prompt prefix.
- **Decode-backend ablations on the same row1/pos26 probe:** the one-off SGLang pods changed only
  the linear-attention decode backend and kept prefill on Triton. A debug worktree at
  `/home/apanda/xorl-sglang-k3-gdn-debug` added an env-gated
  `SGLANG_K3_GDN_FORCE_NONPACKED=1` switch that only forces
  `GDNKernelDispatcher.supports_packed_decode=False`; the patch is preserved at
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_triton_nonpacked_debug_patch.diff`.
  The forced non-packed Triton probe wrote
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_row1_pos26_triton_nonpacked_decode_probe.json`;
  its pod log confirms `decode=triton`, `prefill=triton`, and `packed_decode=False`. It did not
  reproduce the packed-Triton continuation. At pos26 it generated token `26003`, the same token that
  prefill ranked first, with decode logprob `-1.1475313902`; same-sequence prefill scored that
  generated token `-0.3353516161` (`decode-minus-prefill=-0.8121797740`), and a repeat decode
  matched exactly. This is the strongest current isolation: the wrong argmax at row1/pos26 is on
  the Triton packed-decode path, not on SGLang's shared input projection, short-conv update,
  non-packed Triton recurrent update, top-k extraction, or xorl's fixed-sequence forward.
  A second no-CUDA-graph packed run added an in-process comparator that, on each packed decode call,
  reran non-packed Triton from the same already-convolved `mixed_qkv`, `a`, `b`, `A_log`,
  `dt_bias`, and pre-update `ssm_states`. It wrote
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_packed_vs_nonpacked_decode_compare.jsonl`
  and the compact summary
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_packed_vs_nonpacked_decode_compare_summary.json`.
  The packed no-graph decode still exactly reproduced the saved bad continuation and pos26 token
  `5454`; the comparator emitted 2,340 same-call rows across ranks 0/1, 30 linear-attention layers,
  and call indices 0-38. Using only `max_abs` fields from that artifact, packed-vs-nonpacked
  differences begin immediately (`first_call_with_out_abs_gt_0=0`, `first_call_with_state_abs_gt_1e_minus_3=0`),
  with global max core-output delta `0.0078125` and global max recurrent-state delta
  `0.0217790604`. At call26, the worst one-step core-output delta is only `0.0002441406`, while
  the recurrent-state delta is `0.0157604218`. So the bad argmax is more likely accumulated
  recurrent-state drift from repeated packed updates than a single catastrophic one-step packed
  output at pos26.
  The follow-up candidate fix in the same debug worktree changed packed decode from
  `tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)` to `tl.sigmoid(b_val)` inside
  `fused_recurrent_gated_delta_rule_packed_decode_kernel`, matching the non-packed fp32 beta math.
  The first validation pod reached startup but could not bind host port `30000`; its log is
  preserved at
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_packed_beta_fp32_nograph_probe_port30000_failed_pod.log`.
  The successful port-`31000` validation pod
  `sglang-gdn-packed-beta-fp32-nograph-20260626t1842z` ran with Triton decode/prefill,
  `packed_decode=True`, and `--disable-cuda-graph`. Its row1 probe wrote
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_row1_pos26_packed_beta_fp32_nograph_decode_probe.json`:
  pos26 generated token `26003` rather than the saved bad packed token `5454`; top-5 was
  `[26003:-1.1475313902, 5454:-1.2725313902, 9951:-1.8975313902, 11411:-2.5225315094,
  22319:-3.3975315094]`. The same-call comparator wrote
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_packed_vs_nonpacked_decode_compare_beta_fp32.jsonl`
  and summary
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_packed_vs_nonpacked_decode_compare_beta_fp32_summary.json`;
  over 1,920 rows across ranks 0/1, 30 linear-attention layers, and call indices 7-38,
  `global_max_out_abs=0.0`, `global_max_state_abs=0.0`, and call26 also had zero output/state
  delta. The debug patch, including the comparator hooks, is preserved at
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_gdn_debug_hooks_beta_fp32_patch.diff`,
  and the production-relevant one-line fix alone is preserved at
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_gdn_beta_fp32_minimal_patch.diff`.
  The successful pod log is preserved at
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_packed_beta_fp32_nograph_probe_pod.log`.
  This confirms the row1/pos26 wrong argmax came from the packed kernel's bf16 beta round-trip.
  A clean minimal-fix worktree was then created at `/home/apanda/xorl-sglang-k3-beta-fp32` with only
  the one-line `fused_recurrent.py` patch. A production-like fixed reference pod
  `sglang-gdn-beta-fp32-fixed-ref-20260626t1905z` ran from that worktree with CUDA graph enabled,
  Triton decode/prefill, `packed_decode=True`, and `--enable-return-routed-experts`. Its log is
  preserved at
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_beta_fp32_fixed_ref_pod.log`;
  the current minimal diff is preserved at
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_beta_fp32_minimal_worktree_current.diff`.
  Regenerating the four current worst traces from this fixed server wrote
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/refreshed_traces_beta_fp32_fixed.json`.
  The refresh captured routed experts for all four traces; three outputs changed. The key expected
  change was `train.parquet:rg0:row1` pos26, where the stale packed output `5454` changed to
  `26003`.
  Replaying that fixed bundle through xorl EP4 Triton/alltoall with `--xorl-replay-routing` wrote
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/k3_result_r3_ep4_triton_alltoall_beta_fp32_fixed.json`:
  total tokens `282`, mean K3 `0.0146352545`, median `7.46e-10`, p95 `0.0147035919`,
  max `1.9126869478`, `flip_count=7`, `flip_rate=0.02482`, and non-flip mean K3 `0.00163343`.
  The new worst token moved to `train.parquet:rg0:row0` pos124 token `318` (`" ("`), with
  SGLang generation logprob `-0.2279991955` and xorl logprob `-1.7079229355`. The offline
  diagnosis at
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/diagnosis_r3_ep4_triton_alltoall_beta_fp32_fixed.json`
  again confirmed `best_shift=0`; using SGLang prefill/fixed-sequence logprobs as reference drops
  the same xorl result to mean K3 `0.0011510748`, p95 `0.0014324672`, and max `0.1205935681`.
  A direct fixed-server row0 decode-vs-prefill probe at
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_row0_worst_tokens_beta_fp32_decode_prefill_probe.json`
  reproduced the regenerated row0 output and shows the remaining worst tokens are still SGLang
  decode-vs-prefill rank/logprob splits. At row0 pos124, decode ranked token `318` first at
  `-0.2279991955`, while fixed-sequence prefill ranked token `198` first at `-0.1177402288` and
  token `318` second at `-2.2427401543`.
  A prefill-beta diagnostic worktree at `/home/apanda/xorl-sglang-k3-prefill-beta-fp32` applied
  both the packed decode beta fix and a one-line `fused_gdn_gating.py` probe that stores
  `blk_beta_output` directly into the float32 `beta_output` tensor instead of first casting it back
  to `b.dtype`. The probe diff is preserved at
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_gdn_decode_and_prefill_beta_fp32_probe_patch.diff`,
  the pod manifest at
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_gdn_prefill_beta_fp32_probe_pod.yaml`,
  and the pod log at
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_prefill_beta_fp32_probe_pod.log`.
  The direct row0 probe
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_row0_worst_tokens_prefill_beta_fp32_decode_prefill_probe.json`
  proves this is a real numerical lever: row0 generation first changed at pos92 (`539 -> 4417`),
  and the old fixed-sequence prefill scores moved toward the previous decode surface (for example,
  old row0 pos84 prefill changed from `-0.9741321` to `-0.3132895`). Regenerating all four traces
  against that patched SGLang wrote
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/refreshed_traces_prefill_beta_fp32_fixed.json`;
  three of four outputs changed, with generation self-consistency means `0.0260` row25, `0.0340`
  row1, `0.0195` row0, and `0.0461` row2.
  However, replaying that patched-SGLang bundle against unchanged xorl EP4 Triton/alltoall wrote
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/k3_result_r3_ep4_triton_alltoall_prefill_beta_fp32_fixed.json`
  and was much worse: total tokens `266`, mean K3 `3.5107392183`, median `4.56e-10`, p95
  `0.0444147441`, max `845.0978675086`, `flip_count=9`, and non-flip mean `0.0028925622`.
  The damage is almost entirely the new row2 tail: row2 mean K3 `22.7364934262`, worst pos40
  token `248046` (`<|im_end|>`), SGLang logprob `-0.0087642530`, xorl logprob `-6.7573437691`.
  The diagnosis
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/diagnosis_r3_ep4_triton_alltoall_prefill_beta_fp32_fixed.json`
  shows this is not fixed by using SGLang prefill reference (`mean=3.4830723670`, max
  `842.0111290908`).
  A focused row2 replay with `--xorl-diagnostic-topk 5` wrote
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/k3_result_r3_ep4_triton_alltoall_prefill_beta_fp32_row2_topk.json`;
  its pod log and manifest are preserved at
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/xorl_row2_prefill_beta_topk_pod.log`
  and
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/xorl_row2_prefill_beta_topk_pod.yaml`.
  At the worst row2 pos40 terminal token, SGLang's top-5 has `<|im_end|>` rank1 at
  `-0.0123944273`, then token `271` at `-5.0123944283`; xorl's top-5 has token `198` rank1 at
  `-0.0073439162`, then `<|im_end|>` rank2 at `-6.7573437691`. The preceding row2 tail already
  diverges: at pos38, SGLang ranks token `3359` first while xorl ranks token `29` first; at pos37,
  SGLang ranks token `13766` first while xorl's top three are `198`, `14`, and `248046`.
  A paired xorl beta-fp32 probe was also tried in `/home/apanda/xorl-k3-beta-fp32-xorl`, changing
  `_sglang_compatible_beta_gate` to return `b_input.float().sigmoid()` without the bf16 round-trip.
  Its patch, manifest, pod log, result, and diagnosis are preserved as
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/xorl_beta_fp32_probe_patch.diff`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/xorl_beta_fp32_probe_pod.yaml`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/xorl_beta_fp32_probe_pod.log`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/k3_result_r3_ep4_triton_alltoall_sglang_xorl_prefill_beta_fp32.json`,
  and
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/diagnosis_r3_ep4_triton_alltoall_sglang_xorl_prefill_beta_fp32.json`.
  That paired run was also worse (`mean=3.9481627284`, p95 `0.0633113192`, max `959.2363145338`)
  on the same row2 `<|im_end|>` token. Conclusion: prefill beta precision explains part of the
  SGLang decode/prefill surface mismatch and improves row0/row1 locally, but changing that contract
  moves SGLang onto a row2 continuation that xorl strongly disagrees with; it should stay a
  diagnostic branch, not the next production fix.
  The fixed-beta trace was then re-scored offline against SGLang's fixed-sequence prefill
  logprobs instead of generation-time logprobs, using the saved xorl per-token logprobs from
  `k3_result_r3_ep4_triton_alltoall_beta_fp32_fixed.json`. The compare-style artifact
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/k3_result_r3_ep4_triton_alltoall_beta_fp32_fixed_prefill_reference_offline.json`
  and diagnosis
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/diagnosis_r3_ep4_triton_alltoall_beta_fp32_fixed_prefill_reference_offline.json`
  show the same xorl forward path is much closer to zero under that fixed-sequence reference:
  total tokens `282`, mean K3 `0.0011510748`, median `2.80e-11`, p95 `0.0014324672`,
  p99 `0.0151254098`, max `0.1205935681`, and best alignment shift `0`. The remaining prefill
  reference tail is sparse: row0 pos124 token `318` (`" ("`) and row0 pos93 token `1510`
  (`" `"`) dominate the max, while row0 p95 is only `0.0002676023`.
  A focused row0 replay with `--xorl-diagnostic-topk 5` wrote
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/k3_result_r3_ep4_triton_alltoall_beta_fp32_fixed_row0_topk.json`;
  its prefill-reference offline artifact and diagnosis are
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/k3_result_r3_ep4_triton_alltoall_beta_fp32_fixed_row0_topk_prefill_reference_offline.json`
  and
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/diagnosis_r3_ep4_triton_alltoall_beta_fp32_fixed_row0_topk_prefill_reference_offline.json`;
  the pod log/manifest/launch args are preserved as
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/xorl_row0_beta_fixed_topk_pod.log`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/xorl_row0_beta_fixed_topk_pod.yaml`,
  and
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/xorl_row0_beta_fixed_topk_launch_args.json`.
  That focused replay confirms the row0 tail is not a top-token fork under the prefill reference:
  at positions `67`, `84`, `92`, `93`, and `124`, SGLang and xorl have identical top-5 token ids
  in the same order. The residual is logit-margin calibration inside the same ranked distribution
  (for example, at pos124 both rank token `198` first and token `318` second; SGLang scores token
  `318` at `-2.2427401543`, xorl at `-1.7079229355`).
  The follow-up row0 final-hidden/component probes are preserved under the same artifact root. The
  fixed-path SGLang final-hidden request wrote
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_sglang_return_hidden_fixed.json`
  with `rl_on_policy_target=xorl-batch-invariant`, `hidden_kind=final`, `input_mode=full`, and score
  rows `2047..2174`; the paired xorl final-hidden rows are in
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_xorl_final_hidden_rows_pos93_pos124.pt`.
  The compact comparison
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_sglang_xorl_final_hidden_compare_pos93_pos124.json`
  shows the residual is already present in the final hidden state. At pos93/source row `2140`, xorl
  minus SGLang final hidden has max abs `0.62890625`, mean abs `0.0835710466`, and rms
  `0.1059719324`; selected HF `lm_head` rows give margin `88461-1510 = 1.0694427490` for SGLang
  and `0.3099594116` for xorl, a delta of `-0.7594833374` versus the API top-logprob margin delta
  `-0.75`. At pos124/source row `2171`, final hidden max abs is `0.328125`, mean abs
  `0.0662549287`, and rms `0.0837234259`; selected margins are `198-318 = 2.0804119110` for SGLang
  and `1.5232563019` for xorl, a delta of `-0.5571556091` versus the API margin delta `-0.625`.
  The SGLang returned top-logprobs in that same artifact exactly recover the fixed prefill targets
  (`-1.5019326210` at pos93 token `1510`, `-2.2427401543` at pos124 token `318`). The xorl
  hidden/reference-logit run
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/k3_result_r3_ep4_triton_alltoall_beta_fp32_fixed_row0_xorl_hidden_ref_logits.json`
  showed `xorl_loss_logprob_delta=0.0` for both positions; its fp32 raw-weight reference differs
  from the xorl loss path by only `-0.1140214801` at pos93 and `+0.0185151100` at pos124. So the
  remaining row0 error is not target extraction, top-k ordering, or xorl CE/logsoftmax precision.
  Late component dumps exist at
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_finalcomp_sglang_components_fixed.pt`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_finalcomp_xorl_components_fixed.rank0.pt`,
  and
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_finalcomp_xorl_vs_sglang_component_compare_fixed_rank0.json`,
  but those SGLang tensor names are input-side captures and do not expose the full final logits
  hidden for rows `2140`/`2171`. Treat the return-hidden artifact as the authoritative final-hidden
  boundary.
  A follow-up final-norm closure report is
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_final_norm_boundary_closure_pos93_pos124.json`.
  It corrects the boundary further: xorl `model.layers.39.layer_output` and SGLang raw
  `model.norm[1]` are the pre-final-norm vectors, and their row deltas are small before final norm
  (pos93 max `0.044921875`, mean `0.0068286955`, rms `0.008606622`; pos124 max `0.03515625`,
  mean `0.0065105408`, rms `0.008318231`). Applying the effective Qwen3.6 final RMSNorm as
  zero-centered `(1.0 + model.language_model.norm.weight)` exactly closes xorl's captured final
  hidden at both positions; using the raw checkpoint weight as a standard multiplier does not
  close xorl (mean error about `0.79`/`0.77`). Applying the same zero-centered norm to SGLang's
  pre-final vector closes the returned SGLang final hidden within bf16 dump noise (mean
  `0.0024585724` at pos93 and `0.0020594597` at pos124). The final-hidden gap is therefore the
  small pre-final row gap after final-norm amplification, not a final RMSNorm implementation or
  xorl CE/logsoftmax bug.
  One caution from the attempted layer-ladder capture: the current xorl hidden-component hook is not
  transparent for earlier-layer captures. The perturbation summary
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_component_capture_perturbation_summary.json`
  shows the authoritative row0 `38-39` capture kept the prior xorl logprobs (`pos93=-0.9740812182`,
  `pos124=-1.7079229355`), but adding layer `30` changed the same forward to `pos93=-0.6931496859`
  and `pos124=-1.2301137447`; a layer-30-only capture produced the same changed values. The
  stricter row-only negative control
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/k3_result_row0_l30_rowonly_xorl_components_fixed_20260626T204215Z.json`
  installed only layer-30 `layer_input,layer_output` hooks and copied rows `2047-2174` to CPU
  immediately; it still reproduced the same changed logprobs (`pos93=-0.6931496859`,
  `pos124=-1.2301137447`). Its compact dumps are row-only (`128 x 2048` bf16 rows per component,
  about `1.1M` per rank). The comparison artifact
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_l30_rowonly_xorl_hook_perturbation_compare_20260626T204215Z.json`
  compares that run against the transparent xorl `output_hidden_states` baseline: at pos93/source row
  `2140`, layer-30 input already differs from transparent hidden-state index `30` by rms
  `0.0055154883` and layer-30 output differs from index `31` by rms `0.0070040952`; at pos124/source
  row `2171`, the corresponding rms deltas are `0.0052066757` and `0.0082584191`. A nearest-index
  check confirms those are the closest hidden-state indices, so this is not an off-by-one mapping
  issue. The non-authoritative ladder artifacts are still preserved as
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_ladder_sglang_artifact_fixed.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_ladder_xorl_components_fixed.rank0.pt`,
  and
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_ladder_component_input_progression_hypothesis.json`,
  but they should be treated as a memory/lifetime-sensitivity diagnostic, not as authoritative layer
  localization.
  Superseded note: the earlier no-xorl-hook parent-boundary, same-input GDN/MLP, and SGLang-context
  rowpatch artifacts from `20260626T210047Z` were generated from a SGLang request that did not match
  the fixed trace scoring surface and whose dumped top-k routes did not exactly match the trace
  `routed_experts` payload. Keep those artifacts only as diagnostics for that stale request. They
  should not be cited as authoritative layer36-38 localization for the exact fixed trace replay; use
  the route-validated artifacts and full-prefix replay above instead.

  The no-hook full-prefix capture path is now wired through the xorl runner and K3 launcher:
  `/home/apanda/xorl-apanda-dev/src/xorl/server/runner/model_runner.py` accepts
  `diagnostic_hidden_state_path`, `diagnostic_hidden_state_layers`, and
  `diagnostic_hidden_state_row_indices` in `loss_fn_params` and writes
  `<path>.rankN.pt` with `model.hidden_states.<idx>` tensors plus original-shape/row metadata.
  The same runner-side hook-free dump is mirrored in the launcher checkout at
  `/home/apanda/xorl-slime-parity-low-precision/src/xorl/server/runner/model_runner.py`, which is
  the source tree used by the command below via `PYTHONPATH=${REPO_DIR}/src`.
  `/home/apanda/xorl-slime-parity-low-precision/experiments/k3_tests/compare_static_traces.py` and
  `launch_k3_test.py` expose the same flags as
  `--xorl-diagnostic-hidden-state-path`, `--xorl-diagnostic-hidden-state-layers`, and
  `--xorl-diagnostic-hidden-state-row-indices`. For the row0 late-layer probe, capture
  output-hidden-state indices `36-38` and flattened rows `0-2175`: index `36` is input to decoder
  layer36, `37` is output of layer36/input to layer37, and `38` is output of layer37/input to layer38.
  The full prefix row range matters; `2047-2174` would only cover label/response rows and would not
  give an xorl prior-row context for the GDN recurrent-state replay.

  Relaunch recipe:

  ```bash
  cd /home/apanda/xorl-slime-parity-low-precision
  BASE=/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z
  RUN=q36-row0-fullprefix-hidden-$(date -u +%Y%m%dT%H%M%SZ)
  K8S_NAMESPACE=apanda K8S_PVC_NAME=home-apanda \
    /home/apanda/xorl-internal/.venv/bin/python experiments/k3_tests/launch_k3_test.py \
    --model qwen3.6-35b \
    --xorl-config experiments/k3_tests/configs/qwen3_6_35b_ep4_triton_alltoall.yaml \
    --xorl-gpus 4 \
    --static-traces-file "$BASE/refreshed_traces_beta_fp32_fixed.json" \
    --trace-id train.parquet:rg0:row0 \
    --num-prompts 1 \
    --xorl-replay-routing \
    --xorl-venv /home/apanda/xorl-internal/.venv \
    --client-python /home/apanda/xorl-internal/.venv/bin/python \
    --xorl-node-selector-from-capacity \
    --gpu-capacity-wait-timeout-sec 0 \
    --pod-suffix "$RUN" \
    --log-dir "/tmp/$RUN" \
    --output-json "$BASE/k3_result_${RUN}.json" \
    --xorl-diagnostic-topk 5 \
    --xorl-diagnostic-hidden-state-path "$BASE/xorl_fullprefix_hidden_${RUN}" \
    --xorl-diagnostic-hidden-state-layers 36-38 \
    --xorl-diagnostic-hidden-state-row-indices 0-2175
	  ```

  This capture has now been run as `q36-row0-fullprefix-hidden-20260626T212634Z`. Artifacts:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/k3_result_q36-row0-fullprefix-hidden-20260626T212634Z.json`
  and
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/xorl_fullprefix_hidden_q36-row0-fullprefix-hidden-20260626T212634Z.rank{0,1,2,3}.pt`.
  The hidden-state rank files are identical for indices `36`, `37`, and `38`; each saved tensor is
  `(2176, 2048)` bf16 with rows `0-2175`. The replay reproduced the known row0 tail with `auto`
  reference logprobs: mean K3 `0.0258559484`, median `1.7e-12`, max `1.9126869478` at
  pos124/token318, and `5/128` tokens above K3 `0.1`.

  The new full-prefix offline replay artifact is
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer36_37_fullprefix_replay_20260626T212634Z.json`,
  with compact diagnosis
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_fullprefix_route_prefill_diagnosis_20260626T212634Z.json`.
  It replays layers36/37 with three contexts: base SGLang dump context, target-row patch, and full
  prefix xorl `hidden[L]` context. Critical result: with full-prefix xorl context and xorl/trace-equivalent
  unforced expert sets, the replay matches actual xorl `hidden[L+1]` within bf16-scale RMS on the four
  checked layer/row pairs: layer36/pos93 `0.0006647`, layer36/pos124 `0.0007291`,
  layer37/pos93 `0.0007401`, layer37/pos124 `0.0008485`. Using the older component dump's captured
  IDs instead leaves large residuals on the bad rows: layer36/pos124 `0.0192885` and layer37/pos93
  `0.0134731` RMS vs xorl.

  The trace `routed_experts` payload explains why: for layer36/row2171 the trace expert set is
  `{2,10,39,74,124,142,153,227}` while the component dump used
  `{2,39,74,115,124,142,153,227}`; for layer37/row2140 the trace uses expert `99` where the component
  dump used expert `84`. Therefore the earlier SGLang component dump is not route-aligned to the exact
  trace replay on the two bad rows, and its residual-stream comparison should not be treated as the
  exact prefill-reference residual boundary.

  The same diagnosis recomputed K3 against `sglang_prefill_logprob` stored per token rather than
  `auto`/generation logprobs. Row0 drops from auto mean/max `0.025856`/`1.912687` to prefill-target
  mean/max `0.0028763`/`0.172319`; p95 drops to `0.0002635`. The largest two auto tails are dominated
  by SGLang generation-vs-prefill disagreement: pos124/token318 generation `-0.2279992`, prefill
  `-2.2427402`, xorl `-1.7079229`; pos93/token1510 generation `-0.0788911`, prefill `-1.5019326`,
  xorl `-0.9740812`. So the biggest row0 `auto` K3 is not a direct xorl-vs-SGLang-prefill forward
  mismatch; it is mostly SGLang generation logprob inconsistency. The remaining prefill-reference
  mismatch is still real but much smaller, concentrated at pos124 and pos93.

  Route-aligned SGLang follow-up was then completed. The debug-dump converter and launcher now request
  `return_routed_experts`, record `response_route_alignment`, record `dump_route_alignment` from
  dumped `model.layers.N.mlp.topk` ids, and support
  `--sglang-debug-dump-validate-routed-experts` to fail a dump that does not exactly match the trace
  route on the compared rows/layers. Focused tests in
  `/home/apanda/xorl-slime-parity-low-precision/tests/experiments/test_k3_static_traces.py` cover the
  request payload and decoded route-alignment summary.

  Two initial live SGLang-only route-validated captures were run from
  `/home/apanda/xorl-sglang-k3-beta-fp32/python` with
  `--rl-on-policy-target xorl-batch-invariant`, `--enable-return-routed-experts`, and the fixed
  trace
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/refreshed_traces_beta_fp32_fixed.json`:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_routealigned_artifact_q36-row0-sglang-routealigned-20260626T214525Z.json`
  and
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_routeparent_artifact_q36-row0-sglang-routeparent-20260626T215049Z.json`.
  Both have `response_route_alignment.matches=true`, `dump_route_alignment.matches=true`, and
  `mismatch_count=0` over `2048` checked route entries for layers `36-37` across the 128 target
  rows. Both also exactly reproduce the saved fixed-prefill trace logprobs (`max_abs_delta=0.0`).
  Their pods scheduled on `research-common-h100-100.cloud.together.ai` and were cleaned up.

  This supersedes the earlier parent dump as an exact trace residual reference. The old parent dump
  routecheck artifact
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_old_sglang_dump_routecheck_20260626T214419Z.json`
  shows `dump_route_alignment.matches=false`: `743/2048` exact target-route entries mismatched across
  layers `36-37`. Its request logprobs were also not the fixed trace scoring surface: max absolute
  delta versus the trace was `0.8316159248`; pos93 returned `-0.9740803838` instead of
  `-1.5019326210`, and pos124 returned `-1.4111242294` instead of `-2.2427401543`. Therefore the
  old parent residual-stream growth and same-input GDN/MLP artifacts remain useful as diagnostics for
  that stale request, but they are not authoritative evidence for the exact fixed trace replay.

  A third route-validated capture then used the same fixed trace and same minimal SGLang scoring
  surface, but with only the debug tensor-dump hook patched to preserve parent/module input captures:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_routealias_artifact_q36-row0-sglang-routealias-20260626T220504Z.json`.
  It has `response_route_alignment.matches=true`, `dump_route_alignment.matches=true`, and
  `mismatch_count=0` over the same `2048` layer36/37 route entries. Returned prefill logprobs match
  the fixed trace exactly (`max_abs_delta=0.0`), including pos93 `-1.5019326210` and pos124
  `-2.2427401543`. The dump now contains `layer_input`, `forward_input.residual`,
  `input_layernorm`, `post_attention_layernorm`, `post_attention_residual`, `mlp`, and
  `layer_output` aliases for both layers 36 and 37. The routealias pod was cleaned up after writing
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_routealias_dump_q36-row0-sglang-routealias-20260626T220504Z`.

  The routealias parent-boundary replay artifact is
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer36_37_routealias_fullprefix_replay_20260626T220504Z.json`.
  Across layer36/37 at pos93/pos124, the observed xorl-vs-SGLang `layer_output` gap is `0.00488`
  to `0.00651` RMS, while the incoming residual-stream gap is already `0.00420` to `0.00544` RMS.
  The base SGLang-context replay matches captured SGLang `layer_output` to at most `0.0011223` RMS.
  Replacing the full prefix with xorl hidden states matches actual xorl `hidden[L+1]` at bf16 scale
  (`fullprefix_final_regathered_weights_vs_xorl_output` max `0.0008475257`) and projects
  `0.9469` to `0.9754` of the observed SGLang-vs-xorl output delta, with remaining RMS
  `0.00115` to `0.00129`. A row-only patch explains less (`0.7538` to `0.8019` projection,
  remaining RMS `0.00228` to `0.00290`). The unforced full-prefix router has full expert-set overlap
  with captured route in all four rows (`8/8`), with `regathered_weight_l1_vs_captured` from
  `0.01114` to `0.01581`.

  Upstream follow-up captured xorl `output_hidden_states` indices `34-38` on the same row0 fixed
  trace with `--xorl-replay-routing`, so the xorl forward reused the trace's SGLang
  `routed_experts`. Artifact:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer34_38_routealias_xorl_routeforced_hidden_boundary_20260626T221755Z.json`.
  The route-forced hidden dump is
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/xorl_fullprefix_hidden_q36-row0-xorl-routeforced-hidden34-38-20260626T221755Z.rank0.pt`;
  it exactly matches the previous fullprefix hidden dump for indices `36-38` (`routeforced_vs_previous`
  target-row RMS and max are both `0.0`). A no-route repeat is not interpretable for boundary
  localization: it differs from the route-forced hidden dump by target-row RMS `0.0217`, `0.0247`,
  and `0.0295` at indices `36`, `37`, and `38`.

  The route-forced boundary table shows the residual gap is already present before layer34. Across all
  128 target rows, xorl hidden index `L` vs SGLang layer `L` residual input has RMS
  `0.0035306`, `0.0036916`, `0.0047933`, `0.0054385`, and `0.0064657` for layers `34-38`.
  Output gaps for layers `34-37` are `0.0037013`, `0.0048079`, `0.0054498`, and `0.0064789`.
  On the two bad rows, input gaps grow similarly: row2140 goes `0.0030398 -> 0.0030905 ->
  0.0042024 -> 0.0048407 -> 0.0061140`; row2171 goes `0.0039400 -> 0.0041491 ->
  0.0047737 -> 0.0054415 -> 0.0065045`. So layer34/35 contribute additional drift, but they
  are not the source; the next boundary probe needs layers before 34.

  That earlier-layer coarse probe is now complete. SGLang routevalidated parent capture:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_coarse0_34_artifact_q36-row0-sglang-coarse0-34-20260626T222447Z.json`.
  Matching xorl route-forced hidden capture:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/xorl_fullprefix_hidden_q36-row0-xorl-routeforced-coarse0-34-20260626T223019Z.rank0.pt`.
  Joined boundary artifact:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer0_34_routealias_xorl_routeforced_coarse_boundary_20260626T223019Z.json`.
  The SGLang capture has `response_route_alignment.matches=true`,
  `dump_route_alignment.matches=true`, and `0/10240` route mismatches for layers
  `0,4,8,12,16,20,24,28,32,34`; returned prefill logprobs still match the fixed trace at pos93
  `-1.5019326210` and pos124 `-2.2427401543`. The xorl coarse hidden dump exactly reproduces the
  previous route-forced layer34/35 tensors (`target_rows` RMS and max both `0.0`).

  Coarse result: embeddings/layer0 input match exactly (`0.0` RMS), then the target-row residual
  input gap grows monotonically: layer0 `0.0`, layer4 `0.0002433`, layer8 `0.0003781`,
  layer12 `0.0005335`, layer16 `0.0007055`, layer20 `0.0010474`, layer24 `0.0012944`,
  layer28 `0.0016828`, layer32 `0.0027861`, layer34 `0.0035306`. Layer output gaps follow the
  same shape: layer0 `0.0000900`, layer4 `0.0002722`, layer8 `0.0003953`, layer12 `0.0005431`,
  layer16 `0.0007603`, layer20 `0.0010518`, layer24 `0.0013116`, layer28 `0.0017443`,
  layer32 `0.0033328`, layer34 `0.0037013`. The largest coarse target-row input jump is between
  layer28 and layer32 (`+0.0011033` RMS); layer32 to layer34 adds another `+0.0007445`. On the bad
  rows, layer28 input is already nonzero (`0.0015064` at row2140 and `0.0016600` at row2171), then
  layer32 rises to `0.0023185`/`0.0028701` and layer34 to `0.0030398`/`0.0039400`.

  The consecutive narrow layer28-32 probe is now complete. SGLang routevalidated parent capture:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_narrow28_32_artifact_q36-row0-sglang-narrow28-32-20260626T223526Z.json`.
  Matching xorl route-forced hidden capture:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/xorl_fullprefix_hidden_q36-row0-xorl-routeforced-narrow28-32-20260626T224037Z.rank0.pt`.
  Joined boundary artifact:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer28_32_routealias_xorl_routeforced_narrow_boundary_20260626T224037Z.json`.
  The SGLang capture has `response_route_alignment.matches=true`,
  `dump_route_alignment.matches=true`, and `0/5120` route mismatches for layers `28-32`. TP0 and
  TP1 dumps are identical for `layer_input`, `forward_input.residual`, and `layer_output` on all
  five captured layers. The matching xorl run wrote
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/k3_result_q36-row0-xorl-routeforced-narrow28-32-20260626T224037Z.json`
  with one-trace mean K3 `0.0258559484`, p95 `0.0097971739`, max `1.9126869478`, and the same
  worst fixed-prefill row0 token at pos124.

  Narrow result: target-row residual input gaps are layer28 `0.0016895`, layer29 `0.0017443`,
  layer30 `0.0018183`, layer31 `0.0023010`, and layer32 `0.0027961` RMS. Layer output gaps are
  layer28 `0.0017443`, layer29 `0.0018183`, layer30 `0.0023010`, layer31 `0.0027961`, and
  layer32 `0.0033328`. So the coarse layer28->32 jump is not a single layer28/29 event; most of
  it arrives in the late subintervals, with input-gap growth `+0.0004827` at `30->31` and
  `+0.0004952` at `31->32`, and the largest output-gap growth `+0.0005366` at `31->32`. On the bad
  rows, layer32 input is `0.0022998` at row2140 and `0.0028732` at row2171; layer32 output rises to
  `0.0029439` and `0.0037002`. Overlapping route-forced hidden indices `28`, `29`, `32`, and `33`
  reproduce the coarse capture exactly (`target_rows` RMS and max both `0.0`).

  The layer30-32 component probe is also complete. The normalized SGLang component tensor is
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_component_tensor_layers30_32_q36-row0-sglang-narrow28-32-20260626T223526Z.pt`.
  The matching xorl route-forced component dump is
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/xorl_component_tensor_q36-row0-xorl-routeforced-components30-32-20260626T224810Z.rank0.pt`,
  and its K3 replay result is
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/k3_result_q36-row0-xorl-routeforced-components30-32-20260626T224810Z.json`
  with the same one-trace K3 as the narrow boundary run: mean `0.025856`, p95 `0.009797`, max
  `1.912687`, worst fixed-prefill row0 token at pos124.

  Component artifacts:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer30_32_component_tensor_compare_20260626T224810Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer30_32_residual_source_terms_20260626T224810Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer30_32_component_target_summary_20260626T224810Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer30_32_input_sensitivity_20260626T224810Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer30_32_gdn_input_sensitivity_tp2_20260626T224810Z.json`,
  and
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer31_full_attention_input_sensitivity_20260626T224810Z.json`.
  The component join has `36/36` common keys, no missing components, and no shape mismatches.

  Component result: the layer output gap is dominated by inherited residual drift, not by a bad
  residual add closure. For target rows, captured `layer_output` RMS is layer30 `0.0023010`,
  layer31 `0.0027961`, and layer32 `0.0033328`; replacing the local block output with zero keeps
  the inherited-only gap at `0.0022048`, `0.0026485`, and `0.0031284`. The local-only output term is
  smaller but real: `0.0009611`, `0.0012017`, and `0.0014614`. Within that local term, attention is
  the larger contributor (`0.0013853`, `0.0014656`, `0.0018604` RMS) and MLP is smaller
  (`0.0009244`, `0.0011664`, `0.0014266` RMS). Candidate residual-add closures are exact `0.0`;
  the small SGLang layer-output closure is bf16-scale (`0.0002116`, `0.0002127`, `0.0002517` RMS).

  The norm tensors amplify residual drift, but they are not themselves final layer-output terms.
  Target-row mean RMS for `input_norm` is `0.0179`, `0.0159`, `0.0218` across layers `30-32`;
  `post_attention_norm`/`shared_expert_input` rises to `0.0307`, `0.0352`, `0.0398`. The weighted
  shared-expert term stays small by comparison: mean output RMS `0.000430`, `0.000562`, `0.000576`
  in the input-sensitivity artifact. The raw shared-expert tensor has some large max entries, but
  the gate-scaled `shared_expert_weighted` path is not the main residual-growth source.

  Same-input attention replays largely account for the local attention paths. Layers `30` and `32`
  are `linear_attention`; the GDN replay reports captured-attention mean RMS `0.0012914` and
  `0.0017991`, while xorl recomputed final-minus-captured is only `0.0002622`/`0.0002931` mean RMS
  on target rows. The raw SGLang TP rank dump sums exactly to the normalized SGLang captured
  attention (`dump_out_proj_sum_minus_reference_captured_attention_max_abs=0.0`); TP2
  recompute-vs-dump max is `0.0078125`/`0.015625`. Layer `31` is `full_attention`, not GDN; its
  captured-attention mean RMS is `0.0013295`, and the full-attention replay's computed
  final-minus-captured mean RMS is `0.0006700` on target rows. So the remaining layer30-32 evidence
  points to upstream residual/input-norm drift feeding attention blocks that mostly replay from the
  captured inputs, not to MLP, residual-add, routing, or GDN out-projection closure as the primary
  issue.

  The earlier layer28-30 component probe is now complete as well. It reused the same routevalidated
  SGLang narrow dump for layers `28-32`; normalized SGLang tensor:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_component_tensor_layers28_30_q36-row0-sglang-narrow28-32-20260626T223526Z.pt`.
  Matching xorl route-forced component dump:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/xorl_component_tensor_q36-row0-xorl-routeforced-components28-30-20260626T230036Z.rank0.pt`.
  K3 result:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/k3_result_q36-row0-xorl-routeforced-components28-30-20260626T230036Z.json`,
  again with mean `0.0258559484`, p95 `0.0097971739`, max `1.9126869478`, and the same worst
  fixed-prefill row0 pos124 token.

  Layer28-30 component artifacts:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer28_30_component_tensor_compare_20260626T230036Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer28_30_residual_source_terms_20260626T230036Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer28_30_input_sensitivity_20260626T230036Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer28_30_component_target_summary_20260626T230036Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer28_30_gdn_input_sensitivity_tp2_20260626T230036Z.json`,
  and
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer28_30_full_attention_input_sensitivity_20260626T230036Z.json`.
  The join again has `36/36` common component keys, no missing components, and no shape mismatches.

  Layer28-30 result: residual drift is already present at layer28 input. Target-row `layer_input`
  RMS is layer28 `0.0016895`, layer29 `0.0017443`, layer30 `0.0018183`; target-row `layer_output`
  RMS is layer28 `0.0017443`, layer29 `0.0018183`, layer30 `0.0023010`. So layer28/29 mostly pass
  the inherited drift along, while layer30 is the first notable local jump in this window and becomes
  the layer31 input jump seen above. Component RMS for attention is `0.0007215`, `0.0009148`,
  `0.0013853`; MLP is `0.0007003`, `0.0007502`, `0.0009244`; weighted shared expert remains smaller
  at `0.0003649`, `0.0003535`, `0.0005196`. The layer-output source decomposition is consistent:
  inherited-only RMS is `0.0016058`, `0.0016762`, `0.0021382`; local-only RMS is `0.0006988`,
  `0.0007284`, `0.0008906`.

  Layers `28`, `29`, and `30` are all `linear_attention`; the full-attention replay artifact records
  all three as unsupported for full attention. The GDN same-input replay covers all three and mostly
  reproduces the captured attention deltas: captured-attention mean RMS is `0.0006871`, `0.0008801`,
  `0.0012914`, while xorl recomputed final-minus-captured is only `0.0001630`, `0.0002693`,
  `0.0002622` mean RMS. The raw SGLang TP rank dump again sums exactly to normalized SGLang captured
  attention (`dump_out_proj_sum_minus_reference_captured_attention_max_abs=0.0`) with TP2
  recompute-vs-dump max `0.015625`. This makes a pure GDN formula/out-projection bug unlikely for
  this window; the remaining unexplained residual is already entering layer28, with layer30 adding a
  smaller but visible GDN response to that drift.

  The layer24-28 routevalidated component probe is now complete. SGLang narrow artifact:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_narrow24_28_artifact_q36-row0-sglang-narrow24-28-20260626T231044Z.json`.
  Its response and dump route alignment both match the fixed trace, with `0/5120` mismatches across
  layers `24-28`. Normalized SGLang tensor:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_component_tensor_layers24_28_q36-row0-sglang-narrow24-28-20260626T231044Z.pt`.
  Matching xorl route-forced component dump:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/xorl_component_tensor_q36-row0-xorl-routeforced-components24-28-20260626T231602Z.rank0.pt`.
  K3 result:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/k3_result_q36-row0-xorl-routeforced-components24-28-20260626T231602Z.json`,
  again with mean `0.0258559484`, p95 `0.0097971739`, max `1.9126869478`, and the same 5/128
  selection-flip tokens.

  Layer24-28 component artifacts:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer24_28_component_tensor_compare_20260626T231602Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer24_28_residual_source_terms_20260626T231602Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer24_28_input_sensitivity_20260626T231602Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer24_28_component_target_summary_20260626T231602Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer24_28_gdn_input_sensitivity_tp2_20260626T231602Z.json`,
  and
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer24_28_full_attention_input_sensitivity_20260626T231602Z.json`.
  The join has `60/60` common component keys, no missing components, and no shape mismatches.

  Layer24-28 result: residual drift is already present at layer24 input. Target-row `layer_input`
  RMS is layer24 `0.0013015`, layer25 `0.0013116`, layer26 `0.0014020`, layer27 `0.0013962`, and
  layer28 `0.0016895`; target-row `layer_output` RMS is `0.0013116`, `0.0014020`, `0.0013962`,
  `0.0016895`, and `0.0017443`. Layers24-26 mostly carry the inherited stream, layer27 is the local
  step that becomes the layer28 input jump, and layer28 then mostly carries that drift forward with a
  smaller GDN response. Component RMS for attention is `0.0004747`, `0.0004352`, `0.0005207`,
  `0.0009538`, `0.0007215`; MLP is `0.0004832`, `0.0005472`, `0.0005675`, `0.0007419`,
  `0.0007003`; weighted shared expert stays smaller at `0.0002074`, `0.0002436`, `0.0003165`,
  `0.0004061`, `0.0003649`.

  The source decomposition again points away from a residual-add bug. Layer-output inherited-only RMS
  is `0.0012192`, `0.0012875`, `0.0013018`, `0.0015554`, `0.0016058`; local-only RMS is
  `0.0004959`, `0.0005535`, `0.0005582`, `0.0007147`, `0.0006988`. Candidate residual closures are
  exact at bf16 pairwise precision for attention-residual and layer-output equations, and the SGLang
  layer-output closure remains small (`<=0.0001388` target-row RMS). This keeps the residual-flow
  diagnosis on inherited upstream drift plus local attention response, not on route replay, MLP, or
  residual-add implementation.

  Attention replays split by layer type. Layers `24`, `25`, `26`, and `28` are `linear_attention`;
  GDN captured-attention mean RMS is `0.0004517`, `0.0004097`, `0.0004942`, and `0.0006871`, while
  xorl TP2 recomputed final-minus-captured is only `0.0001242`, `0.0001044`, `0.0001330`, and
  `0.0001774` mean RMS. The raw SGLang TP rank dump again sums exactly to normalized SGLang captured
  attention (`dump_out_proj_sum_minus_reference_captured_attention_max_abs=0.0`) with TP2
  recompute-vs-dump max `0.015625`. Layer `27` is `full_attention`; its captured-attention mean RMS
  is `0.0008962`, and the same-input full-attention replay's computed final-minus-captured mean RMS is
  `0.0004275`. That leaves layer27 as the main local response inside 24-28, but still as a response to
  upstream input-norm/residual drift that is already visible before layer24.

  The layer20-24 routevalidated component probe is also complete. SGLang narrow artifact:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_narrow20_24_artifact_q36-row0-sglang-narrow20-24-20260626T232924Z.json`.
  Its response and dump route alignment both match the fixed trace, again with `0/5120` mismatches
  across layers `20-24`. Normalized SGLang tensor:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_component_tensor_layers20_24_q36-row0-sglang-narrow20-24-20260626T232924Z.pt`.
  Matching xorl route-forced component dump:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/xorl_component_tensor_q36-row0-xorl-routeforced-components20-24-20260626T233214Z.rank0.pt`.
  K3 result:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/k3_result_q36-row0-xorl-routeforced-components20-24-20260626T233214Z.json`,
  with the same one-trace values: mean `0.0258559484`, p95 `0.0097971739`, max
  `1.9126869478`, and 5/128 selection-flip tokens.

  Layer20-24 component artifacts:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer20_24_component_tensor_compare_20260626T233214Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer20_24_residual_source_terms_20260626T233214Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer20_24_input_sensitivity_20260626T233214Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer20_24_component_target_summary_20260626T233214Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer20_24_gdn_input_sensitivity_tp2_20260626T233214Z.json`,
  and
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer20_24_full_attention_input_sensitivity_20260626T233214Z.json`.
  The join has `60/60` common component keys, no missing components, and no shape mismatches.

  Layer20-24 result: residual drift is already present at layer20 input. Target-row `layer_input`
  RMS is layer20 `0.0010539`, layer21 `0.0010518`, layer22 `0.0010432`, layer23 `0.0010813`, and
  layer24 `0.0013015`; target-row `layer_output` RMS is `0.0010518`, `0.0010432`, `0.0010813`,
  `0.0013015`, and `0.0013116`. Layers20-22 mostly carry a roughly flat inherited residual. Layer23
  is the visible local step that becomes the layer24 input jump; layer24 mostly carries it forward.
  Component RMS for attention is `0.0004121`, `0.0004044`, `0.0004117`, `0.0006262`, `0.0004747`;
  MLP is `0.0004340`, `0.0003893`, `0.0004478`, `0.0005865`, `0.0004832`; weighted shared expert
  remains smaller at `0.0002180`, `0.0002065`, `0.0002239`, `0.0002740`, `0.0002074`.

  The layer-output source decomposition repeats the inherited-drift pattern. Inherited-only RMS is
  `0.0010081`, `0.0009991`, `0.0010086`, `0.0011906`, `0.0012192`; local-only RMS is `0.0004380`,
  `0.0004027`, `0.0004629`, `0.0005850`, `0.0004959`. Candidate layer-output closures are exact
  at bf16 pairwise precision; SGLang layer-output closure remains small (`<=0.0001215` target-row
  RMS). So the upstream edge is not introduced by the residual-add equation inside layers20-24.

  Attention replays again split by layer type. Layers `20`, `21`, `22`, and `24` are
  `linear_attention`; GDN captured-attention mean RMS is `0.0003922`, `0.0003849`, `0.0003948`, and
  `0.0004517`, while xorl TP2 recomputed final-minus-captured is `0.0001323`, `0.0001175`,
  `0.0001311`, and `0.0001242` mean RMS. The raw SGLang TP rank dump sums exactly to normalized
  captured attention (`dump_out_proj_sum_minus_reference_captured_attention_max_abs=0.0`) with TP2
  recompute-vs-dump max `0.0078125`. Layer `23` is `full_attention`; its captured-attention mean RMS
  is `0.0005759`, and same-input full-attention replay's computed final-minus-captured mean RMS is
  `0.0003486`. This makes layer23 the main local response inside 20-24, but the residual drift is
  already nonzero entering layer20.

  The layer16-20 routevalidated component probe is complete as well. SGLang narrow artifact:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_narrow16_20_artifact_q36-row0-sglang-narrow16-20-20260626T233902Z.json`.
  Response and dump route alignment both match the fixed trace, with `0/5120` mismatches across
  layers `16-20`. Normalized SGLang tensor:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_component_tensor_layers16_20_q36-row0-sglang-narrow16-20-20260626T233902Z.pt`.
  Matching xorl route-forced component dump:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/xorl_component_tensor_q36-row0-xorl-routeforced-components16-20-20260626T234503Z.rank0.pt`.
  K3 result:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/k3_result_q36-row0-xorl-routeforced-components16-20-20260626T234503Z.json`,
  still mean `0.0258559484`, p95 `0.0097971739`, max `1.9126869478`, and 5/128 selection flips.

  Layer16-20 component artifacts:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer16_20_component_tensor_compare_20260626T234503Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer16_20_residual_source_terms_20260626T234503Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer16_20_input_sensitivity_20260626T234503Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer16_20_component_target_summary_20260626T234503Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer16_20_gdn_input_sensitivity_tp2_20260626T234503Z.json`,
  and
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer16_20_full_attention_input_sensitivity_20260626T234503Z.json`.
  The join has `60/60` common component keys, no missing components, and no shape mismatches.

  Layer16-20 result: residual drift is already present at layer16 input. Target-row `layer_input`
  RMS is layer16 `0.0007107`, layer17 `0.0007603`, layer18 `0.0007922`, layer19 `0.0008639`, and
  layer20 `0.0010539`; target-row `layer_output` RMS is `0.0007603`, `0.0007922`, `0.0008639`,
  `0.0010539`, and `0.0010518`. Layers16-18 show gradual carried growth. Layer19 is the local step
  that becomes the layer20 input jump; layer20 then mostly carries it forward. Component RMS for
  attention is `0.0002989`, `0.0003388`, `0.0003030`, `0.0005409`, `0.0004121`; MLP is
  `0.0003567`, `0.0003448`, `0.0003942`, `0.0004782`, `0.0004340`; weighted shared expert remains
  smaller at `0.0001618`, `0.0001424`, `0.0001932`, `0.0002224`, `0.0002180`.

  Layer-output source decomposition remains inherited-drift dominated. Inherited-only RMS is
  `0.0006910`, `0.0007280`, `0.0007890`, `0.0009686`, `0.0010081`; local-only RMS is `0.0003661`,
  `0.0003547`, `0.0004094`, `0.0004916`, `0.0004380`. Candidate layer-output closures are again
  exact at bf16 pairwise precision; SGLang layer-output closure stays small (`<=0.0001091` target-row
  RMS). The residual discrepancy is therefore already upstream of layer16, not introduced by the
  residual-add equation in layers16-20.

  Attention replay results match the hybrid layer pattern. Layers `16`, `17`, `18`, and `20` are
  `linear_attention`; GDN captured-attention mean RMS is `0.0002889`, `0.0003248`, `0.0002898`, and
  `0.0003922`, while xorl TP2 recomputed final-minus-captured is `0.0001011`, `0.0001415`,
  `0.0001151`, and `0.0001323` mean RMS. The raw SGLang TP rank dump sums exactly to normalized
  captured attention (`dump_out_proj_sum_minus_reference_captured_attention_max_abs=0.0`) with TP2
  recompute-vs-dump max `0.0078125`. Layer `19` is `full_attention`; its captured-attention mean RMS
  is `0.0005074`, and same-input full-attention replay's computed final-minus-captured mean RMS is
  `0.0003468`. Layer19 is the main local response inside 16-20, but the inherited stream is already
  nonzero entering layer16.

  The layer12-16 routevalidated component probe is now complete. SGLang narrow artifact:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_narrow12_16_artifact_q36-row0-sglang-narrow12-16-20260626T235342Z.json`.
  Response and dump route alignment both match the fixed trace, again with `0/5120` mismatches
  across layers `12-16`. Normalized SGLang tensor:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_component_tensor_layers12_16_q36-row0-sglang-narrow12-16-20260626T235342Z.pt`.
  Matching xorl route-forced component dump:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/xorl_component_tensor_q36-row0-xorl-routeforced-components12-16-20260626T235823Z.rank0.pt`.
  K3 result:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/k3_result_q36-row0-xorl-routeforced-components12-16-20260626T235823Z.json`,
  still mean `0.0258559484`, p95 `0.0097971739`, max `1.9126869478`, and 5/128 selection flips.

  Layer12-16 component artifacts:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer12_16_component_tensor_compare_20260626T235823Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer12_16_residual_source_terms_20260626T235823Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer12_16_input_sensitivity_20260626T235823Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer12_16_component_target_summary_20260626T235823Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer12_16_gdn_input_sensitivity_tp2_20260626T235823Z.json`,
  and
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer12_16_full_attention_input_sensitivity_20260626T235823Z.json`.
  The join has `60/60` common component keys after filtering SGLang-only `raw_mlp`, no missing
  shared components, and no shape mismatches.

  Layer12-16 result: residual drift is already present at layer12 input. Target-row `layer_input`
  RMS is layer12 `0.0005297`, layer13 `0.0005335`, layer14 `0.0005779`, layer15 `0.0005818`, and
  layer16 `0.0006998`; target-row `layer_output` RMS is `0.0005335`, `0.0005779`, `0.0005818`,
  `0.0006998`, and `0.0007481`. Layers12-14 mostly carry a smaller inherited residual. Layer15 is
  the local full-attention step into layer16; layer16 mostly carries that forward. Component RMS for
  attention is `0.0002564`, `0.0002081`, `0.0002486`, `0.0003472`, `0.0002889`; MLP is
  `0.0002252`, `0.0002573`, `0.0002480`, `0.0003337`, `0.0003444`; weighted shared expert remains
  smaller at `0.0000945`, `0.0001011`, `0.0001294`, `0.0001383`, `0.0001488`.

  Layer-output source decomposition remains inherited-drift dominated. Inherited-only RMS is
  `0.0005014`, `0.0005360`, `0.0005460`, `0.0006422`, `0.0006910`; local-only RMS is `0.0002487`,
  `0.0002822`, `0.0002805`, `0.0003537`, `0.0003661`. Candidate layer-output closures are exact
  at bf16 pairwise precision; SGLang layer-output closure stays small (`<=0.0000944` target-row
  RMS). The residual discrepancy is therefore already upstream of layer12, not introduced by the
  residual-add equation in layers12-16.

  Attention replay results again match the hybrid layer pattern. Layers `12`, `13`, `14`, and `16`
  are `linear_attention`; GDN captured-attention mean RMS is `0.0002564`, `0.0002081`,
  `0.0002486`, and `0.0002889`, while xorl TP2 recomputed final-minus-captured is `0.0001101`,
  `0.0000811`, `0.0000993`, and `0.0001011` mean RMS. The raw SGLang TP rank dump sums exactly to
  normalized captured attention (`dump_out_proj_sum_minus_reference_captured_attention_max_abs=0.0`)
  with TP2 recompute-vs-dump max `0.005859375`. Layer `15` is `full_attention`; its
  captured-attention mean RMS is `0.0003472`, and same-input full-attention replay's computed
  final-minus-captured mean RMS is `0.0002452`. Layer15 is the main local response inside 12-16, but
  the inherited stream is already nonzero entering layer12.

  The layer8-12 routevalidated component probe is now complete. SGLang narrow artifact:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_narrow8_12_artifact_q36-row0-sglang-narrow8-12-20260627T000831Z.json`.
  Response and dump route alignment both match the fixed trace, again with `0/5120` mismatches
  across layers `8-12`. Normalized SGLang tensor:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_component_tensor_layers8_12_q36-row0-sglang-narrow8-12-20260627T000831Z.pt`.
  Matching xorl route-forced component dump:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/xorl_component_tensor_q36-row0-xorl-routeforced-components8-12-20260627T001316Z.rank0.pt`.
  K3 result:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/k3_result_q36-row0-xorl-routeforced-components8-12-20260627T001316Z.json`,
  still mean `0.0258559484`, p95 `0.0097971739`, max `1.9126869478`, and 5/128 selection flips.

  Layer8-12 component artifacts:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer8_12_component_tensor_compare_20260627T001316Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer8_12_residual_source_terms_20260627T001316Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer8_12_input_sensitivity_20260627T001316Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer8_12_component_target_summary_20260627T001316Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer8_12_gdn_input_sensitivity_tp2_20260627T001316Z.json`,
  and
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer8_12_full_attention_input_sensitivity_20260627T001316Z.json`.
  The join has `60/60` common component keys after filtering SGLang-only `raw_mlp`, no missing
  shared components, and no shape mismatches.

  Layer8-12 result: residual drift is already present at layer8 input. Target-row `layer_input`
  RMS is layer8 `0.0003722`, layer9 `0.0003873`, layer10 `0.0004059`, layer11 `0.0004370`, and
  layer12 `0.0005297`; target-row `layer_output` RMS is `0.0003873`, `0.0004059`, `0.0004370`,
  `0.0005297`, and `0.0005335`. Layers8-10 show gradual carried growth. Layer11 is the local
  full-attention step into layer12; layer12 mostly carries that forward. Component RMS for
  attention is `0.0001703`, `0.0001659`, `0.0001935`, `0.0002446`, `0.0002564`; MLP is
  `0.0001707`, `0.0001809`, `0.0001970`, `0.0002679`, `0.0002252`; weighted shared expert remains
  smaller at `0.0000725`, `0.0000805`, `0.0000956`, `0.0001215`, `0.0000945`.

  Layer-output source decomposition remains inherited-drift dominated. Inherited-only RMS is
  `0.0003658`, `0.0003828`, `0.0004133`, `0.0004895`, `0.0005014`; local-only RMS is `0.0001906`,
  `0.0001988`, `0.0002247`, `0.0002866`, `0.0002487`. Candidate layer-output closures are exact
  at bf16 pairwise precision; SGLang layer-output closure stays small (`<=0.0000788` target-row
  RMS). The residual discrepancy is therefore already upstream of layer8, not introduced by the
  residual-add equation in layers8-12.

  Attention replay results again match the hybrid layer pattern. Layers `8`, `9`, `10`, and `12`
  are `linear_attention`; GDN captured-attention mean RMS is `0.0001703`, `0.0001659`,
  `0.0001935`, and `0.0002564`, while xorl TP2 recomputed final-minus-captured is `0.0000870`,
  `0.0000721`, `0.0000901`, and `0.0001101` mean RMS. Layer `11` is `full_attention`; its
  captured-attention mean RMS is `0.0002446`, and same-input full-attention replay's computed
  final-minus-captured mean RMS is `0.0001825`. Layer11 is the main local response inside 8-12, but
  the inherited stream is already nonzero entering layer8.

  The layer4-8 routevalidated component probe is now complete. SGLang narrow artifact:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_narrow4_8_artifact_q36-row0-sglang-narrow4-8-20260627T002233Z.json`.
  Response and dump route alignment both match the fixed trace, again with `0/5120` mismatches
  across layers `4-8`. Corrected TP-merged normalized SGLang tensor:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_component_tensor_layers4_8_tpmerged_q36-row0-sglang-narrow4-8-20260627T002233Z.pt`.
  This supersedes the earlier TP0-only normalized 4-8 tensor; the compact summary records
  `reference_rank_glob=TP*_*` and `reference_rank_dump_file_count=2`. Matching xorl route-forced
  component dump:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/xorl_component_tensor_q36-row0-xorl-routeforced-components4-8-20260627T002838Z.rank0.pt`.
  K3 result:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/k3_result_q36-row0-xorl-routeforced-components4-8-20260627T002838Z.json`,
  still mean `0.0258559484`, p95 `0.0097971739`, max `1.9126869478`, and 5/128 selection flips.

  Layer4-8 component artifacts:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer4_8_component_tensor_compare_20260627T002838Z_tpmerged.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer4_8_residual_source_terms_20260627T002838Z_tpmerged.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer4_8_input_sensitivity_20260627T002838Z_tpmerged.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer4_8_component_target_summary_20260627T002838Z_tpmerged.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer4_8_gdn_input_sensitivity_tp2_20260627T002838Z_tpmerged.json`,
  and
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer4_8_full_attention_input_sensitivity_20260627T002838Z_tpmerged.json`.
  The join has `60/60` common component keys after filtering SGLang-only `raw_mlp`, no missing
  shared components, no shape mismatches, and the corrected reference is built from both TP ranks.

  Layer4-8 result: residual drift is already present at layer4 input. Target-row `layer_input`
  RMS is layer4 `0.0002319`, layer5 `0.0002601`, layer6 `0.0002914`, layer7 `0.0003151`, and
  layer8 `0.0003722`; target-row `layer_output` RMS is `0.0002601`, `0.0002914`, `0.0003151`,
  `0.0003722`, and `0.0003873`. Layers4-6 mostly carry gradual inherited growth. Layer7 is the
  local full-attention step into layer8; layer8 mostly carries that forward. Component RMS for
  attention is `0.0001113`, `0.0001417`, `0.0001500`, `0.0001691`, `0.0001703`; MLP is
  `0.0001087`, `0.0001222`, `0.0001340`, `0.0001671`, `0.0001707`; weighted shared expert remains
  smaller at `0.0000435`, `0.0000481`, `0.0000571`, `0.0000710`, `0.0000725`.

  Layer-output source decomposition remains inherited-drift dominated. Inherited-only RMS is
  `0.0002392`, `0.0002731`, `0.0002939`, `0.0003466`, `0.0003658`; local-only RMS is `0.0001286`,
  `0.0001388`, `0.0001566`, `0.0001868`, `0.0001906`. Candidate layer-output closures are exact
  at bf16 pairwise precision; SGLang layer-output closure stays small (`<=0.0000553` target-row
  RMS). The residual discrepancy is therefore already upstream of layer4, not introduced by the
  residual-add equation in layers4-8.

  Attention replay results again match the hybrid layer pattern. Layers `4`, `5`, `6`, and `8`
  are `linear_attention`; GDN captured-attention mean RMS is `0.0001113`, `0.0001417`,
  `0.0001500`, and `0.0001703`, while xorl TP2 recomputed final-minus-captured is `0.0000553`,
  `0.0000853`, `0.0000762`, and `0.0000870` mean RMS. The raw SGLang TP rank dump sums exactly to
  normalized captured attention (`dump_out_proj_sum_minus_reference_captured_attention_max_abs=0.0`)
  with TP2 recompute-vs-dump max `0.002685546875`. Layer `7` is `full_attention`; its
  captured-attention mean RMS is `0.0001691`, and same-input full-attention replay's computed
  final-minus-captured mean RMS is `0.0001141`. Layer7 is the main local response inside 4-8, but
  the inherited stream is already nonzero entering layer4.

  The layer0-4 routevalidated component probe is now complete. SGLang narrow artifact:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_narrow0_4_artifact_q36-row0-sglang-narrow0-4-20260627T010424Z.json`.
  Response and dump route alignment both match the fixed trace, again with `0/5120` mismatches
  across layers `0-4`. TP-merged normalized SGLang tensor:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_component_tensor_layers0_4_tpmerged_q36-row0-sglang-narrow0-4-20260627T010424Z.pt`.
  Matching xorl route-forced component dump:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/xorl_component_tensor_q36-row0-xorl-routeforced-components0-4-20260627T011009Z.rank0.pt`.
  K3 result:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/k3_result_q36-row0-xorl-routeforced-components0-4-20260627T011009Z.json`,
  still mean `0.0258559484`, p95 `0.0097971739`, max `1.9126869478`, and 5/128 selection flips.

  Layer0-4 component artifacts:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer0_4_component_tensor_compare_20260627T011009Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer0_4_residual_source_terms_20260627T011009Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer0_4_input_sensitivity_20260627T011009Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer0_4_component_target_summary_20260627T011009Z.json`,
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer0_4_gdn_input_sensitivity_tp2_20260627T011009Z.json`,
  and
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer0_4_full_attention_input_sensitivity_20260627T011009Z.json`.
  The join has `60/60` common component keys after filtering SGLang-only `raw_mlp`, no missing
  shared components, no shape mismatches, and the corrected reference is built from both TP ranks.

  Layer0-4 result: layer0 `layer_input` and `input_norm` are exactly identical (`0.0` target-row
  RMS), so the residual is not present before the transformer stack. The first nonzero target-row
  drift appears inside layer0: `attention=0.0000536`, `mlp=0.0000469`,
  `post_attention_residual=0.0000593`, and `layer_output=0.0000830`. Target-row `layer_input` RMS
  is layer0 `0.0`,
  layer1 `0.0000830`, layer2 `0.0001361`, layer3 `0.0001800`, and layer4 `0.0002319`; target-row
  `layer_output` RMS is `0.0000830`, `0.0001361`, `0.0001800`, `0.0002319`, and `0.0002601`.
  Layers1-2 carry gradual inherited growth, layer3 is the local full-attention step into layer4,
  and layer4 carries that forward.

  Layer-output source decomposition confirms the first-block source. Layer0 attention-residual
  inherited-only RMS is `0.0` while local-only RMS is `0.0000593`; layer0 layer-output inherited-only
  RMS is `0.0000623` and local-only RMS is `0.0000592`. For layers0-4, layer-output inherited-only
  RMS is `0.0000623`, `0.0001119`, `0.0001568`, `0.0002133`, `0.0002392`; local-only RMS is
  `0.0000592`, `0.0000831`, `0.0000995`, `0.0001146`, `0.0001286`. Candidate layer-output
  closures are exact at bf16 pairwise precision; SGLang layer-output closure stays small
  (`<=0.0000444` target-row RMS). The residual discrepancy is therefore introduced by local
  computation in layer0, not by embeddings, route replay, residual-add equations, or later inherited
  residual flow.

  Attention replay localizes the next debug target to layer0 GDN/captured-output closure. Layers
  `0`, `1`, `2`, and `4` are `linear_attention`; GDN captured-attention mean RMS is `0.0000536`,
  `0.0000672`, `0.0000918`, and `0.0001113`, while xorl TP2 recomputed final-minus-captured is
  `0.0000577`, `0.0000494`, `0.0000578`, and `0.0000553` mean RMS. The raw SGLang TP rank dump
  sums exactly to normalized captured attention (`dump_out_proj_sum_minus_reference_captured_attention_max_abs=0.0`)
  with TP2 recompute-vs-dump max `0.00390625`. Layer `3` is `full_attention`; its captured-attention
  mean RMS is `0.0000967`, and same-input full-attention replay's computed final-minus-captured mean
  RMS is `0.0000730`. Since layer0 input/input_norm are identical, the next probe should compare
  layer0 GDN formula/kernel closure against the captured SGLang TP-summed attention and XoRL
  captured attention at the exact row0 sequence length/input, not move to another residual-flow
  window.

  Follow-up layer0 GDN closure artifact:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer0_gdn_closure_summary_20260627T011009Z.json`.
  It recomputes layer0 from the captured SGLang `input_layernorm` and compares XoRL-style formula,
  SGLang-style formula, XoRL captured attention, and TP-summed SGLang captured attention over the
  same 128 target rows. The target-row input is exact (`xorl_input_norm_vs_sglang_input_norm=
  0.0`). XoRL-style and SGLang-style formulas are close (`mean_row_rms=0.0000172`, max row
  `0.0000870`), and XoRL-style formula vs XoRL captured attention is also close
  (`mean_row_rms=0.0000143`, max row `0.0000611`). The SGLang-style formula vs TP-summed SGLang
  captured attention is much closer to the observed engine gap (`mean_row_rms=0.0000543`, max row
  `0.0001156`), and XoRL captured vs SGLang captured is `mean_row_rms=0.0000536`, max row
  `0.0001134`. This shifts the immediate target from generic GDN parity to SGLang runtime
  captured-attention closure: the formulas mostly agree from the same input, but the actual
  SGLang TP-summed layer0 `linear_attn` dump is offset by the same scale as the engine gap. The raw
  rank dump still sums exactly to the normalized captured attention, so the next probe should inspect
  the SGLang runtime linear-attention internals/TP out-proj partials against the parity formula, not
  another residual or MLP/routing window.

  SGLang rank-local closure now isolates that runtime difference to the conv+gating variant, not
  projection, norm, or out-proj. Base artifact:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer0_sglang_rank_local_closure_20260627T011009Z.json`.
  Projection closure is exact (`qkvz_projection_max_abs=0.0`, `ba_projection_max_abs=0.0`), captured
  norm to out-proj is exact (`norm_to_out_proj_max_abs=0.0`), and rank-local out-proj equals
  `linear_attn` (`out_proj_vs_linear_attn_max_abs=0.0`). With the diagnostic's PyTorch conv plus
  PyTorch gating recompute, `core_to_attn_max_abs=0.015625` and `computed_core_to_norm_max_abs=
  0.005859375`. With SGLang conv plus SGLang fused gating:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer0_sglang_rank_local_closure_sglangconv_sglanggating_none_20260627T011009Z.json`,
  `core_to_attn_max_abs=0.0` and `computed_core_to_norm_max_abs=0.0001220703125` (same with
  `core_cu_seqlens=single`). Axis splits do not close: SGLang conv + PyTorch gating still has
  `core_to_attn_max_abs=0.015625`, and PyTorch conv + SGLang gating also has
  `core_to_attn_max_abs=0.015625`. The next implementation experiment should therefore make XoRL's
  Qwen3.6 GDN layer use the same serving-compatible causal conv and fused beta/gating numerics as
  SGLang, then rerun the one-trace K3 and layer0 closure before touching later layers.

  That implementation experiment now exists behind a diagnostic flag in
  `/home/apanda/xorl-slime-parity-low-precision`: `linear_attention_sglang_forward_compatible`.
  It wires Qwen3.6 `GatedDeltaNet` through SGLang's packed causal-conv helper and fused GDN gating
  helper for forward-only prefill replay, and the K3 launcher now accepts
  `--reference-logprobs prefill` plus `--xorl-extra-pythonpath` so the XoRL pod can import the local
  SGLang helper modules without installing them into the venv. Candidate config:
  `/home/apanda/xorl-slime-parity-low-precision/experiments/k3_tests/configs/qwen3_6_35b_ep4_triton_alltoall_sglang_gdn_forward.yaml`.

  One-trace row0 smoke artifact:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/k3_result_q36-row0-xorl-sglang-gdn-forward-20260627T013528Z.json`.
  This is an improvement but not K3=0. Against SGLang prefill logprobs, the run reports mean K3
  `0.0016030`, median `6.39e-14`, p95 `0.0002771`, p99 `0.0509516`, max `0.1200362`, and
  selection flips `1/128` (`flip_rate=0.0078125`). Recomputing the old route-forced row0 artifact
  against the same prefill references gives mean K3 `0.0020874` and flips `2/128`; the candidate
  removes the old pos93 flip but leaves the pos124/token318 tail (`sglang_prefill=-2.2427402`,
  `xorl=-1.7092699`, K3 `0.1200362`) and worsens some non-flip tail positions.

  Layer0 selected-row component compare:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/row0_layer0_sglang_gdn_forward_candidate_component_compare_20260627T013528Z.json`.
  The four XoRL ranks are identical for the dumped rows. `input_norm` remains exact. Layer0
  attention improves but does not close: old-vs-SGLang mean/rms/max over rows
  `2047,2083,2145,2174` was `2.52e-05 / 4.39e-05 / 9.77e-04`; the new conv+fused-gating path is
  `1.13e-05 / 2.82e-05 / 4.88e-04`. Layer0 output improves only slightly:
  `4.42e-05 / 9.51e-05 / 0.00390625` old vs `3.49e-05 / 8.55e-05 / 0.00390625` new. The next
  probe should therefore inspect the remaining layer0 delta after the packed conv+fused-gating
  change, especially norm-gate/output-proj sensitivity at the surviving pos124 tail, before
  promoting this diagnostic path or moving to later layers.

  Updated conclusion: the biggest row0 `auto` K3 is still SGLang generation-vs-prefill mismatch.
  The smaller row0 prefill-reference tail is real, but the old residual-stream localization was based
  on a stale/non-route-aligned SGLang dump. The route-validated parent-boundary run now shows that
  layer36/37 local replay is internally consistent for both engines: SGLang replays its own captured
  boundary to bf16-scale error, xorl replays its own full-prefix hidden path to bf16-scale error, and
  the remaining SGLang-vs-xorl difference is already present in the residual stream entering layer34.
  The coarse+narrow/component grid now localizes the main fixed-prefill residual growth to inherited
  residual drift introduced inside layer0, with visible local full-attention responses at layers3,
  7, 11, 15, 19, 23, and 27, smaller layer28/layer30 GDN responses, then the larger consecutive
  layer31-32 output growth. The next probe should stop walking residual windows and instead target
  the remaining post-conv+fused-gating layer0 GDN delta on the exact fixed-prefill row0 input; broad
  sampler/numerics knob sweeps and MLP/routing-focused changes are lower priority. In parallel, run
  K3 gates with `--reference-logprobs prefill` when the question is training forward parity; use
  `auto` only when explicitly testing SGLang decode/generation consistency.

  FlashInfer decode wrote
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_row1_pos26_flashinfer_decode_probe.json`;
  logs showed `decode=FlashInferGDNKernel`, `extend=TritonGDNKernel`, and `packed_decode=False`.
  It did not reproduce the Triton continuation, but at pos26 it generated token `26003`, the same
  token that Triton prefill ranked first. FlashInfer decode scored that token `-0.8496439457`, while
  same-sequence prefill scored it `-0.3353516161` (`decode-minus-prefill=-0.5142923295`); a repeat
  decode matched exactly. CuTeDSL decode wrote
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/sglang_row1_pos26_cutedsl_decode_probe.json`;
  logs showed `decode=CuteDSLGDNKernel`, `extend=TritonGDNKernel`, and `packed_decode=False`.
  CuTeDSL produced a deterministic but very different continuation; at pos26 it generated token
  `148368` with decode logprob `-5.1969799995`, while same-sequence prefill scored that generated
  token `-0.0412824750` (`decode-minus-prefill=-5.1556975245`). This narrows the current row1
  failure further: Triton packed decode is the wrong-argmax path versus prefill, non-packed Triton
  and FlashInfer are closer on the argmax for this token but still not logprob-identical to prefill,
  and CuTeDSL is not a drop-in prefill-consistent replacement on this prompt.
- **Recurring row25/pos57/token15704 tail:** the same token is top-ranked across kernel/topology
  variants: FlashQLA layout-fix k3 `6.7085201718`, new EP4 Triton/alltoall k3 `1.6984313229`, and
  older FLA worst4 k3 `0.466981687`. This is a shared model/reference/routing-boundary hot token
  with a kernel-path multiplier; it is not only FlashQLA, not only DeepEP/quack, and not label
  alignment (`best_shift=0`).
- **Current best source for the old tail:** row25/pos57 in the saved artifact is primarily a
  SGLang generation-vs-prefill old-logprob inconsistency. For token `15704` (`"parameter"`),
  SGLang generation reported logprob `-0.3921356797`, while SGLang prefill/top-k scored the same
  target at `-2.5796377659` and xorl FlashQLA scored it at `-2.6959538460`; both SGLang prefill and
  xorl ranked token `1628` (`"function"`) first at about `-0.08`/`-0.07`. The EP4 Triton/alltoall
  replay still had xorl `-1.8061547279` against the same old generation reference. So the old
  training ratio tail is not "xorl picked an arbitrary bad distribution"; it is that the sampler's
  decode-time old_logprob was much more optimistic than both full-sequence scoring paths.
- **Current sampler probe changes that reference:** against live new-revision sampler
  `wordle-sci-opsd-sampler-b-7` with `--rl-on-policy-target xorl-batch-invariant` and
  `--enable-return-routed-experts`, the same row25 prefix at pos57 produced matching decode and
  prefill scores: target `15704` had logprob `-0.6976295114` in both paths, top-5 was
  `[1628:-0.6976295114, 15704:-0.6976295114, 1005:-6.1976294518, 48218:-7.6976294518,
  846:-7.8226294518]`, and the decode token was `1628`. This means the old static trace reference
  is stale for the full-parity recipe. The refreshed four-trace replay above is the current source
  of truth for the next xorl K3 conclusion.
- **Standalone GDN layer checks are not the smoking gun:** the existing FlashQLA artifacts show
  packed projection and packed conv match exactly, FlashQLA core vs FLA has max abs
  `0.00048828125` and mean abs `3.57e-7` at seq128, and the full layer comparison passes its
  thresholds. The remaining suspect is the full decode-state/logprob path under hybrid GDN/MoE
  generation, not the isolated GDN core kernel alone.
- **R3 exact/replay artifacts:** do not treat existing `r3_exact_*` results as proof that R3
  fixes or breaks parity globally. Some old artifacts used `sglang_*routed_experts` source fields
  and produced catastrophic k3. The currently green `compare_static_traces.py --replay-routing`
  path expects a plain per-trace `routed_experts` field.
- **TensorRT-LLM:** no existing xorl/K3 harness surfaced in the checkout. TensorRT-LLM has
  generic `SamplingParams(logprobs=..., prompt_logprobs=..., return_context_logits=...,
  return_generation_logits=...)` plumbing, but I found no routed-expert capture or batch-invariant
  K3 replay equivalent. Treat TensorRT-LLM as needing a new fixed-sequence scoring adapter before
  it can be compared on the same K3 axis.

### Next exact K3 commands

Use the complete harness checkout for static K3 work until the missing experiment scripts are
backported to `/home/apanda/xorl-apanda-dev`.

Offline diagnosis of the current FlashQLA failure vs the FLA baseline:

```bash
cd /home/apanda/xorl-slime-parity-low-precision/experiments/k3_tests
PYTHONPATH=. /home/apanda/xorl-internal/.venv/bin/python diagnose_static_k3.py \
  --k3-result /shared/opd-control/er-opd-q36-35b-slots/k3/amdahl087_apandadev_gqa_flashqla_full32_layoutfix_20260616T092310Z/k3_result.json \
  --traces-file /shared/opd-control/er-opd-q36-mtp-perf-replay/k3/sglanggen_trace_refresh_20260615T033726Z/q36-coderforge-pilot-opd32-sglanggen-static-traces-20260615T033726Z.json \
  --compare-k3-result /shared/opd-control/er-opd-q36-35b-slots/k3/amdahl077_quack_deepep_sms36_20260615T1028Z/k3_result.json \
  --top-n 5 \
  --output-json /tmp/flashqla_layoutfix_vs_fla_diagnosis_20260626.json
```

Static worst4 non-R3 replay, using a simpler xorl EP4 Triton/alltoall path:

```bash
cd /home/apanda/xorl-slime-parity-low-precision
RUN=q36-worst4-nonr3-$(date -u +%Y%m%dT%H%M%SZ)
K8S_NAMESPACE=apanda K8S_PVC_NAME=home-apanda \
  /home/apanda/xorl-internal/.venv/bin/python experiments/k3_tests/launch_k3_test.py \
  --model qwen3.6-35b \
  --xorl-config experiments/k3_tests/configs/qwen3_6_35b_ep4_triton_alltoall.yaml \
  --xorl-gpus 4 \
  --static-traces-file /shared/opd-control/er-opd-q36-35b-slots/k3/amdahl087_apandadev_gqa_flashqla_full32_layoutfix_20260616T092310Z/worst4_repro_traces.json \
  --num-prompts 4 \
  --xorl-venv /home/apanda/xorl-internal/.venv \
  --client-python /home/apanda/xorl-internal/.venv/bin/python \
  --xorl-node-selector-from-capacity \
  --gpu-capacity-wait-timeout-sec 0 \
  --pod-suffix "$RUN" \
  --log-dir "/tmp/$RUN" \
  --output-json "/tmp/$RUN.json" \
  --max-median-k3 1e-6 \
  --max-flip-rate 0.05 \
  --k3-flip-threshold 0.1
```

Verified on 2026-06-26 17:37 UTC as
`q36-worst4-nonr3-smoke-20260626T173722Z`: selected
`research-common-h100-089.cloud.together.ai`, replayed four static traces, wrote
`/tmp/q36-worst4-nonr3-smoke-20260626T173722Z.json`, and cleaned up the pod.

Offline diagnosis for that EP4 Triton/alltoall replay:

```bash
cd /home/apanda/xorl-slime-parity-low-precision/experiments/k3_tests
PYTHONPATH=. /home/apanda/xorl-internal/.venv/bin/python diagnose_static_k3.py \
  --k3-result /tmp/q36-worst4-nonr3-smoke-20260626T173722Z.json \
  --traces-file /shared/opd-control/er-opd-q36-35b-slots/k3/amdahl087_apandadev_gqa_flashqla_full32_layoutfix_20260616T092310Z/worst4_repro_traces.json \
  --compare-k3-result /shared/opd-control/er-opd-q36-35b-slots/k3/amdahl087_apandadev_gqa_flashqla_full32_layoutfix_20260616T092310Z/k3_result.json \
  --compare-k3-result /shared/opd-control/er-opd-q36-35b-slots/k3/amdahl087_apandadev_fla_worst4_20260616T082411Z/k3_result.json \
  --top-n 8 \
  --output-json /tmp/q36-worst4-nonr3-smoke-20260626T173722Z.diagnosis.json
```

That diagnosis confirmed `best_shift=0`, generation-reference mean k3 `0.0110965137`, prefill
reference mean k3 `0.0105212`, and SGLang self-consistency mean absolute logprob delta
`0.0422137932` on the worst4 subset.

Do **not** add `--xorl-replay-routing` to the old worst4 bundle above: that bundle has no plain
`routed_experts` field. For current R3 evidence, use the refreshed bundle that was regenerated
against `wordle-sci-opsd-sampler-b-7`:

```bash
cd /home/apanda/xorl-slime-parity-low-precision
RUN=q36-worst4-current-r3-replay-$(date -u +%Y%m%dT%H%M%SZ)
K8S_NAMESPACE=apanda K8S_PVC_NAME=home-apanda \
  /home/apanda/xorl-internal/.venv/bin/python experiments/k3_tests/launch_k3_test.py \
  --model qwen3.6-35b \
  --xorl-config experiments/k3_tests/configs/qwen3_6_35b_ep4_triton_alltoall.yaml \
  --xorl-gpus 4 \
  --static-traces-file /shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/refreshed_traces.json \
  --num-prompts 4 \
  --xorl-replay-routing \
  --xorl-venv /home/apanda/xorl-internal/.venv \
  --client-python /home/apanda/xorl-internal/.venv/bin/python \
  --xorl-node-selector-from-capacity \
  --gpu-capacity-wait-timeout-sec 0 \
  --pod-suffix "$RUN" \
  --log-dir "/tmp/$RUN" \
  --output-json "/tmp/$RUN.json" \
  --max-median-k3 1e-6 \
  --max-flip-rate 0.05 \
  --k3-flip-threshold 0.1
```

Verified on 2026-06-26 17:53 UTC as `q36-worst4-current-r3-replay-20260626T175300Z`:
selected `research-common-h100-073.cloud.together.ai`, replayed four routed static traces, wrote
`/tmp/q36-worst4-current-r3-replay-20260626T175300Z.json`, copied the durable result to the shared
path above, and cleaned up the pod.

Offline diagnosis for the current R3 replay:

```bash
cd /home/apanda/xorl-slime-parity-low-precision/experiments/k3_tests
PYTHONPATH=. /home/apanda/xorl-internal/.venv/bin/python diagnose_static_k3.py \
  --k3-result /shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/k3_result_r3_ep4_triton_alltoall.json \
  --traces-file /shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/refreshed_traces.json \
  --top-n 12 \
  --output-json /shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/diagnosis_r3_ep4_triton_alltoall.json
```

Fixed beta-fp32 SGLang rerun, already verified on 2026-06-26 as
`q36-worst4-beta-fp32-r3-20260626T1856Z`:

```bash
cd /home/apanda/xorl-slime-parity-low-precision
K8S_NAMESPACE=apanda K8S_PVC_NAME=home-apanda \
  /home/apanda/xorl-internal/.venv/bin/python experiments/k3_tests/launch_k3_test.py \
  --model qwen3.6-35b \
  --xorl-config experiments/k3_tests/configs/qwen3_6_35b_ep4_triton_alltoall.yaml \
  --xorl-gpus 4 \
  --static-traces-file /shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/refreshed_traces_beta_fp32_fixed.json \
  --num-prompts 4 \
  --xorl-replay-routing \
  --xorl-venv /home/apanda/xorl-internal/.venv \
  --client-python /home/apanda/xorl-internal/.venv/bin/python \
  --xorl-node-selector-from-capacity \
  --gpu-capacity-wait-timeout-sec 0 \
  --pod-suffix q36-worst4-beta-fp32-r3-20260626T1856Z \
  --log-dir /tmp/q36-worst4-beta-fp32-r3-20260626T1856Z \
  --output-json /shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/k3_result_r3_ep4_triton_alltoall_beta_fp32_fixed.json \
  --max-median-k3 1e-6 \
  --max-flip-rate 0.05 \
  --k3-flip-threshold 0.1
```

Offline diagnosis for that fixed beta-fp32 replay:

```bash
cd /home/apanda/xorl-slime-parity-low-precision/experiments/k3_tests
PYTHONPATH=. /home/apanda/xorl-internal/.venv/bin/python diagnose_static_k3.py \
  --k3-result /shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/k3_result_r3_ep4_triton_alltoall_beta_fp32_fixed.json \
  --traces-file /shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/refreshed_traces_beta_fp32_fixed.json \
  --top-n 12 \
  --output-json /shared/opd-control/er-opd-q36-35b-slots/k3/q36_worst4_current_batchinv_r3_20260626T175049Z/diagnosis_r3_ep4_triton_alltoall_beta_fp32_fixed.json
```

The immediate next K3 validation is no longer another beta-precision relaunch. The packed decode
beta fix should be upstreamed as a minimal SGLang patch because it removes the row1/pos26
wrong-token flip and reduces raw mean K3 to `0.0146352545`. The prefill beta-fp32 experiment should
not be promoted: it proves the row0 residual is beta-sensitive, but it worsens the reference by
creating a row2 `<|im_end|>` tail that xorl does not score. For the training/reference contract,
the best-supported next change is to keep the packed decode beta fix and store fixed-sequence-scored
old logprobs instead of decode-time generation logprobs; under the prefill reference, the same xorl
forward path is already much closer to zero (`mean=0.0011510748`, `p95=0.0014324672`). To drive the
remaining row0 residual from `1e-3` to true zero, do not debug alignment, routing, target extraction,
or xorl CE first: component and final-hidden probes now show identical top-k ordering and xorl
loss-path consistency, while selected `lm_head` margins from final hidden reproduce the sparse
pos93/pos124 gap. Final RMSNorm is now closed as an amplifier rather than the source. The latest
routevalidated parent-boundary captures supersede the older no-xorl-hook boundary localization:
the gap is introduced inside layer0 even though layer0 input/input_norm are identical,
layer3/layer7/layer11/layer15/layer19/layer23/layer27 add full-attention responses,
layer28/layer30 add smaller GDN responses, the larger local growth appears through layer31-32, and
the result is carried into layer34/36/37. The next narrow investigation is layer0
GDN/captured-attention closure at the exact fixed-prefill row0 input, not another sampler,
alignment, or residual-flow window.

---

## Recipe (what to run)

Two arms, sharing the **identical full-parity recipe**, varying one axis each:
- **`grpo-wq36-2n-isr3`** — IS loss (`importance_sampling`) + full parity. Isolates *parity* vs the
  old IS run (does it fix the over-training collapse / climb higher?).
- **`grpo-wq36-2n-clophi`** — `policy_loss` with **Clip-Higher** (`eps_low 0.2`, `eps_high 0.5`) +
  full parity. Isolates *loss* (can tuned clipping match unclipped IS?).

Both need the batch-invariant samplers. To run in parallel, use the second pool
(`launch/wordle-sci-opsd-sampler-c.yaml`) + a second SMG.

Builders/configs: `build_grpo_wq36_2node_isr3.py` / `…_clophi.py`,
`configs/grpo-ep8x2node-muon-lowlr-{isr3,clophi}.yaml`.

---

## References
- slime deterministic recipe: `~/slime/docs/en/advanced/reproducibility.md`
  (SGLang `--enable-deterministic-inference` + `--attention-backend flashinfer` + Megatron
  `--deterministic-mode` + `NCCL_ALGO=Ring`/`CUBLAS_WORKSPACE_CONFIG=:4096:8`).
- SGLang deterministic inference: lmsys blog 2025-09-22 ("batch-invariant" kernels).
- Train/infer mismatch correction (TIS/MIS/decoupled-PPO, batch-invariance):
  `~/slime/examples/train_infer_mismatch_helper/README.md`.
- Routing replay: arXiv:2507.18071 (slime `--use-routing-replay`).
- xorl knobs: `~/xorl-apanda-dev/src/xorl/server/server_arguments.py` (rmsnorm_mode, activation_native,
  rope_native, attention_cast_bf16, flash_attention_deterministic, lm_head_fp32); the R3 path:
  `server/runner/model_runner.py` (`RoutingReplayHandler`) + `server/orchestrator/request_processor.py`.

## Trade-offs / caveats
- Deterministic SGLang inference is **slower** (pytorch sampling, no radix cache). Run is rollout-bound,
  so steps get slower; acceptable for the parity/stability win.
- `rmsnorm_mode` and `routed_expert_logits` and the DeepEP-determinism angle remain as further levers
  if k3 is still non-trivial after this round.

---

## 🏁 FINAL OUTCOME (2026-06-26 late) — parity is GDN-gated; batch-invariant REVERTED

Ran the full recipe live (batch-invariant samplers + clean R3 routing via `--enable-return-routed-experts`
+ trainer numerics + env). Result:
- **The TAIL dropped** (`is_ratio_max` 12 → 5–6 → **3.1**) — routing-replay + batch-invariant kernels
  work on the MoE-selection-boundary tail.
- **The MEAN k3 did NOT drop** (~0.055–0.069, unchanged). So the mean is a *pervasive per-token* gap,
  not the tail → **it's the GDN/FlashQLA linear-attention layers** (every token traverses them; the
  batch-invariant kernels + R3 cover matmul/log_softmax/rmsnorm/MoE, NOT the GDN/mamba path). Matches the
  other agent's "FlashQLA GDN is still not cleared." **k3→0 is GDN-gated — not reachable via sampler/
  numerics knobs.**
- **The batch-invariant samplers CRASH-LOOP** (exitCode 1 under generation load) → during the down
  windows the trainer's weight-sync health-check fails and it diverges to NaN. So batch-invariant here is
  **both ineffective (mean k3) AND destabilizing → REVERTED** to the stable flashinfer samplers.

### The stable recipe (what to actually run)
- **Samplers:** flashinfer, **NO** `--rl-on-policy-target`, **NO** `--enable-return-routed-experts`
  (`launch/wordle-sci-opsd-sampler-b.yaml` is reverted to this proven config, `replicas: 8`).
- **Trainer:** `importance_sampling` (the loss-lever — climbs from base; do NOT use `policy_loss`+clip
  from base). `flash_attention_deterministic` MUST be OFF (crashes hdim-256). The `rope/activation/
  attention_cast` numerics are harmless to keep. The client `--return-routed-experts` flag is a no-op
  without the server flag — harmless to leave.
- **k3 ~0.055 is the accepted floor** until GDN parity lands — it did NOT block training/climbing before
  (the earlier IS run climbed 0→0.48 in-training at k3~0.055).

### Operational gotchas hit this round (mirrored in the infra runbook)
1. **Bounce the SMG whenever you bounce the samplers** — `wordle-grpo-smg` caches backends; after a
   sampler restart it 503s on stale endpoints, silently starving the trainer into NaN.
   `kubectl rollout restart deployment wordle-grpo-smg`.
2. **`kubectl apply` resets `spec.replicas`** to the file's value — keep the manifest at `replicas: 8`
   (it had `1`, which scaled the live pool 8→1).
3. **`flash_attention_deterministic: true` crashes the trainer at hdim-256** (Qwen3.6) — keep OFF.
4. **2-node capacity wall → single-node EP8 fallback** (`build_grpo_wq36_1node_isr3.py`, `dp_replicate 1`)
   — scientifically equivalent (ptmqx was single-node), needs only one 8-GPU node.

### Remaining levers (GDN / late-layer workstream's domain)
`routed_expert_logits` (gate weights, needs an xorl_client upgrade) plus the route-validated
late-layer residual path. The latest routealias row0 evidence validates both the route-aligned
SGLang parent boundary and xorl's route-forced layer34-38 hidden path at bf16 scale, and demotes the
older non-route-aligned SGLang parent-boundary localization. The layer30-32 component probe is now
complete: local attention replays mostly from captured inputs, MLP/shared-expert weighted terms are
smaller, and the layer-output gap is mostly inherited residual drift. The layer28-30 component probe
then shows all three layers are GDN, layer28 already has nonzero inherited residual drift, and
layer30 adds a visible local jump in that window. The layer24-28 component probe moves the upstream
edge earlier again: layer24 already has nonzero inherited residual drift, layers24-26 mostly carry it,
and layer27 is the main local full-attention response before layer28. The layer20-24 component probe
moves the upstream edge earlier again: layer20 already has nonzero inherited residual drift, layers20-22
mostly carry it, and layer23 is the main local full-attention response before layer24. The layer16-20
component probe moves the upstream edge earlier again: layer16 already has nonzero
inherited residual drift, layers16-18 show gradual carried growth, and layer19 is the main local
full-attention response before layer20. The layer12-16 component probe moves the upstream edge
earlier again: layer12 already has nonzero inherited residual drift, layers12-14 mostly carry it,
and layer15 is the main local full-attention response before layer16. The layer8-12 component probe
moves the upstream edge earlier again: layer8 already has nonzero inherited residual drift, layers8-10
mostly carry it, and layer11 is the main local full-attention response before layer12. The layer4-8
component probe moves the upstream edge earlier again: layer4 already has nonzero inherited residual
drift, layers4-6 mostly carry it, and layer7 is the main local full-attention response before layer8.
The layer0-4 component probe finds layer0 input/input_norm exact and the first nonzero residual
introduced inside layer0 local attention/MLP. The remaining path to push k3 below the live-run floor
is layer0 GDN/captured-attention closure at the exact fixed-prefill row0 input, not another broad
sampler/numerics knob sweep.
