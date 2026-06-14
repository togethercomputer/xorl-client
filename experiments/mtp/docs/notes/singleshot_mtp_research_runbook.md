# Workstream 09: SingleShot MTP Research Runbook

Date: 2026-06-04

---

## ⏩ STATUS 2026-06-13 — Qwen3.6-35B-A3B OPD-MTP: k=2 BOOTSTRAP WORKS (supersedes the 06-04 plan below for the live OPD line)

The 06-04 executive summary/ladder below is provenance. Current truth for the Qwen3.6 Coderforge OPD-MTP line:

**Result:** the SingleShot MTP draft is **learning**. At fixed `k=2` (ConfAdapt threshold unchanged at 0.3),
offset-1 (first mask) draft acceptance climbed **3% → ~78%**, draft confidence **0.29 → 0.84**, and
**commit_len 1.03 → 1.85** (runs `110939Z`/`194614Z`/`200325Z`, step ~500). Trajectory = slow creep then a
phase-transition around step 500; reproduced across 3 runs. Confidence now correlates with correctness.

**What unblocked it (the journey, so it isn't re-litigated):**
- Stuck-at-1.03 era root cause = (a) replay attention leaked rejected drafts + (b) only the committed slice got
  loss. Fixed by trajectory-slot visibility + `supervise_emit_window=True` (emit window sized by `attempt_k`, not
  `commit_len`, so all draft slots get teacher-targeted gradient ~every step regardless of acceptance). Branch
  `fix/mtp-replay-visibility-emit-supervision` (live worktree `/home/apanda/xorl-mtp-commitlen-fix-20260612`).
- Those fixes were necessary but NOT sufficient at `k=4`: fixed `k=4` cold-starts three mask offsets at once and
  dilutes gradient, so offset-1 never bootstrapped (the long stuck `014150Z` run). **The lever was the cold-start:
  `k=2` concentrates all mask-slot gradient on offset-1.** Loss/optimizer/architecture were NOT the problem
  (hard_teacher_ce target is correct; teacher argmax matches the trajectory at supervised positions ~0.91).
- The earlier "architectural ceiling → pivot to EAGLE" verdict (H3) is **REFUTED** by the k=2 success. Do not pivot
  to EAGLE/separate-draft-model. The acceptance gate is `hf_exact` (draft argmax must equal AR-verify argmax), not
  confidence — that's why sharpening alone wasn't enough; k=2 fixed the argmax.
- The trainer-replay-vs-sglang-sampler forward gap (~50% argmax disagreement at low-margin draft rows, surfaced by
  the perf agent's clean-region proof) is **benign**: k=2 training on the current replay transferred to the
  *sampler's* commit_len (1.85), which a real forward ceiling could not produce.

**Throughput context (perf agent's lane):** step is **sampling-bound** (sample ~85-95s vs FB ~20s overlapped), so
the clean-region GDN-replay 26× FB/MFU win is hidden — sampling (q-banding / more samplers) is the wall-clock lever.
Clean-region is argmax-equivalent (proof) → deploy-safe for the FB win, just not the bottleneck right now.

**Next science (specced):** widen-k curriculum from the k=2 checkpoint — `k=2 → [2,4] → [2,8] → [2,16]` — and judge
**offset-2/3 acceptance + teacher-scored rollout quality**, NOT loss/val_loss (val_loss 0.22 is driven by easy
committed positions; not a draft signal). Hold the widen until the sampler reprogram (`--mtp-static-cuda-graph-k-list`)
can be coordinated with the perf agent.

**Detail docs (this directory):** `mtp_science_verdict_20260613.md` (full verdict + per-offset metrics + §5b
trace-level audit of the supervision/acceptance gates), `mtp_science_next_experiments_20260613.md` (EXP-1 continue-k2 /
EXP-2 widen-k / EXP-3 offline probe, with exact configs + stop criteria), `mtp_clean_region_replay_semantics_answer.md`
(the replay-fidelity Q&A + proof outcome). Memory: `project_mtp_commit_len_floor_root_cause`.

---

This is the operational runbook for the SingleShot-style MTP reproduction in XoRL. It compresses the current research
state, run history, launch procedure, debugging loop, and next hypotheses into one file. The full chronological lab
notebook remains in
`/home/apanda/xorl-apanda-dev/docs/notes/slime-parity-workstreams/08_singleshot_mtp_debugging_plan.md`.

Primary local references:

- Paper draft: `/home/apanda/singleshot/icml_main.tex`
- Original SingleShot code: `/home/apanda/singleshot`
- XoRL implementation worktree: `/home/apanda/xorl-mtp-singleshot-port-20260602`
- Training harness:
  `/home/apanda/xorl-mtp-singleshot-port-20260602/scripts/opd/train_static_singleshot_mtp_smoke.py`
- Decode evaluator:
  `/home/apanda/xorl-mtp-singleshot-port-20260602/scripts/opd/eval_static_singleshot_mtp_decode.py`
- OPD Qwen3.6 driver:
  `/home/apanda/xorl-mtp-singleshot-port-20260602/scripts/opd/run_opd_pipeline.py`
- Qwen3.6 Coderforge ConfAdapt manifests:
  `/home/apanda/xorl-mtp-singleshot-port-20260602/experiments/opd_profile/k8s/qwen36_singleshot_mtp_coderforge_confadapt_4node/`
- Qwen3.6 SGLang student/teacher manifests:
  `/home/apanda/xorl-mtp-singleshot-port-20260602/experiments/opd_profile/k8s/qwen36_singleshot_mtp_lora_coderforge_ep4/`
- Qwen3.6 trainer examples:
  `/home/apanda/xorl-mtp-singleshot-port-20260602/examples/server/opd_singleshot_mtp_qwen36/`
- Current 10k MetaMath manifest:
  `/home/apanda/xorl-mtp-singleshot-port-20260602/experiments/opd_profile/k8s/qwen3_4b_static_mtp_metamath_gradaccum_10k.yaml`
- Decode eval manifest:
  `/home/apanda/xorl-mtp-singleshot-port-20260602/experiments/opd_profile/k8s/qwen3_4b_static_mtp_metamath_decode_eval_best499.yaml`

## Executive Summary

The current launch target is Qwen3.6-35B-A3B on Coderforge with ConfAdapt-native decoding from the start. The Qwen3-4B
MetaMath reproduction remains useful background and a small-model diagnostic, but it should no longer block the first
Qwen3.6/Coderforge scale ladder. The working assumption is that Qwen3.6-35B-A3B is already strong enough on Coderforge
for on-policy ConfAdapt MTP training to produce useful traces.

The data methodology should be assistant-turn based. A long Coderforge conversation with `N` assistant turns should
become `N` candidate prompts: for each turn, use all context up to the start of that assistant content as the prompt,
retain the assistant content span as reference/eval metadata, and let the student generate an on-policy ConfAdapt
continuation up to the configured `max_new_tokens`. Do not train by teacher-forcing the stored assistant target into the
MTP slots. The backward pass must replay the student-accepted ConfAdapt trace.

The historical paper-positive recipe was short-context MetaMathQA training with a frozen same-checkpoint AR teacher, full
student finetuning, learned `<MTP>` token, hard teacher-argmax labels, student-forced MTP proposals, causal blocked MTP
attention, random `k in [2,16]`, random offsets, `N=160`, and `M=5`.

The most important correction from the latest discussion is that confidence adaptation is not only an evaluation policy.
During training, the rollout sampler should use ConfAdapt: the student proposes up to `k`, accepts only the confident
contiguous prefix, re-prefills exactly those accepted proposals for the next teacher/student replay, and computes loss
only on the accepted positions. Fixed static `k` remains a diagnostic stress test, not the deployment or training policy.

Current state:

- The XoRL static-batch tensor geometry matches the SingleShot `truncate_and_mask` semantics in preflight.
- The training path now uses Flex/block-mask attention and avoids the known production static q-len CUDA graph risk.
- Learned-token, low-k, and curriculum training all produce real optimization signal.
- The first 10k MetaMath run produced useful confidence-adaptive decode behavior but still failed static high-k decode.
- Qwen3.6/Qwen3.5 linear-attention layers now have a SingleShot mask path for GatedDeltaNet, and Qwen3.6 OPD MTP no
  longer needs to be treated as a mask-unsupported smoke-only path.
- A Qwen3.6 Coderforge ConfAdapt smoke (`q36lm2fix-06041918`) completed one OPD step with native SGLang MTP sampling,
  trainer-side native trace replay, trace coverage for all generated targets, a successful optimizer step, and NCCL
  weight sync.
- The old Coderforge tokenized `input_ids` are not tokenizer-compatible with Qwen3.6. Retokenize
  `chat_template_applied` with the Qwen3.6 tokenizer and train from materialized assistant-turn `prompt_ids`.
- Run 0 assistant-turn materialization is complete at
  `/shared/opd-datasets/coderforge_assistant_turns_ctx8k_tgt256_qwen36_pilot64`.
- Run 1 fit/debug `q36k4a06042041` completed the 3-step 1024/64 gate using 4 full trainer nodes plus separate
  teacher/student SGLang pods. After fixes for ConfAdapt trace length, empty native-trace ranks, and bootstrap-only
  rows, all three profile rows had native trace coverage and weight-sync success. The Kubernetes resources were deleted
  after the run to release the 36 requested GPUs.
- Follow-up low-LR fit `q36k4b06042218` completed 10 steps at the same 1024/64 fit point with `OPD_OPTIM_LR=1e-6`.
  All 10 rows had native trace coverage and weight-sync success; the resources were deleted after row 10.
- A 1536/64 memory-boundary probe (`q36k4c06042308`) OOMed on step 0 in the fused linear-attention norm/gate path. The
  current fit ceiling remains `OPD_MAX_NEW_TOKENS=64` and `OPD_PROMPT_DATASET_PROMPT_LEN=1024`.
- The 16-shard one-off run (`q36k4d06042335`) was closed after four profile rows and its Kubernetes resources were
  deleted to release 36 GPUs. It proves the 16-shard dataset can drive native MTP OPD, but it is not
  throughput-shaped: `opd_pipeline_enabled=false`, `prompt_count=1`, and `sync_endpoint_count=1`. Step 2 regressed to
  `362.28s` total, with `326.77s` in trainer forward/backward and `24.17s` in sync, while the trainer log showed
  repeated RDMA endpoint/device errors during weight sync.
- The next Qwen3.6 throughput plan is the reprogrammable-slot/SMG runbook at
  `/home/apanda/xorl-mtp-singleshot-port-20260602/experiments/opd_profile/runbooks/q36_singleshot_reprogrammable_smg_runbook.md`.
  Do not launch another long one-off Q36 run until the native SMG route or direct native fanout path is resolved.
- The active small-model follow-up run is training with ConfAdapt at threshold `0.6`, because threshold `0.9` did not
  bootstrap gradients beyond the first MTP token from a cold start.

Small-model reference run, last recorded:

- Job: `q34meta-static-lmtp-ga32-10k-r5-conf06-0604`
- W&B run ID: `4a1v4giz`
- Output:
  `/shared/opd-coord/static-mtp-smoke/q34meta-static-lmtp-ga32-10k-r5-conf06-0604/`
- Strategy: `--train-rollout-strategy confidence`
- Threshold: `--train-confidence-threshold 0.6`
- Status at last check: alive, using Kubernetes-assigned plugin GPU, logging W&B/JSONL metrics, and supervising deeper
  positions than the failed threshold-0.9 run. Re-check before acting on this status; it is not part of the Qwen3.6
  launch gate.

## Research Goal

The goal is not merely to make OPD loss decrease. Several runs already showed that loss, teacher/student top-1 agreement,
and even hard-CE optimization can improve while the rollout collapses into repeated tokens. The target is behavioral:
convert an AR model into a standalone MTP model that can generate multiple future tokens per step when it is confident,
without repeated-token collapse and with teacher-scored quality close to AR.

Historical paper target for the Qwen3-4B diagnostic branch:

- Model: Qwen3-4B-Instruct-2507.
- Data: MetaMathQA, formatted as BOS plus `input + "\n\n" + response`, not Qwen chat template.
- Geometry: `N=160`, `k_max=16`, `M=N/(2*k_max)=5`, random `k in [2,16]`, random offsets.
- Objective: hard teacher-argmax CE under student-forced prefixes.
- Teacher: frozen same checkpoint as the initial student.
- Student: full finetune.
- Mask token: learned `<MTP>` token initialized from embedding statistics.
- Attention: causal blocked MTP attention.
- Evaluation: static `k` plus confidence-adaptive decoding. The practical frontier is confidence-adaptive.

Paper-level Qwen3-4B numbers to aim toward:

- Post-training static `k=1`: about 89.1% GSM8K.
- Static `k=2`: about 82.3% GSM8K.
- ConfAdapt threshold `0.95`: effective `k` about 2.7 at about 86.9%.
- ConfAdapt threshold `0.9`: effective `k` about 3.1 at about 83.6%.

## Implementation State

The current training harness supports the required pieces:

- `--train-rollout-strategy confidence|static`, defaulting to `confidence`.
- `--train-confidence-threshold`, defaulting to `0.9`.
- ConfAdapt training replay: student proposes up to `k`, accepts a contiguous confident prefix with at least one token,
  re-prefills accepted proposal tokens, and computes hard-CE only over accepted positions.
- Metrics for attempted versus accepted/supervised tokens:
  `attempted_tokens`, `supervised_tokens`, `accepted_token_count`, `accepted_token_rate`, `accepted_k_mean`,
  `accepted_k_min`, `accepted_k_max`, and `accepted_pos<N>_rate`.
- Accepted-only metrics:
  `student_teacher_top1_agreement_accepted`, `student_gt_top1_agreement_accepted`,
  `teacher_gt_top1_agreement_accepted`, `student_entropy_accepted`, `teacher_entropy_accepted`,
  `student_top1_conf_accepted`, `teacher_top1_conf_accepted`, and proposal confidence.
- Examples JSONL includes `accepted_counts` and per-token `student_token_conf`.
- Manifest GPU selection now prefers the Kubernetes device-plugin GPU UUID before any high-free-memory fallback. This is
  required in privileged pods so the run uses the scheduler-assigned GPU.
- GPU Kubernetes jobs must include `team: turbo` on the pod template labels.

Important constraints:

- Do not train by prefeeding ground-truth future tokens. The backward pass must replay the MTP rollout structure.
- If the student samples two tokens and then another two tokens, the training replay must represent those as two accepted
  MTP regions. Within a region, later positions cannot attend to earlier simultaneous predictions as if they were AR.
- For deployment/promotion, decode with ConfAdapt. Static fixed `k` is useful for stress testing but should not be the
  sole checkpoint selector.
- For Qwen3.6/Coderforge, the current `run_opd_pipeline.py` assistant-turn loader is not sufficient by itself for the
  proposed training data methodology: it finds assistant spans but selects only one usable assistant turn per source row.
  Add or use a materialized "explode all assistant turns" preprocessing step before treating the dataset as complete.
- For 128k-context training, do not assume that tail-window trainer truncation is equivalent to full-context training. If
  the trainer input drops the long prefix, Qwen3.6 linear-attention state and full-attention context are not represented
  the same way as the rollout. Treat 128k as rollout/eval first unless the trainer topology explicitly processes the
  full prefix or has a validated state-replay path.

## Qwen3.6 Coderforge ConfAdapt Plan

This section is the canonical handoff for the next scale-run agent. It supersedes older notes that said Qwen3-30B or
Coderforge should wait for a small-model MetaMath gate.

### Validated Starting Point

Use Qwen3.6-35B-A3B for the student, teacher, tokenizer, and SGLang rollout endpoints:

- HF snapshot:
  `/shared/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0`
- XORL DCP checkpoint:
  `/shared/huggingface/Qwen3.6-35B-A3B-xorl-dcp-ep8-20260522`
- SGLang repo expected by manifests: `/workspace/xorl-sglang-internal`
- XoRL repo expected by trainer manifests: `/workspace/xorl-mtp-singleshot-port-20260602`

Known-good smoke:

- Run ID: `q36lm2fix-06041918`
- Artifact:
  `/shared/opd-coord/q36lm2fix-06041918/artifacts/qwen36_confadapt_4node_opd_profile.jsonl`
- Result: one OPD step completed with `loss=0.1450418010354042`, `student_sampling_mode=sglang_native_mtp`,
  `student_sampling_mtp_native=True`, `student_sampling_mtp_debug_trace_q_lens=2`,
  `student_sampling_mtp_replay_trace_covered_all_targets=True`,
  `rollout/native_trace_covered_generated_all=True`, `sync_success=True`, and `singleshot_mtp_sampling_mode=native`.
- Limitation: this was a minimal k2, short-context, small-step plumbing smoke, not a throughput or quality result. It
  also used the old Coderforge `input_ids`, which decode incorrectly under the Qwen3.6 tokenizer, so treat it as a
  server/trainer plumbing proof only.

Relevant tests after the Qwen3.6 mask work:

```bash
cd /home/apanda/xorl-mtp-singleshot-port-20260602
PYTHONPATH=src pytest \
  tests/models/test_qwen3_5_singleshot_mask.py \
  tests/ops/test_linear_attention_singleshot_mask.py \
  tests/server/runner/test_opd_runner.py::test_singleshot_mtp_allows_linear_attention_after_mask_support \
  -q
```

Expected result from the last validation: `6 passed`.

### Data Methodology

The desired Coderforge training unit is one assistant turn, not one full document row.

Materialize a new parquet/HF-disk dataset with at least these columns:

- `prompt_ids`: tokenized context ending at the start of assistant content.
- `target_ids`: assistant content tokens until the turn terminator, for filtering and evaluation only.
- `doc_id`: stable conversation/task/document identifier.
- `turn_idx`: assistant-turn index within the document.
- `source_row`: original dataset row index or source key.
- `context_len`: length before context capping.
- `prompt_len`: actual prompt length after capping.
- `target_len`: full assistant target length before target capping.
- `target_cap`: configured max target/eval span.
- Optional but useful: repo/task metadata, language, split, and truncation flags.

Implementation notes:

- Reuse the ChatML token-span logic in `run_opd_pipeline.py`, but do not reuse the old Qwen-Coder tokenizer IDs for
  Qwen3.6. The raw Coderforge tokenized parquet uses old ChatML IDs `[151644, 77091, 151645]`; the Qwen3.6 tokenizer
  encodes `<|im_start|>assistant\n` as `[248045, 74455, 198]` and `<|im_end|>` as `[248046]`.
- For Qwen3.6, retokenize `chat_template_applied` from the raw parquet with
  `/shared/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0`, derive
  the marker IDs from that tokenizer, and train on the resulting `prompt_ids`.
- Split by `doc_id`, not by individual assistant turn, so train/eval do not share the same conversation.
- Include all qualifying assistant turns, not only the last turn. The existing `--prompt-dataset-turn-strategy=assistant`
  path is a useful parser reference but is not sufficient because it returns one assistant prompt per source row.
- Drop or tag turns with malformed role markers, empty assistant content, target length below the minimum, or severe
  prompt/target truncation.
- The stored target is not the main training target. The student should roll out on-policy with ConfAdapt, then the
  trainer should replay the accepted native MTP trace.

Initial buckets:

- `coderforge_assistant_turns_ctx8k_tgt256`: first real training bucket.
- `coderforge_assistant_turns_ctx16k_tgt512`: first scale bucket if 8k is healthy.
- `coderforge_assistant_turns_ctx32k_tgt512`: scale and throughput bucket.
- `coderforge_assistant_turns_ctx128k_tgt512_or_1024`: rollout/eval first; training only after full-context trainer
  handling is validated.

The currently available short Coderforge tokenized dataset is:

```text
/shared/glm-5-1-coderforge-truncated/l8192/data-00000-of-00001.parquet
```

It has 1184 rows with `input_ids` and `labels`. Treat it as a smoke source, not the final assistant-turn dataset.

Run 0 pilot materialization is complete:

- Script:
  `/home/apanda/xorl-mtp-singleshot-port-20260602/scripts/opd/build_qwen_mtp_prompt_dataset.py`
- Raw source:
  `/shared/huggingface/hub/datasets--togethercomputer--CoderForge-Preview/snapshots/060fca96cf723b2ebab3181e9e59fafd273df3cb/trajectories-tokenized_qwencoder`
- Output:
  `/shared/opd-datasets/coderforge_assistant_turns_ctx8k_tgt256_qwen36_pilot64`
- Build:

```bash
.venv/bin/python scripts/opd/build_qwen_mtp_prompt_dataset.py \
  --source-tokenized-path /shared/huggingface/hub/datasets--togethercomputer--CoderForge-Preview/snapshots/060fca96cf723b2ebab3181e9e59fafd273df3cb/trajectories-tokenized_qwencoder \
  --output-dir /shared/opd-datasets/coderforge_assistant_turns_ctx8k_tgt256_qwen36_pilot64 \
  --ctx-len 8192 \
  --target-cap 256 \
  --min-target-tokens 16 \
  --eval-fraction 0.02 \
  --max-input-files 1 \
  --max-source-rows 64 \
  --metadata-columns reward \
  --retokenize-tokenizer-path /shared/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --derive-chatml-token-ids
```

Run 0 pilot summary:

- 64 source rows produced 3618 assistant-turn rows.
- Train split: 3573 turns from 63 documents.
- Eval split: 45 turns from 1 document.
- Malformed marker rate: 0.0.
- Target-too-short count: 0.
- Prompt truncation rate: about 0.948.
- Target truncation rate: about 0.230.
- Train file: `train.parquet`; eval file: `eval.parquet`; summary: `summary.json`.
- This is a small pilot bucket for fit/debug. Build a larger uncapped bucket before longer training.

Run 0 larger streaming materialization checkpoint:

- Output:
  `/shared/opd-datasets/coderforge_assistant_turns_ctx8k_tgt256_qwen36_files16_stream`
- Build log:
  `/shared/opd-datasets/coderforge_assistant_turns_ctx8k_tgt256_qwen36_files16_stream.build.log`
- Summary hash:
  `25274bd14eb890f69d11d9ef647e3d1eb848f0434d8a23f046f31567a4c7ec1d`
- Source coverage: first 16 parquet shards, 2368 source rows/documents.
- Assistant turns: 130429 total rows.
- Train split: 128034 turns from 2326 documents.
- Eval split: 2395 turns from 42 documents.
- Malformed marker rate: 0.0.
- Prompt truncation rate: about 0.937.
- Target truncation rate: about 0.223.
- Size on shared storage: about 882 MiB.
- Parquet row groups: train 126, eval 3, written with `--write-batch-size 1024`.
- Spot check: prompts end in the Qwen3.6-derived assistant header `[248045, 74455, 198]`; `target_ids` are capped to
  256 tokens while `target_len` preserves the original assistant span length.
- This is large enough for a longer 1024/64 smoke or short quality probe. It is still not the full uncapped all-shard
  materialization.

### Training Methodology

For each prompt batch:

1. Student SGLang endpoint performs native MTP generation with `sampling_mode=native` and `mtp_strategy=["conf_adapt",
   threshold]`.
2. The generated continuation and native debug trace are recorded.
3. Teacher endpoint computes hidden/logprob targets for the prompt plus generated continuation.
4. Trainer rewrites the batch using `rollout_replay=true` and validates that the native trace covers all supervised
   generated tokens.
5. Trainer optimizes the Qwen3.6 student with OPD/KL loss on the replayed accepted positions.
6. Updated weights sync back to the student SGLang endpoint.

Start conservative:

- `OPD_MTP_K_TOKS=4` for the first real run, then `8`, then `16`.
- `OPD_MTP_CONF_THRESHOLD=0.6` for bootstrap. Compare `0.8`, `0.9`, or a schedule only after deeper positions appear.
- `OPD_MAX_NEW_TOKENS=64` with `OPD_PROMPT_DATASET_PROMPT_LEN=1024` is the known fit point for the current 4x8 trainer.
  The original `256` target and a 1536/64 probe OOMed and should be retried only after a memory plan is explicit.
- Set both `OPD_FULL_FT_LR=0.000001` and `OPD_OPTIM_LR=0.000001` for real training. `OPD_FULL_FT_LR` controls the
  trainer config, but `run_opd_pipeline.py` passes `OPD_OPTIM_LR` to the optimizer-step API and defaults to `1e-4`.
- `train_router=false`, `router_fp32=true`, `lm_head_fp32=true`.
- Keep `native_mtp_debug_trace=true` and `validate_native_mtp_trace=true` until the data/topology ladder is stable.

Do not promote a run on loss alone. The acceptance depth, replay coverage, teacher NLL, repeat metrics, and weight-sync
health matter more.

### Topology

Immediate cluster reality is dynamic. Re-check before every launch and do not assume an old free-node snapshot is still
true. During `q36k4a06042041`, the run used 36 GPUs:

- Trainer: 32 GPUs across full nodes `research-common-h100-117`, `116`, `040`, and `047`.
- Student SGLang: 2 GPUs on `research-common-h100-105`.
- Teacher SGLang: 2 GPUs on `research-common-h100-041`.
- Cluster-wide inventory during the run: 50 H100 nodes, 397 allocatable GPUs, 352 requested, 45 free.
- Completely free 8-GPU nodes during the run: `research-common-h100-049` and `071`.
- Partial free nodes during the run: `001:2`, `041:2`, `058:6`, `059:1`, `079:1`, `080:4`, `094:4`, `105:2`,
  `114:7`.

After `q36k4a` cleanup, the same inventory reported 316 GPUs requested and 81 free. Completely free 8-GPU nodes were
`research-common-h100-040`, `047`, `049`, `071`, `116`, and `117`; partial free nodes were `001:2`, `041:4`, `058:6`,
`059:1`, `079:1`, `080:4`, `094:4`, `105:4`, and `114:7`.

After `q36k4b` cleanup, the cluster reported 313 GPUs requested and 84 free. Completely free 8-GPU nodes were
`research-common-h100-040`, `047`, `049`, `116`, and `117`; partial free nodes were `001:2`, `036:7`, `041:4`, `058:6`,
`059:1`, `071:4`, `079:1`, `080:4`, `094:4`, `105:4`, and `114:7`.

After `q36k4c` cleanup, the cluster reported 308 GPUs requested and 89 free. Completely free 8-GPU nodes were
`research-common-h100-040`, `047`, `049`, `071`, `116`, and `117`; partial free nodes were `001:2`, `036:7`, `041:4`,
`058:6`, `059:1`, `079:1`, `080:4`, `094:5`, `105:4`, and `114:7`.

On 2026-06-04 23:32 UTC, the cluster reported 389 schedulable GPUs, 298 requested, and 91 free. Completely free 8-GPU
nodes were `research-common-h100-036`, `040`, `047`, `049`, `071`, `116`, and `117`; partial free nodes were `001:2`,
`041:4`, `058:6`, `059:1`, `079:1`, `080:4`, `094:6`, and `105:4`. There were no pending GPU pods. The current
SingleShot Qwen3.6 ladder was not consuming GPUs at that moment; work was on dataset materialization/validation.

On 2026-06-04 23:59 UTC, with the warm `er-opd-q36-35b-slots` stack and the active `q36k4d06042335` one-off run both
allocated, the cluster reported 397 allocatable GPUs, 342 GPUs used by live pods, and 55 free GPUs. Completely free
8-GPU nodes were `research-common-h100-036`, `116`, and `117`; partial free nodes were `001:2`, `041:2`, `058:6`,
`059:1`, `079:1`, `080:4`, `094:6`, `105:2`, and `114:7`. The warm stack used 88 GPUs; `q36k4d06042335` used another
36 GPUs.

After `q36k4d06042335` cleanup, the cluster reported 397 allocatable GPUs, 311 GPUs used by live pods, and 86 free GPUs.
Completely free 8-GPU nodes were `research-common-h100-036`, `040`, `047`, `049`, and `071`; partial free nodes were
`001:2`, `041:4`, `058:6`, `059:1`, `079:1`, `080:4`, `094:6`, `105:4`, `114:7`, `116:4`, and `117:7`. The warm
`er-opd-q36-35b-slots` stack remains allocated and ready for reprogrammable-slot work.

For future fit/debug runs, do not scale a live StatefulSet mid-flight just because extra 8-GPU nodes appear. These smokes
use batch size 1 and are bottlenecked by trainer forward/backward plus sync/prefill mechanics, not by insufficient node
count. Use extra full nodes for an independent parallel smoke or for the next topology run after the current rung is
settled.

Current exception: for Qwen3.6 SingleShot throughput work, prefer the reprogrammable-slot/SMG path over another one-off
4-node run. The one-off path has already shown underfilled, serial behavior and P2P sync warnings.

Cluster snapshot command:

```bash
python - <<'PY'
import json, subprocess

nodes = json.loads(subprocess.check_output(["kubectl", "get", "nodes", "-o", "json"], text=True))["items"]
pods = json.loads(subprocess.check_output(["kubectl", "get", "pods", "-A", "-o", "json"], text=True))["items"]
used = {}
for pod in pods:
    if pod.get("status", {}).get("phase") not in {"Pending", "Running"}:
        continue
    node = pod.get("spec", {}).get("nodeName")
    if not node:
        continue
    gpu_count = 0
    for container in pod.get("spec", {}).get("containers", []):
        resources = container.get("resources", {})
        requests = resources.get("requests", {}) or {}
        limits = resources.get("limits", {}) or {}
        value = requests.get("nvidia.com/gpu", limits.get("nvidia.com/gpu", 0))
        try:
            gpu_count += int(value)
        except Exception:
            pass
    if gpu_count:
        used[node] = used.get(node, 0) + gpu_count

rows = []
for node in nodes:
    name = node["metadata"]["name"]
    alloc = int(node["status"].get("allocatable", {}).get("nvidia.com/gpu", 0))
    if alloc:
        rows.append((alloc - used.get(name, 0), used.get(name, 0), alloc, name))
rows.sort(reverse=True)
print("free\tused\talloc\tnode")
for row in rows[:80]:
    print("%s\t%s\t%s\t%s" % row)
print("free_8gpu_nodes=", sum(1 for free, _, alloc, _ in rows if alloc >= 8 and free >= 8))
PY
```

Preferred launch topology when clean full nodes are available:

- Trainer: 4 nodes x 8 GPUs = 32 GPUs.
- Trainer parallelism: `dp_shard=32`, `expert_parallel_size=8`, `ep_fsdp=4`, `tp=1`, `pp=1`, `ulysses=1`,
  `ringattn=1`.
- Student SGLang: TP2 for 8k/16k; TP4 if context or batch pressure requires it.
- Teacher SGLang: TP4 for 8k/16k; TP8 for 32k+ if prefill memory or latency dominates.
- MoE dispatch: use the current manifest defaults for the first fit; after correctness is stable, prefer DeepEP with
  `deepep_num_sms=72` for throughput comparisons.

Fragmented-node fallback if full 8-GPU nodes are not available:

- Use 4-GPU trainer pods across clean partial nodes and keep student/teacher at TP2 or TP4.
- Avoid this path for the next Qwen3.6 ladder attempt unless the assigned device-plugin GPU UUIDs and actual GPU memory
  are verified for every pod. A 2x4 fallback run (`q36cf8k4-06042026`) placed `trainer-1` on
  `research-common-h100-105` with `CUDA_VISIBLE_DEVICES=0,1,2,3`; DCP materialization OOMed because those GPUs had only
  about 30-144 MiB free. The run was deleted before any profile row.
- Do not request GPUs on a node that already has unmanaged or live workload memory pressure. The scheduler view is
  necessary but not sufficient.
- Before pinning nodes, inspect actual GPU memory with the pod logs or a short privileged diagnostic. Scheduler
  accounting is necessary but not sufficient.

All GPU pod templates must carry `team: turbo`. Do not manually set Volcano scheduler fields unless there is a specific
reason; the team label should let the webhook inject the queue/scheduler.

### Initial Run Ladder

Run 0: data audit/materialization.

- Build the assistant-turn dataset.
- Output a JSON summary with rows, documents, assistant turns per document, prompt-length histogram, target-length
  histogram, target truncation rate, malformed marker rate, and train/eval split sizes.
- Status: pilot complete at `/shared/opd-datasets/coderforge_assistant_turns_ctx8k_tgt256_qwen36_pilot64`.

Run 1: `q36-cf-8k-k4-t256-fit`.

- Dataset: assistant-turn `ctx8k_tgt256`.
- Trainer: 16-32 GPUs depending on availability.
- Student/teacher: TP2 or TP4.
- `OPD_MTP_K_TOKS=4`, `OPD_MAX_NEW_TOKENS=256`, `OPD_MTP_CONF_THRESHOLD=0.6`.
- Duration: 10-20 steps.
- Gate: no OOM, finite loss, `student_sampling_mode=sglang_native_mtp`, native trace coverage for all generated
  targets, no unexplained large unused-trace tail, successful optimizer step, successful weight sync.
- First fit/debug realization: `q36k4a06042041`.
  - Dataset:
    `/shared/opd-datasets/coderforge_assistant_turns_ctx8k_tgt256_qwen36_pilot64`.
  - Profile:
    `/shared/opd-coord/q36k4a06042041/artifacts/qwen36_confadapt_4node_opd_profile.jsonl`.
  - W&B run: `https://wandb.ai/together-research/singleshot/runs/efkgogbi`.
  - Conservative fit knobs after OOM ladder: `OPD_PROMPT_DATASET_PROMPT_LEN=1024`, `OPD_MAX_NEW_TOKENS=64`,
    `OPD_NUM_STEPS=3`, `OPD_MAX_OPD_STEPS=3`, `OPD_MTP_K_TOKS=4`, `OPD_MTP_CONF_THRESHOLD=0.6`.
  - Step 0 succeeded with `loss=0.447278`, `valid_tokens=512`, `step_total_s=354.497`,
    `teacher_prefill_s=117.584`, `forward_backward_roundtrip_s=163.786`, `sync_inference_weights_s=61.782`,
    `trace_steps=17`, `trace_committed=65`, replay coverage true, and sync success true.
  - Step 1 succeeded with `loss=2.027064`, `valid_tokens=512`, `step_total_s=108.424`,
    `teacher_prefill_s=0.667`, `forward_backward_roundtrip_s=80.262`, `sync_inference_weights_s=20.586`,
    `trace_steps=17`, `trace_committed=65`, replay coverage true, and sync success true.
  - Step 2 succeeded with `loss=1.232681`, `valid_tokens=512`, `step_total_s=359.901`,
    `teacher_prefill_s=0.599`, `forward_backward_roundtrip_s=326.871`, `sync_inference_weights_s=20.583`,
    `trace_steps=63`, `trace_committed=63`, replay coverage true, sync success true, and
    `rollout/max_repeated_token_run=1`.
  - The warm steps prove the bootstrap-only native-trace case no longer crashes on cluster. Step 2 also shows a much
    longer trainer roundtrip than step 1, so the next run should keep per-step timings in the gate.
- Follow-up low-LR realization: `q36k4b06042218`.
  - Same dataset and 1024/64 fit point, `OPD_NUM_STEPS=10`, `OPD_MAX_OPD_STEPS=10`, `OPD_OPTIM_LR=0.000001`,
    `OPD_FULL_FT_LR=0.000001`.
  - Active profile:
    `/shared/opd-coord/q36k4b06042218/artifacts/qwen36_confadapt_4node_opd_profile.jsonl`.
  - W&B run:
    `https://wandb.ai/together-research/singleshot/runs/0wd45fqk`.
  - Placement: trainer pods on `research-common-h100-040`, `116`, `047`, and `049`; student SGLang on
    `research-common-h100-105`; teacher SGLang on `research-common-h100-041`.
  - Result: 10/10 rows completed with `learning_rate=1e-6`, replay coverage true on every row, and sync success true on
    every row.
  - Aggregate metrics: mean loss `0.7867`, min/max loss `0.0381/4.6066`; mean step time `245.9s`; mean trainer
    forward/backward roundtrip `205.9s`; mean sync `19.37s`; mean student sampling `2.80s`.
  - Trace-step distribution: `{4: 1, 17: 3, 18: 1, 63: 5}`. The `trace_steps=63` rows had long trainer roundtrips near
    `311-334s`.
  - Repeat behavior was mixed: 3/10 rows had repeat runs at least 16, while 7/10 had `rollout/max_repeated_token_run=1`.
  - Cleanup: deleted the trainer StatefulSet, student job, teacher job, and services after row 10 landed.
- Failed memory-boundary probe: `q36k4c06042308`.
  - Same dataset, `OPD_PROMPT_DATASET_PROMPT_LEN=1536`, `OPD_MAX_NEW_TOKENS=64`, `OPD_NUM_STEPS=3`,
    `OPD_OPTIM_LR=0.000001`.
  - W&B run:
    `https://wandb.ai/together-research/singleshot/runs/rhuljm50`.
  - Step 0 sampled and fetched teacher hidden states for 1599 tokens, then trainer forward/backward OOMed before any
    profile row was written.
  - Failure: `torch.OutOfMemoryError` in `src/xorl/ops/linear_attention/modules/fused_norm_gate.py`, allocating
    `2.28 GiB` with `2.17 GiB` free on rank 0 GPU 0. The packed trainer batch was 1664 tokens.
  - Cleanup: deleted the trainer StatefulSet, student job, teacher job, and services. Pods were gone at the final check.

Run 2: `q36-cf-16k-k8-t512-scale32`.

- Dataset: assistant-turn `ctx16k_tgt512`.
- Trainer: target 32 GPUs.
- Student/teacher: TP4 if TP2 prefill is slow or near memory limits.
- `OPD_MTP_K_TOKS=8`, `OPD_MAX_NEW_TOKENS=512`.
- Duration: 20-50 steps.
- Gate: Run 1 gates plus stable step-time variance, nonzero acceptance beyond position 1, and no replay coverage
  failures.

Run 3: `q36-cf-32k-k8-t512-scale`.

- Dataset: assistant-turn `ctx32k_tgt512`.
- Trainer: 32-64 GPUs as available.
- Teacher: likely TP4/TP8.
- Use this run to decide whether teacher prefill, student rollout, trainer forward/backward, or weight sync is the main
  bottleneck.

Run 4: `q36-cf-128k-rollout-eval`.

- Dataset: assistant-turn `ctx128k_tgt512_or_1024`.
- Purpose: evaluate whether Qwen3.6 is good enough on true long Coderforge contexts before paying for full training.
- Metrics: completion validity, target overlap or task-specific eval if available, teacher NLL, repeat metrics, effective
  k, commit-length distribution, and prefill latency.
- Do not treat this as a training run unless the trainer full-context plan is explicit and validated.

### Launch Template

Use the existing manifests as the starting point:

```bash
cd /home/apanda/xorl-mtp-singleshot-port-20260602
RUN_ID=q36-cf-8k-k4-t256-fit-$(date -u +%m%d%H%M)
COORD_DIR=/shared/opd-coord/${RUN_ID}
mkdir -p "${COORD_DIR}"

sed "s#/shared/opd-coord/CHANGE-ME-RUN-ID#${COORD_DIR}#g; s#CHANGE-ME-RUN-ID#${RUN_ID}#g" \
  experiments/opd_profile/k8s/qwen36_singleshot_mtp_lora_coderforge_ep4/student-sglang.yaml \
  | kubectl apply --dry-run=server -f -
sed "s#/shared/opd-coord/CHANGE-ME-RUN-ID#${COORD_DIR}#g; s#CHANGE-ME-RUN-ID#${RUN_ID}#g" \
  experiments/opd_profile/k8s/qwen36_singleshot_mtp_lora_coderforge_ep4/teacher-sglang.yaml \
  | kubectl apply --dry-run=server -f -
sed "s#/shared/opd-coord/CHANGE-ME-RUN-ID#${COORD_DIR}#g; s#CHANGE-ME-RUN-ID#${RUN_ID}#g" \
  experiments/opd_profile/k8s/qwen36_singleshot_mtp_coderforge_confadapt_4node/trainer.yaml \
  | kubectl apply --dry-run=server -f -
```

After server-side dry-run passes, apply the same three rendered manifests without `--dry-run=server`. Override knobs via
manifest env edits or `kubectl set env` before pods start. For Run 1, set at least:

```bash
OPD_PROMPT_DATASET_PATH=<materialized-ctx8k-assistant-turn-dataset>
OPD_PROMPT_DATASET_TYPE=parquet
OPD_PROMPT_DATASET_COLUMN=prompt_ids
OPD_PROMPT_DATASET_PROMPT_LEN=8192
OPD_PROMPT_DATASET_NUM_PROMPTS=<batch-per-opd-step>
OPD_MTP_K_TOKS=4
OPD_MTP_CONF_THRESHOLD=0.6
OPD_MAX_NEW_TOKENS=256
OPD_FULL_FT_LR=0.000001
OPD_OPTIM_LR=0.000001
```

The current `q36k4a06042041` and `q36k4b06042218` fit/debug runs had to lower the fit knobs to
`OPD_PROMPT_DATASET_PROMPT_LEN=1024` and `OPD_MAX_NEW_TOKENS=64`. A follow-up `1536/64` probe (`q36k4c06042308`) OOMed
in trainer forward/backward. Treat `8192/256` as the desired Run 1 target, not as a known fit point for the current 4x8
trainer topology.

Do not use the existing old-tokenizer smoke dataset for Qwen3.6 fit runs. If a pure plumbing smoke intentionally uses it,
keep the run name clearly marked as a smoke and leave `OPD_PROMPT_DATASET_COLUMN=input_ids`.

### First Checks After Launch

```bash
kubectl get pods -A -l run-id=${RUN_ID} -o wide
kubectl get pods -A -l run-id=${RUN_ID} -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.metadata.labels.team}{"\n"}{end}'
find "${COORD_DIR}/artifacts" -maxdepth 3 -name job.log -o -name server.log -o -name server.err
tail -n 80 "${COORD_DIR}/artifacts/trainer-node0/job.log"
tail -n 1 "${COORD_DIR}/artifacts/qwen36_confadapt_4node_opd_profile.jsonl" | python -m json.tool
```

The first profile row must show:

- `student_sampling_mode=sglang_native_mtp`.
- `student_sampling_mtp_native=True`.
- `student_sampling_mtp_replay_trace_covered_all_targets=True`.
- `rollout/native_trace_covered_generated_all=True`.
- `student_sampling_mtp_trace_tokens_unused` small and explained by final-block overrun, or zero. Coverage failure is a
  blocker; a small unused tail is not by itself a blocker for ConfAdapt.
- `sync_success=True` unless intentionally running with `OPD_SKIP_OPTIM_STEP=1`.
- `prompt_dataset_path`, `prompt_dataset_column`, and `prompt_lengths` matching the intended materialized dataset.

### Missing Information Before The Next Real Launch

- Larger canonical materialized assistant-turn Coderforge dataset path. The current pilot dataset is intentionally capped
  at 64 source rows.
- Final metadata fields to preserve in the larger materialized dataset.
- Target held-out evaluation set beyond the pilot doc-hash split.
- Clean-node availability at the launch time. The previous snapshot is stale as soon as the cluster changes.
- Exact SGLang commit/image containing the Qwen3.6 GDN q>1 native MTP decode patch; verify this from pod startup logs.
- DeepEP availability in the trainer Python environment for the selected image and worktree.
- Decision on 128k trainer semantics: full-prefix training with CP/SP versus rollout-only eval until state replay exists.
- Real LR and optimizer schedule after the first 8k fit. Start low, then adjust from measured loss/acceptance behavior.

### Unnecessary Or Potentially Misleading Information

- The full Qwen3-4B MetaMath run ledger is historical context for debugging, not required reading before launching the
  first Qwen3.6 Coderforge fit. Keep it for provenance but do not use it as a blocker.
- Static high-k decode quality is a stress diagnostic, not the promotion criterion for this workstream.
- Older Qwen3-30B Coderforge manifests are superseded by Qwen3.6-35B-A3B for the next runs.
- Old notes saying Qwen3.6 linear attention lacks SingleShot mask support are stale for this worktree.
- Raw profile JSON is too verbose for handoff. Record only the gate metrics, run ID, artifact paths, and conclusions.

## Run Ledger

### Early Online OPD Runs

`q34gsm-rk2-06032308`

- Full-vocab reverse-KL OPD by accident, not paper-faithful hard-teacher CE.
- Loss improved from about 2.36 to 0.55 by step 16, top-1 agreement rose from 0.73 to 0.93, and entropy collapsed.
- Rollouts collapsed: 32/32 sampled continuations had repeat runs at least 16; digit-token rate was about 98.7%.
- Conclusion: loss can improve while behavior is unusable.

`q34gsm-hardce8-06032324`

- Hard teacher CE path, stopped at step 11.
- Loss improved from about 3.98 to 0.85 by step 10 and top-1 agreement rose, but generated text collapsed into repeated
  digits.
- Teacher NLL audit showed student rollouts were much worse than AR teacher and even worse than simple repeated-token
  baselines in some cases.
- Conclusion: online poisoned-prefix hard CE can train the student into attractors because the teacher may also prefer
  repeated tokens after a poisoned prefix.

`q34gsm-frozenprobe-06032345`

- Frozen native MTP probe.
- AR was healthy; native static `k=16` was badly repetitive.
- ConfAdapt at `k=16`, threshold `0.9`, avoided long repeats but accepted shallow continuations.
- Conclusion: confidence gating is necessary, but the native runtime still needed correctness work.

### Runtime Correctness Runs

`q34gsm-pendingfix-0604`

- Fixed SGLang pending-token handoff: scheduler now uses `mtp_pending_token_ids` rather than stale generic sampler tokens.
- Native `k=2` improved in tiny probes; `k=8` remained weak.

`q34gsm-singleparity-0604`

- Singleton native `k=1` matched AR exactly for one prompt.
- Earlier AR mismatch was a batching state splice issue, not true MTP behavior.

Batch isolation series:

- Pre-fix `q34gsm-batchiso-0604`: batched AR mismatched due stale `next_token_id`.
- Post-fix `q34gsm-batchiso-k28-r3-0604`: AR, native `k=1`, native `k=2` matched 4/4 prompts; native `k=8` still
  mismatched 2/4.
- `q34gsm-batchiso-k2468-diag-0604`: AR 4/4, k1 4/4, k2 4/4, k4 3/4, k6 4/4, k8 2/4.
- Later no-CUDA-graph FlashInfer and deterministic torch-native reruns matched all prompts for `k in {2,4,6,8}`.
- Production static q-len CUDA graph path still showed mismatches even when trace replay passed.
- Conclusion: avoid q-len-specific static CUDA graphs for this reproduction path unless that path is separately fixed.

### Static Batch And Training Preflights

Static batch preflight:

- Script: `preflight_mtp_static_batch.py`.
- Verifies XoRL `prepare_singleshot_mtp_batch` against SingleShot `truncate_and_mask`.
- Settings: `N=160`, `M=5`, random `k in [2,16]`, random negative offsets, no pad/EOS/prelude/prefix loss.
- Artifacts:
  `/shared/opd-coord/static-mtp-paper-preflight-20260604/summary.json` and
  `/shared/opd-coord/static-mtp-paper-preflight-20260604/summary_pad128.json`.
- Result: pass.
- Unit tests: `pytest tests/mtp/test_singleshot.py -q` passed with 24 tests.

`q34gsm-static-smoke2-0604`

- 10-step GSM proxy static smoke.
- W&B run ID: `jzu9arhi`.
- Loss improved from 7.194 to 2.945 and student/teacher top-1 from 0.178 to 0.433.
- Still repetitive; this was only a mechanical pass.

`q34gsm-static-500-0604`

- 500-step static run.
- W&B run ID: `bclrmvts`.
- First50 to last50 loss improved from 2.869 to 1.862.
- Student/teacher top-1 improved from 0.416 to 0.531.
- Repeats improved but remained present.
- Conclusion: static training learns, but fixed high-k behavior is not solved.

`q34gsm-static-learned100-0604`

- Learned `<MTP>` token path.
- W&B run ID: `d3vo6gs0`.
- Final eval improved at low k; k8/k16 remained repeat-heavy.
- Learned token alone is not sufficient.

`q34gsm-static-lmtp-k4-500-0604`

- Low-k curriculum with learned token and random `k in [2,4]`.
- W&B run ID: `s50dxqo6`.
- First50 to last50 loss improved from 1.873 to 1.181.
- Repeats became mild at k2/k4; k8/k16 still weak.
- Conclusion: low-k staging is promising.

`q34gsm-static-lmtp-k4ckpt-0604` and `q34gsm-static-lmtp-k4ckpt-eval2-0604`

- Checkpoint save/load validation.
- W&B run IDs: `52u2kf6j` and `t9i6lx3y`.
- Reloaded checkpoint behaved consistently; no reinitialization issue.

`q34gsm-static-lmtp-k8fromk4-0604`

- Continued from k4 checkpoint to random `k in [2,8]`.
- W&B run ID: `ah1ny6vv`.
- Loss and agreement improved across k2/k4/k6/k8 buckets, but high-k decode still regressed at points.
- Conclusion: curriculum helps but is not enough by itself.

### Decode Evaluations

`q34gsm-static-lmtp-decodeeval-med4-0604`

- Decode evaluator validation on k4 and k8 checkpoints.
- W&B run ID: `gfivka4a`.
- Static k8/k16 remained repetitive.
- ConfAdapt prevented collapse but accepted only about 1.5 to 1.8 tokens per MTP step.
- Conclusion: ConfAdapt is the right decode policy, but the trained model is still far below the paper frontier.

### MetaMath Runs

`q34meta-static-lmtp-ga8-50-0604`

- First MetaMath smoke.
- W&B run ID: `uqq1s2wo`.
- Loss improved from 3.350 to 2.132 over 50 steps.
- Fixed eval improved at every k, but high-k repeats remained.
- Conclusion: launch mechanics, dataset, W&B, checkpointing, and learned token path are healthy.

`q34meta-static-lmtp-ga32-10k-0604`

- Main 10k static MetaMath run.
- W&B run ID: `ui60if2c`.
- Output:
  `/shared/opd-coord/static-mtp-smoke/q34meta-static-lmtp-ga32-10k-0604/`
- Dataset:
  `/shared/opd-datasets/qwen3_4b_metamathqa_bos_min192_8192.jsonl`
- Geometry: `N=160`, `M=5`, random `k in [2,16]`, learned `<MTP>`, random offsets.
- Effective batch: 32 via gradient accumulation.
- Fixed final eval at step 9999:
  - k2 loss 0.432, student/teacher 0.850, student/GT 0.825, max repeat 1.0.
  - k4 loss 0.676, student/teacher 0.806, student/GT 0.713, max repeat 1.75.
  - k8 loss 1.525, student/teacher 0.625, student/GT 0.384, max repeat 2.25.
  - k16 loss 1.887, student/teacher 0.502, student/GT 0.266, max repeat 4.875.
- Final9999 32-sample decode:
  - Static k8: effective k 8, teacher NLL 3.248, argmax 0.604, max repeat 14.594.
  - ConfAdapt k8 threshold 0.95: effective k 2.110, teacher NLL 0.181, argmax 0.957, max repeat 1.406.
  - ConfAdapt k8 threshold 0.9: effective k 2.389, teacher NLL 0.234, argmax 0.951, max repeat 1.438.
- Conclusion: fixed k8 CE selected a different checkpoint than the practical ConfAdapt frontier. Final9999 was better
  than best7499 for confidence-adaptive decode despite worse fixed-eval CE. Static k8 remained unusable as a deployment
  policy.

### ConfAdapt Training Runs

`q34meta-static-lmtp-ga32-10k-r2-0604`

- Stopped after startup because it still supervised the full static sampled k region.
- Conclusion: wrong training objective for the current plan.

`q34meta-static-lmtp-ga32-10k-r3-confadapt-0604`

- W&B run ID: `unwu6w2g`.
- Training strategy: ConfAdapt threshold `0.9`.
- Startup and train metrics confirmed the implementation was replaying accepted tokens only.
- Stopped because cold-start threshold `0.9` accepted almost exactly one token per region:
  `accepted_k_mean` stayed around 1.0 to 1.04 and `accepted_pos2_rate` stayed near 0.0 to 0.04.
- Conclusion: valid implementation, bad bootstrap threshold.

`q34meta-static-lmtp-ga32-10k-r4-conf06-0604`

- Training strategy: ConfAdapt threshold `0.6`.
- Showed nontrivial accepted depth immediately: startup k8 had `accepted_k_mean=1.425`, `accepted_pos2_rate=0.125`,
  `accepted_pos3_rate=0.05`.
- Stopped because the manifest preferred a high-free GPU over the Kubernetes plugin GPU in a privileged pod.
- Conclusion: threshold `0.6` is worth running, but GPU selection had to be fixed.

`q34meta-static-lmtp-ga32-10k-r5-conf06-0604`

- Active replacement run.
- W&B run ID: `4a1v4giz`.
- Training strategy: ConfAdapt threshold `0.6`.
- Startup selected the plugin GPU and logged W&B/JSONL metrics correctly.
- Deeper supervision is sustained beyond the first warmup rows. At the latest local check, it had 1803 train rows
  through step 1802:
  - loss: first 1.011, last 0.298, tail-100 mean 0.386.
  - `accepted_k_mean`: first 1.375, last 1.888, tail-100 mean 1.731.
  - `accepted_pos2_rate`: tail-100 mean 0.549.
  - `accepted_pos3_rate`: tail-100 mean 0.165.
  - `accepted_pos4_rate`: tail-100 mean 0.019.
  - `student_teacher_top1_agreement_accepted`: tail-100 mean 0.905.
- Conclusion: this is the current most relevant run. It is intentionally noisier than threshold `0.9` because it trains
  lower-confidence later positions.

### Qwen3.6 Coderforge Runs

`q36lm2fix-06041918`

- One-step server/trainer plumbing smoke.
- Completed native SGLang MTP sampling, trainer native-trace replay, optimizer step, and weight sync.
- Limitation: used old Coderforge `input_ids`, so do not use it as evidence that the Qwen3.6 training data path is
  semantically correct.

`coderforge_assistant_turns_ctx8k_tgt256_qwen36_pilot64`

- Run 0 data materialization pilot.
- Output:
  `/shared/opd-datasets/coderforge_assistant_turns_ctx8k_tgt256_qwen36_pilot64`
- Summary: 3618 assistant turns from 64 source rows, 3573 train rows, 45 eval rows, malformed marker rate 0.0.
- Status: usable for Run 1 fit/plumbing, not large enough for a longer quality run.

`q36cf8k4-06042026`

- Attempted 2-node x 4-GPU Run 1 fallback using the Qwen3.6-retokenized assistant-turn pilot dataset.
- Rendered manifests:
  `/shared/opd-coord/q36cf8k4-06042026/rendered/`
- Trainer config artifact:
  `/shared/opd-coord/q36cf8k4-06042026/artifacts/qwen36_student_trainer_confadapt_4node_ep8.yaml`
- Trainer settings: `NNODES=2`, `data_parallel_shard_size=8`, `expert_parallel_size=8`, `OPD_MTP_K_TOKS=4`,
  `OPD_PROMPT_DATASET_PATH=/shared/opd-datasets/coderforge_assistant_turns_ctx8k_tgt256_qwen36_pilot64`,
  `OPD_PROMPT_DATASET_COLUMN=prompt_ids`, `OPD_PROMPT_DATASET_PROMPT_LEN=8192`,
  `OPD_MAX_NEW_TOKENS=256`, `OPD_FULL_FT_LR=0.000001`.
- Failure: `trainer-1` landed on `research-common-h100-105` with `CUDA_VISIBLE_DEVICES=0,1,2,3` and DCP load OOMed
  during meta-tensor materialization because those GPUs were already effectively full.
- Result: deleted the StatefulSet, student job, teacher job, and services before any valid profile row. Do not treat this
  as a model/data failure.

`q36cf8k4full-06042035`

- Attempted clean 4-node x 8-GPU full-node Run 1 after extra 8-GPU nodes appeared.
- Failure: the long run ID caused a generated StatefulSet selector/controller label to exceed Kubernetes' 63-character
  label value limit.
- Result: deleted and relaunched with shorter ID `q36k4a06042041`.

`q36k4a06042041`

- Completed Qwen3.6 Run 1 fit/debug ladder.
- Coord dir:
  `/shared/opd-coord/q36k4a06042041`
- Rendered manifests:
  `/shared/opd-coord/q36k4a06042041/rendered/`
- Profile:
  `/shared/opd-coord/q36k4a06042041/artifacts/qwen36_confadapt_4node_opd_profile.jsonl`
- W&B run after latest relaunch:
  `https://wandb.ai/together-research/singleshot/runs/efkgogbi`
- Placement:
  - trainer pods on `research-common-h100-117`, `116`, `040`, and `047`.
  - student SGLang on `research-common-h100-105`.
  - teacher SGLang on `research-common-h100-041`.
- Current fit settings:
  - Dataset:
    `/shared/opd-datasets/coderforge_assistant_turns_ctx8k_tgt256_qwen36_pilot64`
  - `OPD_PROMPT_DATASET_COLUMN=prompt_ids`
  - `OPD_PROMPT_DATASET_PROMPT_LEN=1024`
  - `OPD_MAX_NEW_TOKENS=64`
  - `OPD_MTP_K_TOKS=4`
  - `OPD_MTP_CONF_THRESHOLD=0.6`
  - `OPD_NUM_STEPS=3`
  - `OPD_MAX_OPD_STEPS=3`
- Failure/fix ladder before the current successful rows:
  - Batch size 4 failed in student SGLang with insufficient Mamba-cache space; switched prompt/chunk/prefetch to 1.
  - Teacher hidden-cache initially failed because chunked prefill returned only the final chunk; increased teacher
    chunked prefill/prefill limits to 9216.
  - Teacher initially consumed a trainer node; pinned teacher away from trainer full nodes.
  - Trainer pod scheduling initially conflicted with mutated `node-pool=compute`; set trainer `nodeSelector` to
    `node-group: default`.
  - ConfAdapt trace cap initially requested only enough rows for fixed-k blocks and failed coverage for 256 generated
    tokens. `run_opd_pipeline.py` now requests `max_new_tokens` native trace rows for ConfAdapt because a row can commit
    only one token.
  - Non-owning local ranks failed native trace replay on all-ignore rows. `src/xorl/mtp/singleshot.py` now treats those
    rows as inert.
  - One-token `<|im_end|>` generations produced bootstrap-only rows with no native trace steps. The native-trace replay
    preparer now accepts `generated_count == 1` with an empty trace.
  - Memory ladder: 256 new tokens OOMed; 64 new tokens with full 4719-token prompt OOMed; 64 new tokens with 2048-token
    prompt OOMed; 64 new tokens with 1024-token prompt fits.
- Successful profile rows at latest check:
  - Step 0: `loss=0.447278`, `valid_tokens=512`, `step_total_s=354.497`, `teacher_prefill_s=117.584`,
    `forward_backward_roundtrip_s=163.786`, `sync_inference_weights_s=61.782`, `trace_steps=17`,
    `trace_committed=65`, replay coverage true, sync success true.
  - Step 1: `loss=2.027064`, `valid_tokens=512`, `step_total_s=108.424`, `teacher_prefill_s=0.667`,
    `forward_backward_roundtrip_s=80.262`, `sync_inference_weights_s=20.586`, `trace_steps=17`,
    `trace_committed=65`, replay coverage true, sync success true.
  - Step 2: `loss=1.232681`, `valid_tokens=512`, `step_total_s=359.901`, `teacher_prefill_s=0.599`,
    `forward_backward_roundtrip_s=326.871`, `sync_inference_weights_s=20.583`, `trace_steps=63`,
    `trace_committed=63`, replay coverage true, sync success true, `rollout/max_repeated_token_run=1`, and
    `rollout/unique_token_fraction=0.484375`.
- Cleanup: deleted the trainer StatefulSet, student job, teacher job, and services after the third row landed.
- Conclusion: this is the first retokenized assistant-turn Qwen3.6 run to clear native SGLang MTP sampling, trainer
  native-trace replay, optimizer step, and weight sync on a 4x8 trainer topology. It is a memory/correctness fit, not a
  quality result.

`q36k4b06042218`

- Completed 10-step low-LR follow-up at the proven 1024/64 fit point.
- Coord dir:
  `/shared/opd-coord/q36k4b06042218`
- Rendered manifests:
  `/shared/opd-coord/q36k4b06042218/rendered/`
- Profile:
  `/shared/opd-coord/q36k4b06042218/artifacts/qwen36_confadapt_4node_opd_profile.jsonl`
- W&B run:
  `https://wandb.ai/together-research/singleshot/runs/0wd45fqk`
- Placement:
  - trainer pods on `research-common-h100-040`, `116`, `047`, and `049`.
  - student SGLang on `research-common-h100-105`.
  - teacher SGLang on `research-common-h100-041`.
- Settings:
  - Dataset:
    `/shared/opd-datasets/coderforge_assistant_turns_ctx8k_tgt256_qwen36_pilot64`
  - `OPD_PROMPT_DATASET_COLUMN=prompt_ids`
  - `OPD_PROMPT_DATASET_PROMPT_LEN=1024`
  - `OPD_MAX_NEW_TOKENS=64`
  - `OPD_MTP_K_TOKS=4`
  - `OPD_MTP_CONF_THRESHOLD=0.6`
  - `OPD_NUM_STEPS=10`
  - `OPD_MAX_OPD_STEPS=10`
  - `OPD_OPTIM_LR=0.000001`
  - `OPD_FULL_FT_LR=0.000001`
- Gate result:
  - 10/10 rows completed.
  - All rows used `learning_rate=1e-6`.
  - Native trace coverage true on every row.
  - Weight sync success true on every row.
  - No trainer-0 error patterns were detected by the monitor loop.
- Timing summary:
  - mean `step_total_s=245.9`, min/max `59.7/361.4`.
  - mean `forward_backward_roundtrip_s=205.9`, min/max `38.4/334.2`.
  - mean `teacher_prefill_s=12.45`; this is dominated by cold step 0 at `118.5s`, with warm prefill around
    `0.56-1.06s`.
  - mean `sync_inference_weights_s=19.37`, with cold step 0 at `52.15s` and later sync around `13.6-16.9s`.
- Behavior summary:
  - mean loss `0.7867`, min/max `0.0381/4.6066`.
  - `rollout/max_repeated_token_run >= 16` on 3/10 rows.
  - `rollout/max_repeated_token_run=1` on 7/10 rows.
  - mean `rollout/unique_token_fraction=0.369`.
  - Trace-step counts: `{4: 1, 17: 3, 18: 1, 63: 5}`. Rows with `trace_steps=63` were the slow rows.
- Cleanup: deleted the trainer StatefulSet, student job, teacher job, and services after row 10 landed; pods were gone at
  the final check.
- Conclusion: the 1024/64 topology is stable enough for a longer low-LR smoke. It is still not a quality run because the
  pilot64 dataset is tiny and repeat behavior is mixed.

`q36k4c06042308`

- Failed 1536/64 memory-boundary probe.
- Coord dir:
  `/shared/opd-coord/q36k4c06042308`
- Rendered manifests:
  `/shared/opd-coord/q36k4c06042308/rendered/`
- Profile path:
  `/shared/opd-coord/q36k4c06042308/artifacts/qwen36_confadapt_4node_opd_profile.jsonl`
- W&B run:
  `https://wandb.ai/together-research/singleshot/runs/rhuljm50`
- Placement:
  - trainer pods on `research-common-h100-071`, `047`, `040`, and `117`.
  - student SGLang on `research-common-h100-105`.
  - teacher SGLang on `research-common-h100-041`.
- Settings:
  - Dataset:
    `/shared/opd-datasets/coderforge_assistant_turns_ctx8k_tgt256_qwen36_pilot64`
  - `OPD_PROMPT_DATASET_COLUMN=prompt_ids`
  - `OPD_PROMPT_DATASET_PROMPT_LEN=1536`
  - `OPD_MAX_NEW_TOKENS=64`
  - `OPD_MTP_K_TOKS=4`
  - `OPD_MTP_CONF_THRESHOLD=0.6`
  - `OPD_NUM_STEPS=3`
  - `OPD_MAX_OPD_STEPS=3`
  - `OPD_OPTIM_LR=0.000001`
  - `OPD_FULL_FT_LR=0.000001`
- Failure:
  - Step 0 started and wrote `rollout_samples.jsonl` plus `teacher_hidden_step0.safetensors`.
  - Student sampling took `5.858s` for 64 new tokens.
  - Teacher prefill took `119.152s` for 1599 tokens.
  - Trainer packed 1 sample into a 1664-token batch and OOMed during forward/backward before any profile row was
    written.
  - Error: `torch.OutOfMemoryError` in `src/xorl/ops/linear_attention/modules/fused_norm_gate.py`, allocating
    `2.28 GiB` with `2.17 GiB` free on rank 0 GPU 0.
- Cleanup: deleted the trainer StatefulSet, student job, teacher job, and services; pods were gone at the final check.
- Conclusion: 1536 prompt tokens does not fit the current 4x8 trainer at 64 generated tokens without a memory change.

## What Is Working

- Static SingleShot tensor geometry is validated against the original implementation.
- Flex/block-mask training can run forward/backward on GPU.
- W&B and JSONL logging cover the critical metrics for training rollout acceptance, entropy, agreement, and repetition.
- Learned MTP token setup works without resizing Qwen's physical vocab beyond reserved capacity.
- Checkpoint save/load works.
- Low-k curriculum improves low-k behavior.
- The 10k MetaMath run achieved a useful confidence-adaptive decode frontier near effective k 2.1 to 2.4 with low repeat
  rates.
- ConfAdapt-training implementation is in place and r5 is the first correct long run for it.
- Qwen3.6 SingleShot replay can run through the mixed full-attention/linear-attention stack with the GatedDeltaNet mask
  path, native MTP debug trace validation, optimizer stepping, and weight sync.
- Qwen3.6 assistant-turn prompts retokenized with the Qwen3.6 tokenizer can drive OPD from `prompt_ids`.
- Native ConfAdapt trace replay now handles empty local ranks and bootstrap-only one-token generations.
- The 1024/64 Qwen3.6 fit point is stable for at least 10 optimizer/sync steps at `OPD_OPTIM_LR=1e-6`.
- Streaming Qwen3.6 assistant-turn materialization works beyond the pilot bucket; the 16-shard Coderforge checkpoint has
  130429 usable assistant-turn rows with zero malformed markers.
- A self-contained Qwen3.6 reprogrammable-slot/SMG throughput plan now exists at
  `experiments/opd_profile/runbooks/q36_singleshot_reprogrammable_smg_runbook.md`.

## What Did Not Work

- Full-vocab reverse-KL OPD did not reproduce the paper target and collapsed.
- Online poisoned-prefix hard CE did not prevent repeated-token attractors.
- Fixed static high-k decode is not yet a usable deployment policy.
- Loss, hard CE, and teacher/student top-1 agreement alone are insufficient success metrics.
- ConfAdapt threshold `0.9` from cold start does not bootstrap deeper MTP positions.
- The q-len-specific static CUDA graph path is correctness-risky for batched native MTP.
- Selecting the highest-free GPU inside a privileged pod can violate Kubernetes GPU assignment.
- The nominal 8k prompt / 256 new-token Qwen3.6 Run 1 target does not fit the current 4x8 trainer configuration. The
  current fit point is 1024 prompt tokens and 64 generated tokens; 1536/64 also OOMs.
- Setting `OPD_FULL_FT_LR` alone is not enough for OPD runs; the optimizer-step request uses `OPD_OPTIM_LR` and otherwise
  defaults to `1e-4`.
- The one-off `q36k4d06042335` topology is not an adequate throughput substrate: it uses one prompt per step, one
  student sampler, one teacher endpoint, no pipeline, and only one synced sampler endpoint.
- Native SingleShot MTP routing through SMG is not proven yet. The local driver uses raw SGLang `/generate`,
  `/model_info`, and `/teacher_hidden_cache`, while the documented SMG path was validated for OpenAI-compatible chat
  completions.

## Current Hypotheses

1. ConfAdapt should be used during training, not only during decode. The student should train on what it would actually
   accept, and the backward pass should replay that accepted structure.
2. A fixed threshold of `0.6` may bootstrap deeper positions, after which the threshold should possibly be annealed toward
   `0.9` or `0.95`.
3. Checkpoint selection should use decode frontier metrics, not fixed k8 CE alone.
4. Curriculum remains promising: `k in [2,4]`, then `[2,8]`, then `[2,16]`, or an equivalent confidence-threshold schedule.
5. Static high-k failure is acceptable as a diagnostic if confidence-adaptive decode quality and effective k improve.
6. Qwen3.6-35B-A3B should be the next Coderforge target. The small-model MetaMath gate is no longer a blocker for the
   initial Qwen3.6 assistant-turn data audit, fit smoke, and 8k/16k/32k scale ladder.

## Promotion Metrics

For training:

- `loss`, but only as a weak optimization sanity check.
- `accepted_k_mean`, `accepted_k_min`, `accepted_k_max`.
- `accepted_pos2_rate`, `accepted_pos3_rate`, `accepted_pos4_rate`, `accepted_pos8_rate`.
- `accepted_token_rate`.
- `student_entropy`, `teacher_entropy`, and accepted-only variants.
- `student_top1_conf`, `teacher_top1_conf`, accepted-only variants.
- `student_teacher_top1_agreement_accepted`.
- `student_region_max_repeat_run`, `student_regions_with_full_repeat_rate`.
- `grad_norm`, `step_time_s`, and W&B logging liveness.

For decode promotion:

- Effective k under ConfAdapt, especially k8/conf0.9 and k8/conf0.95.
- Teacher NLL of student continuation.
- Teacher argmax match.
- Repeat metrics.
- Acceptance by position.
- Static k decode only as a stress test and regression detector.

Stop or investigate if:

- W&B is absent or stale.
- `accepted_k_mean` is pinned near 1.0 for a long window when the purpose is deeper training.
- `accepted_pos2_rate` stays near zero after warmup.
- Repeats increase while loss decreases.
- Teacher NLL of student decode worsens materially while fixed CE improves.
- `allocated_cuda_visible_devices` does not match `cuda_visible_devices` in `job.log`.
- Any pod lacks `team: turbo` on the pod template.

## Operating The Machinery

All commands below assume:

```bash
cd /home/apanda/xorl-mtp-singleshot-port-20260602
```

### Local Validation

Run syntax and unit checks:

```bash
.venv/bin/python -m py_compile scripts/opd/train_static_singleshot_mtp_smoke.py
.venv/bin/ruff check scripts/opd/train_static_singleshot_mtp_smoke.py
.venv/bin/python -m pytest tests/mtp/test_singleshot.py -q
```

### Check The Active Run

```bash
kubectl get job,pods -n apanda -l run-id=q34meta-static-lmtp-ga32-10k-r5-conf06-0604 -o wide
tail -n 80 /shared/opd-coord/static-mtp-smoke/q34meta-static-lmtp-ga32-10k-r5-conf06-0604/job.log
tail -n 1 /shared/opd-coord/static-mtp-smoke/q34meta-static-lmtp-ga32-10k-r5-conf06-0604/metrics.jsonl \
  | .venv/bin/python -m json.tool
```

Summarize early acceptance metrics:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

run = "q34meta-static-lmtp-ga32-10k-r5-conf06-0604"
path = Path("/shared/opd-coord/static-mtp-smoke") / run / "metrics.jsonl"
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
train = [r for r in rows if "eval_k_toks" not in r and "accepted_k_mean" in r]
print("rows", len(train), "last_step", train[-1]["step"] if train else None)
for key in ["loss", "accepted_k_mean", "accepted_pos2_rate", "accepted_pos3_rate", "accepted_pos4_rate"]:
    vals = [r[key] for r in train if r.get(key) is not None]
    if vals:
        print(key, "first", vals[0], "last", vals[-1], "mean", sum(vals) / len(vals), "min", min(vals), "max", max(vals))
PY
```

### Launch A New Historical Qwen3-4B ConfAdapt Training Run

This section is retained for small-model diagnostics. For the next Coderforge work, use the Qwen3.6 launch template
above instead. Use a unique run ID. Keep `team: turbo` on the pod template. Prefer the plugin GPU UUID in the manifest.

Dry run:

```bash
RUN_ID=q34meta-static-lmtp-ga32-10k-r6-conf06-0604
perl -pe "s/q34meta-static-lmtp-ga32-10k-0604/${RUN_ID}/g; s/STATIC_MTP_TRAIN_CONFIDENCE_THRESHOLD:-0\\.9/STATIC_MTP_TRAIN_CONFIDENCE_THRESHOLD:-0.6/g" \
  experiments/opd_profile/k8s/qwen3_4b_static_mtp_metamath_gradaccum_10k.yaml \
  | kubectl apply -n apanda --dry-run=server -f -
```

Launch:

```bash
RUN_ID=q34meta-static-lmtp-ga32-10k-r6-conf06-0604
perl -pe "s/q34meta-static-lmtp-ga32-10k-0604/${RUN_ID}/g; s/STATIC_MTP_TRAIN_CONFIDENCE_THRESHOLD:-0\\.9/STATIC_MTP_TRAIN_CONFIDENCE_THRESHOLD:-0.6/g" \
  experiments/opd_profile/k8s/qwen3_4b_static_mtp_metamath_gradaccum_10k.yaml \
  | kubectl apply -n apanda -f -
```

Stop a run:

```bash
kubectl delete job -n apanda q34meta-static-lmtp-ga32-10k-r6-conf06-0604
```

### Decode A Checkpoint

Decode checkpoints once a training checkpoint is worth promotion, usually at eval checkpoint steps or final. Use the decode
evaluator to compare static and ConfAdapt policies on the same prompt set.

Typical inputs:

- Checkpoint:
  `/shared/opd-coord/static-mtp-smoke/<TRAIN_RUN>/student_checkpoint`
- Eval checkpoint:
  `/shared/opd-coord/static-mtp-smoke/<TRAIN_RUN>/student_checkpoint_step4999`
- Best checkpoint:
  `/shared/opd-coord/static-mtp-smoke/<TRAIN_RUN>/student_checkpoint_best`

Run through Kubernetes by rendering the decode eval manifest with a unique decode run ID and the chosen checkpoint path.
After launch, compare k8 static, k8/conf0.9, and k8/conf0.95 first.

### Artifact Layout

Training outputs live under:

```text
/shared/opd-coord/static-mtp-smoke/<RUN_ID>/
```

Important files:

- `job.log`: launch, GPU selection, W&B startup, stack traces.
- `metrics.jsonl`: train and fixed-eval scalar metrics.
- `examples.jsonl`: sampled examples, accepted counts, proposal confidence.
- `student_checkpoint/`: final checkpoint.
- `student_checkpoint_best/`: fixed-eval selected checkpoint.
- `student_checkpoint_step<N>/`: saved eval checkpoints when configured.
- `best_checkpoint_summary.json`: fixed-eval best checkpoint metadata.

Decode outputs live under:

```text
/shared/opd-coord/static-mtp-decode-eval/<DECODE_RUN_ID>/
```

## Autoresearch Loop

1. Preflight code and manifest.
2. Launch a small or long run with W&B on.
3. Confirm scheduling and GPU assignment from `job.log`.
4. Confirm W&B and JSONL metrics within the first few steps.
5. Watch acceptance metrics, entropy, repeats, and loss. Do not use loss alone.
6. Save/evaluate checkpoints at configured intervals.
7. Decode candidate checkpoints with static and ConfAdapt strategies.
8. Promote by ConfAdapt quality frontier: effective k at acceptable teacher NLL, argmax match, and repeat metrics.
9. Stop or alter the run when metrics show the hypothesis is false.
10. Record the decision in this runbook or the detailed debug log before launching the next variant.

Current loop for r5:

- Let it reach at least the first meaningful eval/checkpoint unless metrics collapse earlier.
- At checkpoint steps, decode with k8/conf0.9 and k8/conf0.95.
- Compare against final9999 from `q34meta-static-lmtp-ga32-10k-0604`.
- If threshold `0.6` improves accepted depth but hurts decode quality, try a threshold schedule rather than a fixed
  threshold.
- If it does not improve accepted depth, return to a low-k curriculum or lower threshold only for early warmup.

## Next Experiments

Most useful next actions:

1. Use the closed `q36k4d06042335` artifacts only as baseline/failure evidence; its one-off resources have been deleted.
2. Use the 16-shard checkpoint for the next throughput-shaped run, but drive it through the reprogrammable-slot workflow
   rather than another one-off topology.
3. Probe native SingleShot MTP compatibility through SMG, or implement direct native fanout in `run_opd_pipeline.py`.
4. Port or derive `experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py` from the existing Q36
   reprogrammable generator so trainer controls launch this repo's native-MTP OPD driver.
5. Run the first two-sampler reprogrammable smoke with `prompts_per_step=2`, `pipeline_chunk_size=1`,
   `pipeline_prefetch_chunks=2`, `teacher_concurrency=1`, and `sampler_layout=spare-teacher1`.
6. Continue materialization toward the full uncapped all-shard Qwen3.6-retokenized assistant-turn dataset. Do not use the
   pilot64 dataset for quality.
7. Probe the next memory boundary only after changing a memory lever. Good candidates are `1024/96`, shorter
   SingleShot truncation length, lower `pad_to_multiple`, checkpointing/offload changes, or a topology that reduces the
   linear-attention forward/backward memory pressure. Do not retry `1536/64` unchanged.
8. Use the 10-step `q36k4b06042218` timing evidence to prioritize reducing the `trace_steps=63` trainer roundtrip, since
   those rows spend about 311-334s in trainer forward/backward.
9. Re-try the original Run 1 target (`ctx8k`, 256 generated tokens) only after a memory plan is explicit: shorter MTP
   training window, lower prompt cap, different checkpointing, smaller teacher/student batch, or a larger trainer
   topology that actually changes the failing memory dimension.
10. After Run 1 has a stable real fit point on the larger dataset, launch `q36-cf-16k-k8-t512-scale32` or a reduced
   16k/64 bridge run depending on the measured memory headroom.
11. Run `q36-cf-128k-rollout-eval` before any 128k training attempt.
12. Continue the Qwen3-4B r5/threshold-schedule work only as a diagnostic branch, not as a blocker for the Qwen3.6 ladder.

Do not promote a Qwen3.6 Coderforge run beyond the current rung until:

- The native MTP trace covers every generated supervised token.
- Small final-block unused committed trace tails are explained and bounded.
- Weight sync succeeds unless the run intentionally sets `OPD_SKIP_OPTIM_STEP=1`.
- The prompt dataset path/column/lengths in the profile row match the intended assistant-turn materialization.
- Effective k or accepted position depth improves without teacher NLL or repeat metrics regressing.
- The run is scheduler-correct, W&B-complete, and artifact paths are recorded here.
