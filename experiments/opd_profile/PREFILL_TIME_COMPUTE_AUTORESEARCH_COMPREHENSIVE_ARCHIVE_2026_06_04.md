# Prefill-Time-Compute Autoresearch Comprehensive Archive - 2026-06-04

This is the deliberately oversized archival companion to `PREFILL_TIME_COMPUTE_AUTORESEARCH_RUNBOOK_2026_06_04.md`. The active runbook is kept short for operating the next run; this file preserves the historical detail, removed ledgers, raw queue state, candidate YAMLs, scorecards, and source runbooks in one searchable place.


Generated/updated from local workspace state at `2026-06-04T20:15:28Z`.


## 0. How To Use This Archive


- Use the active runbook for what to launch next.
- Use this archive when reconstructing why a branch was accepted, rejected, or deprioritized.
- Nothing here should override newer queue state; if `ideas.yaml` changes after this file is written, regenerate or update this archive.
- Raw profile JSONL files are not embedded in full because they live under `/shared/.../opd_profile.jsonl`; paths are preserved in scorecards and source docs.


## 1. Source Document Index


| Source | Lines | Bytes | Role |
| --- | ---: | ---: | --- |
| `experiments/opd_profile/PREFILL_TIME_COMPUTE_AUTORESEARCH_RUNBOOK_2026_06_04.md` | 702 | 23247 | Active concise handoff and current operating procedure. |
| `experiments/opd_profile/OPD_CONFIG_A_B_RUNBOOK_2026_06_02.md` | 499 | 19571 | 235B Config A/B baseline, interpretation, and early operating hazards. |
| `experiments/opd_profile/Q36_35B_OPD_REPROGRAMMABLE_SLOTS_RUNBOOK_2026_06_02.md` | 1667 | 73659 | Qwen3.6 A-X/Y sweep, runtime fixes, active stack, and decision criteria. |
| `experiments/opd_profile/K8S_REPROGRAMMABLE_SLOTS_RUNBOOK_2026_06_03.md` | 3137 | 139992 | Reprogrammable-slot stack, Z-AO decision runs, AM/AN/AO details, hazards. |
| `experiments/opd_profile/PREFILL_TIME_COMPUTE_OPSD_RESEARCH_MEMO_2026_06_03.md` | 42 | 4133 | OPSD research position and autoresearch implication. |
| `experiments/opd_profile/PREFILL_TIME_COMPUTE_OPSD_RUNBOOK_2026_06_03.md` | 76 | 2064 | Early OPSD/autoresearch launch and scoring commands. |
| `experiments/opd_profile/ENCODED_REASONING_MASTER_RUNBOOK_2026_06_02.md` | 362 | 23462 | Early encoded-reasoning concept/runbook and baseline framing. |


## 2. Current Queue Snapshot


### `controller.py next`


```yaml
id: PTC-023
status: queued
priority: 92
candidate: candidates/PTC-023.yaml
requires:
- PTC-020
requires_statuses:
  PTC-020:
  - strong_signal
  - promote_retest
  - weak_signal
hypothesis: If AM's operational pause-vs-no-pause lift does not require hidden-state
  matching, then the same short static-filler answer-causal recipe with hidden matching
  disabled should retain a positive step-5 1k control delta and answer-logprob support.
rationale: This is the clean AM-minus-hidden ablation requested after PTC-020. It
  keeps the corrupt buffer and corrupt-answer contrastive training objective, FlashQLA,
  P2P sync, parallel endpoint sync, and the same final-only 1k operational gate, while
  setting `opd_hidden_match_coef=0.0`. Corrupt exact-match eval is skipped because
  the queue is judging pause-vs-no-pause performance.
default_num_steps: 6
default_prompts_per_step: 128
```

### Generator Status


```text
dispatch                 pid=  410613 running_since=2026-06-04T18:49:47Z pid=410613 hash=e61ee682ad9ae990885dcd4370350cf75819fac86badbdda2754992ff7f10262 log=/shared/opd-control/er-opd-q36-35b-slots/dispatch/logs/20260604T184947Z-run.log
sglang-0                 pid=  556484 running_since=2026-06-04T18:49:47Z pid=556484 hash=5c6fdcb82b473606b7c43faf65766b41816347492d21a8ed9312c9a7e7e59333 log=/shared/opd-control/er-opd-q36-35b-slots/sglang-0/logs/20260604T184947Z-run.log
sglang-1                 pid=       - stopped_at=2026-06-02T19:08:19Z
teacher-sglang-0         pid=  501863 running_since=2026-06-04T19:11:31Z pid=501863 hash=4b7082193b4252d496909c26f657c94098a855953bb52967bead42d161488dbb log=/shared/opd-control/er-opd-q36-35b-slots/teacher-sglang-0/logs/20260604T191131Z-run.log
teacher-sglang-1         pid=  456939 running_since=2026-06-04T18:49:48Z pid=456939 hash=11fbc47eebe96067ab6c675cd875399e1baf479e2931c191fe27e5718c016833 log=/shared/opd-control/er-opd-q36-35b-slots/teacher-sglang-1/logs/20260604T184948Z-run.log
teacher-smg              pid=       - stopped_at=2026-06-04T18:49:41Z
trainer-head             pid=       - stopped_at=2026-06-04T19:44:28Z
trainer-worker-1         pid=       - stopped_at=2026-06-04T19:44:28Z
trainer-worker-2         pid=       - stopped_at=2026-06-04T19:44:27Z
trainer-worker-3         pid=       - stopped_at=2026-06-04T19:44:28Z
trainer-worker-4         pid=       - stopped_at=2026-06-04T19:44:27Z
trainer-worker-5         pid=       - stopped_at=2026-06-04T19:44:28Z
trainer-worker-6         pid=       - stopped_at=2026-06-04T19:44:28Z
trainer-worker-7         pid=       - stopped_at=2026-06-04T19:44:28Z
```

### Candidate / Idea Summary


| ID | Status | Priority | Steps | Prompts/step | Base | Score | Hidden | Corrupt buf | Corrupt ans | Positive ans | Sync | Serial endpoints | Eval corrupt | Last verdict | Last reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| PTC-001 | infra_invalid | 100 | 9 | 128 | AN |  | 0.0 | 0.0 | 0.0 | 0.0 | nccl_broadcast |  | False | infra_invalid | request_failure_frac_max=0.3720 |
| PTC-002 | rejected | 95 | 9 | 128 | AN |  | 0.0 | 0.0 | 0.0 | 0.0 | nccl_broadcast |  | False | superseded | Superseded by PTC-006, which tested the same gold-answer objective on the FlashQLA and fast-sync path and was science_reject. |
| PTC-006 | rejected | 96 | 9 | 128 | AN |  | 0.0 | 0.0 | 0.0 | 0.0 | nccl_broadcast |  | False | science_reject | no positive accuracy or answer-logprob signal |
| PTC-003 | rejected | 90 | 9 | 128 | AN |  | 0.5 | 0.0 | 0.0 | 0.0 | nccl_broadcast |  | False | science_reject | no positive accuracy or answer-logprob signal |
| PTC-004 | rejected | 70 | 13 | 256 | AN |  | 0.0 | 0.0 | 0.0 | 0.0 | nccl_broadcast |  | False | skipped_precondition_failed | Not launched because PTC-006 and PTC-003 were clean FlashQLA runs but both had negative accuracy deltas and negative answer-logprob margins. |
| PTC-007 | rejected | 94 | 9 | 128 | AN |  | 0.0 | 0.0 | 0.0 | 0.0 | nccl_broadcast |  | False | science_reject | no positive accuracy or answer-logprob signal |
| PTC-008 | rejected | 93 | 9 | 128 | AN |  | 0.0 | 1.0 | 0.125 | 0.0 | nccl_broadcast |  | True | science_reject | no positive accuracy or answer-logprob signal |
| PTC-009 | rejected | 92 | 9 | 128 | AN |  | 0.5 | 0.0 | 0.0 | 1.0 | nccl_broadcast |  | True | science_reject | no positive accuracy or answer-logprob signal |
| PTC-010 | rejected | 91 | 9 | 128 | AN |  | 1.0 | 0.0 | 0.0 | 0.0 | nccl_broadcast |  | True | science_reject | no positive accuracy or answer-logprob signal |
| PTC-011 | inconclusive | 92 | 9 | 128 | AN |  | 1.0 | 0.0 | 0.0 | 0.0 | nccl_broadcast |  | True | inconclusive | completed cleanly but missed both reject and promotion thresholds |
| PTC-012 | rejected | 90 | 9 | 128 | AN |  | 0.0 | 0.0 | 0.25 | 0.75 | nccl_broadcast |  | True | inconclusive | Step-8 accuracy delta decayed to +0.0195 while answer logprob and answer selection worsened; non-promotable recipe despite shuffled-memory control still proving generated memory carries signal. |
| PTC-013 | rejected | 89 | 9 | 128 | AN |  | 0.0 | 0.0 | 0.25 | 0.75 | nccl_broadcast |  | True | science_reject | no positive accuracy or answer-logprob signal |
| PTC-014 | rejected | 88 | 9 | 128 | AN |  | 0.0 | 0.0 | 0.0 | 1.0 | nccl_broadcast |  | True | science_reject | no positive accuracy or answer-logprob signal |
| PTC-015 | rejected | 87 | 9 | 128 | AN |  | 0.0 | 0.0 | 0.0 | 2.0 | nccl_broadcast |  | True | inconclusive | completed cleanly but missed both reject and promotion thresholds |
| PTC-016 | rejected | 86 | 11 | 128 | AN |  | 1.0 | 0.0 | 0.0 | 2.0 | nccl_broadcast |  | True | early_science_reject | Step-5 control rejected the hidden-anchor hypothesis despite the controller's generic inconclusive label: buffer_delta was only +0.0078, corrupt-pause accuracy was 0.5391, answer_logprob_margin was negative, and answer_select_delta was strongly negative. |
| PTC-017 | rejected | 85 | 11 | 128 | AN |  | 1.0 | 0.0 | 0.0 | 2.0 | nccl_broadcast |  | True | superseded | PTC-016 step-5 control already showed hidden anchoring at 0.5 made corrupt-pause accuracy high and answer-selection strongly negative; doubling the same hidden-only pressure is a poor next experiment. |
| PTC-018 | rejected | 86 | 11 | 128 | AN |  | 0.0 | 0.0 | 0.0 | 1.5 | nccl_broadcast |  | True | science_reject | no positive accuracy or answer-logprob signal |
| PTC-019 | infra_invalid | 86 | 11 | 128 | AN |  | 0.0 | 0.0 | 0.0 | 1.0 | nccl_broadcast |  | True | infra_invalid | request_failure_frac_max=0.0039 |
| PTC-020 | promote_retest | 90 | 6 | 128 | AM | pause_vs_nopause | 2.0 | 1.0 | 0.125 | 0.0 | p2p | False | True | promote_retest | pause-vs-nopause accuracy warrants retest with positive answer-logprob support |
| PTC-021 | queued | 89 | 11 | 128 | AM | pause_vs_nopause | 2.0 | 1.0 | 0.125 | 0.0 | p2p | False | True |  |  |
| PTC-022 | queued | 88 | 6 | 256 | AM | pause_vs_nopause | 2.0 | 1.0 | 0.125 | 0.0 | p2p | False | True |  |  |
| PTC-023 | queued | 92 | 6 | 128 | AM | pause_vs_nopause | 0.0 | 1.0 | 0.125 | 0.0 | p2p | False | False |  |  |
| PTC-024 | queued | 91 | 6 | 128 | AM | pause_vs_nopause | 2.0 | 0.0 | 0.0 | 0.125 | p2p | False | False |  |  |
| PTC-025 | queued | 90 | 6 | 128 | AM | pause_vs_nopause | 0.0 | 0.0 | 0.0 | 0.125 | p2p | False | False |  |  |


### Scorecard Summary


| Scorecard | Verdict | Step | N | Delta | Z | Answer margin | Answer Z | Profile |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `20260604T011532Z-PTC-001-step8-infra_invalid.json` | infra_invalid | 8 | 1000.0 | -0.006000000000000005 | -0.33863625839560096 | 0.0011698661124863946 | 0.5921812725913325 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T225859Z-configPTC-001-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T011532Z-PTC-005-step8-infra_invalid.json` | infra_invalid | 8 | 1000.0 | 0.0 | 0.0 | -0.004177647228492835 | -2.164316609339693 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T000006Z-configPTC-005-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T011537Z-PTC-001-step8-infra_invalid.json` | infra_invalid | 8 | 1000.0 | -0.006000000000000005 | -0.33863625839560096 | 0.0011698661124863946 | 0.5921812725913325 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T225859Z-configPTC-001-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T021202Z-PTC-006-step8-science_reject.json` | science_reject | 8 | 1000.0 | -0.0050000000000000044 | -0.28471338904946075 | -0.0005920925329430714 | -0.3690380389251624 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T013603Z-configPTC-006-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T021205Z-PTC-006-step8-science_reject.json` | science_reject | 8 | 1000.0 | -0.0050000000000000044 | -0.28471338904946075 | -0.0005920925329430714 | -0.3690380389251624 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T013603Z-configPTC-006-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T045539Z-PTC-003-step8-science_reject.json` | science_reject | 8 | 1000.0 | -0.006000000000000005 | -0.3354196304347396 | -0.002851773091628856 | -1.550627938419657 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T021246Z-configPTC-003-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T045543Z-PTC-003-step8-science_reject.json` | science_reject | 8 | 1000.0 | -0.006000000000000005 | -0.3354196304347396 | -0.002851773091628856 | -1.550627938419657 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T021246Z-configPTC-003-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T053033Z-PTC-007-step4-science_reject.json` | science_reject | 4 | 1000.0 | -0.08899999999999997 | -4.461643350547222 | -0.05189284104206759 | -8.495199167432943 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T050810Z-configPTC-007-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T060104Z-PTC-008-step4-science_reject.json` | science_reject | 4 | 1000.0 | -0.08599999999999997 | -4.097450014496387 | -0.004445023867406381 | -0.7627423018328828 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T053133Z-configPTC-008-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T063058Z-PTC-009-step4-science_reject.json` | science_reject | 4 | 1000.0 | -0.08200000000000007 | -4.313173687822969 | -0.06802329225159871 | -7.5156321540708575 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T060143Z-configPTC-009-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T070229Z-PTC-010-step4-science_reject.json` | science_reject | 4 | 1000.0 | -0.10499999999999998 | -5.367891918486458 | -0.07165765871649975 | -8.41706506088195 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T063316Z-configPTC-010-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T074639Z-PTC-011-stepna-incomplete.json` | incomplete |  |  |  |  |  |  | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T074650Z-PTC-011-stepna-incomplete.json` | incomplete |  |  |  |  |  |  | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T074659Z-PTC-011-stepna-incomplete.json` | incomplete |  |  |  |  |  |  | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T075411Z-PTC-011-stepna-incomplete.json` | incomplete |  |  |  |  |  |  | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T075543Z-PTC-011-stepna-incomplete.json` | incomplete |  |  |  |  |  |  | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T075929Z-PTC-011-stepna-incomplete.json` | incomplete |  |  |  |  |  |  | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T075957Z-PTC-011-stepna-incomplete.json` | incomplete |  |  |  |  |  |  | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T080016Z-PTC-011-stepna-incomplete.json` | incomplete |  |  |  |  |  |  | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T082853Z-PTC-011-step4-inconclusive.json` | inconclusive | 4 | 256.0 | 0.046875 | 1.2467006799461504 | -0.1338996193211561 | -4.934445616493985 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T082854Z-PTC-011-step4-inconclusive.json` | inconclusive | 4 | 256.0 | 0.046875 | 1.2467006799461504 | -0.1338996193211561 | -4.934445616493985 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T091135Z-PTC-011-step8-inconclusive.json` | inconclusive | 8 | 256.0 | 0.02734375 | 0.7666379087775477 | -0.027456654039798866 | -2.1209689010269392 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T091212Z-PTC-011-step8-inconclusive.json` | inconclusive | 8 | 256.0 | 0.02734375 | 0.7666379087775477 | -0.027456654039798866 | -2.1209689010269392 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T100716Z-PTC-012-step4-inconclusive.json` | inconclusive | 4 | 256.0 | 0.109375 | 3.0399616129719975 | -0.11067124789628861 | -3.0825148106625977 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T091215Z-configPTC-012-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T100734Z-PTC-012-step4-inconclusive.json` | inconclusive | 4 | 256.0 | 0.109375 | 3.0399616129719975 | -0.11067124789628861 | -3.0825148106625977 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T091215Z-configPTC-012-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T105011Z-PTC-012-step8-inconclusive.json` | inconclusive | 8 | 256.0 | 0.01953125 | 0.5474446288295078 | -0.1596965955282831 | -5.496740775541098 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T091215Z-configPTC-012-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T105043Z-PTC-012-step8-inconclusive.json` | inconclusive | 8 | 256.0 | 0.01953125 | 0.5474446288295078 | -0.1596965955282831 | -5.496740775541098 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T091215Z-configPTC-012-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T114657Z-PTC-013-step4-science_reject.json` | science_reject | 4 | 256.0 | -0.15234375 | -3.7259874043167356 | -0.31010046812613806 | -9.973705393936111 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T105110Z-configPTC-013-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T114724Z-PTC-013-step4-science_reject.json` | science_reject | 4 | 256.0 | -0.15234375 | -3.7259874043167356 | -0.31010046812613806 | -9.973705393936111 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T105110Z-configPTC-013-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T124228Z-PTC-014-step4-inconclusive.json` | inconclusive | 4 | 256.0 | 0.13671875 | 3.7110514492390805 | -0.06823835960594374 | -2.393474203845524 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T114759Z-configPTC-014-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T124319Z-PTC-014-step4-inconclusive.json` | inconclusive | 4 | 256.0 | 0.13671875 | 3.7110514492390805 | -0.06823835960594374 | -2.393474203845524 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T114759Z-configPTC-014-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T132437Z-PTC-014-step8-science_reject.json` | science_reject | 8 | 256.0 | -0.0078125 | -0.20866508438634226 | -0.10689015139499593 | -5.096342933926368 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T114759Z-configPTC-014-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T132455Z-PTC-014-step8-science_reject.json` | science_reject | 8 | 256.0 | -0.0078125 | -0.20866508438634226 | -0.10689015139499593 | -5.096342933926368 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T114759Z-configPTC-014-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T142024Z-PTC-015-step4-inconclusive.json` | inconclusive | 4 | 256.0 | 0.03515625 | 1.000506750675012 | 0.006929151958554657 | 0.4353864050668189 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T132457Z-configPTC-015-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T142105Z-PTC-015-step4-inconclusive.json` | inconclusive | 4 | 256.0 | 0.03515625 | 1.000506750675012 | 0.006929151958554657 | 0.4353864050668189 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T132457Z-configPTC-015-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T142112Z-PTC-015-step4-inconclusive.json` | inconclusive | 4 | 256.0 | 0.03515625 | 1.000506750675012 | 0.006929151958554657 | 0.4353864050668189 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T132457Z-configPTC-015-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T150312Z-PTC-015-step8-inconclusive.json` | inconclusive | 8 | 256.0 | 0.0234375 | 0.6852250139532997 | -0.044649495764718904 | -1.9204682948730094 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T132457Z-configPTC-015-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T160508Z-PTC-016-step5-inconclusive.json` | inconclusive | 5 | 256.0 | 0.0078125 | 0.19659669487640224 | -0.023487769558334422 | -1.8648381345662142 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T150313Z-configPTC-016-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T170915Z-PTC-018-step5-science_reject.json` | science_reject | 5 | 256.0 | -0.00390625 | -0.10869775914448827 | -0.14569518343434454 | -4.566378284576971 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T160640Z-configPTC-018-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T170927Z-PTC-018-step5-science_reject.json` | science_reject | 5 | 256.0 | -0.00390625 | -0.10869775914448827 | -0.14569518343434454 | -4.566378284576971 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T160640Z-configPTC-018-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T182846Z-PTC-019-step5-infra_invalid.json` | infra_invalid | 5 | 256.0 | 0.03515625 | 0.8798334176828828 | -0.29823434144685007 | -7.985485988385205 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T171139Z-configPTC-019-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T182857Z-PTC-019-step5-infra_invalid.json` | infra_invalid | 5 | 256.0 | 0.03515625 | 0.8798334176828828 | -0.29823434144685007 | -7.985485988385205 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T171139Z-configPTC-019-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T182907Z-PTC-019-step5-infra_invalid.json` | infra_invalid | 5 | 256.0 | 0.03515625 | 0.8798334176828828 | -0.29823434144685007 | -7.985485988385205 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T171139Z-configPTC-019-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |
| `20260604T194427Z-PTC-020-step5-promote_retest.json` | promote_retest | 5 | 1024.0 | 0.0458984375 | 2.1189846284303124 | 0.1110034145900177 | 23.75572011440179 | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T191133Z-configPTC-020-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl` |


## 3. Current Queue And Candidate YAMLs


### `experiments/opd_profile/autoresearch/ideas.yaml`

```yaml
programme: prefill_time_compute_opsd
created_utc: '2026-06-03T00:00:00Z'
notes:
- Focus on objective-level OPSD changes before filler search or decode-compression
  work.
- Default decision eval target is 1000 examples; anything below 900 scored examples
  is incomplete.
ideas:
- id: PTC-001
  status: infra_invalid
  priority: 100
  candidate: candidates/PTC-001.yaml
  hypothesis: Pure positive OPSD KL on filler plus sampled answer can transfer teacher
    prefill computation into student filler states.
  rationale: This is the cleanest version of the user-proposed objective and removes
    prompt-position KL from the denominator.
  default_num_steps: 9
  default_prompts_per_step: 128
  last_launched_utc: '2026-06-03T22:58:59Z'
  last_num_steps: 9
  last_prompts_per_step: 128
  last_verdict: infra_invalid
  last_reason: request_failure_frac_max=0.3720
  last_profile: /shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T225859Z-configPTC-001-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl
  last_scorecard: /home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T011537Z-PTC-001-step8-infra_invalid.json
  last_scored_utc: '2026-06-04T01:15:37Z'
- id: PTC-002
  status: rejected
  priority: 95
  candidate: candidates/PTC-002.yaml
  hypothesis: Gold answer tails make the teacher continuation stable enough for the
    filler KL objective to train.
  rationale: If sampled answer noise is masking the effect, fixing the answer tail
    should improve answer-logprob support.
  default_num_steps: 9
  default_prompts_per_step: 128
  last_verdict: superseded
  last_reason: Superseded by PTC-006, which tested the same gold-answer objective
    on the FlashQLA and fast-sync path and was science_reject.
- id: PTC-006
  status: rejected
  priority: 96
  candidate: candidates/PTC-006.yaml
  hypothesis: Gold answer tails make the teacher continuation stable enough under
    the FlashQLA and faster-sync infrastructure.
  rationale: This is the PTC-002 science variant on the current throughput path, avoiding
    a fallback to the older FLA backend while testing the gold-answer hypothesis.
  default_num_steps: 9
  default_prompts_per_step: 128
  last_launched_utc: '2026-06-04T01:36:00Z'
  last_num_steps: 9
  last_prompts_per_step: 128
  last_verdict: science_reject
  last_reason: no positive accuracy or answer-logprob signal
  last_profile: /shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T013603Z-configPTC-006-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl
  last_scorecard: /home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T021205Z-PTC-006-step8-science_reject.json
  last_scored_utc: '2026-06-04T02:12:05Z'
- id: PTC-003
  status: rejected
  priority: 90
  candidate: candidates/PTC-003.yaml
  hypothesis: Filler-position hidden matching adds a useful representation target
    once prompt KL is masked out.
  rationale: Hidden matching was weak historically, but the old loss mixed it with
    other objective confounds.
  default_num_steps: 9
  default_prompts_per_step: 128
  last_launched_utc: '2026-06-04T02:12:44Z'
  last_num_steps: 9
  last_prompts_per_step: 128
  last_verdict: science_reject
  last_reason: no positive accuracy or answer-logprob signal
  last_profile: /shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T021246Z-configPTC-003-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl
  last_scorecard: /home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T045543Z-PTC-003-step8-science_reject.json
  last_scored_utc: '2026-06-04T04:55:43Z'
- id: PTC-004
  status: rejected
  priority: 70
  candidate: candidates/PTC-004.yaml
  hypothesis: A larger batch/eval rerun of the best PTC variant will reduce variance
    enough to distinguish weak signal from noise.
  rationale: Scale only after at least one earlier PTC candidate has nonnegative accuracy
    delta and strong answer-logprob support.
  default_num_steps: 13
  default_prompts_per_step: 256
  last_verdict: skipped_precondition_failed
  last_reason: Not launched because PTC-006 and PTC-003 were clean FlashQLA runs but
    both had negative accuracy deltas and negative answer-logprob margins.
- id: PTC-007
  status: rejected
  priority: 94
  candidate: candidates/PTC-007.yaml
  hypothesis: The previous 9-token filler made the latent channel too small; a 108-token
    filler may provide enough capacity for teacher prefill computation to become causally
    recoverable by the student.
  rationale: PTC-006 cleanly rejected the short-filler gold-answer KL variant, but
    the research memo explicitly treats compression as secondary until a causal signal
    exists. This retests the same objective with a much larger pause budget before
    changing hidden targets or scaling examples.
  default_num_steps: 9
  default_prompts_per_step: 128
  last_launched_utc: '2026-06-04T05:08:09Z'
  last_num_steps: 9
  last_prompts_per_step: 128
  last_verdict: science_reject
  last_reason: no positive accuracy or answer-logprob signal
  last_profile: /shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T050810Z-configPTC-007-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl
  last_scorecard: /home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T053033Z-PTC-007-step4-science_reject.json
  last_scored_utc: '2026-06-04T05:30:33Z'
- id: PTC-008
  status: rejected
  priority: 93
  candidate: candidates/PTC-008.yaml
  hypothesis: PTC-007 failed because long positive-only filler KL mostly trained easy
    filler-token imitation and diluted answer-causal credit; a RiM-invariant-style
    contrastive objective should make real memory help the answer and corrupted memory
    hurt it while keeping the 108-token capacity budget.
  rationale: This returns to the earlier AH/AI causal mechanism signal but ports it
    onto the current FlashQLA, fast-sync, long-filler path. It is not another filler
    sweep; the key change is the signed answer and corrupt-memory contrast.
  default_num_steps: 9
  default_prompts_per_step: 128
  last_launched_utc: '2026-06-04T05:31:33Z'
  last_num_steps: 9
  last_prompts_per_step: 128
  last_verdict: science_reject
  last_reason: no positive accuracy or answer-logprob signal
  last_profile: /shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T053133Z-configPTC-008-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl
  last_scorecard: /home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T060104Z-PTC-008-step4-science_reject.json
  last_scored_utc: '2026-06-04T06:01:04Z'
- id: PTC-009
  status: rejected
  priority: 92
  candidate: candidates/PTC-009.yaml
  hypothesis: PTC-008 only proved corrupted memory can be made worse; it still failed
    to make real pause beat no-pause. Same-visible-input cache mismatch with active
    hidden matching and stronger positive answer KL should pressure prompt-specific
    memory without training a visible corrupt-token anti-target.
  rationale: This is the long-filler version of the cache-mismatch/RiM invariant.
    It keeps the 108-token capacity budget, removes signed corrupt-answer KL, turns
    on hidden matching for the memory mismatch arm, and raises answer KL weight from
    0.125 to 1.0 to counter the no-pause advantage seen in PTC-007/PTC-008.
  default_num_steps: 9
  default_prompts_per_step: 128
  last_launched_utc: '2026-06-04T06:01:42Z'
  last_num_steps: 9
  last_prompts_per_step: 128
  last_verdict: science_reject
  last_reason: no positive accuracy or answer-logprob signal
  last_profile: /shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T060143Z-configPTC-009-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl
  last_scorecard: /home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T063058Z-PTC-009-step4-science_reject.json
  last_scored_utc: '2026-06-04T06:30:58Z'
- id: PTC-010
  status: rejected
  priority: 91
  candidate: candidates/PTC-010.yaml
  hypothesis: PTC-009 failed because filler-position KL gave the optimizer an easy
    non-causal target and did not force answer dependence on the learned memory. Removing
    filler-token KL while keeping hidden-state memory matching, a stronger cache-mismatch
    negative, and direct gold-answer KL should reveal whether the pause buffer can
    carry useful prompt-specific state when the easy filler imitation objective is
    gone.
  rationale: This is not another filler-length sweep. It preserves the 108-token capacity,
    FlashQLA/SMG/fast-sync path, and step-4 gate, but changes the loss surface from
    filler KL plus hidden/cache pressure to hidden-only memory plus answer KL. If
    pause still equals corrupt and trails no-pause, the fixed-slot teacher-cache target
    is likely the wrong abstraction and the loop should move to a generated-memory/RiM
    objective rather than more weight tuning.
  default_num_steps: 9
  default_prompts_per_step: 128
  last_launched_utc: '2026-06-04T06:33:14Z'
  last_num_steps: 9
  last_prompts_per_step: 128
  last_verdict: science_reject
  last_reason: no positive accuracy or answer-logprob signal
  last_profile: /shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T063316Z-configPTC-010-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl
  last_scorecard: /home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T070229Z-PTC-010-step4-science_reject.json
  last_scored_utc: '2026-06-04T07:02:29Z'
- id: PTC-011
  status: inconclusive
  priority: 92
  candidate: candidates/PTC-011.yaml
  hypothesis: The fixed-slot variants failed because the pause region was not a prompt-specific
    memory surface at the causal corruption boundary. Have the student generate 100
    memory tokens from each prompt before answering, then train hidden memory plus
    gold-answer KL with cache-mismatch pressure across the generated memory span.
  rationale: This preserves the FlashQLA/SMG/fast-sync path and the same step-4 gate,
    but replaces static filler with generated prompt-specific memory. The corrupt
    control shuffles generated memories across prompts; a real RiM-like memory should
    beat both no-pause and shuffled-memory controls on accuracy and answer logprob.
    If it still trails no-pause or barely beats shuffled memory, the teacher-cache
    hidden target is not producing causally useful externalized reasoning state. The
    generated-memory control gate uses 256 held-out prompts because each prompt first
    samples 100 exact memory tokens; the fixed-filler 1000-prompt gate would spend
    too much of the run on eval plumbing.
  default_num_steps: 9
  default_prompts_per_step: 128
  last_launched_utc: '2026-06-04T07:35:01Z'
  last_num_steps: 9
  last_prompts_per_step: 128
  last_verdict: inconclusive
  last_reason: completed cleanly but missed both reject and promotion thresholds
  last_profile: /shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl
  last_scorecard: /home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T091212Z-PTC-011-step8-inconclusive.json
  last_scored_utc: '2026-06-04T09:12:12Z'
- id: PTC-012
  status: rejected
  priority: 90
  candidate: candidates/PTC-012.yaml
  hypothesis: 'If PTC-011 rejects, the generated memory surface itself was not enough
    because the teacher-cache hidden/cache-mismatch target was still not aligned with
    answer utility. Remove hidden/cache targets and train a RiM-style answer utility
    objective: real generated memory gets strong gold-answer KL, while corrupted generated
    memory gets a small signed anti-target so answers must depend on the memory contents.'
  rationale: This keeps the 100-token prompt-specific generated memory, FlashQLA,
    SMG, fast-sync path, and generated-memory eval gate, but changes the learning
    signal from matching teacher hidden memory to making memory causally useful for
    the final answer. It should beat no-pause and shuffled-memory controls if the
    bottleneck was the hidden/cache target rather than generated-memory plumbing.
  default_num_steps: 9
  default_prompts_per_step: 128
  last_launched_utc: '2026-06-04T09:12:13Z'
  last_num_steps: 9
  last_prompts_per_step: 128
  last_verdict: inconclusive
  last_reason: Step-8 accuracy delta decayed to +0.0195 while answer logprob and answer
    selection worsened; non-promotable recipe despite shuffled-memory control still
    proving generated memory carries signal.
  last_profile: /shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T091215Z-configPTC-012-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl
  last_scorecard: /home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T105043Z-PTC-012-step8-inconclusive.json
  last_scored_utc: '2026-06-04T10:50:43Z'
- id: PTC-013
  status: rejected
  priority: 89
  candidate: candidates/PTC-013.yaml
  requires:
  - PTC-012
  hypothesis: If PTC-012 rejects, answer-only RiM utility was likely too sparse to
    teach the model how to make its generated memory useful. Add a small explicit
    teacher-conditioned memory KL bootstrap while keeping answer utility and corrupt-memory
    contrast dominant.
  rationale: 'This keeps generated 100-token prompt-specific memory and the same FlashQLA/SMG
    eval gate, but changes only the memory supervision strength: no hidden/cache target,
    no strong filler imitation, just a 0.1 buffer KL bootstrap plus the real-vs-corrupt
    answer objective. If this fails too, the generated-memory path is probably not
    receiving effective credit assignment from these supervised OPD losses.'
  default_num_steps: 9
  default_prompts_per_step: 128
  last_launched_utc: '2026-06-04T10:51:10Z'
  last_num_steps: 9
  last_prompts_per_step: 128
  last_verdict: science_reject
  last_reason: no positive accuracy or answer-logprob signal
  last_profile: /shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T105110Z-configPTC-013-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl
  last_scorecard: /home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T114724Z-PTC-013-step4-science_reject.json
  last_scored_utc: '2026-06-04T11:47:24Z'
- id: PTC-014
  status: rejected
  priority: 88
  candidate: candidates/PTC-014.yaml
  requires:
  - PTC-013
  hypothesis: PTC-012 showed that generated memory can carry causal answer signal,
    but the corrupt-answer anti-target appears to damage likelihood alignment; PTC-013
    showed that adding buffer KL makes pause worse. Remove both corrupt-answer training
    pressure and memory KL, and train only positive gold-answer utility on generated
    memory.
  rationale: This keeps the generated 100-token prompt-specific memory, FlashQLA,
    SMG, and generated-memory control gate, but changes the loss to the least adversarial
    answer-utility objective. If answer logprob remains below no-pause, the supervised
    answer-utility path is likely unable to overcome the no-pause prior without a
    different credit-assignment mechanism.
  default_num_steps: 9
  default_prompts_per_step: 128
  last_launched_utc: '2026-06-04T11:47:56Z'
  last_num_steps: 9
  last_prompts_per_step: 128
  last_verdict: science_reject
  last_reason: no positive accuracy or answer-logprob signal
  last_scored_utc: '2026-06-04T13:24:55Z'
  last_profile: /shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T114759Z-configPTC-014-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl
  last_scorecard: /home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T132455Z-PTC-014-step8-science_reject.json
- id: PTC-015
  status: rejected
  priority: 87
  candidate: candidates/PTC-015.yaml
  requires:
  - PTC-014
  hypothesis: PTC-014 showed that removing corrupt-answer and buffer-KL pressure improves
    the causal generated-memory signal, but answer logprob still lags no-pause. If
    the answer objective is underweighted rather than misdirected, doubling the positive
    gold-answer KL weight should move answer_logprob_margin and answer_select_delta
    toward or above zero while preserving the corrupt-memory collapse.
  rationale: This keeps the generated 100-token prompt-specific memory, FlashQLA,
    SMG, zero corrupt-answer pressure, zero buffer KL, and zero hidden/cache targets.
    The only substantive change from PTC-014 is answer-position teacher KL weight
    1.001 -> 2.001. If this also fails, the failure is less likely simple answer underweighting
    and more likely a credit-assignment/objective mismatch.
  default_num_steps: 9
  default_prompts_per_step: 128
  last_launched_utc: '2026-06-04T13:24:55Z'
  last_num_steps: 9
  last_prompts_per_step: 128
  last_verdict: inconclusive
  last_reason: completed cleanly but missed both reject and promotion thresholds
  last_profile: /shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T132457Z-configPTC-015-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl
  last_scorecard: /home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T150312Z-PTC-015-step8-inconclusive.json
  last_scored_utc: '2026-06-04T15:03:12Z'
- id: PTC-016
  status: rejected
  priority: 86
  candidate: candidates/PTC-016.yaml
  requires:
  - PTC-015
  hypothesis: If PTC-015 still fails, the issue is probably not just answer KL underweighting.
    PTC-011 was the only recent recipe with positive answer-selection at step 8, and
    its distinguishing useful knob was hidden-state memory shaping, not buffer KL.
    Add a small hidden-only generated-memory bootstrap to the PTC-015 answer objective
    while keeping corrupt-answer, buffer KL, and cache-mismatch negatives off.
  rationale: This tests whether the generated memory needs a teacher-state anchor
    to remain useful past the first control point. It preserves the 100 generated
    memory tokens, FlashQLA, SMG, answer weight 2.0, and zero buffer/corrupt losses.
    It changes only hidden memory pressure, at 0.5 instead of PTC-011's 1.5. Control
    evaluation moves to steps 5 and 10 so the loop wakes only on useful eval points.
  default_num_steps: 11
  default_prompts_per_step: 128
  last_launched_utc: '2026-06-04T15:03:13Z'
  last_num_steps: 11
  last_prompts_per_step: 128
  last_verdict: early_science_reject
  last_reason: 'Step-5 control rejected the hidden-anchor hypothesis despite the controller''s
    generic inconclusive label: buffer_delta was only +0.0078, corrupt-pause accuracy
    was 0.5391, answer_logprob_margin was negative, and answer_select_delta was strongly
    negative.'
  last_profile: /shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T150313Z-configPTC-016-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl
  last_scorecard: /home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T160508Z-PTC-016-step5-inconclusive.json
  last_scored_utc: '2026-06-04T16:05:08Z'
- id: PTC-017
  status: rejected
  priority: 85
  candidate: candidates/PTC-017.yaml
  requires:
  - PTC-016
  hypothesis: If PTC-016 rejects, hidden memory shaping may still be directionally
    right but too weak. PTC-011's hidden weight 1.5 was the only setting that produced
    positive step-8 answer-selection, but it mixed in cache-mismatch. PTC-017 raises
    the hidden-only memory bootstrap to 1.0 while preserving PTC-015's stronger answer
    objective and keeping cache/corrupt/buffer-KL losses off.
  rationale: This is the highest-confidence supported fallback before spending time
    on true PG/RiM plumbing. It tests hidden-anchor strength on the clean generated-memory
    substrate with 100 generated tokens, FlashQLA, SMG, and eval only at steps 5 and
    10.
  default_num_steps: 11
  default_prompts_per_step: 128
  last_verdict: superseded
  last_reason: PTC-016 step-5 control already showed hidden anchoring at 0.5 made
    corrupt-pause accuracy high and answer-selection strongly negative; doubling the
    same hidden-only pressure is a poor next experiment.
- id: PTC-018
  status: rejected
  priority: 86
  candidate: candidates/PTC-018.yaml
  requires:
  - PTC-016
  hypothesis: 'PTC-014 with answer weight 1.0 produced the strongest early generated-memory
    buffer delta but negative answer selection, while PTC-015 with answer weight 2.0
    fixed answer selection but weakened the causal buffer gap. PTC-016 showed hidden
    anchoring makes the generated memory less causal. Test the interpolation point:
    answer-only weight 1.5, no hidden/cache/buffer/corrupt losses.'
  rationale: This keeps the 100-token generated-memory substrate, FlashQLA, SMG, gold
    answers, and corrupt generated-memory control. It avoids the now-rejected hidden-anchor
    line and asks whether the useful PTC-014 buffer signal and the PTC-015 answer-selection
    improvement can coexist at an intermediate answer weight.
  default_num_steps: 11
  default_prompts_per_step: 128
  last_launched_utc: '2026-06-04T16:06:38Z'
  last_num_steps: 11
  last_prompts_per_step: 128
  last_verdict: science_reject
  last_reason: no positive accuracy or answer-logprob signal
  last_profile: /shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T160640Z-configPTC-018-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl
  last_scorecard: /home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T170927Z-PTC-018-step5-science_reject.json
  last_scored_utc: '2026-06-04T17:09:27Z'
- id: PTC-019
  status: infra_invalid
  priority: 86
  candidate: candidates/PTC-019.yaml
  requires:
  - PTC-018
  hypothesis: The generated-memory answer-weight and hidden-anchor sweeps failed because
    they train rewritten gold-answer sequences or teacher hidden targets rather than
    giving policy-gradient credit to the actual memory and answer tokens the student
    sampled. Use sampled-answer OPD PG with behavior-policy old_logprobs, k1 distillation
    advantages, a small memory KL weight, and no corrupt/cache synthetic negatives.
  rationale: 'This is the first real RiM/PG plumbing test: old logprobs are aligned
    to the sampled generated-memory and answer tokens, gold-answer replacement is
    off, and invalid non-sampled negative datums are disabled. If it fails cleanly,
    the loop should stop sweeping supervised weights and either implement a true task-reward
    RiM objective or change the memory-generation surface.'
  default_num_steps: 11
  default_prompts_per_step: 128
  last_launched_utc: '2026-06-04T17:11:37Z'
  last_num_steps: 11
  last_prompts_per_step: 128
  last_verdict: infra_invalid
  last_reason: request_failure_frac_max=0.0039
  last_profile: /shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T171139Z-configPTC-019-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl
  last_scorecard: /home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T182907Z-PTC-019-step5-infra_invalid.json
  last_scored_utc: '2026-06-04T18:29:07Z'
- id: PTC-020
  status: promote_retest
  priority: 90
  candidate: candidates/PTC-020.yaml
  hypothesis: 'AM already showed the operational effect we care about: prefilling
    prompt plus arbitrary filler tokens improved exact-match accuracy over no-pause
    at n=1024, with strong post-hoc pause-vs-no-pause answer-logprob support. Retest
    the AM recipe on the current FlashQLA/SMG/P2P-parallel path and score it by pause-vs-no-pause
    performance, not corrupt-control or prompt-specific-memory diagnostics.'
  rationale: The previous non-promotion was mostly a gate-design choice. In-loop answer-logprob
    failed from oversized scorer batches and was fixed post-hoc by chunking; corrupt
    exact-match was contaminated by cap/boundary artifacts; and answer-selection is
    not the immediate objective. This branch asks only whether extra prefetched filler
    tokens improve the answer distribution and exact-match accuracy versus no filler.
  default_num_steps: 6
  default_prompts_per_step: 128
  last_launched_utc: '2026-06-04T19:11:32Z'
  last_num_steps: 6
  last_prompts_per_step: 128
  last_verdict: promote_retest
  last_reason: pause-vs-nopause accuracy warrants retest with positive answer-logprob
    support
  last_profile: /shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T191133Z-configPTC-020-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl
  last_scored_utc: '2026-06-04T19:44:27Z'
  last_scorecard: /home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T194427Z-PTC-020-step5-promote_retest.json
- id: PTC-021
  status: queued
  priority: 89
  candidate: candidates/PTC-021.yaml
  requires:
  - PTC-020
  requires_statuses:
    PTC-020:
    - strong_signal
    - promote_retest
    - weak_signal
  hypothesis: If PTC-020 reproduces the AM pause-vs-no-pause lift, continue the same
    recipe to the second scheduled 1k control point to check whether the operational
    effect is stable rather than a step-5 transient.
  rationale: This is the lowest-risk positive follow-up. It changes only the run length
    from 6 to 11 steps while keeping FlashQLA, parallel multi-endpoint P2P sync, the
    1k pause-vs-no-pause gate, and the AM random-symbol filler objective.
  default_num_steps: 11
  default_prompts_per_step: 128
- id: PTC-022
  status: queued
  priority: 88
  candidate: candidates/PTC-022.yaml
  requires:
  - PTC-021
  requires_statuses:
    PTC-021:
    - strong_signal
    - promote_retest
    - weak_signal
  hypothesis: If the AM pause-vs-no-pause lift survives the stability retest, doubling
    prompts per step should reduce optimizer/eval variance and make the final 1k operational
    signal more reliable.
  rationale: This is a scale-up of the same positive branch, not a new objective sweep.
    It keeps the 6-step final-control schedule but raises prompts per step to 256
    on the parallel multi-endpoint P2P path.
  default_num_steps: 6
  default_prompts_per_step: 256
- id: PTC-023
  status: queued
  priority: 92
  candidate: candidates/PTC-023.yaml
  requires:
  - PTC-020
  requires_statuses:
    PTC-020:
    - strong_signal
    - promote_retest
    - weak_signal
  hypothesis: If AM's operational pause-vs-no-pause lift does not require hidden-state
    matching, then the same short static-filler answer-causal recipe with hidden
    matching disabled should retain a positive step-5 1k control delta and answer-logprob
    support.
  rationale: This is the clean AM-minus-hidden ablation requested after PTC-020.
    It keeps the corrupt buffer and corrupt-answer contrastive training objective,
    FlashQLA, P2P sync, parallel endpoint sync, and the same final-only 1k operational
    gate, while setting `opd_hidden_match_coef=0.0`. Corrupt exact-match eval is
    skipped because the queue is judging pause-vs-no-pause performance.
  default_num_steps: 6
  default_prompts_per_step: 128
- id: PTC-024
  status: queued
  priority: 91
  candidate: candidates/PTC-024.yaml
  requires:
  - PTC-020
  requires_statuses:
    PTC-020:
    - strong_signal
    - promote_retest
    - weak_signal
  hypothesis: If AM's operational lift mostly comes from short-filler answer supervision
    and hidden anchoring rather than the corrupt-negative arm, replacing corrupt
    answer/buffer penalties with a small positive-answer KL should preserve a positive
    pause-vs-no-pause signal.
  rationale: This isolates AM-minus-corrupt while keeping hidden matching active.
    It removes both corrupt training weights, keeps the AM static filler, step-5
    1k operational control, FlashQLA, P2P sync, and parallel endpoint sync, and
    uses `opd_positive_answer_weight=0.125` as the non-corrupt answer-side pressure.
  default_num_steps: 6
  default_prompts_per_step: 128
- id: PTC-025
  status: queued
  priority: 90
  candidate: candidates/PTC-025.yaml
  requires:
  - PTC-020
  requires_statuses:
    PTC-020:
    - strong_signal
    - promote_retest
    - weak_signal
  hypothesis: If neither hidden matching nor corrupt-negative training is necessary
    for the operational prefill-performance effect, then a minimal AM-style static-filler
    run with only positive answer-side pressure should still beat no-pause at the
    step-5 1k control.
  rationale: This is the clean AM-minus-hidden-minus-corrupt ablation. It keeps
    AM's short filler, final-only 1k pause-vs-no-pause scoring, FlashQLA, P2P sync,
    and parallel endpoint sync, but sets hidden and corrupt weights to zero and
    adds `opd_positive_answer_weight=0.125` so the answer path is not left without
    an explicit training signal.
  default_num_steps: 6
  default_prompts_per_step: 128
```


### `experiments/opd_profile/autoresearch/candidates/PTC-001.yaml`

```yaml
id: PTC-001
base_config: AN
slug: ptc-001-pure-kl
buffer_label: randsymbol-ptc-pure-kl
trainer_config: experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node_nccl.yaml
student_prefill_text: " ! | ~ _ * ^ # @ "
student_prefill_count: 1
student_prefill_suffix: "Answer: "
student_stop_sequences: '["\n"]'
teacher_cot_json_path: /shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
prompts_json_path: /shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
num_prompts: 8185
default_num_steps: 9
default_prompts_per_step: 128
supervise_student_cot: true
opd_supervise_buffer_only: false
opd_hidden_match_coef: 0.0
learning_rate: 3e-6
max_new_tokens: 64
eval_max_new_tokens: 64
eval_num_problems: 1000
eval_accuracy_every: 4
eval_control_start_step: 4
eval_control_max_concurrency: 128
eval_answer_logprob_control: true
eval_answer_logprob_batch_size: 128
eval_answer_logprob_max_concurrency: 8
eval_answer_logprob_distractor_control: false
opd_pipeline_rl: false
sampler_quiesce_before_sync: true
sync_method: nccl_broadcast
request_timeout: 1200
weight_sync_timeout: 900
opd_contrastive_corrupt_buffer_weight: 0.0
opd_contrastive_corrupt_answer_weight: 0.0
opd_positive_answer_weight: 0.0
wandb_group: q36-ptc-autoresearch
client_args:
  opd_ptc_positive_buffer_kl_weight: 1.0
  opd_ptc_positive_answer_kl_weight: 0.125
  opd_ptc_positive_hidden_weight: 0.0
  opd_mask_zero_weight_positions: true
  opd_teacher_answer_source: sampled
  eval_corrupt_pause_control: false
```


### `experiments/opd_profile/autoresearch/candidates/PTC-002.yaml`

```yaml
id: PTC-002
base_config: AN
slug: ptc-002-gold-answer-kl
buffer_label: randsymbol-ptc-gold-answer-kl
trainer_config: experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node_nccl.yaml
student_prefill_text: " ! | ~ _ * ^ # @ "
student_prefill_count: 1
student_prefill_suffix: "Answer: "
student_stop_sequences: '["\n"]'
teacher_cot_json_path: /shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
prompts_json_path: /shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
num_prompts: 8185
default_num_steps: 9
default_prompts_per_step: 128
supervise_student_cot: true
opd_supervise_buffer_only: false
opd_hidden_match_coef: 0.0
learning_rate: 3e-6
max_new_tokens: 64
eval_max_new_tokens: 64
eval_num_problems: 1000
eval_accuracy_every: 4
eval_control_start_step: 4
eval_control_max_concurrency: 128
eval_answer_logprob_control: true
eval_answer_logprob_batch_size: 64
eval_answer_logprob_max_concurrency: 2
eval_answer_logprob_distractor_control: false
opd_pipeline_rl: false
sampler_quiesce_before_sync: true
sync_method: nccl_broadcast
request_timeout: 1200
weight_sync_timeout: 900
opd_contrastive_corrupt_buffer_weight: 0.0
opd_contrastive_corrupt_answer_weight: 0.0
opd_positive_answer_weight: 0.0
wandb_group: q36-ptc-autoresearch
client_args:
  opd_ptc_positive_buffer_kl_weight: 1.0
  opd_ptc_positive_answer_kl_weight: 0.125
  opd_ptc_positive_hidden_weight: 0.0
  opd_mask_zero_weight_positions: true
  opd_teacher_answer_source: gold
  eval_corrupt_pause_control: false
```


### `experiments/opd_profile/autoresearch/candidates/PTC-003.yaml`

```yaml
id: PTC-003
base_config: AN
slug: ptc-003-gold-answer-hidden-flashqla
buffer_label: randsymbol-ptc-gold-answer-hidden-flashqla
trainer_config: experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node_nccl.yaml
gdn_backend: flashqla
student_prefill_text: " ! | ~ _ * ^ # @ "
student_prefill_count: 1
student_prefill_suffix: "Answer: "
student_stop_sequences: '["\n"]'
teacher_cot_json_path: /shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
prompts_json_path: /shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
num_prompts: 8185
default_num_steps: 9
default_prompts_per_step: 128
supervise_student_cot: true
opd_supervise_buffer_only: false
opd_hidden_match_coef: 0.5
learning_rate: 3e-6
max_new_tokens: 64
eval_max_new_tokens: 64
eval_num_problems: 1000
eval_accuracy_every: 4
eval_control_start_step: 4
eval_control_max_concurrency: 128
eval_answer_logprob_control: true
eval_answer_logprob_batch_size: 64
eval_answer_logprob_max_concurrency: 2
eval_answer_logprob_distractor_control: false
opd_pipeline_rl: false
sampler_quiesce_before_sync: true
sync_method: nccl_broadcast
request_timeout: 1200
weight_sync_timeout: 900
opd_contrastive_corrupt_buffer_weight: 0.0
opd_contrastive_corrupt_answer_weight: 0.0
opd_positive_answer_weight: 0.0
wandb_group: q36-ptc-autoresearch
client_args:
  opd_ptc_positive_buffer_kl_weight: 1.0
  opd_ptc_positive_answer_kl_weight: 0.125
  opd_ptc_positive_hidden_weight: 1.0
  opd_mask_zero_weight_positions: true
  opd_teacher_answer_source: gold
  eval_corrupt_pause_control: false
```


### `experiments/opd_profile/autoresearch/candidates/PTC-004.yaml`

```yaml
id: PTC-004
base_config: AN
slug: ptc-004-gold-answer-scale-flashqla
buffer_label: randsymbol-ptc-gold-answer-scale-flashqla
trainer_config: experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node_nccl.yaml
gdn_backend: flashqla
student_prefill_text: " ! | ~ _ * ^ # @ "
student_prefill_count: 1
student_prefill_suffix: "Answer: "
student_stop_sequences: '["\n"]'
teacher_cot_json_path: /shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
prompts_json_path: /shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
num_prompts: 8185
default_num_steps: 13
default_prompts_per_step: 256
supervise_student_cot: true
opd_supervise_buffer_only: false
opd_hidden_match_coef: 0.0
learning_rate: 3e-6
max_new_tokens: 64
eval_max_new_tokens: 64
eval_num_problems: 1000
eval_accuracy_every: 4
eval_control_start_step: 4
eval_control_max_concurrency: 128
eval_answer_logprob_control: true
eval_answer_logprob_batch_size: 64
eval_answer_logprob_max_concurrency: 2
eval_answer_logprob_distractor_control: false
opd_pipeline_rl: false
sampler_quiesce_before_sync: true
sync_method: nccl_broadcast
request_timeout: 1200
weight_sync_timeout: 900
opd_contrastive_corrupt_buffer_weight: 0.0
opd_contrastive_corrupt_answer_weight: 0.0
opd_positive_answer_weight: 0.0
wandb_group: q36-ptc-autoresearch
client_args:
  opd_ptc_positive_buffer_kl_weight: 1.0
  opd_ptc_positive_answer_kl_weight: 0.125
  opd_ptc_positive_hidden_weight: 0.0
  opd_mask_zero_weight_positions: true
  opd_teacher_answer_source: gold
  eval_corrupt_pause_control: false
```


### `experiments/opd_profile/autoresearch/candidates/PTC-005.yaml`

```yaml
id: PTC-005
base_config: AN
slug: ptc-005-pure-kl-flashqla
buffer_label: randsymbol-ptc-pure-kl-flashqla
trainer_config: experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node_nccl.yaml
gdn_backend: flashqla
student_prefill_text: " ! | ~ _ * ^ # @ "
student_prefill_count: 1
student_prefill_suffix: "Answer: "
student_stop_sequences: '["\n"]'
teacher_cot_json_path: /shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
prompts_json_path: /shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
num_prompts: 8185
default_num_steps: 9
default_prompts_per_step: 128
supervise_student_cot: true
opd_supervise_buffer_only: false
opd_hidden_match_coef: 0.0
learning_rate: 3e-6
max_new_tokens: 64
eval_max_new_tokens: 64
eval_num_problems: 1000
eval_accuracy_every: 4
eval_control_start_step: 4
eval_control_max_concurrency: 128
eval_answer_logprob_control: true
eval_answer_logprob_batch_size: 64
eval_answer_logprob_max_concurrency: 2
eval_answer_logprob_distractor_control: false
opd_pipeline_rl: false
sampler_quiesce_before_sync: true
sync_method: nccl_broadcast
request_timeout: 1200
weight_sync_timeout: 900
opd_contrastive_corrupt_buffer_weight: 0.0
opd_contrastive_corrupt_answer_weight: 0.0
opd_positive_answer_weight: 0.0
wandb_group: q36-ptc-autoresearch-throughput
client_args:
  opd_ptc_positive_buffer_kl_weight: 1.0
  opd_ptc_positive_answer_kl_weight: 0.125
  opd_ptc_positive_hidden_weight: 0.0
  opd_mask_zero_weight_positions: true
  opd_teacher_answer_source: sampled
  eval_corrupt_pause_control: false
```


### `experiments/opd_profile/autoresearch/candidates/PTC-006.yaml`

```yaml
id: PTC-006
base_config: AN
slug: ptc-006-gold-answer-kl-flashqla
buffer_label: randsymbol-ptc-gold-answer-kl-flashqla
trainer_config: experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node_nccl.yaml
gdn_backend: flashqla
student_prefill_text: " ! | ~ _ * ^ # @ "
student_prefill_count: 1
student_prefill_suffix: "Answer: "
student_stop_sequences: '["\n"]'
teacher_cot_json_path: /shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
prompts_json_path: /shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
num_prompts: 8185
default_num_steps: 9
default_prompts_per_step: 128
supervise_student_cot: true
opd_supervise_buffer_only: false
opd_hidden_match_coef: 0.0
learning_rate: 3e-6
max_new_tokens: 64
eval_max_new_tokens: 64
eval_num_problems: 1000
eval_accuracy_every: 4
eval_control_start_step: 4
eval_control_max_concurrency: 128
eval_answer_logprob_control: true
eval_answer_logprob_batch_size: 64
eval_answer_logprob_max_concurrency: 2
eval_answer_logprob_distractor_control: false
opd_pipeline_rl: false
sampler_quiesce_before_sync: true
sync_method: nccl_broadcast
request_timeout: 1200
weight_sync_timeout: 900
opd_contrastive_corrupt_buffer_weight: 0.0
opd_contrastive_corrupt_answer_weight: 0.0
opd_positive_answer_weight: 0.0
wandb_group: q36-ptc-autoresearch
client_args:
  opd_ptc_positive_buffer_kl_weight: 1.0
  opd_ptc_positive_answer_kl_weight: 0.125
  opd_ptc_positive_hidden_weight: 0.0
  opd_mask_zero_weight_positions: true
  opd_teacher_answer_source: gold
  eval_corrupt_pause_control: false
```


### `experiments/opd_profile/autoresearch/candidates/PTC-007.yaml`

```yaml
id: PTC-007
base_config: AN
slug: ptc-007-gold-answer-kl-longfiller108-flashqla
buffer_label: randsymbol-ptc-gold-answer-kl-longfiller108-flashqla
trainer_config: experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node_nccl.yaml
gdn_backend: flashqla
student_prefill_text: " ! | ~ _ * ^ # @ "
student_prefill_count: 12
student_prefill_suffix: "Answer: "
student_stop_sequences: '["\n"]'
teacher_cot_json_path: /shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
prompts_json_path: /shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
num_prompts: 8185
default_num_steps: 9
default_prompts_per_step: 128
supervise_student_cot: true
opd_supervise_buffer_only: false
opd_hidden_match_coef: 0.0
learning_rate: 3e-6
max_new_tokens: 64
eval_max_new_tokens: 64
eval_num_problems: 1000
eval_accuracy_every: 4
eval_control_start_step: 4
eval_control_max_concurrency: 128
eval_answer_logprob_control: true
eval_answer_logprob_batch_size: 64
eval_answer_logprob_max_concurrency: 2
eval_answer_logprob_distractor_control: false
opd_pipeline_rl: false
sampler_quiesce_before_sync: true
sync_method: nccl_broadcast
request_timeout: 1200
weight_sync_timeout: 900
opd_contrastive_corrupt_buffer_weight: 0.0
opd_contrastive_corrupt_answer_weight: 0.0
opd_positive_answer_weight: 0.0
wandb_group: q36-ptc-autoresearch
client_args:
  opd_ptc_positive_buffer_kl_weight: 1.0
  opd_ptc_positive_answer_kl_weight: 0.125
  opd_ptc_positive_hidden_weight: 0.0
  opd_mask_zero_weight_positions: true
  opd_teacher_answer_source: gold
  eval_corrupt_pause_control: false
```


### `experiments/opd_profile/autoresearch/candidates/PTC-008.yaml`

```yaml
id: PTC-008
base_config: AN
slug: ptc-008-rimstyle-answercontrast-longfiller108-flashqla
buffer_label: randsymbol-rimstyle-answercontrast-longfiller108-flashqla
trainer_config: experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node_nccl.yaml
gdn_backend: flashqla
student_prefill_text: " ! | ~ _ * ^ # @ "
student_prefill_count: 12
student_prefill_suffix: "Answer: "
student_stop_sequences: '["\n"]'
teacher_cot_json_path: /shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
prompts_json_path: /shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
num_prompts: 8185
default_num_steps: 9
default_prompts_per_step: 128
supervise_student_cot: true
opd_supervise_buffer_only: false
opd_hidden_match_coef: 0.0
learning_rate: 3e-6
max_new_tokens: 64
eval_max_new_tokens: 64
eval_num_problems: 1000
eval_accuracy_every: 4
eval_control_start_step: 4
eval_control_max_concurrency: 128
eval_answer_logprob_control: true
eval_answer_logprob_batch_size: 64
eval_answer_logprob_max_concurrency: 2
eval_answer_logprob_distractor_control: true
eval_answer_logprob_distractor_offset: 1
opd_pipeline_rl: false
sampler_quiesce_before_sync: true
sync_method: nccl_broadcast
request_timeout: 1200
weight_sync_timeout: 900
opd_contrastive_corrupt_buffer_weight: 1.0
opd_contrastive_corrupt_answer_weight: 0.125
opd_contrastive_corrupt_buffer_mode: rotate_preserve_ws
opd_contrastive_corrupt_buffer_span: memory_only
opd_positive_answer_weight: 0.0
opd_loss_max_clamp: 5.0
wandb_group: q36-ptc-autoresearch
client_args:
  opd_ptc_positive_buffer_kl_weight: 0.0
  opd_ptc_positive_answer_kl_weight: 0.0
  opd_ptc_positive_hidden_weight: 0.0
  opd_mask_zero_weight_positions: true
  opd_teacher_answer_source: gold
  eval_corrupt_pause_control: true
```


### `experiments/opd_profile/autoresearch/candidates/PTC-009.yaml`

```yaml
id: PTC-009
base_config: AN
slug: ptc-009-cachemismatch-answer1-longfiller108-flashqla
buffer_label: randsymbol-cachemismatch-answer1-longfiller108-flashqla
trainer_config: experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node_nccl.yaml
gdn_backend: flashqla
student_prefill_text: " ! | ~ _ * ^ # @ "
student_prefill_count: 12
student_prefill_suffix: "Answer: "
student_stop_sequences: '["\n"]'
teacher_cot_json_path: /shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
prompts_json_path: /shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
num_prompts: 8185
default_num_steps: 9
default_prompts_per_step: 128
supervise_student_cot: true
opd_supervise_buffer_only: false
opd_hidden_match_coef: 0.5
learning_rate: 3e-6
max_new_tokens: 64
eval_max_new_tokens: 64
eval_num_problems: 1000
eval_accuracy_every: 4
eval_control_start_step: 4
eval_control_max_concurrency: 128
eval_answer_logprob_control: true
eval_answer_logprob_batch_size: 64
eval_answer_logprob_max_concurrency: 2
eval_answer_logprob_distractor_control: true
eval_answer_logprob_distractor_offset: 1
opd_pipeline_rl: false
sampler_quiesce_before_sync: true
sync_method: nccl_broadcast
request_timeout: 1200
weight_sync_timeout: 900
opd_contrastive_corrupt_buffer_weight: 0.0
opd_contrastive_corrupt_answer_weight: 0.0
opd_contrastive_corrupt_buffer_mode: rotate_preserve_ws
opd_contrastive_corrupt_buffer_span: memory_only
opd_positive_answer_weight: 1.0
opd_cache_mismatch_memory_weight: 0.25
opd_cache_mismatch_balance_positive_hidden: true
opd_teacher_memory_pair_diagnostics: true
opd_loss_max_clamp: 5.0
wandb_group: q36-ptc-autoresearch
client_args:
  opd_ptc_positive_buffer_kl_weight: 0.0
  opd_ptc_positive_answer_kl_weight: 0.0
  opd_ptc_positive_hidden_weight: 0.0
  opd_mask_zero_weight_positions: true
  opd_teacher_answer_source: gold
  eval_corrupt_pause_control: true
```


### `experiments/opd_profile/autoresearch/candidates/PTC-010.yaml`

```yaml
id: PTC-010
base_config: AN
slug: ptc-010-hiddenonly-answer1-cachemis-longfiller108-flashqla
buffer_label: randsymbol-hiddenonly-answer1-cachemis-longfiller108-flashqla
trainer_config: experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node_nccl.yaml
gdn_backend: flashqla
student_prefill_text: " ! | ~ _ * ^ # @ "
student_prefill_count: 12
student_prefill_suffix: "Answer: "
student_stop_sequences: '["\n"]'
teacher_cot_json_path: /shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
prompts_json_path: /shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
num_prompts: 8185
default_num_steps: 9
default_prompts_per_step: 128
supervise_student_cot: true
opd_supervise_buffer_only: false
opd_hidden_match_coef: 1.0
learning_rate: 3e-6
max_new_tokens: 64
eval_max_new_tokens: 64
eval_num_problems: 1000
eval_accuracy_every: 4
eval_control_start_step: 4
eval_control_max_concurrency: 128
eval_answer_logprob_control: true
eval_answer_logprob_batch_size: 64
eval_answer_logprob_max_concurrency: 2
eval_answer_logprob_distractor_control: true
eval_answer_logprob_distractor_offset: 1
opd_pipeline_rl: false
sampler_quiesce_before_sync: true
sync_method: nccl_broadcast
request_timeout: 1200
weight_sync_timeout: 900
opd_contrastive_corrupt_buffer_weight: 0.0
opd_contrastive_corrupt_answer_weight: 0.0
opd_contrastive_corrupt_buffer_mode: rotate_preserve_ws
opd_contrastive_corrupt_buffer_span: memory_only
opd_positive_answer_weight: 0.0
opd_cache_mismatch_memory_weight: 0.5
opd_teacher_memory_pair_diagnostics: true
opd_loss_max_clamp: 5.0
wandb_group: q36-ptc-autoresearch
client_args:
  opd_ptc_positive_buffer_kl_weight: 0.0
  opd_ptc_positive_answer_kl_weight: 1.0
  opd_ptc_positive_hidden_weight: 1.5
  opd_mask_zero_weight_positions: true
  opd_teacher_answer_source: gold
  eval_corrupt_pause_control: true
```


### `experiments/opd_profile/autoresearch/candidates/PTC-011.yaml`

```yaml
id: PTC-011
base_config: AN
slug: ptc-011-generated-memory100-hidden-answer-cachemis-flashqla
buffer_label: generated-memory100-hidden-answer1-cachemis-flashqla
trainer_config: experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node_nccl.yaml
gdn_backend: flashqla
student_prefill_text: "Memory: "
student_prefill_count: 1
student_prefill_suffix: "\nAnswer: "
student_stop_sequences: '["\n"]'
teacher_cot_json_path: /shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
prompts_json_path: /shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
num_prompts: 8185
default_num_steps: 9
default_prompts_per_step: 128
supervise_student_cot: true
opd_supervise_buffer_only: false
opd_hidden_match_coef: 1.0
learning_rate: 3e-6
max_new_tokens: 64
eval_max_new_tokens: 64
eval_num_problems: 256
score_min_control_n: 250
eval_accuracy_every: 4
eval_control_start_step: 4
eval_control_max_concurrency: 128
eval_answer_logprob_control: true
eval_answer_logprob_batch_size: 64
eval_answer_logprob_max_concurrency: 2
eval_answer_logprob_distractor_control: true
eval_answer_logprob_distractor_offset: 1
opd_pipeline_rl: false
sampler_quiesce_before_sync: true
sync_method: nccl_broadcast
request_timeout: 1200
weight_sync_timeout: 900
opd_contrastive_corrupt_buffer_weight: 0.0
opd_contrastive_corrupt_answer_weight: 0.0
opd_contrastive_corrupt_buffer_mode: rotate_preserve_ws
opd_contrastive_corrupt_buffer_span: memory_only
opd_positive_answer_weight: 0.0
opd_cache_mismatch_memory_weight: 0.5
opd_teacher_memory_pair_diagnostics: true
opd_loss_max_clamp: 5.0
wandb_group: q36-ptc-autoresearch
client_args:
  student_generated_memory_tokens: 100
  eval_generated_memory_control: true
  eval_generated_memory_tokens: 100
  eval_generated_memory_shuffle_offset: 1
  eval_generated_memory_temperature: 0.0
  opd_ptc_positive_buffer_kl_weight: 0.0
  opd_ptc_positive_answer_kl_weight: 1.0
  opd_ptc_positive_hidden_weight: 1.5
  opd_mask_zero_weight_positions: true
  opd_teacher_answer_source: gold
  eval_corrupt_pause_control: true
```


### `experiments/opd_profile/autoresearch/candidates/PTC-012.yaml`

```yaml
id: PTC-012
base_config: AN
slug: ptc-012-generated-memory100-rim-answercontrast-flashqla
buffer_label: generated-memory100-rim-answercontrast-flashqla
trainer_config: experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node_nccl.yaml
gdn_backend: flashqla
student_prefill_text: "Memory: "
student_prefill_count: 1
student_prefill_suffix: "\nAnswer: "
student_stop_sequences: '["\n"]'
teacher_cot_json_path: /shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
prompts_json_path: /shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
num_prompts: 8185
default_num_steps: 9
default_prompts_per_step: 128
supervise_student_cot: true
opd_supervise_buffer_only: false
opd_hidden_match_coef: 0.0
learning_rate: 3e-6
max_new_tokens: 64
eval_max_new_tokens: 64
eval_num_problems: 256
score_min_control_n: 250
eval_accuracy_every: 4
eval_control_start_step: 4
eval_control_max_concurrency: 128
eval_answer_logprob_control: true
eval_answer_logprob_batch_size: 64
eval_answer_logprob_max_concurrency: 2
eval_answer_logprob_distractor_control: true
eval_answer_logprob_distractor_offset: 1
opd_pipeline_rl: false
sampler_quiesce_before_sync: true
sync_method: nccl_broadcast
request_timeout: 1200
weight_sync_timeout: 900
opd_contrastive_corrupt_buffer_weight: 0.0
opd_contrastive_corrupt_answer_weight: 0.25
opd_contrastive_corrupt_buffer_mode: rotate_preserve_ws
opd_contrastive_corrupt_buffer_span: memory_only
opd_positive_answer_weight: 0.75
opd_cache_mismatch_memory_weight: 0.0
opd_teacher_memory_pair_diagnostics: false
opd_loss_max_clamp: 5.0
wandb_group: q36-ptc-autoresearch
client_args:
  student_generated_memory_tokens: 100
  eval_generated_memory_control: true
  eval_generated_memory_tokens: 100
  eval_generated_memory_shuffle_offset: 1
  eval_generated_memory_temperature: 0.0
  opd_ptc_positive_buffer_kl_weight: 0.0
  # Tiny positive-answer PTC weight activates explicit PTC position weights,
  # keeping buffer KL/hidden weights at zero instead of the custom-pair default
  # positive buffer KL=1.0. The substantive answer utility still comes from
  # opd_positive_answer_weight and opd_contrastive_corrupt_answer_weight above.
  opd_ptc_positive_answer_kl_weight: 0.001
  opd_ptc_positive_hidden_weight: 0.0
  opd_mask_zero_weight_positions: true
  opd_teacher_answer_source: gold
  eval_corrupt_pause_control: true
```


### `experiments/opd_profile/autoresearch/candidates/PTC-013.yaml`

```yaml
id: PTC-013
base_config: AN
slug: ptc-013-generated-memory100-rim-answercontrast-bufferboot-flashqla
buffer_label: generated-memory100-rim-answercontrast-bufferboot-flashqla
trainer_config: experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node_nccl.yaml
gdn_backend: flashqla
student_prefill_text: "Memory: "
student_prefill_count: 1
student_prefill_suffix: "\nAnswer: "
student_stop_sequences: '["\n"]'
teacher_cot_json_path: /shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
prompts_json_path: /shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
num_prompts: 8185
default_num_steps: 9
default_prompts_per_step: 128
supervise_student_cot: true
opd_supervise_buffer_only: false
opd_hidden_match_coef: 0.0
learning_rate: 3e-6
max_new_tokens: 64
eval_max_new_tokens: 64
eval_num_problems: 256
score_min_control_n: 250
eval_accuracy_every: 4
eval_control_start_step: 4
eval_control_max_concurrency: 128
eval_answer_logprob_control: true
eval_answer_logprob_batch_size: 64
eval_answer_logprob_max_concurrency: 2
eval_answer_logprob_distractor_control: true
eval_answer_logprob_distractor_offset: 1
opd_pipeline_rl: false
sampler_quiesce_before_sync: true
sync_method: nccl_broadcast
request_timeout: 1200
weight_sync_timeout: 900
opd_contrastive_corrupt_buffer_weight: 0.0
opd_contrastive_corrupt_answer_weight: 0.25
opd_contrastive_corrupt_buffer_mode: rotate_preserve_ws
opd_contrastive_corrupt_buffer_span: memory_only
opd_positive_answer_weight: 0.75
opd_cache_mismatch_memory_weight: 0.0
opd_teacher_memory_pair_diagnostics: false
opd_loss_max_clamp: 5.0
wandb_group: q36-ptc-autoresearch
client_args:
  student_generated_memory_tokens: 100
  eval_generated_memory_control: true
  eval_generated_memory_tokens: 100
  eval_generated_memory_shuffle_offset: 1
  eval_generated_memory_temperature: 0.0
  opd_ptc_positive_buffer_kl_weight: 0.1
  opd_ptc_positive_answer_kl_weight: 0.001
  opd_ptc_positive_hidden_weight: 0.0
  opd_mask_zero_weight_positions: true
  opd_teacher_answer_source: gold
  eval_corrupt_pause_control: true
```


### `experiments/opd_profile/autoresearch/candidates/PTC-014.yaml`

```yaml
id: PTC-014
base_config: AN
slug: ptc-014-generated-memory100-positive-answeronly-flashqla
buffer_label: generated-memory100-positive-answeronly-flashqla
trainer_config: experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node_nccl.yaml
gdn_backend: flashqla
student_prefill_text: "Memory: "
student_prefill_count: 1
student_prefill_suffix: "\nAnswer: "
student_stop_sequences: '["\n"]'
teacher_cot_json_path: /shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
prompts_json_path: /shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
num_prompts: 8185
default_num_steps: 9
default_prompts_per_step: 128
supervise_student_cot: true
opd_supervise_buffer_only: false
opd_hidden_match_coef: 0.0
learning_rate: 3e-6
max_new_tokens: 64
eval_max_new_tokens: 64
eval_num_problems: 256
score_min_control_n: 250
eval_accuracy_every: 4
eval_control_start_step: 4
eval_control_max_concurrency: 128
eval_answer_logprob_control: true
eval_answer_logprob_batch_size: 64
eval_answer_logprob_max_concurrency: 2
eval_answer_logprob_distractor_control: true
eval_answer_logprob_distractor_offset: 1
opd_pipeline_rl: false
sampler_quiesce_before_sync: true
sync_method: nccl_broadcast
request_timeout: 1200
weight_sync_timeout: 900
opd_contrastive_corrupt_buffer_weight: 0.0
opd_contrastive_corrupt_answer_weight: 0.0
opd_contrastive_corrupt_buffer_mode: rotate_preserve_ws
opd_contrastive_corrupt_buffer_span: memory_only
opd_positive_answer_weight: 1.0
opd_cache_mismatch_memory_weight: 0.0
opd_teacher_memory_pair_diagnostics: false
opd_loss_max_clamp: 5.0
wandb_group: q36-ptc-autoresearch
client_args:
  student_generated_memory_tokens: 100
  eval_generated_memory_control: true
  eval_generated_memory_tokens: 100
  eval_generated_memory_shuffle_offset: 1
  eval_generated_memory_temperature: 0.0
  opd_ptc_positive_buffer_kl_weight: 0.0
  # Tiny positive-answer PTC weight activates explicit PTC position weights,
  # keeping buffer KL/hidden weights at zero instead of the custom-pair default
  # positive buffer KL=1.0. The substantive answer utility comes from
  # opd_positive_answer_weight above.
  opd_ptc_positive_answer_kl_weight: 0.001
  opd_ptc_positive_hidden_weight: 0.0
  opd_mask_zero_weight_positions: true
  opd_teacher_answer_source: gold
  eval_corrupt_pause_control: true
```


### `experiments/opd_profile/autoresearch/candidates/PTC-015.yaml`

```yaml
id: PTC-015
base_config: AN
slug: ptc-015-generated-memory100-positive-answer2-flashqla
buffer_label: generated-memory100-positive-answer2-flashqla
trainer_config: experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node_nccl.yaml
gdn_backend: flashqla
student_prefill_text: "Memory: "
student_prefill_count: 1
student_prefill_suffix: "\nAnswer: "
student_stop_sequences: '["\n"]'
teacher_cot_json_path: /shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
prompts_json_path: /shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
num_prompts: 8185
default_num_steps: 9
default_prompts_per_step: 128
supervise_student_cot: true
opd_supervise_buffer_only: false
opd_hidden_match_coef: 0.0
learning_rate: 3e-6
max_new_tokens: 64
eval_max_new_tokens: 64
eval_num_problems: 256
score_min_control_n: 250
eval_accuracy_every: 4
eval_control_start_step: 4
eval_control_max_concurrency: 128
eval_answer_logprob_control: true
eval_answer_logprob_batch_size: 64
eval_answer_logprob_max_concurrency: 2
eval_answer_logprob_distractor_control: true
eval_answer_logprob_distractor_offset: 1
opd_pipeline_rl: false
sampler_quiesce_before_sync: true
sync_method: nccl_broadcast
request_timeout: 1200
weight_sync_timeout: 900
opd_contrastive_corrupt_buffer_weight: 0.0
opd_contrastive_corrupt_answer_weight: 0.0
opd_contrastive_corrupt_buffer_mode: rotate_preserve_ws
opd_contrastive_corrupt_buffer_span: memory_only
opd_positive_answer_weight: 2.0
opd_cache_mismatch_memory_weight: 0.0
opd_teacher_memory_pair_diagnostics: false
opd_loss_max_clamp: 5.0
wandb_group: q36-ptc-autoresearch
client_args:
  student_generated_memory_tokens: 100
  eval_generated_memory_control: true
  eval_generated_memory_tokens: 100
  eval_generated_memory_shuffle_offset: 1
  eval_generated_memory_temperature: 0.0
  opd_ptc_positive_buffer_kl_weight: 0.0
  # Keep PTC-positive masking active so prompt and memory rows stay zero-weighted;
  # the substantive answer utility comes from opd_positive_answer_weight above.
  opd_ptc_positive_answer_kl_weight: 0.001
  opd_ptc_positive_hidden_weight: 0.0
  opd_mask_zero_weight_positions: true
  opd_teacher_answer_source: gold
  eval_corrupt_pause_control: true
```


### `experiments/opd_profile/autoresearch/candidates/PTC-016.yaml`

```yaml
id: PTC-016
base_config: AN
slug: ptc-016-generated-memory100-answer2-hidden05-flashqla
buffer_label: generated-memory100-answer2-hidden05-flashqla
trainer_config: experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node_nccl.yaml
gdn_backend: flashqla
student_prefill_text: "Memory: "
student_prefill_count: 1
student_prefill_suffix: "\nAnswer: "
student_stop_sequences: '["\n"]'
teacher_cot_json_path: /shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
prompts_json_path: /shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
num_prompts: 8185
default_num_steps: 11
default_prompts_per_step: 128
supervise_student_cot: true
opd_supervise_buffer_only: false
opd_hidden_match_coef: 1.0
learning_rate: 3e-6
max_new_tokens: 64
eval_max_new_tokens: 64
eval_num_problems: 256
score_min_control_n: 250
eval_accuracy_every: 5
eval_control_start_step: 5
eval_control_max_concurrency: 128
eval_answer_logprob_control: true
eval_answer_logprob_batch_size: 64
eval_answer_logprob_max_concurrency: 2
eval_answer_logprob_distractor_control: true
eval_answer_logprob_distractor_offset: 1
opd_pipeline_rl: false
sampler_quiesce_before_sync: true
sync_method: nccl_broadcast
request_timeout: 1200
weight_sync_timeout: 900
opd_contrastive_corrupt_buffer_weight: 0.0
opd_contrastive_corrupt_answer_weight: 0.0
opd_contrastive_corrupt_buffer_mode: rotate_preserve_ws
opd_contrastive_corrupt_buffer_span: memory_only
opd_positive_answer_weight: 2.0
opd_cache_mismatch_memory_weight: 0.0
opd_teacher_memory_pair_diagnostics: false
opd_loss_max_clamp: 5.0
wandb_group: q36-ptc-autoresearch
client_args:
  student_generated_memory_tokens: 100
  eval_generated_memory_control: true
  eval_generated_memory_tokens: 100
  eval_generated_memory_shuffle_offset: 1
  eval_generated_memory_temperature: 0.0
  opd_ptc_positive_buffer_kl_weight: 0.0
  opd_ptc_positive_answer_kl_weight: 0.001
  opd_ptc_positive_hidden_weight: 0.5
  opd_mask_zero_weight_positions: true
  opd_teacher_answer_source: gold
  eval_corrupt_pause_control: true
```


### `experiments/opd_profile/autoresearch/candidates/PTC-017.yaml`

```yaml
id: PTC-017
base_config: AN
slug: ptc-017-generated-memory100-answer2-hidden10-flashqla
buffer_label: generated-memory100-answer2-hidden10-flashqla
trainer_config: experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node_nccl.yaml
gdn_backend: flashqla
student_prefill_text: "Memory: "
student_prefill_count: 1
student_prefill_suffix: "\nAnswer: "
student_stop_sequences: '["\n"]'
teacher_cot_json_path: /shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
prompts_json_path: /shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
num_prompts: 8185
default_num_steps: 11
default_prompts_per_step: 128
supervise_student_cot: true
opd_supervise_buffer_only: false
opd_hidden_match_coef: 1.0
learning_rate: 3e-6
max_new_tokens: 64
eval_max_new_tokens: 64
eval_num_problems: 256
score_min_control_n: 250
eval_accuracy_every: 5
eval_control_start_step: 5
eval_control_max_concurrency: 128
eval_answer_logprob_control: true
eval_answer_logprob_batch_size: 64
eval_answer_logprob_max_concurrency: 2
eval_answer_logprob_distractor_control: true
eval_answer_logprob_distractor_offset: 1
opd_pipeline_rl: false
sampler_quiesce_before_sync: true
sync_method: nccl_broadcast
request_timeout: 1200
weight_sync_timeout: 900
opd_contrastive_corrupt_buffer_weight: 0.0
opd_contrastive_corrupt_answer_weight: 0.0
opd_contrastive_corrupt_buffer_mode: rotate_preserve_ws
opd_contrastive_corrupt_buffer_span: memory_only
opd_positive_answer_weight: 2.0
opd_cache_mismatch_memory_weight: 0.0
opd_teacher_memory_pair_diagnostics: false
opd_loss_max_clamp: 5.0
wandb_group: q36-ptc-autoresearch
client_args:
  student_generated_memory_tokens: 100
  eval_generated_memory_control: true
  eval_generated_memory_tokens: 100
  eval_generated_memory_shuffle_offset: 1
  eval_generated_memory_temperature: 0.0
  opd_ptc_positive_buffer_kl_weight: 0.0
  opd_ptc_positive_answer_kl_weight: 0.001
  opd_ptc_positive_hidden_weight: 1.0
  opd_mask_zero_weight_positions: true
  opd_teacher_answer_source: gold
  eval_corrupt_pause_control: true
```


### `experiments/opd_profile/autoresearch/candidates/PTC-018.yaml`

```yaml
id: PTC-018
base_config: AN
slug: ptc-018-generated-memory100-positive-answer15-flashqla
buffer_label: generated-memory100-positive-answer15-flashqla
trainer_config: experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node_nccl.yaml
gdn_backend: flashqla
student_prefill_text: "Memory: "
student_prefill_count: 1
student_prefill_suffix: "\nAnswer: "
student_stop_sequences: '["\n"]'
teacher_cot_json_path: /shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
prompts_json_path: /shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
num_prompts: 8185
default_num_steps: 11
default_prompts_per_step: 128
supervise_student_cot: true
opd_supervise_buffer_only: false
opd_hidden_match_coef: 0.0
learning_rate: 3e-6
max_new_tokens: 64
eval_max_new_tokens: 64
eval_num_problems: 256
score_min_control_n: 250
eval_accuracy_every: 5
eval_control_start_step: 5
eval_control_max_concurrency: 128
eval_answer_logprob_control: true
eval_answer_logprob_batch_size: 64
eval_answer_logprob_max_concurrency: 2
eval_answer_logprob_distractor_control: true
eval_answer_logprob_distractor_offset: 1
opd_pipeline_rl: false
sampler_quiesce_before_sync: true
sync_method: nccl_broadcast
request_timeout: 1200
weight_sync_timeout: 900
opd_contrastive_corrupt_buffer_weight: 0.0
opd_contrastive_corrupt_answer_weight: 0.0
opd_contrastive_corrupt_buffer_mode: rotate_preserve_ws
opd_contrastive_corrupt_buffer_span: memory_only
opd_positive_answer_weight: 1.5
opd_cache_mismatch_memory_weight: 0.0
opd_teacher_memory_pair_diagnostics: false
opd_loss_max_clamp: 5.0
wandb_group: q36-ptc-autoresearch
client_args:
  student_generated_memory_tokens: 100
  eval_generated_memory_control: true
  eval_generated_memory_tokens: 100
  eval_generated_memory_shuffle_offset: 1
  eval_generated_memory_temperature: 0.0
  opd_ptc_positive_buffer_kl_weight: 0.0
  # Keep PTC-positive masking active so prompt and memory rows stay zero-weighted;
  # the substantive answer utility comes from opd_positive_answer_weight above.
  opd_ptc_positive_answer_kl_weight: 0.001
  opd_ptc_positive_hidden_weight: 0.0
  opd_mask_zero_weight_positions: true
  opd_teacher_answer_source: gold
  eval_corrupt_pause_control: true
```


### `experiments/opd_profile/autoresearch/candidates/PTC-019.yaml`

```yaml
id: PTC-019
base_config: AN
slug: ptc-019-generated-memory100-sampled-pg-k1-flashqla
buffer_label: generated-memory100-sampled-pg-k1-flashqla
trainer_config: experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node_nccl.yaml
gdn_backend: flashqla
student_prefill_text: "Memory: "
student_prefill_count: 1
student_prefill_suffix: "\nAnswer: "
student_stop_sequences: '["\n"]'
teacher_cot_json_path: /shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
prompts_json_path: /shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
num_prompts: 8185
default_num_steps: 11
default_prompts_per_step: 128
supervise_student_cot: true
opd_supervise_buffer_only: false
opd_hidden_match_coef: 0.0
opd_loss_mode: k1
opd_use_policy_gradient: true
learning_rate: 1e-6
max_new_tokens: 64
eval_max_new_tokens: 64
eval_num_problems: 256
score_min_control_n: 250
eval_accuracy_every: 5
eval_control_start_step: 5
eval_control_max_concurrency: 128
eval_answer_logprob_control: true
eval_answer_logprob_batch_size: 64
eval_answer_logprob_max_concurrency: 2
eval_answer_logprob_distractor_control: true
eval_answer_logprob_distractor_offset: 1
opd_pipeline_rl: false
sampler_quiesce_before_sync: true
sync_method: nccl_broadcast
request_timeout: 1200
weight_sync_timeout: 900
opd_contrastive_corrupt_buffer_weight: 0.0
opd_contrastive_corrupt_answer_weight: 0.0
opd_contrastive_corrupt_buffer_mode: rotate_preserve_ws
opd_contrastive_corrupt_buffer_span: memory_only
opd_positive_answer_weight: 1.0
opd_cache_mismatch_memory_weight: 0.0
opd_teacher_memory_pair_diagnostics: false
opd_loss_max_clamp: 5.0
wandb_group: q36-ptc-autoresearch
client_args:
  student_generated_memory_tokens: 100
  eval_generated_memory_control: true
  eval_generated_memory_tokens: 100
  eval_generated_memory_shuffle_offset: 1
  eval_generated_memory_temperature: 0.0
  opd_ptc_positive_buffer_kl_weight: 0.1
  opd_ptc_positive_answer_kl_weight: 0.001
  opd_ptc_positive_hidden_weight: 0.0
  opd_mask_zero_weight_positions: true
  opd_teacher_answer_source: sampled
  eval_corrupt_pause_control: true
```


### `experiments/opd_profile/autoresearch/candidates/PTC-020.yaml`

```yaml
id: PTC-020
base_config: AM
slug: ptc-020-am-pause-vs-nopause-final1k-flashqla
buffer_label: randsymbol-am-pause-vs-nopause-final1k-flashqla
trainer_config: experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node.yaml
gdn_backend: flashqla
score_mode: pause_vs_nopause
score_min_control_n: 1000
student_prefill_text: " ! | ~ _ * ^ # @ "
student_prefill_count: 1
student_prefill_suffix: "Answer: "
student_stop_sequences: '["\n"]'
teacher_cot_json_path: /shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
prompts_json_path: /shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
num_prompts: 8185
default_num_steps: 6
default_prompts_per_step: 128
supervise_student_cot: true
opd_supervise_buffer_only: false
opd_hidden_match_coef: 2.0
learning_rate: 3e-6
max_new_tokens: 64
eval_max_new_tokens: 64
eval_num_problems: 1024
eval_accuracy_every: 5
eval_control_start_step: 5
eval_control_max_concurrency: 128
eval_answer_logprob_control: true
eval_answer_logprob_batch_size: 64
eval_answer_logprob_max_concurrency: 2
eval_answer_logprob_distractor_control: false
eval_answer_logprob_distractor_offset: 1
opd_pipeline_rl: false
sampler_quiesce_before_sync: true
sync_method: p2p
serial_endpoint_sync: false
request_timeout: 1200
weight_sync_timeout: 900
opd_contrastive_corrupt_buffer_weight: 1.0
opd_contrastive_corrupt_answer_weight: 0.125
opd_contrastive_corrupt_buffer_mode: rotate
opd_contrastive_corrupt_buffer_span: memory_only
opd_positive_answer_weight: 0.0
opd_loss_max_clamp: 5.0
wandb_group: q36-ptc-autoresearch
client_args:
  eval_corrupt_pause_control: true
```


### `experiments/opd_profile/autoresearch/candidates/PTC-021.yaml`

```yaml
id: PTC-021
base_config: AM
slug: ptc-021-am-pause-vs-nopause-stability1k-flashqla
buffer_label: randsymbol-am-pause-vs-nopause-stability1k-flashqla
trainer_config: experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node.yaml
gdn_backend: flashqla
score_mode: pause_vs_nopause
score_min_control_n: 1000
student_prefill_text: " ! | ~ _ * ^ # @ "
student_prefill_count: 1
student_prefill_suffix: "Answer: "
student_stop_sequences: '["\n"]'
teacher_cot_json_path: /shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
prompts_json_path: /shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
num_prompts: 8185
default_num_steps: 11
default_prompts_per_step: 128
supervise_student_cot: true
opd_supervise_buffer_only: false
opd_hidden_match_coef: 2.0
learning_rate: 3e-6
max_new_tokens: 64
eval_max_new_tokens: 64
eval_num_problems: 1024
eval_accuracy_every: 5
eval_control_start_step: 5
eval_control_max_concurrency: 128
eval_answer_logprob_control: true
eval_answer_logprob_batch_size: 64
eval_answer_logprob_max_concurrency: 2
eval_answer_logprob_distractor_control: false
eval_answer_logprob_distractor_offset: 1
opd_pipeline_rl: false
sampler_quiesce_before_sync: true
sync_method: p2p
serial_endpoint_sync: false
request_timeout: 1200
weight_sync_timeout: 900
opd_contrastive_corrupt_buffer_weight: 1.0
opd_contrastive_corrupt_answer_weight: 0.125
opd_contrastive_corrupt_buffer_mode: rotate
opd_contrastive_corrupt_buffer_span: memory_only
opd_positive_answer_weight: 0.0
opd_loss_max_clamp: 5.0
wandb_group: q36-ptc-autoresearch
client_args:
  eval_corrupt_pause_control: true
```


### `experiments/opd_profile/autoresearch/candidates/PTC-022.yaml`

```yaml
id: PTC-022
base_config: AM
slug: ptc-022-am-pause-vs-nopause-batch256-final1k-flashqla
buffer_label: randsymbol-am-pause-vs-nopause-batch256-final1k-flashqla
trainer_config: experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node.yaml
gdn_backend: flashqla
score_mode: pause_vs_nopause
score_min_control_n: 1000
student_prefill_text: " ! | ~ _ * ^ # @ "
student_prefill_count: 1
student_prefill_suffix: "Answer: "
student_stop_sequences: '["\n"]'
teacher_cot_json_path: /shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
prompts_json_path: /shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
num_prompts: 8185
default_num_steps: 6
default_prompts_per_step: 256
supervise_student_cot: true
opd_supervise_buffer_only: false
opd_hidden_match_coef: 2.0
learning_rate: 3e-6
max_new_tokens: 64
eval_max_new_tokens: 64
eval_num_problems: 1024
eval_accuracy_every: 5
eval_control_start_step: 5
eval_control_max_concurrency: 128
eval_answer_logprob_control: true
eval_answer_logprob_batch_size: 64
eval_answer_logprob_max_concurrency: 2
eval_answer_logprob_distractor_control: false
eval_answer_logprob_distractor_offset: 1
opd_pipeline_rl: false
sampler_quiesce_before_sync: true
sync_method: p2p
serial_endpoint_sync: false
request_timeout: 1200
weight_sync_timeout: 900
opd_contrastive_corrupt_buffer_weight: 1.0
opd_contrastive_corrupt_answer_weight: 0.125
opd_contrastive_corrupt_buffer_mode: rotate
opd_contrastive_corrupt_buffer_span: memory_only
opd_positive_answer_weight: 0.0
opd_loss_max_clamp: 5.0
wandb_group: q36-ptc-autoresearch
client_args:
  eval_corrupt_pause_control: true
```


### `experiments/opd_profile/autoresearch/candidates/PTC-023.yaml`

```yaml
id: PTC-023
base_config: AM
slug: ptc-023-am-nohidden-pause-vs-nopause-final1k-flashqla
buffer_label: randsymbol-am-nohidden-pause-vs-nopause-final1k-flashqla
trainer_config: experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node.yaml
gdn_backend: flashqla
score_mode: pause_vs_nopause
score_min_control_n: 1000
student_prefill_text: " ! | ~ _ * ^ # @ "
student_prefill_count: 1
student_prefill_suffix: "Answer: "
student_stop_sequences: '["\n"]'
teacher_cot_json_path: /shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
prompts_json_path: /shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
num_prompts: 8185
default_num_steps: 6
default_prompts_per_step: 128
supervise_student_cot: true
opd_supervise_buffer_only: false
opd_hidden_match_coef: 0.0
learning_rate: 3e-6
max_new_tokens: 64
eval_max_new_tokens: 64
eval_num_problems: 1024
eval_accuracy_every: 5
eval_control_start_step: 5
eval_control_max_concurrency: 128
eval_answer_logprob_control: true
eval_answer_logprob_batch_size: 64
eval_answer_logprob_max_concurrency: 2
eval_answer_logprob_distractor_control: false
eval_answer_logprob_distractor_offset: 1
opd_pipeline_rl: false
sampler_quiesce_before_sync: true
sync_method: p2p
serial_endpoint_sync: false
request_timeout: 1200
weight_sync_timeout: 900
opd_contrastive_corrupt_buffer_weight: 1.0
opd_contrastive_corrupt_answer_weight: 0.125
opd_contrastive_corrupt_buffer_mode: rotate
opd_contrastive_corrupt_buffer_span: memory_only
opd_positive_answer_weight: 0.0
opd_loss_max_clamp: 5.0
wandb_group: q36-ptc-autoresearch
client_args:
  eval_corrupt_pause_control: false
```


### `experiments/opd_profile/autoresearch/candidates/PTC-024.yaml`

```yaml
id: PTC-024
base_config: AM
slug: ptc-024-am-nocorrupt-keephidden-positive-answer-final1k-flashqla
buffer_label: randsymbol-am-nocorrupt-keephidden-positive-answer-final1k-flashqla
trainer_config: experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node.yaml
gdn_backend: flashqla
score_mode: pause_vs_nopause
score_min_control_n: 1000
student_prefill_text: " ! | ~ _ * ^ # @ "
student_prefill_count: 1
student_prefill_suffix: "Answer: "
student_stop_sequences: '["\n"]'
teacher_cot_json_path: /shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
prompts_json_path: /shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
num_prompts: 8185
default_num_steps: 6
default_prompts_per_step: 128
supervise_student_cot: true
opd_supervise_buffer_only: false
opd_hidden_match_coef: 2.0
learning_rate: 3e-6
max_new_tokens: 64
eval_max_new_tokens: 64
eval_num_problems: 1024
eval_accuracy_every: 5
eval_control_start_step: 5
eval_control_max_concurrency: 128
eval_answer_logprob_control: true
eval_answer_logprob_batch_size: 64
eval_answer_logprob_max_concurrency: 2
eval_answer_logprob_distractor_control: false
eval_answer_logprob_distractor_offset: 1
opd_pipeline_rl: false
sampler_quiesce_before_sync: true
sync_method: p2p
serial_endpoint_sync: false
request_timeout: 1200
weight_sync_timeout: 900
opd_contrastive_corrupt_buffer_weight: 0.0
opd_contrastive_corrupt_answer_weight: 0.0
opd_contrastive_corrupt_buffer_mode: rotate
opd_contrastive_corrupt_buffer_span: memory_only
opd_positive_answer_weight: 0.125
opd_loss_max_clamp: 5.0
wandb_group: q36-ptc-autoresearch
client_args:
  eval_corrupt_pause_control: false
```


### `experiments/opd_profile/autoresearch/candidates/PTC-025.yaml`

```yaml
id: PTC-025
base_config: AM
slug: ptc-025-am-nohidden-nocorrupt-positive-answer-final1k-flashqla
buffer_label: randsymbol-am-nohidden-nocorrupt-positive-answer-final1k-flashqla
trainer_config: experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_8node.yaml
gdn_backend: flashqla
score_mode: pause_vs_nopause
score_min_control_n: 1000
student_prefill_text: " ! | ~ _ * ^ # @ "
student_prefill_count: 1
student_prefill_suffix: "Answer: "
student_stop_sequences: '["\n"]'
teacher_cot_json_path: /shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
prompts_json_path: /shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
num_prompts: 8185
default_num_steps: 6
default_prompts_per_step: 128
supervise_student_cot: true
opd_supervise_buffer_only: false
opd_hidden_match_coef: 0.0
learning_rate: 3e-6
max_new_tokens: 64
eval_max_new_tokens: 64
eval_num_problems: 1024
eval_accuracy_every: 5
eval_control_start_step: 5
eval_control_max_concurrency: 128
eval_answer_logprob_control: true
eval_answer_logprob_batch_size: 64
eval_answer_logprob_max_concurrency: 2
eval_answer_logprob_distractor_control: false
eval_answer_logprob_distractor_offset: 1
opd_pipeline_rl: false
sampler_quiesce_before_sync: true
sync_method: p2p
serial_endpoint_sync: false
request_timeout: 1200
weight_sync_timeout: 900
opd_contrastive_corrupt_buffer_weight: 0.0
opd_contrastive_corrupt_answer_weight: 0.0
opd_contrastive_corrupt_buffer_mode: rotate
opd_contrastive_corrupt_buffer_span: memory_only
opd_positive_answer_weight: 0.125
opd_loss_max_clamp: 5.0
wandb_group: q36-ptc-autoresearch
client_args:
  eval_corrupt_pause_control: false
```


## 4. Autoresearch Runs Log


### `experiments/opd_profile/autoresearch/runs.jsonl`

```jsonl
{"candidate": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/candidates/PTC-001.yaml", "event": "launch", "idea_id": "PTC-001", "num_steps": 9, "prompts_per_step": 128, "sampler_layout": "spare-teacher1", "sampler_replicas": 2, "time_utc": "2026-06-03T22:58:59Z"}
{"event": "score", "idea_id": "PTC-001", "score": {"control_rows": 2, "metrics": {"acc_corrupt": 0.0, "acc_nopause": 0.808, "acc_pause": 0.802, "answer_logprob_margin": 0.0011698661124863946, "answer_logprob_z": 0.5921812725913325, "answer_select_delta": 0.0, "answer_select_z": 0.0, "cap_hit_frac_max": 0.0, "control_n": 1000.0, "delta": -0.006000000000000005, "delta_z": -0.33863625839560096, "failure_frac_max": 0.372, "step": 8, "vs_corrupt_delta": 0.802, "vs_corrupt_z": 63.643578234611006}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T225859Z-configPTC-001-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "request_failure_frac_max=0.3720", "rows": 9, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T225859Z-configPTC-001-er-opd-q36-35b-slots-trainer-head", "verdict": "infra_invalid"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T011532Z-PTC-001-step8-infra_invalid.json", "time_utc": "2026-06-04T01:15:32Z"}
{"event": "score", "idea_id": "PTC-005", "score": {"control_rows": 2, "metrics": {"acc_corrupt": 0.0, "acc_nopause": 0.799, "acc_pause": 0.799, "answer_logprob_margin": -0.004177647228492835, "answer_logprob_z": -2.164316609339693, "answer_select_delta": 0.0, "answer_select_z": 0.0, "cap_hit_frac_max": 0.0, "control_n": 1000.0, "delta": 0.0, "delta_z": 0.0, "failure_frac_max": 0.436, "step": 8, "vs_corrupt_delta": 0.799, "vs_corrupt_z": 63.04858743944589}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T000006Z-configPTC-005-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "request_failure_frac_max=0.4360", "rows": 9, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T000006Z-configPTC-005-er-opd-q36-35b-slots-trainer-head", "verdict": "infra_invalid"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T011532Z-PTC-005-step8-infra_invalid.json", "time_utc": "2026-06-04T01:15:32Z"}
{"event": "advance", "idea_id": "PTC-001", "score": {"control_rows": 2, "metrics": {"acc_corrupt": 0.0, "acc_nopause": 0.808, "acc_pause": 0.802, "answer_logprob_margin": 0.0011698661124863946, "answer_logprob_z": 0.5921812725913325, "answer_select_delta": 0.0, "answer_select_z": 0.0, "cap_hit_frac_max": 0.0, "control_n": 1000.0, "delta": -0.006000000000000005, "delta_z": -0.33863625839560096, "failure_frac_max": 0.372, "step": 8, "vs_corrupt_delta": 0.802, "vs_corrupt_z": 63.643578234611006}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T225859Z-configPTC-001-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "request_failure_frac_max=0.3720", "rows": 9, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T225859Z-configPTC-001-er-opd-q36-35b-slots-trainer-head", "verdict": "infra_invalid"}, "status": "infra_invalid", "time_utc": "2026-06-04T01:15:37Z"}
{"candidate": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/candidates/PTC-006.yaml", "event": "launch", "idea_id": "PTC-006", "num_steps": 9, "prompts_per_step": 128, "sampler_layout": "spare-teacher1", "sampler_replicas": 2, "time_utc": "2026-06-04T01:15:56Z"}
{"candidate": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/candidates/PTC-006.yaml", "event": "launch", "idea_id": "PTC-006", "num_steps": 9, "prompts_per_step": 128, "sampler_layout": "spare-teacher1", "sampler_replicas": 2, "time_utc": "2026-06-04T01:36:00Z"}
{"event": "score", "idea_id": "PTC-006", "score": {"control_rows": 2, "metrics": {"acc_corrupt": 0.0, "acc_nopause": 0.812, "acc_pause": 0.807, "answer_logprob_margin": -0.0005920925329430714, "answer_logprob_z": -0.3690380389251624, "answer_select_delta": 0.0, "answer_select_z": 0.0, "cap_hit_frac_max": 0.0, "control_n": 1000.0, "delta": -0.0050000000000000044, "delta_z": -0.28471338904946075, "failure_frac_max": 0.0, "step": 8, "vs_corrupt_delta": 0.807, "vs_corrupt_z": 64.6633369867274}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T013603Z-configPTC-006-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "no positive accuracy or answer-logprob signal", "rows": 9, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T013603Z-configPTC-006-er-opd-q36-35b-slots-trainer-head", "verdict": "science_reject"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T021202Z-PTC-006-step8-science_reject.json", "time_utc": "2026-06-04T02:12:02Z"}
{"event": "advance", "idea_id": "PTC-006", "score": {"control_rows": 2, "metrics": {"acc_corrupt": 0.0, "acc_nopause": 0.812, "acc_pause": 0.807, "answer_logprob_margin": -0.0005920925329430714, "answer_logprob_z": -0.3690380389251624, "answer_select_delta": 0.0, "answer_select_z": 0.0, "cap_hit_frac_max": 0.0, "control_n": 1000.0, "delta": -0.0050000000000000044, "delta_z": -0.28471338904946075, "failure_frac_max": 0.0, "step": 8, "vs_corrupt_delta": 0.807, "vs_corrupt_z": 64.6633369867274}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T013603Z-configPTC-006-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "no positive accuracy or answer-logprob signal", "rows": 9, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T013603Z-configPTC-006-er-opd-q36-35b-slots-trainer-head", "verdict": "science_reject"}, "status": "rejected", "time_utc": "2026-06-04T02:12:05Z"}
{"candidate": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/candidates/PTC-003.yaml", "event": "launch", "idea_id": "PTC-003", "num_steps": 9, "prompts_per_step": 128, "sampler_layout": "spare-teacher1", "sampler_replicas": 2, "time_utc": "2026-06-04T02:12:44Z"}
{"event": "score", "idea_id": "PTC-003", "score": {"control_rows": 2, "metrics": {"acc_corrupt": 0.0, "acc_nopause": 0.803, "acc_pause": 0.797, "answer_logprob_margin": -0.002851773091628856, "answer_logprob_z": -1.550627938419657, "answer_select_delta": 0.0, "answer_select_z": 0.0, "cap_hit_frac_max": 0.0, "control_n": 1000.0, "delta": -0.006000000000000005, "delta_z": -0.3354196304347396, "failure_frac_max": 0.0, "step": 8, "vs_corrupt_delta": 0.797, "vs_corrupt_z": 62.65866559690078}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T021246Z-configPTC-003-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "no positive accuracy or answer-logprob signal", "rows": 9, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T021246Z-configPTC-003-er-opd-q36-35b-slots-trainer-head", "verdict": "science_reject"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T045539Z-PTC-003-step8-science_reject.json", "time_utc": "2026-06-04T04:55:39Z"}
{"event": "advance", "idea_id": "PTC-003", "score": {"control_rows": 2, "metrics": {"acc_corrupt": 0.0, "acc_nopause": 0.803, "acc_pause": 0.797, "answer_logprob_margin": -0.002851773091628856, "answer_logprob_z": -1.550627938419657, "answer_select_delta": 0.0, "answer_select_z": 0.0, "cap_hit_frac_max": 0.0, "control_n": 1000.0, "delta": -0.006000000000000005, "delta_z": -0.3354196304347396, "failure_frac_max": 0.0, "step": 8, "vs_corrupt_delta": 0.797, "vs_corrupt_z": 62.65866559690078}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T021246Z-configPTC-003-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "no positive accuracy or answer-logprob signal", "rows": 9, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T021246Z-configPTC-003-er-opd-q36-35b-slots-trainer-head", "verdict": "science_reject"}, "status": "rejected", "time_utc": "2026-06-04T04:55:43Z"}
{"candidate": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/candidates/PTC-007.yaml", "event": "launch", "idea_id": "PTC-007", "num_steps": 9, "prompts_per_step": 128, "sampler_layout": "spare-teacher1", "sampler_replicas": 2, "time_utc": "2026-06-04T05:08:09Z"}
{"event": "advance", "idea_id": "PTC-007", "score": {"control_rows": 1, "metrics": {"acc_corrupt": 0.0, "acc_nopause": 0.766, "acc_pause": 0.677, "answer_logprob_margin": -0.05189284104206759, "answer_logprob_z": -8.495199167432943, "answer_select_delta": 0.0, "answer_select_z": 0.0, "cap_hit_frac_max": 0.0, "control_n": 1000.0, "delta": -0.08899999999999997, "delta_z": -4.461643350547222, "failure_frac_max": 0.0, "step": 4, "vs_corrupt_delta": 0.677, "vs_corrupt_z": 45.781822071627325}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T050810Z-configPTC-007-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "no positive accuracy or answer-logprob signal", "rows": 5, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T050810Z-configPTC-007-er-opd-q36-35b-slots-trainer-head", "verdict": "science_reject"}, "status": "rejected", "time_utc": "2026-06-04T05:30:33Z"}
{"candidate": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/candidates/PTC-008.yaml", "event": "launch", "idea_id": "PTC-008", "num_steps": 9, "prompts_per_step": 128, "sampler_layout": "spare-teacher1", "sampler_replicas": 2, "time_utc": "2026-06-04T05:31:33Z"}
{"event": "score", "idea_id": "PTC-008", "score": {"control_rows": 1, "metrics": {"acc_corrupt": 0.591, "acc_nopause": 0.71, "acc_pause": 0.624, "answer_logprob_margin": -0.004445023867406381, "answer_logprob_z": -0.7627423018328828, "answer_select_delta": -0.025082884714261792, "answer_select_z": -3.434489752279603, "cap_hit_frac_max": 0.004, "control_n": 1000.0, "delta": -0.08599999999999997, "delta_z": -4.097450014496387, "failure_frac_max": 0.0, "step": 4, "vs_corrupt_delta": 0.03300000000000003, "vs_corrupt_z": 1.5120078506650454}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T053133Z-configPTC-008-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "no positive accuracy or answer-logprob signal", "rows": 5, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T053133Z-configPTC-008-er-opd-q36-35b-slots-trainer-head", "verdict": "science_reject"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T060104Z-PTC-008-step4-science_reject.json", "time_utc": "2026-06-04T06:01:04Z"}
{"event": "advance", "idea_id": "PTC-008", "score": {"control_rows": 1, "metrics": {"acc_corrupt": 0.591, "acc_nopause": 0.71, "acc_pause": 0.624, "answer_logprob_margin": -0.004445023867406381, "answer_logprob_z": -0.7627423018328828, "answer_select_delta": -0.025082884714261792, "answer_select_z": -3.434489752279603, "cap_hit_frac_max": 0.004, "control_n": 1000.0, "delta": -0.08599999999999997, "delta_z": -4.097450014496387, "failure_frac_max": 0.0, "step": 4, "vs_corrupt_delta": 0.03300000000000003, "vs_corrupt_z": 1.5120078506650454}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T053133Z-configPTC-008-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "no positive accuracy or answer-logprob signal", "rows": 5, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T053133Z-configPTC-008-er-opd-q36-35b-slots-trainer-head", "verdict": "science_reject"}, "status": "rejected", "time_utc": "2026-06-04T06:01:04Z"}
{"candidate": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/candidates/PTC-009.yaml", "event": "launch", "idea_id": "PTC-009", "num_steps": 9, "prompts_per_step": 128, "sampler_layout": "spare-teacher1", "sampler_replicas": 2, "time_utc": "2026-06-04T06:01:42Z"}
{"event": "advance", "idea_id": "PTC-009", "score": {"control_rows": 1, "metrics": {"acc_corrupt": 0.719, "acc_nopause": 0.801, "acc_pause": 0.719, "answer_logprob_margin": -0.06802329225159871, "answer_logprob_z": -7.5156321540708575, "answer_select_delta": -0.17130115062077741, "answer_select_z": -14.448557610375241, "cap_hit_frac_max": 0.0, "control_n": 1000.0, "delta": -0.08200000000000007, "delta_z": -4.313173687822969, "failure_frac_max": 0.0, "step": 4, "vs_corrupt_delta": 0.0, "vs_corrupt_z": 0.0}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T060143Z-configPTC-009-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "no positive accuracy or answer-logprob signal", "rows": 5, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T060143Z-configPTC-009-er-opd-q36-35b-slots-trainer-head", "verdict": "science_reject"}, "status": "rejected", "time_utc": "2026-06-04T06:30:58Z"}
{"candidate": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/candidates/PTC-010.yaml", "event": "launch", "idea_id": "PTC-010", "num_steps": 9, "prompts_per_step": 128, "sampler_layout": "spare-teacher1", "sampler_replicas": 2, "time_utc": "2026-06-04T06:33:14Z"}
{"event": "advance", "idea_id": "PTC-010", "score": {"control_rows": 1, "metrics": {"acc_corrupt": 0.677, "acc_nopause": 0.789, "acc_pause": 0.684, "answer_logprob_margin": -0.07165765871649975, "answer_logprob_z": -8.41706506088195, "answer_select_delta": -0.19453478057920348, "answer_select_z": -16.804226108679384, "cap_hit_frac_max": 0.002, "control_n": 1000.0, "delta": -0.10499999999999998, "delta_z": -5.367891918486458, "failure_frac_max": 0.0, "step": 4, "vs_corrupt_delta": 0.007000000000000006, "vs_corrupt_z": 0.3356957021998745}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T063316Z-configPTC-010-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "no positive accuracy or answer-logprob signal", "rows": 5, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T063316Z-configPTC-010-er-opd-q36-35b-slots-trainer-head", "verdict": "science_reject"}, "status": "rejected", "time_utc": "2026-06-04T07:02:29Z"}
{"candidate": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/candidates/PTC-011.yaml", "event": "launch", "idea_id": "PTC-011", "num_steps": 9, "prompts_per_step": 128, "sampler_layout": "spare-teacher1", "sampler_replicas": 2, "time_utc": "2026-06-04T07:08:21Z"}
{"candidate": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/candidates/PTC-011.yaml", "event": "launch", "idea_id": "PTC-011", "num_steps": 9, "prompts_per_step": 128, "sampler_layout": "spare-teacher1", "sampler_replicas": 2, "time_utc": "2026-06-04T07:16:41Z"}
{"candidate": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/candidates/PTC-011.yaml", "event": "launch", "idea_id": "PTC-011", "num_steps": 9, "prompts_per_step": 128, "sampler_layout": "spare-teacher1", "sampler_replicas": 2, "time_utc": "2026-06-04T07:23:28Z"}
{"candidate": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/candidates/PTC-011.yaml", "event": "launch", "idea_id": "PTC-011", "num_steps": 9, "prompts_per_step": 128, "sampler_layout": "spare-teacher1", "sampler_replicas": 2, "time_utc": "2026-06-04T07:35:01Z"}
{"event": "score", "idea_id": "PTC-011", "score": {"profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "no control rows", "rows": 1, "verdict": "incomplete"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T074639Z-PTC-011-stepna-incomplete.json", "time_utc": "2026-06-04T07:46:39Z"}
{"event": "score", "idea_id": "PTC-011", "score": {"profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "no control rows", "rows": 1, "verdict": "incomplete"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T074650Z-PTC-011-stepna-incomplete.json", "time_utc": "2026-06-04T07:46:50Z"}
{"event": "score", "idea_id": "PTC-011", "score": {"profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "no control rows", "rows": 1, "verdict": "incomplete"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T074659Z-PTC-011-stepna-incomplete.json", "time_utc": "2026-06-04T07:46:59Z"}
{"event": "score", "idea_id": "PTC-011", "score": {"profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "no control rows", "rows": 2, "verdict": "incomplete"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T075411Z-PTC-011-stepna-incomplete.json", "time_utc": "2026-06-04T07:54:11Z"}
{"event": "score", "idea_id": "PTC-011", "score": {"profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "no control rows", "rows": 2, "verdict": "incomplete"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T075543Z-PTC-011-stepna-incomplete.json", "time_utc": "2026-06-04T07:55:43Z"}
{"event": "score", "idea_id": "PTC-011", "score": {"profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "no control rows", "rows": 3, "verdict": "incomplete"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T075929Z-PTC-011-stepna-incomplete.json", "time_utc": "2026-06-04T07:59:29Z"}
{"event": "score", "idea_id": "PTC-011", "score": {"profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "no control rows", "rows": 3, "verdict": "incomplete"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T075957Z-PTC-011-stepna-incomplete.json", "time_utc": "2026-06-04T07:59:57Z"}
{"event": "score", "idea_id": "PTC-011", "score": {"profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "no control rows", "rows": 3, "verdict": "incomplete"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T080016Z-PTC-011-stepna-incomplete.json", "time_utc": "2026-06-04T08:00:16Z"}
{"event": "autopilot_score", "idea_id": "PTC-011", "score": {"control_rows": 1, "metrics": {"acc_corrupt": 0.00390625, "acc_nopause": 0.73828125, "acc_pause": 0.78515625, "answer_logprob_margin": -0.1338996193211561, "answer_logprob_z": -4.934445616493985, "answer_select_delta": 0.05212081693119896, "answer_select_z": 1.4338520989113466, "cap_hit_frac_max": 0.0, "control_n": 256.0, "delta": 0.046875, "delta_z": 1.2467006799461504, "failure_frac_max": 0.0, "step": 4, "vs_corrupt_delta": 0.78125, "vs_corrupt_z": 30.08975668093039}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "completed cleanly but missed both reject and promotion thresholds", "rows": 5, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head", "verdict": "inconclusive"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T082853Z-PTC-011-step4-inconclusive.json", "time_utc": "2026-06-04T08:28:53Z"}
{"event": "score", "idea_id": "PTC-011", "score": {"control_rows": 1, "metrics": {"acc_corrupt": 0.00390625, "acc_nopause": 0.73828125, "acc_pause": 0.78515625, "answer_logprob_margin": -0.1338996193211561, "answer_logprob_z": -4.934445616493985, "answer_select_delta": 0.05212081693119896, "answer_select_z": 1.4338520989113466, "cap_hit_frac_max": 0.0, "control_n": 256.0, "delta": 0.046875, "delta_z": 1.2467006799461504, "failure_frac_max": 0.0, "step": 4, "vs_corrupt_delta": 0.78125, "vs_corrupt_z": 30.08975668093039}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "completed cleanly but missed both reject and promotion thresholds", "rows": 5, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head", "verdict": "inconclusive"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T082854Z-PTC-011-step4-inconclusive.json", "time_utc": "2026-06-04T08:28:54Z"}
{"event": "score", "idea_id": "PTC-011", "score": {"control_rows": 2, "metrics": {"acc_corrupt": 0.30078125, "acc_nopause": 0.78125, "acc_pause": 0.80859375, "answer_logprob_margin": -0.027456654039798866, "answer_logprob_z": -2.1209689010269392, "answer_select_delta": 0.1930973367662947, "answer_select_z": 7.5565314530967775, "cap_hit_frac_max": 0.0, "control_n": 256.0, "delta": 0.02734375, "delta_z": 0.7666379087775477, "failure_frac_max": 0.0, "step": 8, "vs_corrupt_delta": 0.5078125, "vs_corrupt_z": 13.44708920081176}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "completed cleanly but missed both reject and promotion thresholds", "rows": 9, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head", "verdict": "inconclusive"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T091135Z-PTC-011-step8-inconclusive.json", "time_utc": "2026-06-04T09:11:35Z"}
{"event": "advance", "idea_id": "PTC-011", "score": {"control_rows": 2, "metrics": {"acc_corrupt": 0.30078125, "acc_nopause": 0.78125, "acc_pause": 0.80859375, "answer_logprob_margin": -0.027456654039798866, "answer_logprob_z": -2.1209689010269392, "answer_select_delta": 0.1930973367662947, "answer_select_z": 7.5565314530967775, "cap_hit_frac_max": 0.0, "control_n": 256.0, "delta": 0.02734375, "delta_z": 0.7666379087775477, "failure_frac_max": 0.0, "step": 8, "vs_corrupt_delta": 0.5078125, "vs_corrupt_z": 13.44708920081176}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "completed cleanly but missed both reject and promotion thresholds", "rows": 9, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head", "verdict": "inconclusive"}, "status": "inconclusive", "time_utc": "2026-06-04T09:12:12Z"}
{"candidate": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/candidates/PTC-012.yaml", "event": "launch", "idea_id": "PTC-012", "num_steps": 9, "prompts_per_step": 128, "sampler_layout": "spare-teacher1", "sampler_replicas": 2, "time_utc": "2026-06-04T09:12:13Z"}
{"event": "autopilot_wait", "idea_id": "PTC-012", "message": "autopilot: profile/score unavailable for PTC-012: profile not available yet for PTC-012 launched at 2026-06-04T09:12:13Z", "time_utc": "2026-06-04T09:13:14Z"}
{"event": "autopilot_score", "idea_id": "PTC-012", "score": {"control_rows": 1, "metrics": {"acc_corrupt": 0.0, "acc_nopause": 0.73046875, "acc_pause": 0.83984375, "answer_logprob_margin": -0.11067124789628861, "answer_logprob_z": -3.0825148106625977, "answer_select_delta": -0.3257433384510804, "answer_select_z": -6.977632527602614, "cap_hit_frac_max": 0.0, "control_n": 256.0, "delta": 0.109375, "delta_z": 3.0399616129719975, "failure_frac_max": 0.0, "step": 4, "vs_corrupt_delta": 0.83984375, "vs_corrupt_z": 36.63930982415258}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T091215Z-configPTC-012-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "completed cleanly but missed both reject and promotion thresholds", "rows": 5, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T091215Z-configPTC-012-er-opd-q36-35b-slots-trainer-head", "verdict": "inconclusive"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T100716Z-PTC-012-step4-inconclusive.json", "time_utc": "2026-06-04T10:07:16Z"}
{"event": "score", "idea_id": "PTC-012", "score": {"control_rows": 1, "metrics": {"acc_corrupt": 0.0, "acc_nopause": 0.73046875, "acc_pause": 0.83984375, "answer_logprob_margin": -0.11067124789628861, "answer_logprob_z": -3.0825148106625977, "answer_select_delta": -0.3257433384510804, "answer_select_z": -6.977632527602614, "cap_hit_frac_max": 0.0, "control_n": 256.0, "delta": 0.109375, "delta_z": 3.0399616129719975, "failure_frac_max": 0.0, "step": 4, "vs_corrupt_delta": 0.83984375, "vs_corrupt_z": 36.63930982415258}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T091215Z-configPTC-012-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "completed cleanly but missed both reject and promotion thresholds", "rows": 5, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T091215Z-configPTC-012-er-opd-q36-35b-slots-trainer-head", "verdict": "inconclusive"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T100734Z-PTC-012-step4-inconclusive.json", "time_utc": "2026-06-04T10:07:34Z"}
{"event": "score", "idea_id": "PTC-012", "score": {"control_rows": 2, "metrics": {"acc_corrupt": 0.11328125, "acc_nopause": 0.78515625, "acc_pause": 0.8046875, "answer_logprob_margin": -0.1596965955282831, "answer_logprob_z": -5.496740775541098, "answer_select_delta": -0.7489891492578584, "answer_select_z": -14.397340349047914, "cap_hit_frac_max": 0.0, "control_n": 256.0, "delta": 0.01953125, "delta_z": 0.5474446288295078, "failure_frac_max": 0.0, "step": 8, "vs_corrupt_delta": 0.69140625, "vs_corrupt_z": 21.795580416961943}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T091215Z-configPTC-012-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "completed cleanly but missed both reject and promotion thresholds", "rows": 9, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T091215Z-configPTC-012-er-opd-q36-35b-slots-trainer-head", "verdict": "inconclusive"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T105011Z-PTC-012-step8-inconclusive.json", "time_utc": "2026-06-04T10:50:11Z"}
{"event": "advance", "idea_id": "PTC-012", "score": {"control_rows": 2, "metrics": {"acc_corrupt": 0.11328125, "acc_nopause": 0.78515625, "acc_pause": 0.8046875, "answer_logprob_margin": -0.1596965955282831, "answer_logprob_z": -5.496740775541098, "answer_select_delta": -0.7489891492578584, "answer_select_z": -14.397340349047914, "cap_hit_frac_max": 0.0, "control_n": 256.0, "delta": 0.01953125, "delta_z": 0.5474446288295078, "failure_frac_max": 0.0, "step": 8, "vs_corrupt_delta": 0.69140625, "vs_corrupt_z": 21.795580416961943}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T091215Z-configPTC-012-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "completed cleanly but missed both reject and promotion thresholds", "rows": 9, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T091215Z-configPTC-012-er-opd-q36-35b-slots-trainer-head", "verdict": "inconclusive"}, "status": "inconclusive", "time_utc": "2026-06-04T10:50:43Z"}
{"event": "manual_decision", "idea_id": "PTC-012", "reason": "Step-8 accuracy delta decayed to +0.0195 while answer logprob and answer selection worsened; marking rejected to launch PTC-013 memory-KL bootstrap.", "status": "rejected", "time_utc": "2026-06-04T10:50:55Z"}
{"candidate": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/candidates/PTC-013.yaml", "event": "launch", "idea_id": "PTC-013", "num_steps": 9, "prompts_per_step": 128, "sampler_layout": "spare-teacher1", "sampler_replicas": 2, "time_utc": "2026-06-04T10:51:10Z"}
{"event": "autopilot_wait", "idea_id": "PTC-013", "message": "autopilot: profile/score unavailable for PTC-013: profile not available yet for PTC-013 launched at 2026-06-04T10:51:10Z", "time_utc": "2026-06-04T10:51:19Z"}
{"event": "autopilot_wait", "idea_id": "PTC-013", "message": "autopilot: profile/score unavailable for PTC-013: profile not available yet for PTC-013 launched at 2026-06-04T10:51:10Z", "time_utc": "2026-06-04T10:54:19Z"}
{"event": "score", "idea_id": "PTC-013", "score": {"control_rows": 1, "metrics": {"acc_corrupt": 0.1328125, "acc_nopause": 0.75, "acc_pause": 0.59765625, "answer_logprob_margin": -0.31010046812613806, "answer_logprob_z": -9.973705393936111, "answer_select_delta": -0.24107080985158527, "answer_select_z": -6.486670231064018, "cap_hit_frac_max": 0.0, "control_n": 256.0, "delta": -0.15234375, "delta_z": -3.7259874043167356, "failure_frac_max": 0.0, "step": 4, "vs_corrupt_delta": 0.46484375, "vs_corrupt_z": 12.47164544836176}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T105110Z-configPTC-013-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "no positive accuracy or answer-logprob signal", "rows": 5, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T105110Z-configPTC-013-er-opd-q36-35b-slots-trainer-head", "verdict": "science_reject"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T114657Z-PTC-013-step4-science_reject.json", "time_utc": "2026-06-04T11:46:57Z"}
{"event": "advance", "idea_id": "PTC-013", "score": {"control_rows": 1, "metrics": {"acc_corrupt": 0.1328125, "acc_nopause": 0.75, "acc_pause": 0.59765625, "answer_logprob_margin": -0.31010046812613806, "answer_logprob_z": -9.973705393936111, "answer_select_delta": -0.24107080985158527, "answer_select_z": -6.486670231064018, "cap_hit_frac_max": 0.0, "control_n": 256.0, "delta": -0.15234375, "delta_z": -3.7259874043167356, "failure_frac_max": 0.0, "step": 4, "vs_corrupt_delta": 0.46484375, "vs_corrupt_z": 12.47164544836176}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T105110Z-configPTC-013-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "no positive accuracy or answer-logprob signal", "rows": 5, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T105110Z-configPTC-013-er-opd-q36-35b-slots-trainer-head", "verdict": "science_reject"}, "status": "rejected", "time_utc": "2026-06-04T11:47:24Z"}
{"candidate": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/candidates/PTC-014.yaml", "event": "launch", "idea_id": "PTC-014", "num_steps": 9, "prompts_per_step": 128, "sampler_layout": "spare-teacher1", "sampler_replicas": 2, "time_utc": "2026-06-04T11:47:56Z"}
{"event": "autopilot_wait", "idea_id": "PTC-014", "message": "autopilot: profile/score unavailable for PTC-014: profile not available yet for PTC-014 launched at 2026-06-04T11:47:56Z", "time_utc": "2026-06-04T11:48:26Z"}
{"event": "autopilot_score", "idea_id": "PTC-014", "score": {"control_rows": 1, "metrics": {"acc_corrupt": 0.015625, "acc_nopause": 0.69921875, "acc_pause": 0.8359375, "answer_logprob_margin": -0.06823835960594374, "answer_logprob_z": -2.393474203845524, "answer_select_delta": -0.18082758527490222, "answer_select_z": -4.051345401410779, "cap_hit_frac_max": 0.0, "control_n": 256.0, "delta": 0.13671875, "delta_z": 3.7110514492390805, "failure_frac_max": 0.0, "step": 4, "vs_corrupt_delta": 0.8203125, "vs_corrupt_z": 33.60672201667224}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T114759Z-configPTC-014-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "completed cleanly but missed both reject and promotion thresholds", "rows": 5, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T114759Z-configPTC-014-er-opd-q36-35b-slots-trainer-head", "verdict": "inconclusive"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T124228Z-PTC-014-step4-inconclusive.json", "time_utc": "2026-06-04T12:42:28Z"}
{"event": "score", "idea_id": "PTC-014", "score": {"control_rows": 1, "metrics": {"acc_corrupt": 0.015625, "acc_nopause": 0.69921875, "acc_pause": 0.8359375, "answer_logprob_margin": -0.06823835960594374, "answer_logprob_z": -2.393474203845524, "answer_select_delta": -0.18082758527490222, "answer_select_z": -4.051345401410779, "cap_hit_frac_max": 0.0, "control_n": 256.0, "delta": 0.13671875, "delta_z": 3.7110514492390805, "failure_frac_max": 0.0, "step": 4, "vs_corrupt_delta": 0.8203125, "vs_corrupt_z": 33.60672201667224}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T114759Z-configPTC-014-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "completed cleanly but missed both reject and promotion thresholds", "rows": 5, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T114759Z-configPTC-014-er-opd-q36-35b-slots-trainer-head", "verdict": "inconclusive"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T124319Z-PTC-014-step4-inconclusive.json", "time_utc": "2026-06-04T12:43:19Z"}
{"event": "autopilot_infra_invalid", "idea_id": "PTC-014", "reason": "trainer-head:pid=       - exited_at=2026-06-04T13:24:14Z rc=0; trainer-worker-1:pid=       - stopped_at=2026-06-04T13:24:18Z; trainer-worker-2:pid=       - stopped_at=2026-06-04T13:24:19Z; trainer-worker-3:pid=       - stopped_at=2026-06-04T13:24:18Z; trainer-worker-4:pid=       - stopped_at=2026-06-04T13:24:20Z; trainer-worker-5:pid=       - stopped_at=2026-06-04T13:24:19Z; trainer-worker-6:pid=       - stopped_at=2026-06-04T13:24:18Z; trainer-worker-7:pid=       - stopped_at=2026-06-04T13:24:20Z", "time_utc": "2026-06-04T13:24:30Z"}
{"event": "score", "idea_id": "PTC-014", "score": {"control_rows": 2, "metrics": {"acc_corrupt": 0.3203125, "acc_nopause": 0.76953125, "acc_pause": 0.76171875, "answer_logprob_margin": -0.10689015139499593, "answer_logprob_z": -5.096342933926368, "answer_select_delta": -0.14860751767676922, "answer_select_z": -3.269913442281971, "cap_hit_frac_max": 0.0, "control_n": 256.0, "delta": -0.0078125, "delta_z": -0.20866508438634226, "failure_frac_max": 0.0, "step": 8, "vs_corrupt_delta": 0.44140625, "vs_corrupt_z": 11.177756782299781}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T114759Z-configPTC-014-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "no positive accuracy or answer-logprob signal", "rows": 9, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T114759Z-configPTC-014-er-opd-q36-35b-slots-trainer-head", "verdict": "science_reject"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T132437Z-PTC-014-step8-science_reject.json", "time_utc": "2026-06-04T13:24:37Z"}
{"event": "advance", "idea_id": "PTC-014", "score": {"control_rows": 2, "metrics": {"acc_corrupt": 0.3203125, "acc_nopause": 0.76953125, "acc_pause": 0.76171875, "answer_logprob_margin": -0.10689015139499593, "answer_logprob_z": -5.096342933926368, "answer_select_delta": -0.14860751767676922, "answer_select_z": -3.269913442281971, "cap_hit_frac_max": 0.0, "control_n": 256.0, "delta": -0.0078125, "delta_z": -0.20866508438634226, "failure_frac_max": 0.0, "step": 8, "vs_corrupt_delta": 0.44140625, "vs_corrupt_z": 11.177756782299781}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T114759Z-configPTC-014-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "no positive accuracy or answer-logprob signal", "rows": 9, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T114759Z-configPTC-014-er-opd-q36-35b-slots-trainer-head", "verdict": "science_reject"}, "status": "rejected", "time_utc": "2026-06-04T13:24:55Z"}
{"candidate": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/candidates/PTC-015.yaml", "event": "launch", "idea_id": "PTC-015", "num_steps": 9, "prompts_per_step": 128, "sampler_layout": "spare-teacher1", "sampler_replicas": 2, "time_utc": "2026-06-04T13:24:55Z"}
{"event": "score", "idea_id": "PTC-015", "score": {"control_rows": 1, "metrics": {"acc_corrupt": 0.0, "acc_nopause": 0.78515625, "acc_pause": 0.8203125, "answer_logprob_margin": 0.006929151958554657, "answer_logprob_z": 0.4353864050668189, "answer_select_delta": 0.11405637547581872, "answer_select_z": 3.151441197561941, "cap_hit_frac_max": 0.00390625, "control_n": 256.0, "delta": 0.03515625, "delta_z": 1.000506750675012, "failure_frac_max": 0.0, "step": 4, "vs_corrupt_delta": 0.8203125, "vs_corrupt_z": 34.18619095737215}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T132457Z-configPTC-015-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "completed cleanly but missed both reject and promotion thresholds", "rows": 5, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T132457Z-configPTC-015-er-opd-q36-35b-slots-trainer-head", "verdict": "inconclusive"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T142024Z-PTC-015-step4-inconclusive.json", "time_utc": "2026-06-04T14:20:24Z"}
{"event": "score", "idea_id": "PTC-015", "score": {"control_rows": 1, "metrics": {"acc_corrupt": 0.0, "acc_nopause": 0.78515625, "acc_pause": 0.8203125, "answer_logprob_margin": 0.006929151958554657, "answer_logprob_z": 0.4353864050668189, "answer_select_delta": 0.11405637547581872, "answer_select_z": 3.151441197561941, "cap_hit_frac_max": 0.00390625, "control_n": 256.0, "delta": 0.03515625, "delta_z": 1.000506750675012, "failure_frac_max": 0.0, "step": 4, "vs_corrupt_delta": 0.8203125, "vs_corrupt_z": 34.18619095737215}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T132457Z-configPTC-015-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "completed cleanly but missed both reject and promotion thresholds", "rows": 5, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T132457Z-configPTC-015-er-opd-q36-35b-slots-trainer-head", "verdict": "inconclusive"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T142105Z-PTC-015-step4-inconclusive.json", "time_utc": "2026-06-04T14:21:05Z"}
{"event": "autopilot_score", "idea_id": "PTC-015", "score": {"control_rows": 1, "metrics": {"acc_corrupt": 0.0, "acc_nopause": 0.78515625, "acc_pause": 0.8203125, "answer_logprob_margin": 0.006929151958554657, "answer_logprob_z": 0.4353864050668189, "answer_select_delta": 0.11405637547581872, "answer_select_z": 3.151441197561941, "cap_hit_frac_max": 0.00390625, "control_n": 256.0, "delta": 0.03515625, "delta_z": 1.000506750675012, "failure_frac_max": 0.0, "step": 4, "vs_corrupt_delta": 0.8203125, "vs_corrupt_z": 34.18619095737215}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T132457Z-configPTC-015-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "completed cleanly but missed both reject and promotion thresholds", "rows": 5, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T132457Z-configPTC-015-er-opd-q36-35b-slots-trainer-head", "verdict": "inconclusive"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T142112Z-PTC-015-step4-inconclusive.json", "time_utc": "2026-06-04T14:21:12Z"}
{"event": "autopilot_score", "idea_id": "PTC-015", "score": {"control_rows": 2, "metrics": {"acc_corrupt": 0.00390625, "acc_nopause": 0.8046875, "acc_pause": 0.828125, "answer_logprob_margin": -0.044649495764718904, "answer_logprob_z": -1.9204682948730094, "answer_select_delta": 0.16635415123834688, "answer_select_z": 2.769171540539168, "cap_hit_frac_max": 0.0, "control_n": 256.0, "delta": 0.0234375, "delta_z": 0.6852250139532997, "failure_frac_max": 0.0, "step": 8, "vs_corrupt_delta": 0.82421875, "vs_corrupt_z": 34.48670434808517}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T132457Z-configPTC-015-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "completed cleanly but missed both reject and promotion thresholds", "rows": 9, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T132457Z-configPTC-015-er-opd-q36-35b-slots-trainer-head", "verdict": "inconclusive"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T150312Z-PTC-015-step8-inconclusive.json", "time_utc": "2026-06-04T15:03:12Z"}
{"event": "advance", "idea_id": "PTC-015", "score": {"control_rows": 2, "metrics": {"acc_corrupt": 0.00390625, "acc_nopause": 0.8046875, "acc_pause": 0.828125, "answer_logprob_margin": -0.044649495764718904, "answer_logprob_z": -1.9204682948730094, "answer_select_delta": 0.16635415123834688, "answer_select_z": 2.769171540539168, "cap_hit_frac_max": 0.0, "control_n": 256.0, "delta": 0.0234375, "delta_z": 0.6852250139532997, "failure_frac_max": 0.0, "step": 8, "vs_corrupt_delta": 0.82421875, "vs_corrupt_z": 34.48670434808517}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T132457Z-configPTC-015-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "completed cleanly but missed both reject and promotion thresholds", "rows": 9, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T132457Z-configPTC-015-er-opd-q36-35b-slots-trainer-head", "verdict": "inconclusive"}, "status": "inconclusive", "time_utc": "2026-06-04T15:03:12Z"}
{"candidate": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/candidates/PTC-016.yaml", "event": "launch", "idea_id": "PTC-016", "num_steps": 11, "prompts_per_step": 128, "sampler_layout": "spare-teacher1", "sampler_replicas": 2, "time_utc": "2026-06-04T15:03:13Z"}
{"event": "autopilot_wait", "idea_id": "PTC-016", "message": "autopilot: profile/score unavailable for PTC-016: profile not available yet for PTC-016 launched at 2026-06-04T15:03:13Z", "time_utc": "2026-06-04T15:03:43Z"}
{"event": "autopilot_wait", "idea_id": "PTC-016", "message": "autopilot: profile/score unavailable for PTC-016: profile not available yet for PTC-016 launched at 2026-06-04T15:03:13Z", "time_utc": "2026-06-04T15:06:43Z"}
{"event": "score", "idea_id": "PTC-016", "score": {"control_rows": 1, "metrics": {"acc_corrupt": 0.5390625, "acc_nopause": 0.71484375, "acc_pause": 0.72265625, "answer_logprob_margin": -0.023487769558334422, "answer_logprob_z": -1.8648381345662142, "answer_select_delta": -0.17712099054024782, "answer_select_z": -6.542153945802289, "cap_hit_frac_max": 0.0, "control_n": 256.0, "delta": 0.0078125, "delta_z": 0.19659669487640224, "failure_frac_max": 0.0, "step": 5, "vs_corrupt_delta": 0.18359375, "vs_corrupt_z": 4.384336584684985}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T150313Z-configPTC-016-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "completed cleanly but missed both reject and promotion thresholds", "rows": 6, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T150313Z-configPTC-016-er-opd-q36-35b-slots-trainer-head", "verdict": "inconclusive"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T160508Z-PTC-016-step5-inconclusive.json", "time_utc": "2026-06-04T16:05:08Z"}
{"candidate": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/candidates/PTC-018.yaml", "event": "launch", "idea_id": "PTC-018", "num_steps": 11, "prompts_per_step": 128, "sampler_layout": "spare-teacher1", "sampler_replicas": 2, "time_utc": "2026-06-04T16:06:38Z"}
{"event": "autopilot_wait", "idea_id": "PTC-018", "message": "autopilot: profile/score unavailable for PTC-018: profile not available yet for PTC-018 launched at 2026-06-04T16:06:38Z", "time_utc": "2026-06-04T16:06:45Z"}
{"event": "autopilot_wait", "idea_id": "PTC-018", "message": "autopilot: profile/score unavailable for PTC-018: profile not available yet for PTC-018 launched at 2026-06-04T16:06:38Z", "time_utc": "2026-06-04T16:09:45Z"}
{"event": "score", "idea_id": "PTC-018", "score": {"control_rows": 1, "metrics": {"acc_corrupt": 0.05078125, "acc_nopause": 0.79296875, "acc_pause": 0.7890625, "answer_logprob_margin": -0.14569518343434454, "answer_logprob_z": -4.566378284576971, "answer_select_delta": -0.4431691200380204, "answer_select_z": -8.050098350765134, "cap_hit_frac_max": 0.0, "control_n": 256.0, "delta": -0.00390625, "delta_z": -0.10869775914448827, "failure_frac_max": 0.0, "step": 5, "vs_corrupt_delta": 0.73828125, "vs_corrupt_z": 25.496527933431814}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T160640Z-configPTC-018-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "no positive accuracy or answer-logprob signal", "rows": 6, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T160640Z-configPTC-018-er-opd-q36-35b-slots-trainer-head", "verdict": "science_reject"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T170915Z-PTC-018-step5-science_reject.json", "time_utc": "2026-06-04T17:09:15Z"}
{"event": "advance", "idea_id": "PTC-018", "score": {"control_rows": 1, "metrics": {"acc_corrupt": 0.05078125, "acc_nopause": 0.79296875, "acc_pause": 0.7890625, "answer_logprob_margin": -0.14569518343434454, "answer_logprob_z": -4.566378284576971, "answer_select_delta": -0.4431691200380204, "answer_select_z": -8.050098350765134, "cap_hit_frac_max": 0.0, "control_n": 256.0, "delta": -0.00390625, "delta_z": -0.10869775914448827, "failure_frac_max": 0.0, "step": 5, "vs_corrupt_delta": 0.73828125, "vs_corrupt_z": 25.496527933431814}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T160640Z-configPTC-018-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "no positive accuracy or answer-logprob signal", "rows": 6, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T160640Z-configPTC-018-er-opd-q36-35b-slots-trainer-head", "verdict": "science_reject"}, "status": "rejected", "time_utc": "2026-06-04T17:09:27Z"}
{"event": "autopilot_idle", "message": "autopilot: no launched or queued ideas", "time_utc": "2026-06-04T17:09:47Z"}
{"candidate": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/candidates/PTC-019.yaml", "event": "launch", "idea_id": "PTC-019", "num_steps": 11, "prompts_per_step": 128, "sampler_layout": "spare-teacher1", "sampler_replicas": 2, "time_utc": "2026-06-04T17:11:37Z"}
{"event": "score", "idea_id": "PTC-020", "score": {"control_rows": 1, "metrics": {"acc_corrupt": 0.5810546875, "acc_nopause": 0.5556640625, "acc_pause": 0.6494140625, "answer_logprob_margin": 0.0, "answer_logprob_z": 0.0, "answer_select_delta": 0.0, "answer_select_z": 0.0, "cap_hit_frac_max": 0.009765625, "control_n": 1024.0, "delta": 0.09375, "delta_z": 4.3547971922577755, "failure_frac_max": 0.0, "score_mode": "pause_vs_nopause", "step": 5, "vs_corrupt_delta": 0.068359375, "vs_corrupt_z": 3.18705045981604}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T171221Z-configAM-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "pause-vs-nopause accuracy gate passed without positive answer-logprob support", "rows": 6, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T171221Z-configAM-er-opd-q36-35b-slots-trainer-head", "verdict": "strong_ptc_signal"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T174343Z-PTC-020-step5-strong_ptc_signal.json", "time_utc": "2026-06-04T17:43:43Z"}
{"event": "score", "idea_id": "PTC-019", "score": {"control_rows": 1, "metrics": {"acc_corrupt": 0.0390625, "acc_nopause": 0.6953125, "acc_pause": 0.73046875, "answer_logprob_margin": -0.29823434144685007, "answer_logprob_z": -7.985485988385205, "answer_select_delta": -0.14551921726336287, "answer_select_z": -3.0982369496584097, "cap_hit_frac_max": 0.00390625, "control_n": 256.0, "delta": 0.03515625, "delta_z": 0.8798334176828828, "failure_frac_max": 0.00390625, "score_mode": "causal_control", "step": 5, "vs_corrupt_delta": 0.69140625, "vs_corrupt_z": 22.848370565374616}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T171139Z-configPTC-019-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "request_failure_frac_max=0.0039", "rows": 6, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T171139Z-configPTC-019-er-opd-q36-35b-slots-trainer-head", "verdict": "infra_invalid"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T182846Z-PTC-019-step5-infra_invalid.json", "time_utc": "2026-06-04T18:28:46Z"}
{"event": "advance", "idea_id": "PTC-019", "score": {"control_rows": 1, "metrics": {"acc_corrupt": 0.0390625, "acc_nopause": 0.6953125, "acc_pause": 0.73046875, "answer_logprob_margin": -0.29823434144685007, "answer_logprob_z": -7.985485988385205, "answer_select_delta": -0.14551921726336287, "answer_select_z": -3.0982369496584097, "cap_hit_frac_max": 0.00390625, "control_n": 256.0, "delta": 0.03515625, "delta_z": 0.8798334176828828, "failure_frac_max": 0.00390625, "score_mode": "causal_control", "step": 5, "vs_corrupt_delta": 0.69140625, "vs_corrupt_z": 22.848370565374616}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T171139Z-configPTC-019-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "request_failure_frac_max=0.0039", "rows": 6, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T171139Z-configPTC-019-er-opd-q36-35b-slots-trainer-head", "verdict": "infra_invalid"}, "status": "infra_invalid", "time_utc": "2026-06-04T18:28:57Z"}
{"candidate": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/candidates/PTC-020.yaml", "event": "launch", "idea_id": "PTC-020", "num_steps": 6, "prompts_per_step": 128, "sampler_layout": "spare-teacher1", "sampler_replicas": 2, "time_utc": "2026-06-04T18:28:58Z"}
{"event": "advance", "idea_id": "PTC-019", "score": {"control_rows": 1, "metrics": {"acc_corrupt": 0.0390625, "acc_nopause": 0.6953125, "acc_pause": 0.73046875, "answer_logprob_margin": -0.29823434144685007, "answer_logprob_z": -7.985485988385205, "answer_select_delta": -0.14551921726336287, "answer_select_z": -3.0982369496584097, "cap_hit_frac_max": 0.00390625, "control_n": 256.0, "delta": 0.03515625, "delta_z": 0.8798334176828828, "failure_frac_max": 0.00390625, "score_mode": "causal_control", "step": 5, "vs_corrupt_delta": 0.69140625, "vs_corrupt_z": 22.848370565374616}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T171139Z-configPTC-019-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "request_failure_frac_max=0.0039", "rows": 6, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T171139Z-configPTC-019-er-opd-q36-35b-slots-trainer-head", "verdict": "infra_invalid"}, "status": "infra_invalid", "time_utc": "2026-06-04T18:29:07Z"}
{"candidate": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/candidates/PTC-020.yaml", "event": "launch", "idea_id": "PTC-020", "num_steps": 6, "prompts_per_step": 128, "sampler_layout": "spare-teacher1", "sampler_replicas": 2, "time_utc": "2026-06-04T18:31:37Z"}
{"event": "autopilot_infra_invalid", "idea_id": "PTC-020", "reason": "trainer-head:pid=       - exited_at=2026-06-04T18:45:36Z rc=1; trainer-worker-1:pid=       - stopped_at=2026-06-04T18:45:43Z; trainer-worker-2:pid=       - stopped_at=2026-06-04T18:45:43Z; trainer-worker-3:pid=       - stopped_at=2026-06-04T18:45:42Z; trainer-worker-4:pid=       - stopped_at=2026-06-04T18:45:43Z; trainer-worker-5:pid=       - stopped_at=2026-06-04T18:45:42Z; trainer-worker-6:pid=       - stopped_at=2026-06-04T18:45:42Z; trainer-worker-7:pid=       - stopped_at=2026-06-04T18:45:43Z", "time_utc": "2026-06-04T18:47:32Z"}
{"candidate": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/candidates/PTC-020.yaml", "event": "launch", "idea_id": "PTC-020", "num_steps": 6, "prompts_per_step": 128, "restart_inference": true, "sampler_layout": "spare-teacher1", "sampler_replicas": 2, "time_utc": "2026-06-04T18:49:46Z"}
{"candidate": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/candidates/PTC-020.yaml", "event": "launch", "idea_id": "PTC-020", "num_steps": 6, "prompts_per_step": 128, "restart_inference": false, "sampler_layout": "spare-teacher1", "sampler_replicas": 2, "time_utc": "2026-06-04T19:11:32Z"}
{"event": "monitor", "output": "{\"control_rows\": 0, \"latest_control_answer_select_delta\": null, \"latest_control_answer_select_z\": null, \"latest_control_buffer_delta\": null, \"latest_control_buffer_delta_z\": null, \"latest_control_n\": null, \"latest_control_step\": null, \"latest_eval_accuracy\": 0.4921875, \"latest_step\": 0, \"latest_sync_endpoint_success_count\": 2, \"latest_sync_success\": true, \"native_activity\": [], \"now_utc\": \"2026-06-04 19:18:42 UTC\", \"profile\": \"/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T191133Z-configPTC-020-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl\", \"profile_mtime_age_s\": 54.09980297088623, \"profile_mtime_s\": 1780600668.592968, \"profile_rows\": 1, \"smg\": {\"connections_active\": 11.0, \"inflight_age_count_gt_300s\": 0.0, \"inflight_age_count_gt_30s\": 31.0, \"inflight_age_count_gt_60s\": 0.0, \"router_request_response_gap\": 11.0, \"router_requests_total\": 384.0, \"router_responses_total\": 373.0, \"timestamp_s\": 1780600722.8377576}, \"smg_rates\": {}, \"verdict\": \"live_progress\", \"wandb\": {\"available\": false}}", "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T191133Z-configPTC-020-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "time_utc": "2026-06-04T19:18:42Z"}
{"event": "monitor", "output": "{\"control_rows\": 0, \"latest_control_answer_select_delta\": null, \"latest_control_answer_select_z\": null, \"latest_control_buffer_delta\": null, \"latest_control_buffer_delta_z\": null, \"latest_control_n\": null, \"latest_control_step\": null, \"latest_eval_accuracy\": 0.4609375, \"latest_step\": 1, \"latest_sync_endpoint_success_count\": 2, \"latest_sync_success\": true, \"native_activity\": [], \"now_utc\": \"2026-06-04 19:20:04 UTC\", \"profile\": \"/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T191133Z-configPTC-020-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl\", \"profile_mtime_age_s\": 54.563217878341675, \"profile_mtime_s\": 1780600749.645855, \"profile_rows\": 2, \"smg\": {\"connections_active\": 7.0, \"inflight_age_count_gt_300s\": 0.0, \"inflight_age_count_gt_30s\": 27.0, \"inflight_age_count_gt_60s\": 0.0, \"router_request_response_gap\": 7.0, \"router_requests_total\": 512.0, \"router_responses_total\": 505.0, \"timestamp_s\": 1780600804.34737}, \"smg_rates\": {}, \"verdict\": \"live_progress\", \"wandb\": {\"available\": false}}", "profile": "latest", "time_utc": "2026-06-04T19:20:04Z"}
{"event": "monitor", "output": "{\"control_rows\": 0, \"latest_control_answer_select_delta\": null, \"latest_control_answer_select_z\": null, \"latest_control_buffer_delta\": null, \"latest_control_buffer_delta_z\": null, \"latest_control_n\": null, \"latest_control_step\": null, \"latest_eval_accuracy\": 0.4140625, \"latest_step\": 4, \"latest_sync_endpoint_success_count\": 2, \"latest_sync_success\": true, \"native_activity\": [], \"now_utc\": \"2026-06-04 19:39:25 UTC\", \"profile\": \"/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T191133Z-configPTC-020-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl\", \"profile_mtime_age_s\": 963.5861070156097, \"profile_mtime_s\": 1780601002.120107, \"profile_rows\": 5, \"smg\": {\"connections_active\": 0.0, \"inflight_age_count_gt_300s\": 0.0, \"inflight_age_count_gt_30s\": 0.0, \"inflight_age_count_gt_60s\": 0.0, \"router_request_response_gap\": 0.0, \"router_requests_total\": 3968.0, \"router_responses_total\": 3968.0, \"timestamp_s\": 1780601965.8504558}, \"smg_rates\": {}, \"verdict\": \"idle_or_waiting\", \"wandb\": {\"available\": false}}", "profile": "latest", "time_utc": "2026-06-04T19:39:25Z"}
{"event": "monitor", "output": "{\"control_rows\": 0, \"latest_control_answer_select_delta\": null, \"latest_control_answer_select_z\": null, \"latest_control_buffer_delta\": null, \"latest_control_buffer_delta_z\": null, \"latest_control_n\": null, \"latest_control_step\": null, \"latest_eval_accuracy\": 0.4140625, \"latest_step\": 4, \"latest_sync_endpoint_success_count\": 2, \"latest_sync_success\": true, \"native_activity\": [], \"now_utc\": \"2026-06-04 19:39:54 UTC\", \"profile\": \"/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T191133Z-configPTC-020-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl\", \"profile_mtime_age_s\": 992.0263795852661, \"profile_mtime_s\": 1780601002.120107, \"profile_rows\": 5, \"smg\": {\"connections_active\": 0.0, \"inflight_age_count_gt_300s\": 0.0, \"inflight_age_count_gt_30s\": 0.0, \"inflight_age_count_gt_60s\": 0.0, \"router_request_response_gap\": 0.0, \"router_requests_total\": 3968.0, \"router_responses_total\": 3968.0, \"timestamp_s\": 1780601994.366549}, \"smg_rates\": {}, \"verdict\": \"idle_or_waiting\", \"wandb\": {\"available\": false}}", "profile": "latest", "time_utc": "2026-06-04T19:39:54Z"}
{"event": "monitor", "output": "{\"control_rows\": 0, \"latest_control_answer_select_delta\": null, \"latest_control_answer_select_z\": null, \"latest_control_buffer_delta\": null, \"latest_control_buffer_delta_z\": null, \"latest_control_n\": null, \"latest_control_step\": null, \"latest_eval_accuracy\": 0.4140625, \"latest_step\": 4, \"latest_sync_endpoint_success_count\": 2, \"latest_sync_success\": true, \"native_activity\": [], \"now_utc\": \"2026-06-04 19:41:13 UTC\", \"profile\": \"/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T191133Z-configPTC-020-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl\", \"profile_mtime_age_s\": 1071.4293777942657, \"profile_mtime_s\": 1780601002.120107, \"profile_rows\": 5, \"smg\": {\"connections_active\": 0.0, \"inflight_age_count_gt_300s\": 0.0, \"inflight_age_count_gt_30s\": 0.0, \"inflight_age_count_gt_60s\": 0.0, \"router_request_response_gap\": 0.0, \"router_requests_total\": 3968.0, \"router_responses_total\": 3968.0, \"timestamp_s\": 1780602073.6580732}, \"smg_rates\": {}, \"verdict\": \"idle_or_waiting\", \"wandb\": {\"available\": false}}", "profile": "latest", "time_utc": "2026-06-04T19:41:13Z"}
{"event": "monitor", "output": "{\"control_rows\": 0, \"latest_control_answer_select_delta\": null, \"latest_control_answer_select_z\": null, \"latest_control_buffer_delta\": null, \"latest_control_buffer_delta_z\": null, \"latest_control_n\": null, \"latest_control_step\": null, \"latest_eval_accuracy\": 0.4140625, \"latest_step\": 4, \"latest_sync_endpoint_success_count\": 2, \"latest_sync_success\": true, \"native_activity\": [], \"now_utc\": \"2026-06-04 19:41:33 UTC\", \"profile\": \"/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T191133Z-configPTC-020-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl\", \"profile_mtime_age_s\": 1090.8963859081268, \"profile_mtime_s\": 1780601002.120107, \"profile_rows\": 5, \"smg\": {\"connections_active\": 0.0, \"inflight_age_count_gt_300s\": 0.0, \"inflight_age_count_gt_30s\": 0.0, \"inflight_age_count_gt_60s\": 0.0, \"router_request_response_gap\": 0.0, \"router_requests_total\": 3968.0, \"router_responses_total\": 3968.0, \"timestamp_s\": 1780602093.1251316}, \"smg_rates\": {}, \"verdict\": \"idle_or_waiting\", \"wandb\": {\"available\": false}}", "profile": "latest", "time_utc": "2026-06-04T19:41:33Z"}
{"event": "monitor", "output": "{\"control_rows\": 1, \"latest_control_answer_select_delta\": null, \"latest_control_answer_select_z\": null, \"latest_control_buffer_delta\": 0.0458984375, \"latest_control_buffer_delta_z\": 2.1189846284303124, \"latest_control_n\": 1024.0, \"latest_control_step\": 5, \"latest_eval_accuracy\": 0.4453125, \"latest_step\": 5, \"latest_sync_endpoint_success_count\": 2, \"latest_sync_success\": true, \"native_activity\": [], \"now_utc\": \"2026-06-04 19:43:09 UTC\", \"profile\": \"/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T191133Z-configPTC-020-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl\", \"profile_mtime_age_s\": 48.02815628051758, \"profile_mtime_s\": 1780602140.8391783, \"profile_rows\": 6, \"smg\": {\"connections_active\": 0.0, \"inflight_age_count_gt_300s\": 0.0, \"inflight_age_count_gt_30s\": 0.0, \"inflight_age_count_gt_60s\": 0.0, \"router_request_response_gap\": 0.0, \"router_requests_total\": 3968.0, \"router_responses_total\": 3968.0, \"timestamp_s\": 1780602189.0018802}, \"smg_rates\": {}, \"verdict\": \"profile_control_row_present\", \"wandb\": {\"available\": false}}", "profile": "latest", "time_utc": "2026-06-04T19:43:09Z"}
{"event": "autopilot_score", "idea_id": "PTC-020", "score": {"control_rows": 1, "metrics": {"acc_corrupt": 0.6298828125, "acc_nopause": 0.5732421875, "acc_pause": 0.619140625, "answer_logprob_margin": 0.1110034145900177, "answer_logprob_z": 23.75572011440179, "answer_select_delta": 0.0, "answer_select_z": 0.0, "cap_hit_frac_max": 0.00390625, "control_n": 1024.0, "delta": 0.0458984375, "delta_z": 2.1189846284303124, "failure_frac_max": 0.0, "score_mode": "pause_vs_nopause", "step": 5, "vs_corrupt_delta": -0.0107421875, "vs_corrupt_z": -0.5019794461402117}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T191133Z-configPTC-020-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "pause-vs-nopause accuracy warrants retest with positive answer-logprob support", "rows": 6, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T191133Z-configPTC-020-er-opd-q36-35b-slots-trainer-head", "verdict": "promote_retest"}, "scorecard": "/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/scorecards/20260604T194427Z-PTC-020-step5-promote_retest.json", "time_utc": "2026-06-04T19:44:27Z"}
{"event": "advance", "idea_id": "PTC-020", "score": {"control_rows": 1, "metrics": {"acc_corrupt": 0.6298828125, "acc_nopause": 0.5732421875, "acc_pause": 0.619140625, "answer_logprob_margin": 0.1110034145900177, "answer_logprob_z": 23.75572011440179, "answer_select_delta": 0.0, "answer_select_z": 0.0, "cap_hit_frac_max": 0.00390625, "control_n": 1024.0, "delta": 0.0458984375, "delta_z": 2.1189846284303124, "failure_frac_max": 0.0, "score_mode": "pause_vs_nopause", "step": 5, "vs_corrupt_delta": -0.0107421875, "vs_corrupt_z": -0.5019794461402117}, "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T191133Z-configPTC-020-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "reason": "pause-vs-nopause accuracy warrants retest with positive answer-logprob support", "rows": 6, "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T191133Z-configPTC-020-er-opd-q36-35b-slots-trainer-head", "verdict": "promote_retest"}, "status": "promote_retest", "time_utc": "2026-06-04T19:44:27Z"}
{"event": "autopilot_terminal_stop", "idea_id": "PTC-020", "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T191133Z-configPTC-020-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl", "time_utc": "2026-06-04T19:44:27Z", "verdict": "promote_retest"}
{"event": "monitor", "output": "{\"control_rows\": 1, \"latest_control_answer_select_delta\": null, \"latest_control_answer_select_z\": null, \"latest_control_buffer_delta\": 0.0458984375, \"latest_control_buffer_delta_z\": 2.1189846284303124, \"latest_control_n\": 1024.0, \"latest_control_step\": 5, \"latest_eval_accuracy\": 0.4453125, \"latest_step\": 5, \"latest_sync_endpoint_success_count\": 2, \"latest_sync_success\": true, \"native_activity\": [], \"now_utc\": \"2026-06-04 20:14:37 UTC\", \"profile\": \"/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T191133Z-configPTC-020-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl\", \"profile_mtime_age_s\": 1936.1661853790283, \"profile_mtime_s\": 1780602140.8391783, \"profile_rows\": 6, \"smg\": {\"connections_active\": 0.0, \"inflight_age_count_gt_300s\": 0.0, \"inflight_age_count_gt_30s\": 0.0, \"inflight_age_count_gt_60s\": 0.0, \"router_request_response_gap\": 0.0, \"router_requests_total\": 3968.0, \"router_responses_total\": 3968.0, \"timestamp_s\": 1780604077.1163256}, \"smg_rates\": {}, \"verdict\": \"profile_control_row_present\", \"wandb\": {\"available\": false}}", "profile": "latest", "time_utc": "2026-06-04T20:14:37Z"}
```


## 5. Scorecard JSON Files


### `experiments/opd_profile/autoresearch/scorecards/20260604T011532Z-PTC-001-step8-infra_invalid.json`

```json
{
  "control_rows": 2,
  "metrics": {
    "acc_corrupt": 0.0,
    "acc_nopause": 0.808,
    "acc_pause": 0.802,
    "answer_logprob_margin": 0.0011698661124863946,
    "answer_logprob_z": 0.5921812725913325,
    "answer_select_delta": 0.0,
    "answer_select_z": 0.0,
    "cap_hit_frac_max": 0.0,
    "control_n": 1000.0,
    "delta": -0.006000000000000005,
    "delta_z": -0.33863625839560096,
    "failure_frac_max": 0.372,
    "step": 8,
    "vs_corrupt_delta": 0.802,
    "vs_corrupt_z": 63.643578234611006
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T225859Z-configPTC-001-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "request_failure_frac_max=0.3720",
  "rows": 9,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T225859Z-configPTC-001-er-opd-q36-35b-slots-trainer-head",
  "verdict": "infra_invalid"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T011532Z-PTC-005-step8-infra_invalid.json`

```json
{
  "control_rows": 2,
  "metrics": {
    "acc_corrupt": 0.0,
    "acc_nopause": 0.799,
    "acc_pause": 0.799,
    "answer_logprob_margin": -0.004177647228492835,
    "answer_logprob_z": -2.164316609339693,
    "answer_select_delta": 0.0,
    "answer_select_z": 0.0,
    "cap_hit_frac_max": 0.0,
    "control_n": 1000.0,
    "delta": 0.0,
    "delta_z": 0.0,
    "failure_frac_max": 0.436,
    "step": 8,
    "vs_corrupt_delta": 0.799,
    "vs_corrupt_z": 63.04858743944589
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T000006Z-configPTC-005-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "request_failure_frac_max=0.4360",
  "rows": 9,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T000006Z-configPTC-005-er-opd-q36-35b-slots-trainer-head",
  "verdict": "infra_invalid"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T011537Z-PTC-001-step8-infra_invalid.json`

```json
{
  "control_rows": 2,
  "metrics": {
    "acc_corrupt": 0.0,
    "acc_nopause": 0.808,
    "acc_pause": 0.802,
    "answer_logprob_margin": 0.0011698661124863946,
    "answer_logprob_z": 0.5921812725913325,
    "answer_select_delta": 0.0,
    "answer_select_z": 0.0,
    "cap_hit_frac_max": 0.0,
    "control_n": 1000.0,
    "delta": -0.006000000000000005,
    "delta_z": -0.33863625839560096,
    "failure_frac_max": 0.372,
    "step": 8,
    "vs_corrupt_delta": 0.802,
    "vs_corrupt_z": 63.643578234611006
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T225859Z-configPTC-001-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "request_failure_frac_max=0.3720",
  "rows": 9,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T225859Z-configPTC-001-er-opd-q36-35b-slots-trainer-head",
  "verdict": "infra_invalid"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T021202Z-PTC-006-step8-science_reject.json`

```json
{
  "control_rows": 2,
  "metrics": {
    "acc_corrupt": 0.0,
    "acc_nopause": 0.812,
    "acc_pause": 0.807,
    "answer_logprob_margin": -0.0005920925329430714,
    "answer_logprob_z": -0.3690380389251624,
    "answer_select_delta": 0.0,
    "answer_select_z": 0.0,
    "cap_hit_frac_max": 0.0,
    "control_n": 1000.0,
    "delta": -0.0050000000000000044,
    "delta_z": -0.28471338904946075,
    "failure_frac_max": 0.0,
    "step": 8,
    "vs_corrupt_delta": 0.807,
    "vs_corrupt_z": 64.6633369867274
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T013603Z-configPTC-006-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "no positive accuracy or answer-logprob signal",
  "rows": 9,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T013603Z-configPTC-006-er-opd-q36-35b-slots-trainer-head",
  "verdict": "science_reject"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T021205Z-PTC-006-step8-science_reject.json`

```json
{
  "control_rows": 2,
  "metrics": {
    "acc_corrupt": 0.0,
    "acc_nopause": 0.812,
    "acc_pause": 0.807,
    "answer_logprob_margin": -0.0005920925329430714,
    "answer_logprob_z": -0.3690380389251624,
    "answer_select_delta": 0.0,
    "answer_select_z": 0.0,
    "cap_hit_frac_max": 0.0,
    "control_n": 1000.0,
    "delta": -0.0050000000000000044,
    "delta_z": -0.28471338904946075,
    "failure_frac_max": 0.0,
    "step": 8,
    "vs_corrupt_delta": 0.807,
    "vs_corrupt_z": 64.6633369867274
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T013603Z-configPTC-006-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "no positive accuracy or answer-logprob signal",
  "rows": 9,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T013603Z-configPTC-006-er-opd-q36-35b-slots-trainer-head",
  "verdict": "science_reject"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T045539Z-PTC-003-step8-science_reject.json`

```json
{
  "control_rows": 2,
  "metrics": {
    "acc_corrupt": 0.0,
    "acc_nopause": 0.803,
    "acc_pause": 0.797,
    "answer_logprob_margin": -0.002851773091628856,
    "answer_logprob_z": -1.550627938419657,
    "answer_select_delta": 0.0,
    "answer_select_z": 0.0,
    "cap_hit_frac_max": 0.0,
    "control_n": 1000.0,
    "delta": -0.006000000000000005,
    "delta_z": -0.3354196304347396,
    "failure_frac_max": 0.0,
    "step": 8,
    "vs_corrupt_delta": 0.797,
    "vs_corrupt_z": 62.65866559690078
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T021246Z-configPTC-003-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "no positive accuracy or answer-logprob signal",
  "rows": 9,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T021246Z-configPTC-003-er-opd-q36-35b-slots-trainer-head",
  "verdict": "science_reject"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T045543Z-PTC-003-step8-science_reject.json`

```json
{
  "control_rows": 2,
  "metrics": {
    "acc_corrupt": 0.0,
    "acc_nopause": 0.803,
    "acc_pause": 0.797,
    "answer_logprob_margin": -0.002851773091628856,
    "answer_logprob_z": -1.550627938419657,
    "answer_select_delta": 0.0,
    "answer_select_z": 0.0,
    "cap_hit_frac_max": 0.0,
    "control_n": 1000.0,
    "delta": -0.006000000000000005,
    "delta_z": -0.3354196304347396,
    "failure_frac_max": 0.0,
    "step": 8,
    "vs_corrupt_delta": 0.797,
    "vs_corrupt_z": 62.65866559690078
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T021246Z-configPTC-003-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "no positive accuracy or answer-logprob signal",
  "rows": 9,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T021246Z-configPTC-003-er-opd-q36-35b-slots-trainer-head",
  "verdict": "science_reject"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T053033Z-PTC-007-step4-science_reject.json`

```json
{
  "control_rows": 1,
  "metrics": {
    "acc_corrupt": 0.0,
    "acc_nopause": 0.766,
    "acc_pause": 0.677,
    "answer_logprob_margin": -0.05189284104206759,
    "answer_logprob_z": -8.495199167432943,
    "answer_select_delta": 0.0,
    "answer_select_z": 0.0,
    "cap_hit_frac_max": 0.0,
    "control_n": 1000.0,
    "delta": -0.08899999999999997,
    "delta_z": -4.461643350547222,
    "failure_frac_max": 0.0,
    "step": 4,
    "vs_corrupt_delta": 0.677,
    "vs_corrupt_z": 45.781822071627325
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T050810Z-configPTC-007-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "no positive accuracy or answer-logprob signal",
  "rows": 5,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T050810Z-configPTC-007-er-opd-q36-35b-slots-trainer-head",
  "verdict": "science_reject"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T060104Z-PTC-008-step4-science_reject.json`

```json
{
  "control_rows": 1,
  "metrics": {
    "acc_corrupt": 0.591,
    "acc_nopause": 0.71,
    "acc_pause": 0.624,
    "answer_logprob_margin": -0.004445023867406381,
    "answer_logprob_z": -0.7627423018328828,
    "answer_select_delta": -0.025082884714261792,
    "answer_select_z": -3.434489752279603,
    "cap_hit_frac_max": 0.004,
    "control_n": 1000.0,
    "delta": -0.08599999999999997,
    "delta_z": -4.097450014496387,
    "failure_frac_max": 0.0,
    "step": 4,
    "vs_corrupt_delta": 0.03300000000000003,
    "vs_corrupt_z": 1.5120078506650454
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T053133Z-configPTC-008-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "no positive accuracy or answer-logprob signal",
  "rows": 5,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T053133Z-configPTC-008-er-opd-q36-35b-slots-trainer-head",
  "verdict": "science_reject"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T063058Z-PTC-009-step4-science_reject.json`

```json
{
  "control_rows": 1,
  "metrics": {
    "acc_corrupt": 0.719,
    "acc_nopause": 0.801,
    "acc_pause": 0.719,
    "answer_logprob_margin": -0.06802329225159871,
    "answer_logprob_z": -7.5156321540708575,
    "answer_select_delta": -0.17130115062077741,
    "answer_select_z": -14.448557610375241,
    "cap_hit_frac_max": 0.0,
    "control_n": 1000.0,
    "delta": -0.08200000000000007,
    "delta_z": -4.313173687822969,
    "failure_frac_max": 0.0,
    "step": 4,
    "vs_corrupt_delta": 0.0,
    "vs_corrupt_z": 0.0
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T060143Z-configPTC-009-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "no positive accuracy or answer-logprob signal",
  "rows": 5,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T060143Z-configPTC-009-er-opd-q36-35b-slots-trainer-head",
  "verdict": "science_reject"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T070229Z-PTC-010-step4-science_reject.json`

```json
{
  "control_rows": 1,
  "metrics": {
    "acc_corrupt": 0.677,
    "acc_nopause": 0.789,
    "acc_pause": 0.684,
    "answer_logprob_margin": -0.07165765871649975,
    "answer_logprob_z": -8.41706506088195,
    "answer_select_delta": -0.19453478057920348,
    "answer_select_z": -16.804226108679384,
    "cap_hit_frac_max": 0.002,
    "control_n": 1000.0,
    "delta": -0.10499999999999998,
    "delta_z": -5.367891918486458,
    "failure_frac_max": 0.0,
    "step": 4,
    "vs_corrupt_delta": 0.007000000000000006,
    "vs_corrupt_z": 0.3356957021998745
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T063316Z-configPTC-010-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "no positive accuracy or answer-logprob signal",
  "rows": 5,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T063316Z-configPTC-010-er-opd-q36-35b-slots-trainer-head",
  "verdict": "science_reject"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T074639Z-PTC-011-stepna-incomplete.json`

```json
{
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "no control rows",
  "rows": 1,
  "verdict": "incomplete"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T074650Z-PTC-011-stepna-incomplete.json`

```json
{
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "no control rows",
  "rows": 1,
  "verdict": "incomplete"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T074659Z-PTC-011-stepna-incomplete.json`

```json
{
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "no control rows",
  "rows": 1,
  "verdict": "incomplete"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T075411Z-PTC-011-stepna-incomplete.json`

```json
{
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "no control rows",
  "rows": 2,
  "verdict": "incomplete"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T075543Z-PTC-011-stepna-incomplete.json`

```json
{
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "no control rows",
  "rows": 2,
  "verdict": "incomplete"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T075929Z-PTC-011-stepna-incomplete.json`

```json
{
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "no control rows",
  "rows": 3,
  "verdict": "incomplete"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T075957Z-PTC-011-stepna-incomplete.json`

```json
{
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "no control rows",
  "rows": 3,
  "verdict": "incomplete"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T080016Z-PTC-011-stepna-incomplete.json`

```json
{
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "no control rows",
  "rows": 3,
  "verdict": "incomplete"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T082853Z-PTC-011-step4-inconclusive.json`

```json
{
  "control_rows": 1,
  "metrics": {
    "acc_corrupt": 0.00390625,
    "acc_nopause": 0.73828125,
    "acc_pause": 0.78515625,
    "answer_logprob_margin": -0.1338996193211561,
    "answer_logprob_z": -4.934445616493985,
    "answer_select_delta": 0.05212081693119896,
    "answer_select_z": 1.4338520989113466,
    "cap_hit_frac_max": 0.0,
    "control_n": 256.0,
    "delta": 0.046875,
    "delta_z": 1.2467006799461504,
    "failure_frac_max": 0.0,
    "step": 4,
    "vs_corrupt_delta": 0.78125,
    "vs_corrupt_z": 30.08975668093039
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "completed cleanly but missed both reject and promotion thresholds",
  "rows": 5,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head",
  "verdict": "inconclusive"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T082854Z-PTC-011-step4-inconclusive.json`

```json
{
  "control_rows": 1,
  "metrics": {
    "acc_corrupt": 0.00390625,
    "acc_nopause": 0.73828125,
    "acc_pause": 0.78515625,
    "answer_logprob_margin": -0.1338996193211561,
    "answer_logprob_z": -4.934445616493985,
    "answer_select_delta": 0.05212081693119896,
    "answer_select_z": 1.4338520989113466,
    "cap_hit_frac_max": 0.0,
    "control_n": 256.0,
    "delta": 0.046875,
    "delta_z": 1.2467006799461504,
    "failure_frac_max": 0.0,
    "step": 4,
    "vs_corrupt_delta": 0.78125,
    "vs_corrupt_z": 30.08975668093039
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "completed cleanly but missed both reject and promotion thresholds",
  "rows": 5,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head",
  "verdict": "inconclusive"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T091135Z-PTC-011-step8-inconclusive.json`

```json
{
  "control_rows": 2,
  "metrics": {
    "acc_corrupt": 0.30078125,
    "acc_nopause": 0.78125,
    "acc_pause": 0.80859375,
    "answer_logprob_margin": -0.027456654039798866,
    "answer_logprob_z": -2.1209689010269392,
    "answer_select_delta": 0.1930973367662947,
    "answer_select_z": 7.5565314530967775,
    "cap_hit_frac_max": 0.0,
    "control_n": 256.0,
    "delta": 0.02734375,
    "delta_z": 0.7666379087775477,
    "failure_frac_max": 0.0,
    "step": 8,
    "vs_corrupt_delta": 0.5078125,
    "vs_corrupt_z": 13.44708920081176
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "completed cleanly but missed both reject and promotion thresholds",
  "rows": 9,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head",
  "verdict": "inconclusive"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T091212Z-PTC-011-step8-inconclusive.json`

```json
{
  "control_rows": 2,
  "metrics": {
    "acc_corrupt": 0.30078125,
    "acc_nopause": 0.78125,
    "acc_pause": 0.80859375,
    "answer_logprob_margin": -0.027456654039798866,
    "answer_logprob_z": -2.1209689010269392,
    "answer_select_delta": 0.1930973367662947,
    "answer_select_z": 7.5565314530967775,
    "cap_hit_frac_max": 0.0,
    "control_n": 256.0,
    "delta": 0.02734375,
    "delta_z": 0.7666379087775477,
    "failure_frac_max": 0.0,
    "step": 8,
    "vs_corrupt_delta": 0.5078125,
    "vs_corrupt_z": 13.44708920081176
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "completed cleanly but missed both reject and promotion thresholds",
  "rows": 9,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head",
  "verdict": "inconclusive"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T100716Z-PTC-012-step4-inconclusive.json`

```json
{
  "control_rows": 1,
  "metrics": {
    "acc_corrupt": 0.0,
    "acc_nopause": 0.73046875,
    "acc_pause": 0.83984375,
    "answer_logprob_margin": -0.11067124789628861,
    "answer_logprob_z": -3.0825148106625977,
    "answer_select_delta": -0.3257433384510804,
    "answer_select_z": -6.977632527602614,
    "cap_hit_frac_max": 0.0,
    "control_n": 256.0,
    "delta": 0.109375,
    "delta_z": 3.0399616129719975,
    "failure_frac_max": 0.0,
    "step": 4,
    "vs_corrupt_delta": 0.83984375,
    "vs_corrupt_z": 36.63930982415258
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T091215Z-configPTC-012-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "completed cleanly but missed both reject and promotion thresholds",
  "rows": 5,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T091215Z-configPTC-012-er-opd-q36-35b-slots-trainer-head",
  "verdict": "inconclusive"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T100734Z-PTC-012-step4-inconclusive.json`

```json
{
  "control_rows": 1,
  "metrics": {
    "acc_corrupt": 0.0,
    "acc_nopause": 0.73046875,
    "acc_pause": 0.83984375,
    "answer_logprob_margin": -0.11067124789628861,
    "answer_logprob_z": -3.0825148106625977,
    "answer_select_delta": -0.3257433384510804,
    "answer_select_z": -6.977632527602614,
    "cap_hit_frac_max": 0.0,
    "control_n": 256.0,
    "delta": 0.109375,
    "delta_z": 3.0399616129719975,
    "failure_frac_max": 0.0,
    "step": 4,
    "vs_corrupt_delta": 0.83984375,
    "vs_corrupt_z": 36.63930982415258
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T091215Z-configPTC-012-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "completed cleanly but missed both reject and promotion thresholds",
  "rows": 5,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T091215Z-configPTC-012-er-opd-q36-35b-slots-trainer-head",
  "verdict": "inconclusive"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T105011Z-PTC-012-step8-inconclusive.json`

```json
{
  "control_rows": 2,
  "metrics": {
    "acc_corrupt": 0.11328125,
    "acc_nopause": 0.78515625,
    "acc_pause": 0.8046875,
    "answer_logprob_margin": -0.1596965955282831,
    "answer_logprob_z": -5.496740775541098,
    "answer_select_delta": -0.7489891492578584,
    "answer_select_z": -14.397340349047914,
    "cap_hit_frac_max": 0.0,
    "control_n": 256.0,
    "delta": 0.01953125,
    "delta_z": 0.5474446288295078,
    "failure_frac_max": 0.0,
    "step": 8,
    "vs_corrupt_delta": 0.69140625,
    "vs_corrupt_z": 21.795580416961943
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T091215Z-configPTC-012-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "completed cleanly but missed both reject and promotion thresholds",
  "rows": 9,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T091215Z-configPTC-012-er-opd-q36-35b-slots-trainer-head",
  "verdict": "inconclusive"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T105043Z-PTC-012-step8-inconclusive.json`

```json
{
  "control_rows": 2,
  "metrics": {
    "acc_corrupt": 0.11328125,
    "acc_nopause": 0.78515625,
    "acc_pause": 0.8046875,
    "answer_logprob_margin": -0.1596965955282831,
    "answer_logprob_z": -5.496740775541098,
    "answer_select_delta": -0.7489891492578584,
    "answer_select_z": -14.397340349047914,
    "cap_hit_frac_max": 0.0,
    "control_n": 256.0,
    "delta": 0.01953125,
    "delta_z": 0.5474446288295078,
    "failure_frac_max": 0.0,
    "step": 8,
    "vs_corrupt_delta": 0.69140625,
    "vs_corrupt_z": 21.795580416961943
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T091215Z-configPTC-012-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "completed cleanly but missed both reject and promotion thresholds",
  "rows": 9,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T091215Z-configPTC-012-er-opd-q36-35b-slots-trainer-head",
  "verdict": "inconclusive"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T114657Z-PTC-013-step4-science_reject.json`

```json
{
  "control_rows": 1,
  "metrics": {
    "acc_corrupt": 0.1328125,
    "acc_nopause": 0.75,
    "acc_pause": 0.59765625,
    "answer_logprob_margin": -0.31010046812613806,
    "answer_logprob_z": -9.973705393936111,
    "answer_select_delta": -0.24107080985158527,
    "answer_select_z": -6.486670231064018,
    "cap_hit_frac_max": 0.0,
    "control_n": 256.0,
    "delta": -0.15234375,
    "delta_z": -3.7259874043167356,
    "failure_frac_max": 0.0,
    "step": 4,
    "vs_corrupt_delta": 0.46484375,
    "vs_corrupt_z": 12.47164544836176
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T105110Z-configPTC-013-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "no positive accuracy or answer-logprob signal",
  "rows": 5,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T105110Z-configPTC-013-er-opd-q36-35b-slots-trainer-head",
  "verdict": "science_reject"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T114724Z-PTC-013-step4-science_reject.json`

```json
{
  "control_rows": 1,
  "metrics": {
    "acc_corrupt": 0.1328125,
    "acc_nopause": 0.75,
    "acc_pause": 0.59765625,
    "answer_logprob_margin": -0.31010046812613806,
    "answer_logprob_z": -9.973705393936111,
    "answer_select_delta": -0.24107080985158527,
    "answer_select_z": -6.486670231064018,
    "cap_hit_frac_max": 0.0,
    "control_n": 256.0,
    "delta": -0.15234375,
    "delta_z": -3.7259874043167356,
    "failure_frac_max": 0.0,
    "step": 4,
    "vs_corrupt_delta": 0.46484375,
    "vs_corrupt_z": 12.47164544836176
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T105110Z-configPTC-013-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "no positive accuracy or answer-logprob signal",
  "rows": 5,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T105110Z-configPTC-013-er-opd-q36-35b-slots-trainer-head",
  "verdict": "science_reject"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T124228Z-PTC-014-step4-inconclusive.json`

```json
{
  "control_rows": 1,
  "metrics": {
    "acc_corrupt": 0.015625,
    "acc_nopause": 0.69921875,
    "acc_pause": 0.8359375,
    "answer_logprob_margin": -0.06823835960594374,
    "answer_logprob_z": -2.393474203845524,
    "answer_select_delta": -0.18082758527490222,
    "answer_select_z": -4.051345401410779,
    "cap_hit_frac_max": 0.0,
    "control_n": 256.0,
    "delta": 0.13671875,
    "delta_z": 3.7110514492390805,
    "failure_frac_max": 0.0,
    "step": 4,
    "vs_corrupt_delta": 0.8203125,
    "vs_corrupt_z": 33.60672201667224
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T114759Z-configPTC-014-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "completed cleanly but missed both reject and promotion thresholds",
  "rows": 5,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T114759Z-configPTC-014-er-opd-q36-35b-slots-trainer-head",
  "verdict": "inconclusive"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T124319Z-PTC-014-step4-inconclusive.json`

```json
{
  "control_rows": 1,
  "metrics": {
    "acc_corrupt": 0.015625,
    "acc_nopause": 0.69921875,
    "acc_pause": 0.8359375,
    "answer_logprob_margin": -0.06823835960594374,
    "answer_logprob_z": -2.393474203845524,
    "answer_select_delta": -0.18082758527490222,
    "answer_select_z": -4.051345401410779,
    "cap_hit_frac_max": 0.0,
    "control_n": 256.0,
    "delta": 0.13671875,
    "delta_z": 3.7110514492390805,
    "failure_frac_max": 0.0,
    "step": 4,
    "vs_corrupt_delta": 0.8203125,
    "vs_corrupt_z": 33.60672201667224
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T114759Z-configPTC-014-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "completed cleanly but missed both reject and promotion thresholds",
  "rows": 5,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T114759Z-configPTC-014-er-opd-q36-35b-slots-trainer-head",
  "verdict": "inconclusive"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T132437Z-PTC-014-step8-science_reject.json`

```json
{
  "control_rows": 2,
  "metrics": {
    "acc_corrupt": 0.3203125,
    "acc_nopause": 0.76953125,
    "acc_pause": 0.76171875,
    "answer_logprob_margin": -0.10689015139499593,
    "answer_logprob_z": -5.096342933926368,
    "answer_select_delta": -0.14860751767676922,
    "answer_select_z": -3.269913442281971,
    "cap_hit_frac_max": 0.0,
    "control_n": 256.0,
    "delta": -0.0078125,
    "delta_z": -0.20866508438634226,
    "failure_frac_max": 0.0,
    "step": 8,
    "vs_corrupt_delta": 0.44140625,
    "vs_corrupt_z": 11.177756782299781
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T114759Z-configPTC-014-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "no positive accuracy or answer-logprob signal",
  "rows": 9,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T114759Z-configPTC-014-er-opd-q36-35b-slots-trainer-head",
  "verdict": "science_reject"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T132455Z-PTC-014-step8-science_reject.json`

```json
{
  "control_rows": 2,
  "metrics": {
    "acc_corrupt": 0.3203125,
    "acc_nopause": 0.76953125,
    "acc_pause": 0.76171875,
    "answer_logprob_margin": -0.10689015139499593,
    "answer_logprob_z": -5.096342933926368,
    "answer_select_delta": -0.14860751767676922,
    "answer_select_z": -3.269913442281971,
    "cap_hit_frac_max": 0.0,
    "control_n": 256.0,
    "delta": -0.0078125,
    "delta_z": -0.20866508438634226,
    "failure_frac_max": 0.0,
    "step": 8,
    "vs_corrupt_delta": 0.44140625,
    "vs_corrupt_z": 11.177756782299781
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T114759Z-configPTC-014-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "no positive accuracy or answer-logprob signal",
  "rows": 9,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T114759Z-configPTC-014-er-opd-q36-35b-slots-trainer-head",
  "verdict": "science_reject"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T142024Z-PTC-015-step4-inconclusive.json`

```json
{
  "control_rows": 1,
  "metrics": {
    "acc_corrupt": 0.0,
    "acc_nopause": 0.78515625,
    "acc_pause": 0.8203125,
    "answer_logprob_margin": 0.006929151958554657,
    "answer_logprob_z": 0.4353864050668189,
    "answer_select_delta": 0.11405637547581872,
    "answer_select_z": 3.151441197561941,
    "cap_hit_frac_max": 0.00390625,
    "control_n": 256.0,
    "delta": 0.03515625,
    "delta_z": 1.000506750675012,
    "failure_frac_max": 0.0,
    "step": 4,
    "vs_corrupt_delta": 0.8203125,
    "vs_corrupt_z": 34.18619095737215
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T132457Z-configPTC-015-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "completed cleanly but missed both reject and promotion thresholds",
  "rows": 5,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T132457Z-configPTC-015-er-opd-q36-35b-slots-trainer-head",
  "verdict": "inconclusive"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T142105Z-PTC-015-step4-inconclusive.json`

```json
{
  "control_rows": 1,
  "metrics": {
    "acc_corrupt": 0.0,
    "acc_nopause": 0.78515625,
    "acc_pause": 0.8203125,
    "answer_logprob_margin": 0.006929151958554657,
    "answer_logprob_z": 0.4353864050668189,
    "answer_select_delta": 0.11405637547581872,
    "answer_select_z": 3.151441197561941,
    "cap_hit_frac_max": 0.00390625,
    "control_n": 256.0,
    "delta": 0.03515625,
    "delta_z": 1.000506750675012,
    "failure_frac_max": 0.0,
    "step": 4,
    "vs_corrupt_delta": 0.8203125,
    "vs_corrupt_z": 34.18619095737215
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T132457Z-configPTC-015-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "completed cleanly but missed both reject and promotion thresholds",
  "rows": 5,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T132457Z-configPTC-015-er-opd-q36-35b-slots-trainer-head",
  "verdict": "inconclusive"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T142112Z-PTC-015-step4-inconclusive.json`

```json
{
  "control_rows": 1,
  "metrics": {
    "acc_corrupt": 0.0,
    "acc_nopause": 0.78515625,
    "acc_pause": 0.8203125,
    "answer_logprob_margin": 0.006929151958554657,
    "answer_logprob_z": 0.4353864050668189,
    "answer_select_delta": 0.11405637547581872,
    "answer_select_z": 3.151441197561941,
    "cap_hit_frac_max": 0.00390625,
    "control_n": 256.0,
    "delta": 0.03515625,
    "delta_z": 1.000506750675012,
    "failure_frac_max": 0.0,
    "step": 4,
    "vs_corrupt_delta": 0.8203125,
    "vs_corrupt_z": 34.18619095737215
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T132457Z-configPTC-015-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "completed cleanly but missed both reject and promotion thresholds",
  "rows": 5,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T132457Z-configPTC-015-er-opd-q36-35b-slots-trainer-head",
  "verdict": "inconclusive"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T150312Z-PTC-015-step8-inconclusive.json`

```json
{
  "control_rows": 2,
  "metrics": {
    "acc_corrupt": 0.00390625,
    "acc_nopause": 0.8046875,
    "acc_pause": 0.828125,
    "answer_logprob_margin": -0.044649495764718904,
    "answer_logprob_z": -1.9204682948730094,
    "answer_select_delta": 0.16635415123834688,
    "answer_select_z": 2.769171540539168,
    "cap_hit_frac_max": 0.0,
    "control_n": 256.0,
    "delta": 0.0234375,
    "delta_z": 0.6852250139532997,
    "failure_frac_max": 0.0,
    "step": 8,
    "vs_corrupt_delta": 0.82421875,
    "vs_corrupt_z": 34.48670434808517
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T132457Z-configPTC-015-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "completed cleanly but missed both reject and promotion thresholds",
  "rows": 9,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T132457Z-configPTC-015-er-opd-q36-35b-slots-trainer-head",
  "verdict": "inconclusive"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T160508Z-PTC-016-step5-inconclusive.json`

```json
{
  "control_rows": 1,
  "metrics": {
    "acc_corrupt": 0.5390625,
    "acc_nopause": 0.71484375,
    "acc_pause": 0.72265625,
    "answer_logprob_margin": -0.023487769558334422,
    "answer_logprob_z": -1.8648381345662142,
    "answer_select_delta": -0.17712099054024782,
    "answer_select_z": -6.542153945802289,
    "cap_hit_frac_max": 0.0,
    "control_n": 256.0,
    "delta": 0.0078125,
    "delta_z": 0.19659669487640224,
    "failure_frac_max": 0.0,
    "step": 5,
    "vs_corrupt_delta": 0.18359375,
    "vs_corrupt_z": 4.384336584684985
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T150313Z-configPTC-016-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "completed cleanly but missed both reject and promotion thresholds",
  "rows": 6,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T150313Z-configPTC-016-er-opd-q36-35b-slots-trainer-head",
  "verdict": "inconclusive"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T170915Z-PTC-018-step5-science_reject.json`

```json
{
  "control_rows": 1,
  "metrics": {
    "acc_corrupt": 0.05078125,
    "acc_nopause": 0.79296875,
    "acc_pause": 0.7890625,
    "answer_logprob_margin": -0.14569518343434454,
    "answer_logprob_z": -4.566378284576971,
    "answer_select_delta": -0.4431691200380204,
    "answer_select_z": -8.050098350765134,
    "cap_hit_frac_max": 0.0,
    "control_n": 256.0,
    "delta": -0.00390625,
    "delta_z": -0.10869775914448827,
    "failure_frac_max": 0.0,
    "step": 5,
    "vs_corrupt_delta": 0.73828125,
    "vs_corrupt_z": 25.496527933431814
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T160640Z-configPTC-018-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "no positive accuracy or answer-logprob signal",
  "rows": 6,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T160640Z-configPTC-018-er-opd-q36-35b-slots-trainer-head",
  "verdict": "science_reject"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T170927Z-PTC-018-step5-science_reject.json`

```json
{
  "control_rows": 1,
  "metrics": {
    "acc_corrupt": 0.05078125,
    "acc_nopause": 0.79296875,
    "acc_pause": 0.7890625,
    "answer_logprob_margin": -0.14569518343434454,
    "answer_logprob_z": -4.566378284576971,
    "answer_select_delta": -0.4431691200380204,
    "answer_select_z": -8.050098350765134,
    "cap_hit_frac_max": 0.0,
    "control_n": 256.0,
    "delta": -0.00390625,
    "delta_z": -0.10869775914448827,
    "failure_frac_max": 0.0,
    "step": 5,
    "vs_corrupt_delta": 0.73828125,
    "vs_corrupt_z": 25.496527933431814
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T160640Z-configPTC-018-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "no positive accuracy or answer-logprob signal",
  "rows": 6,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T160640Z-configPTC-018-er-opd-q36-35b-slots-trainer-head",
  "verdict": "science_reject"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T182846Z-PTC-019-step5-infra_invalid.json`

```json
{
  "control_rows": 1,
  "metrics": {
    "acc_corrupt": 0.0390625,
    "acc_nopause": 0.6953125,
    "acc_pause": 0.73046875,
    "answer_logprob_margin": -0.29823434144685007,
    "answer_logprob_z": -7.985485988385205,
    "answer_select_delta": -0.14551921726336287,
    "answer_select_z": -3.0982369496584097,
    "cap_hit_frac_max": 0.00390625,
    "control_n": 256.0,
    "delta": 0.03515625,
    "delta_z": 0.8798334176828828,
    "failure_frac_max": 0.00390625,
    "score_mode": "causal_control",
    "step": 5,
    "vs_corrupt_delta": 0.69140625,
    "vs_corrupt_z": 22.848370565374616
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T171139Z-configPTC-019-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "request_failure_frac_max=0.0039",
  "rows": 6,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T171139Z-configPTC-019-er-opd-q36-35b-slots-trainer-head",
  "verdict": "infra_invalid"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T182857Z-PTC-019-step5-infra_invalid.json`

```json
{
  "control_rows": 1,
  "metrics": {
    "acc_corrupt": 0.0390625,
    "acc_nopause": 0.6953125,
    "acc_pause": 0.73046875,
    "answer_logprob_margin": -0.29823434144685007,
    "answer_logprob_z": -7.985485988385205,
    "answer_select_delta": -0.14551921726336287,
    "answer_select_z": -3.0982369496584097,
    "cap_hit_frac_max": 0.00390625,
    "control_n": 256.0,
    "delta": 0.03515625,
    "delta_z": 0.8798334176828828,
    "failure_frac_max": 0.00390625,
    "score_mode": "causal_control",
    "step": 5,
    "vs_corrupt_delta": 0.69140625,
    "vs_corrupt_z": 22.848370565374616
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T171139Z-configPTC-019-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "request_failure_frac_max=0.0039",
  "rows": 6,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T171139Z-configPTC-019-er-opd-q36-35b-slots-trainer-head",
  "verdict": "infra_invalid"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T182907Z-PTC-019-step5-infra_invalid.json`

```json
{
  "control_rows": 1,
  "metrics": {
    "acc_corrupt": 0.0390625,
    "acc_nopause": 0.6953125,
    "acc_pause": 0.73046875,
    "answer_logprob_margin": -0.29823434144685007,
    "answer_logprob_z": -7.985485988385205,
    "answer_select_delta": -0.14551921726336287,
    "answer_select_z": -3.0982369496584097,
    "cap_hit_frac_max": 0.00390625,
    "control_n": 256.0,
    "delta": 0.03515625,
    "delta_z": 0.8798334176828828,
    "failure_frac_max": 0.00390625,
    "score_mode": "causal_control",
    "step": 5,
    "vs_corrupt_delta": 0.69140625,
    "vs_corrupt_z": 22.848370565374616
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T171139Z-configPTC-019-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "request_failure_frac_max=0.0039",
  "rows": 6,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T171139Z-configPTC-019-er-opd-q36-35b-slots-trainer-head",
  "verdict": "infra_invalid"
}
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T194427Z-PTC-020-step5-promote_retest.json`

```json
{
  "control_rows": 1,
  "metrics": {
    "acc_corrupt": 0.6298828125,
    "acc_nopause": 0.5732421875,
    "acc_pause": 0.619140625,
    "answer_logprob_margin": 0.1110034145900177,
    "answer_logprob_z": 23.75572011440179,
    "answer_select_delta": 0.0,
    "answer_select_z": 0.0,
    "cap_hit_frac_max": 0.00390625,
    "control_n": 1024.0,
    "delta": 0.0458984375,
    "delta_z": 2.1189846284303124,
    "failure_frac_max": 0.0,
    "score_mode": "pause_vs_nopause",
    "step": 5,
    "vs_corrupt_delta": -0.0107421875,
    "vs_corrupt_z": -0.5019794461402117
  },
  "profile": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T191133Z-configPTC-020-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl",
  "reason": "pause-vs-nopause accuracy warrants retest with positive answer-logprob support",
  "rows": 6,
  "run_dir": "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T191133Z-configPTC-020-er-opd-q36-35b-slots-trainer-head",
  "verdict": "promote_retest"
}
```


## 6. Scorecard Markdown Files


### `experiments/opd_profile/autoresearch/scorecards/20260604T011532Z-PTC-001-step8-infra_invalid.md`

```markdown
# PTC-001 scorecard

- verdict: `infra_invalid`
- reason: request_failure_frac_max=0.3720
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T225859Z-configPTC-001-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `1000.0`
- acc_pause / acc_nopause / acc_corrupt: `0.802` / `0.808` / `0.0`
- delta / z: `-0.006000000000000005` / `-0.33863625839560096`
- vs_corrupt_delta / z: `0.802` / `63.643578234611006`
- answer_logprob_margin / z: `0.0011698661124863946` / `0.5921812725913325`
- answer_select_delta / z: `0.0` / `0.0`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T011532Z-PTC-005-step8-infra_invalid.md`

```markdown
# PTC-005 scorecard

- verdict: `infra_invalid`
- reason: request_failure_frac_max=0.4360
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T000006Z-configPTC-005-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `1000.0`
- acc_pause / acc_nopause / acc_corrupt: `0.799` / `0.799` / `0.0`
- delta / z: `0.0` / `0.0`
- vs_corrupt_delta / z: `0.799` / `63.04858743944589`
- answer_logprob_margin / z: `-0.004177647228492835` / `-2.164316609339693`
- answer_select_delta / z: `0.0` / `0.0`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T011537Z-PTC-001-step8-infra_invalid.md`

```markdown
# PTC-001 scorecard

- verdict: `infra_invalid`
- reason: request_failure_frac_max=0.3720
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260603T225859Z-configPTC-001-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `1000.0`
- acc_pause / acc_nopause / acc_corrupt: `0.802` / `0.808` / `0.0`
- delta / z: `-0.006000000000000005` / `-0.33863625839560096`
- vs_corrupt_delta / z: `0.802` / `63.643578234611006`
- answer_logprob_margin / z: `0.0011698661124863946` / `0.5921812725913325`
- answer_select_delta / z: `0.0` / `0.0`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T021202Z-PTC-006-step8-science_reject.md`

```markdown
# PTC-006 scorecard

- verdict: `science_reject`
- reason: no positive accuracy or answer-logprob signal
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T013603Z-configPTC-006-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `1000.0`
- acc_pause / acc_nopause / acc_corrupt: `0.807` / `0.812` / `0.0`
- delta / z: `-0.0050000000000000044` / `-0.28471338904946075`
- vs_corrupt_delta / z: `0.807` / `64.6633369867274`
- answer_logprob_margin / z: `-0.0005920925329430714` / `-0.3690380389251624`
- answer_select_delta / z: `0.0` / `0.0`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T021205Z-PTC-006-step8-science_reject.md`

```markdown
# PTC-006 scorecard

- verdict: `science_reject`
- reason: no positive accuracy or answer-logprob signal
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T013603Z-configPTC-006-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `1000.0`
- acc_pause / acc_nopause / acc_corrupt: `0.807` / `0.812` / `0.0`
- delta / z: `-0.0050000000000000044` / `-0.28471338904946075`
- vs_corrupt_delta / z: `0.807` / `64.6633369867274`
- answer_logprob_margin / z: `-0.0005920925329430714` / `-0.3690380389251624`
- answer_select_delta / z: `0.0` / `0.0`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T045539Z-PTC-003-step8-science_reject.md`

```markdown
# PTC-003 scorecard

- verdict: `science_reject`
- reason: no positive accuracy or answer-logprob signal
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T021246Z-configPTC-003-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `1000.0`
- acc_pause / acc_nopause / acc_corrupt: `0.797` / `0.803` / `0.0`
- delta / z: `-0.006000000000000005` / `-0.3354196304347396`
- vs_corrupt_delta / z: `0.797` / `62.65866559690078`
- answer_logprob_margin / z: `-0.002851773091628856` / `-1.550627938419657`
- answer_select_delta / z: `0.0` / `0.0`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T045543Z-PTC-003-step8-science_reject.md`

```markdown
# PTC-003 scorecard

- verdict: `science_reject`
- reason: no positive accuracy or answer-logprob signal
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T021246Z-configPTC-003-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `1000.0`
- acc_pause / acc_nopause / acc_corrupt: `0.797` / `0.803` / `0.0`
- delta / z: `-0.006000000000000005` / `-0.3354196304347396`
- vs_corrupt_delta / z: `0.797` / `62.65866559690078`
- answer_logprob_margin / z: `-0.002851773091628856` / `-1.550627938419657`
- answer_select_delta / z: `0.0` / `0.0`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T053033Z-PTC-007-step4-science_reject.md`

```markdown
# PTC-007 scorecard

- verdict: `science_reject`
- reason: no positive accuracy or answer-logprob signal
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T050810Z-configPTC-007-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `1000.0`
- acc_pause / acc_nopause / acc_corrupt: `0.677` / `0.766` / `0.0`
- delta / z: `-0.08899999999999997` / `-4.461643350547222`
- vs_corrupt_delta / z: `0.677` / `45.781822071627325`
- answer_logprob_margin / z: `-0.05189284104206759` / `-8.495199167432943`
- answer_select_delta / z: `0.0` / `0.0`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T060104Z-PTC-008-step4-science_reject.md`

```markdown
# PTC-008 scorecard

- verdict: `science_reject`
- reason: no positive accuracy or answer-logprob signal
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T053133Z-configPTC-008-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `1000.0`
- acc_pause / acc_nopause / acc_corrupt: `0.624` / `0.71` / `0.591`
- delta / z: `-0.08599999999999997` / `-4.097450014496387`
- vs_corrupt_delta / z: `0.03300000000000003` / `1.5120078506650454`
- answer_logprob_margin / z: `-0.004445023867406381` / `-0.7627423018328828`
- answer_select_delta / z: `-0.025082884714261792` / `-3.434489752279603`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T063058Z-PTC-009-step4-science_reject.md`

```markdown
# PTC-009 scorecard

- verdict: `science_reject`
- reason: no positive accuracy or answer-logprob signal
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T060143Z-configPTC-009-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `1000.0`
- acc_pause / acc_nopause / acc_corrupt: `0.719` / `0.801` / `0.719`
- delta / z: `-0.08200000000000007` / `-4.313173687822969`
- vs_corrupt_delta / z: `0.0` / `0.0`
- answer_logprob_margin / z: `-0.06802329225159871` / `-7.5156321540708575`
- answer_select_delta / z: `-0.17130115062077741` / `-14.448557610375241`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T070229Z-PTC-010-step4-science_reject.md`

```markdown
# PTC-010 scorecard

- verdict: `science_reject`
- reason: no positive accuracy or answer-logprob signal
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T063316Z-configPTC-010-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `1000.0`
- acc_pause / acc_nopause / acc_corrupt: `0.684` / `0.789` / `0.677`
- delta / z: `-0.10499999999999998` / `-5.367891918486458`
- vs_corrupt_delta / z: `0.007000000000000006` / `0.3356957021998745`
- answer_logprob_margin / z: `-0.07165765871649975` / `-8.41706506088195`
- answer_select_delta / z: `-0.19453478057920348` / `-16.804226108679384`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T074639Z-PTC-011-stepna-incomplete.md`

```markdown
# PTC-011 scorecard

- verdict: `incomplete`
- reason: no control rows
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `None`
- acc_pause / acc_nopause / acc_corrupt: `None` / `None` / `None`
- delta / z: `None` / `None`
- vs_corrupt_delta / z: `None` / `None`
- answer_logprob_margin / z: `None` / `None`
- answer_select_delta / z: `None` / `None`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T074650Z-PTC-011-stepna-incomplete.md`

```markdown
# PTC-011 scorecard

- verdict: `incomplete`
- reason: no control rows
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `None`
- acc_pause / acc_nopause / acc_corrupt: `None` / `None` / `None`
- delta / z: `None` / `None`
- vs_corrupt_delta / z: `None` / `None`
- answer_logprob_margin / z: `None` / `None`
- answer_select_delta / z: `None` / `None`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T074659Z-PTC-011-stepna-incomplete.md`

```markdown
# PTC-011 scorecard

- verdict: `incomplete`
- reason: no control rows
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `None`
- acc_pause / acc_nopause / acc_corrupt: `None` / `None` / `None`
- delta / z: `None` / `None`
- vs_corrupt_delta / z: `None` / `None`
- answer_logprob_margin / z: `None` / `None`
- answer_select_delta / z: `None` / `None`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T075411Z-PTC-011-stepna-incomplete.md`

```markdown
# PTC-011 scorecard

- verdict: `incomplete`
- reason: no control rows
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `None`
- acc_pause / acc_nopause / acc_corrupt: `None` / `None` / `None`
- delta / z: `None` / `None`
- vs_corrupt_delta / z: `None` / `None`
- answer_logprob_margin / z: `None` / `None`
- answer_select_delta / z: `None` / `None`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T075543Z-PTC-011-stepna-incomplete.md`

```markdown
# PTC-011 scorecard

- verdict: `incomplete`
- reason: no control rows
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `None`
- acc_pause / acc_nopause / acc_corrupt: `None` / `None` / `None`
- delta / z: `None` / `None`
- vs_corrupt_delta / z: `None` / `None`
- answer_logprob_margin / z: `None` / `None`
- answer_select_delta / z: `None` / `None`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T075929Z-PTC-011-stepna-incomplete.md`

```markdown
# PTC-011 scorecard

- verdict: `incomplete`
- reason: no control rows
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `None`
- acc_pause / acc_nopause / acc_corrupt: `None` / `None` / `None`
- delta / z: `None` / `None`
- vs_corrupt_delta / z: `None` / `None`
- answer_logprob_margin / z: `None` / `None`
- answer_select_delta / z: `None` / `None`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T075957Z-PTC-011-stepna-incomplete.md`

```markdown
# PTC-011 scorecard

- verdict: `incomplete`
- reason: no control rows
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `None`
- acc_pause / acc_nopause / acc_corrupt: `None` / `None` / `None`
- delta / z: `None` / `None`
- vs_corrupt_delta / z: `None` / `None`
- answer_logprob_margin / z: `None` / `None`
- answer_select_delta / z: `None` / `None`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T080016Z-PTC-011-stepna-incomplete.md`

```markdown
# PTC-011 scorecard

- verdict: `incomplete`
- reason: no control rows
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `None`
- acc_pause / acc_nopause / acc_corrupt: `None` / `None` / `None`
- delta / z: `None` / `None`
- vs_corrupt_delta / z: `None` / `None`
- answer_logprob_margin / z: `None` / `None`
- answer_select_delta / z: `None` / `None`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T082853Z-PTC-011-step4-inconclusive.md`

```markdown
# PTC-011 scorecard

- verdict: `inconclusive`
- reason: completed cleanly but missed both reject and promotion thresholds
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `256.0`
- acc_pause / acc_nopause / acc_corrupt: `0.78515625` / `0.73828125` / `0.00390625`
- delta / z: `0.046875` / `1.2467006799461504`
- vs_corrupt_delta / z: `0.78125` / `30.08975668093039`
- answer_logprob_margin / z: `-0.1338996193211561` / `-4.934445616493985`
- answer_select_delta / z: `0.05212081693119896` / `1.4338520989113466`

## Interpretation

PTC-011 partially supports the generated-memory hypothesis: pause accuracy beats
no-pause by 4.7 points, and shuffled generated memory collapses accuracy to
0.4%, so the prompt-specific memory is carrying task-relevant information. This
is qualitatively different from PTC-009/PTC-010, where pause was basically tied
with corrupt memory and worse than no-pause.

It is not promotable yet because answer logprob under pause is still worse than
no-pause (`-0.134`, z=-4.93). Continue the active run to the next control point
instead of launching RiM immediately. If step 8 keeps the accuracy gain but
answer logprob remains negative, the next recipe should shift to the RiM
answer-utility objective in PTC-012.
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T082854Z-PTC-011-step4-inconclusive.md`

```markdown
# PTC-011 scorecard

- verdict: `inconclusive`
- reason: completed cleanly but missed both reject and promotion thresholds
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `256.0`
- acc_pause / acc_nopause / acc_corrupt: `0.78515625` / `0.73828125` / `0.00390625`
- delta / z: `0.046875` / `1.2467006799461504`
- vs_corrupt_delta / z: `0.78125` / `30.08975668093039`
- answer_logprob_margin / z: `-0.1338996193211561` / `-4.934445616493985`
- answer_select_delta / z: `0.05212081693119896` / `1.4338520989113466`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T091135Z-PTC-011-step8-inconclusive.md`

```markdown
# PTC-011 scorecard

- verdict: `inconclusive`
- reason: completed cleanly but missed both reject and promotion thresholds
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `256.0`
- acc_pause / acc_nopause / acc_corrupt: `0.80859375` / `0.78125` / `0.30078125`
- delta / z: `0.02734375` / `0.7666379087775477`
- vs_corrupt_delta / z: `0.5078125` / `13.44708920081176`
- answer_logprob_margin / z: `-0.027456654039798866` / `-2.1209689010269392`
- answer_select_delta / z: `0.1930973367662947` / `7.5565314530967775`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T091212Z-PTC-011-step8-inconclusive.md`

```markdown
# PTC-011 scorecard

- verdict: `inconclusive`
- reason: completed cleanly but missed both reject and promotion thresholds
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T073501Z-configPTC-011-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `256.0`
- acc_pause / acc_nopause / acc_corrupt: `0.80859375` / `0.78125` / `0.30078125`
- delta / z: `0.02734375` / `0.7666379087775477`
- vs_corrupt_delta / z: `0.5078125` / `13.44708920081176`
- answer_logprob_margin / z: `-0.027456654039798866` / `-2.1209689010269392`
- answer_select_delta / z: `0.1930973367662947` / `7.5565314530967775`

## Interpretation

PTC-011 partially supports the generated-memory hypothesis: real generated memory
still beats no-pause on accuracy and strongly beats shuffled generated memory, so
the prompt-specific memory channel is carrying task-relevant information.

It does not support promoting the hidden/cache target recipe. The pause advantage
over no-pause shrank from `+0.046875` at step 4 to `+0.02734375` at step 8, and
answer logprob remained negative versus no-pause despite improving from
`-0.1338996193211561` to `-0.027456654039798866`. The strong positive answer
selection delta says the memory sometimes points at the right answer, but the
overall likelihood still favors no-pause.

Queue decision: advance PTC-011 as inconclusive and launch PTC-012, which keeps
the generated-memory plumbing but removes the hidden/cache target and directly
optimizes answer utility with real-vs-corrupt generated memory.
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T100716Z-PTC-012-step4-inconclusive.md`

```markdown
# PTC-012 scorecard

- verdict: `inconclusive`
- reason: completed cleanly but missed both reject and promotion thresholds
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T091215Z-configPTC-012-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `256.0`
- acc_pause / acc_nopause / acc_corrupt: `0.83984375` / `0.73046875` / `0.0`
- delta / z: `0.109375` / `3.0399616129719975`
- vs_corrupt_delta / z: `0.83984375` / `36.63930982415258`
- answer_logprob_margin / z: `-0.11067124789628861` / `-3.0825148106625977`
- answer_select_delta / z: `-0.3257433384510804` / `-6.977632527602614`

## Interpretation

PTC-012 strengthens the causal generated-memory story but does not yet validate
the answer-utility recipe. Compared with PTC-011 step 4, pause-vs-no-pause
accuracy improved from `+0.046875` to `+0.109375`, and shuffled generated memory
collapsed completely to `0.0` accuracy. That says the real generated memory is
carrying prompt-specific answer information.

The failure mode moved rather than disappeared. Answer logprob is still worse
than no-pause (`-0.11067124789628861`, z `-3.0825148106625977`), and the answer
selection metric is now strongly negative. The direct real-vs-corrupt answer
objective is making generated memory causally useful under sampling, but it is
not aligning the model's likelihood preference for the pause-conditioned answer.

Queue decision: continue PTC-012 to the step-8 control. If the step-8 accuracy
delta stays strong while answer logprob/selection remain negative, advance
PTC-012 as not promotable and move to PTC-013's small memory-KL bootstrap.
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T100734Z-PTC-012-step4-inconclusive.md`

```markdown
# PTC-012 scorecard

- verdict: `inconclusive`
- reason: completed cleanly but missed both reject and promotion thresholds
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T091215Z-configPTC-012-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `256.0`
- acc_pause / acc_nopause / acc_corrupt: `0.83984375` / `0.73046875` / `0.0`
- delta / z: `0.109375` / `3.0399616129719975`
- vs_corrupt_delta / z: `0.83984375` / `36.63930982415258`
- answer_logprob_margin / z: `-0.11067124789628861` / `-3.0825148106625977`
- answer_select_delta / z: `-0.3257433384510804` / `-6.977632527602614`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T105011Z-PTC-012-step8-inconclusive.md`

```markdown
# PTC-012 scorecard

- verdict: `inconclusive`
- reason: completed cleanly but missed both reject and promotion thresholds
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T091215Z-configPTC-012-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `256.0`
- acc_pause / acc_nopause / acc_corrupt: `0.8046875` / `0.78515625` / `0.11328125`
- delta / z: `0.01953125` / `0.5474446288295078`
- vs_corrupt_delta / z: `0.69140625` / `21.795580416961943`
- answer_logprob_margin / z: `-0.1596965955282831` / `-5.496740775541098`
- answer_select_delta / z: `-0.7489891492578584` / `-14.397340349047914`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T105043Z-PTC-012-step8-inconclusive.md`

```markdown
# PTC-012 scorecard

- verdict: `inconclusive`
- reason: completed cleanly but missed both reject and promotion thresholds
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T091215Z-configPTC-012-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `256.0`
- acc_pause / acc_nopause / acc_corrupt: `0.8046875` / `0.78515625` / `0.11328125`
- delta / z: `0.01953125` / `0.5474446288295078`
- vs_corrupt_delta / z: `0.69140625` / `21.795580416961943`
- answer_logprob_margin / z: `-0.1596965955282831` / `-5.496740775541098`
- answer_select_delta / z: `-0.7489891492578584` / `-14.397340349047914`

## Interpretation

PTC-012 is rejected as a recipe despite the controller's raw `inconclusive`
threshold verdict. The step-4 result briefly looked promising on sampled
accuracy (`+0.109375` pause over no-pause), but by step 8 that advantage decayed
to `+0.01953125` with z `0.5474446288295078`.

The robust finding is still that generated memory is causal: shuffled generated
memory remains far worse than real generated memory (`+0.69140625` real-vs-corrupt
accuracy at step 8). The failure is likelihood alignment and credit assignment.
Answer logprob became more negative than at step 4 (`-0.1596965955282831` versus
`-0.11067124789628861`), and answer selection became strongly worse
(`-0.7489891492578584`).

Queue decision: launch PTC-013. The answer-only RiM utility objective appears too
sparse or misdirected, so the next test adds a small teacher-conditioned memory
KL bootstrap while keeping answer utility and corrupt-memory contrast dominant.
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T114657Z-PTC-013-step4-science_reject.md`

```markdown
# PTC-013 scorecard

- verdict: `science_reject`
- reason: no positive accuracy or answer-logprob signal
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T105110Z-configPTC-013-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `256.0`
- acc_pause / acc_nopause / acc_corrupt: `0.59765625` / `0.75` / `0.1328125`
- delta / z: `-0.15234375` / `-3.7259874043167356`
- vs_corrupt_delta / z: `0.46484375` / `12.47164544836176`
- answer_logprob_margin / z: `-0.31010046812613806` / `-9.973705393936111`
- answer_select_delta / z: `-0.24107080985158527` / `-6.486670231064018`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T114724Z-PTC-013-step4-science_reject.md`

```markdown
# PTC-013 scorecard

- verdict: `science_reject`
- reason: no positive accuracy or answer-logprob signal
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T105110Z-configPTC-013-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `256.0`
- acc_pause / acc_nopause / acc_corrupt: `0.59765625` / `0.75` / `0.1328125`
- delta / z: `-0.15234375` / `-3.7259874043167356`
- vs_corrupt_delta / z: `0.46484375` / `12.47164544836176`
- answer_logprob_margin / z: `-0.31010046812613806` / `-9.973705393936111`
- answer_select_delta / z: `-0.24107080985158527` / `-6.486670231064018`

## Interpretation

PTC-013 rejects the small memory-KL bootstrap. The shuffled-memory control still
shows that generated memory contains some prompt-specific signal, but real pause
is now worse than no-pause by `-0.15234375` accuracy and substantially worse on
answer logprob.

This argues against adding even a small teacher-conditioned buffer KL on top of
the answer-utility objective. It appears to make the generated memory less useful
for the final answer rather than stabilizing credit assignment.

Queue decision: launch PTC-014, which removes both the corrupt-answer anti-target
and memory KL, leaving only positive gold-answer utility on generated memory.
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T124228Z-PTC-014-step4-inconclusive.md`

```markdown
# PTC-014 scorecard

- verdict: `inconclusive`
- reason: completed cleanly but missed both reject and promotion thresholds
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T114759Z-configPTC-014-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `256.0`
- acc_pause / acc_nopause / acc_corrupt: `0.8359375` / `0.69921875` / `0.015625`
- delta / z: `0.13671875` / `3.7110514492390805`
- vs_corrupt_delta / z: `0.8203125` / `33.60672201667224`
- answer_logprob_margin / z: `-0.06823835960594374` / `-2.393474203845524`
- answer_select_delta / z: `-0.18082758527490222` / `-4.051345401410779`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T124319Z-PTC-014-step4-inconclusive.md`

```markdown
# PTC-014 scorecard

- verdict: `inconclusive`
- reason: completed cleanly but missed both reject and promotion thresholds
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T114759Z-configPTC-014-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `256.0`
- acc_pause / acc_nopause / acc_corrupt: `0.8359375` / `0.69921875` / `0.015625`
- delta / z: `0.13671875` / `3.7110514492390805`
- vs_corrupt_delta / z: `0.8203125` / `33.60672201667224`
- answer_logprob_margin / z: `-0.06823835960594374` / `-2.393474203845524`
- answer_select_delta / z: `-0.18082758527490222` / `-4.051345401410779`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T132437Z-PTC-014-step8-science_reject.md`

```markdown
# PTC-014 scorecard

- verdict: `science_reject`
- reason: no positive accuracy or answer-logprob signal
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T114759Z-configPTC-014-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `256.0`
- acc_pause / acc_nopause / acc_corrupt: `0.76171875` / `0.76953125` / `0.3203125`
- delta / z: `-0.0078125` / `-0.20866508438634226`
- vs_corrupt_delta / z: `0.44140625` / `11.177756782299781`
- answer_logprob_margin / z: `-0.10689015139499593` / `-5.096342933926368`
- answer_select_delta / z: `-0.14860751767676922` / `-3.269913442281971`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T132455Z-PTC-014-step8-science_reject.md`

```markdown
# PTC-014 scorecard

- verdict: `science_reject`
- reason: no positive accuracy or answer-logprob signal
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T114759Z-configPTC-014-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `256.0`
- acc_pause / acc_nopause / acc_corrupt: `0.76171875` / `0.76953125` / `0.3203125`
- delta / z: `-0.0078125` / `-0.20866508438634226`
- vs_corrupt_delta / z: `0.44140625` / `11.177756782299781`
- answer_logprob_margin / z: `-0.10689015139499593` / `-5.096342933926368`
- answer_select_delta / z: `-0.14860751767676922` / `-3.269913442281971`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T142024Z-PTC-015-step4-inconclusive.md`

```markdown
# PTC-015 scorecard

- verdict: `inconclusive`
- reason: completed cleanly but missed both reject and promotion thresholds
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T132457Z-configPTC-015-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `256.0`
- acc_pause / acc_nopause / acc_corrupt: `0.8203125` / `0.78515625` / `0.0`
- delta / z: `0.03515625` / `1.000506750675012`
- vs_corrupt_delta / z: `0.8203125` / `34.18619095737215`
- answer_logprob_margin / z: `0.006929151958554657` / `0.4353864050668189`
- answer_select_delta / z: `0.11405637547581872` / `3.151441197561941`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T142105Z-PTC-015-step4-inconclusive.md`

```markdown
# PTC-015 scorecard

- verdict: `inconclusive`
- reason: completed cleanly but missed both reject and promotion thresholds
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T132457Z-configPTC-015-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `256.0`
- acc_pause / acc_nopause / acc_corrupt: `0.8203125` / `0.78515625` / `0.0`
- delta / z: `0.03515625` / `1.000506750675012`
- vs_corrupt_delta / z: `0.8203125` / `34.18619095737215`
- answer_logprob_margin / z: `0.006929151958554657` / `0.4353864050668189`
- answer_select_delta / z: `0.11405637547581872` / `3.151441197561941`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T142112Z-PTC-015-step4-inconclusive.md`

```markdown
# PTC-015 scorecard

- verdict: `inconclusive`
- reason: completed cleanly but missed both reject and promotion thresholds
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T132457Z-configPTC-015-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `256.0`
- acc_pause / acc_nopause / acc_corrupt: `0.8203125` / `0.78515625` / `0.0`
- delta / z: `0.03515625` / `1.000506750675012`
- vs_corrupt_delta / z: `0.8203125` / `34.18619095737215`
- answer_logprob_margin / z: `0.006929151958554657` / `0.4353864050668189`
- answer_select_delta / z: `0.11405637547581872` / `3.151441197561941`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T150312Z-PTC-015-step8-inconclusive.md`

```markdown
# PTC-015 scorecard

- verdict: `inconclusive`
- reason: completed cleanly but missed both reject and promotion thresholds
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T132457Z-configPTC-015-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `256.0`
- acc_pause / acc_nopause / acc_corrupt: `0.828125` / `0.8046875` / `0.00390625`
- delta / z: `0.0234375` / `0.6852250139532997`
- vs_corrupt_delta / z: `0.82421875` / `34.48670434808517`
- answer_logprob_margin / z: `-0.044649495764718904` / `-1.9204682948730094`
- answer_select_delta / z: `0.16635415123834688` / `2.769171540539168`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T160508Z-PTC-016-step5-inconclusive.md`

```markdown
# PTC-016 scorecard

- verdict: `inconclusive`
- reason: completed cleanly but missed both reject and promotion thresholds
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T150313Z-configPTC-016-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `256.0`
- acc_pause / acc_nopause / acc_corrupt: `0.72265625` / `0.71484375` / `0.5390625`
- delta / z: `0.0078125` / `0.19659669487640224`
- vs_corrupt_delta / z: `0.18359375` / `4.384336584684985`
- answer_logprob_margin / z: `-0.023487769558334422` / `-1.8648381345662142`
- answer_select_delta / z: `-0.17712099054024782` / `-6.542153945802289`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T170915Z-PTC-018-step5-science_reject.md`

```markdown
# PTC-018 scorecard

- verdict: `science_reject`
- reason: no positive accuracy or answer-logprob signal
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T160640Z-configPTC-018-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `256.0`
- acc_pause / acc_nopause / acc_corrupt: `0.7890625` / `0.79296875` / `0.05078125`
- delta / z: `-0.00390625` / `-0.10869775914448827`
- vs_corrupt_delta / z: `0.73828125` / `25.496527933431814`
- answer_logprob_margin / z: `-0.14569518343434454` / `-4.566378284576971`
- answer_select_delta / z: `-0.4431691200380204` / `-8.050098350765134`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T170927Z-PTC-018-step5-science_reject.md`

```markdown
# PTC-018 scorecard

- verdict: `science_reject`
- reason: no positive accuracy or answer-logprob signal
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T160640Z-configPTC-018-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `256.0`
- acc_pause / acc_nopause / acc_corrupt: `0.7890625` / `0.79296875` / `0.05078125`
- delta / z: `-0.00390625` / `-0.10869775914448827`
- vs_corrupt_delta / z: `0.73828125` / `25.496527933431814`
- answer_logprob_margin / z: `-0.14569518343434454` / `-4.566378284576971`
- answer_select_delta / z: `-0.4431691200380204` / `-8.050098350765134`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T182846Z-PTC-019-step5-infra_invalid.md`

```markdown
# PTC-019 scorecard

- verdict: `infra_invalid`
- reason: request_failure_frac_max=0.0039
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T171139Z-configPTC-019-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `256.0`
- acc_pause / acc_nopause / acc_corrupt: `0.73046875` / `0.6953125` / `0.0390625`
- delta / z: `0.03515625` / `0.8798334176828828`
- vs_corrupt_delta / z: `0.69140625` / `22.848370565374616`
- answer_logprob_margin / z: `-0.29823434144685007` / `-7.985485988385205`
- answer_select_delta / z: `-0.14551921726336287` / `-3.0982369496584097`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T182857Z-PTC-019-step5-infra_invalid.md`

```markdown
# PTC-019 scorecard

- verdict: `infra_invalid`
- reason: request_failure_frac_max=0.0039
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T171139Z-configPTC-019-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `256.0`
- acc_pause / acc_nopause / acc_corrupt: `0.73046875` / `0.6953125` / `0.0390625`
- delta / z: `0.03515625` / `0.8798334176828828`
- vs_corrupt_delta / z: `0.69140625` / `22.848370565374616`
- answer_logprob_margin / z: `-0.29823434144685007` / `-7.985485988385205`
- answer_select_delta / z: `-0.14551921726336287` / `-3.0982369496584097`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T182907Z-PTC-019-step5-infra_invalid.md`

```markdown
# PTC-019 scorecard

- verdict: `infra_invalid`
- reason: request_failure_frac_max=0.0039
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T171139Z-configPTC-019-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `256.0`
- acc_pause / acc_nopause / acc_corrupt: `0.73046875` / `0.6953125` / `0.0390625`
- delta / z: `0.03515625` / `0.8798334176828828`
- vs_corrupt_delta / z: `0.69140625` / `22.848370565374616`
- answer_logprob_margin / z: `-0.29823434144685007` / `-7.985485988385205`
- answer_select_delta / z: `-0.14551921726336287` / `-3.0982369496584097`
```


### `experiments/opd_profile/autoresearch/scorecards/20260604T194427Z-PTC-020-step5-promote_retest.md`

```markdown
# PTC-020 scorecard

- verdict: `promote_retest`
- reason: pause-vs-nopause accuracy warrants retest with positive answer-logprob support
- profile: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T191133Z-configPTC-020-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl`
- control_n: `1024.0`
- acc_pause / acc_nopause / acc_corrupt: `0.619140625` / `0.5732421875` / `0.6298828125`
- delta / z: `0.0458984375` / `2.1189846284303124`
- vs_corrupt_delta / z: `-0.0107421875` / `-0.5019794461402117`
- answer_logprob_margin / z: `0.1110034145900177` / `23.75572011440179`
- answer_select_delta / z: `0.0` / `0.0`
```


## 7. Embedded Source Runbooks And Memos


The following source documents are embedded verbatim so that details removed from the active runbook remain available in this one file.


### `experiments/opd_profile/PREFILL_TIME_COMPUTE_AUTORESEARCH_RUNBOOK_2026_06_04.md`

```markdown
# Prefill-Time-Compute Autoresearch Runbook - 2026-06-04

This is the active handoff for the OPD / prefill-time-compute research loop.
It is intentionally not a complete historical log. Older runbooks remain the
archive; this file should answer: what is true now, what is missing, what is
unnecessary, what hypotheses we are testing, and how to run the loop without
active babysitting.

The exhaustive companion archive is
`experiments/opd_profile/PREFILL_TIME_COMPUTE_AUTORESEARCH_COMPREHENSIVE_ARCHIVE_2026_06_04.md`.
Use that file when you need the full historical ledgers, source runbooks,
candidate YAMLs, run logs, and scorecards in one place.

## 0. Current State

Primary objective:

- Train a model so `prompt + filler tokens + Answer:` gives better answer
  performance than `prompt + Answer:`.
- The filler does not need to be semantically meaningful. The immediate win is
  operational prefill performance, not a complete prompt-specific memory proof.
- Mechanism diagnostics matter, but should not veto a clean pause-vs-no-pause
  operational win.

Current queue head:

```bash
python experiments/opd_profile/autoresearch/controller.py next
```

selects `PTC-023`.

Last inspected service state in this revision:

- `dispatch`, `sglang-0`, `teacher-sglang-0`, and `teacher-sglang-1` are warm.
- `trainer-head` and `trainer-worker-1..7` are stopped after PTC-020.
- The current sampler layout is `spare-teacher1`: `sglang-0` plus
  `teacher-sglang-1` used as two student sampler endpoints.

The immediate research branch is:

1. `PTC-023`: AM minus hidden matching.
2. `PTC-024`: AM minus corrupt-negative training, hidden matching kept.
3. `PTC-025`: AM minus both hidden matching and corrupt-negative training.
4. `PTC-021`: composite-AM stability retest.
5. `PTC-022`: composite-AM larger-batch retest.

The ablation trio are six-step screens with a step-5 1k control. They are not
stability proofs. A positive ablation should get its own longer stability
follow-up before being treated as scale-ready.

## 1. Important Paths

```bash
REPO=/home/apanda/xorl-apanda-dev-opd-port
cd "$REPO"

STACK=er-opd-q36-35b-slots
NS=apanda
CONTROL_ROOT=/shared/opd-control/er-opd-q36-35b-slots
RESULT_ROOT=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots

GENERATOR=experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py
CONTROLLER=experiments/opd_profile/autoresearch/controller.py
IDEAS=experiments/opd_profile/autoresearch/ideas.yaml
CANDIDATES=experiments/opd_profile/autoresearch/candidates
SCORECARDS=experiments/opd_profile/autoresearch/scorecards
```

Model and data:

```bash
MODEL=/shared/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0
PROMPTS_JSON=/shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
COT_JSON=/shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
NUM_PROMPTS=8185
```

Primary files:

- `experiments/opd_profile/autoresearch/controller.py`
- `experiments/opd_profile/autoresearch/ideas.yaml`
- `experiments/opd_profile/autoresearch/candidates/PTC-*.yaml`
- `experiments/opd_profile/autoresearch/runs.jsonl`
- `experiments/opd_profile/autoresearch/scorecards/*.json`

## 2. Decision Standards

Operational metric:

```text
buffer_delta = acc_pause - acc_nopause
```

Operational claim:

- The pause/filler condition improves answer performance versus no pause.
- Promotion-quality evidence is approximately `buffer_delta >= 0.06` and
  `buffer_delta_z >= 3.0` at `n ~= 1000`.
- A retest-worthy result is approximately `buffer_delta >= 0.04` and
  `buffer_delta_z >= 2.0`, especially with positive answer-logprob support.

Mechanism claim:

- The pause/filler contains prompt-specific useful state.
- This needs boundary-identical corrupt/shuffle controls and positive
  correct-vs-distractor answer-selection behavior.
- AM does not yet prove this stronger claim.

Scoring modes:

- `pause_vs_nopause`: the operational gate. Corrupt-control artifacts and
  answer-selection do not veto a pause-vs-no-pause win.
- `causal_control`: the older stricter mechanism gate. Corrupt-control and
  answer-logprob criteria can veto promotion.

Use `pause_vs_nopause` for the current AM branch.

## 3. What AM Showed

AM was an answer-causal static-filler recipe, not a filler-surface search.

Student visible prefix:

```text
prompt + " ! | ~ _ * ^ # @ " + "Answer: "
```

Teacher context:

```text
prompt + teacher CoT + same visible pause/answer tail
```

Key AM settings:

- `opd_supervise_buffer_only=false`
- `opd_hidden_match_coef=2.0`
- `opd_contrastive_corrupt_buffer_weight=1.0`
- `opd_contrastive_corrupt_answer_weight=0.125`
- `opd_loss_max_clamp=5.0`
- `learning_rate=3e-6`
- `max_new_tokens=64`
- `eval_control_start_step=5`
- `eval_num_problems=1024`

AM only had a large operational control at step 5. Steps 0-4 only had small
ordinary eval rows, so the first reliable pause-vs-no-pause signal appeared at
the final scheduled control:

```text
acc_pause      = 0.6494
acc_nopause    = 0.5557
buffer_delta   = +0.0938
buffer_delta_z = 4.35
```

Post-hoc chunked answer-logprob scoring of the same final sampler weights was
also positive:

```text
answer_logprob_margin = +0.1784
answer_logprob_z      = 36.76
```

Why AM was not promoted at the time:

- The old gate was optimized for a prompt-specific memory claim, not the narrower
  operational claim.
- The legacy `rotate` corrupt control changed the assistant continuation
  boundary and produced cap/format artifacts.
- The in-loop answer-logprob scorer sent oversized batches and got HTTP 503s.
  Chunked post-hoc scoring fixed that.
- The answer-selection distractor probe was genuinely negative. AM raised
  absolute true-answer likelihood but did not improve the correct-vs-wrong
  likelihood margin.

Current interpretation:

- AM should be treated as the strongest operational prefill-performance hit.
- AM should not be cited as a complete prompt-specific memory proof.

## 4. Current Evidence Ledger

Decision-critical completed runs:

| Run | Result | Interpretation |
| --- | --- | --- |
| AH | `delta=+0.2708`, `z=4.06`, answer-logprob `+0.2234`, `z=5.33`, but `n=96` and corrupt cap-hit `0.9375` | First strong answer-causal signal; too small and corrupt-degenerate for promotion. |
| AI | `delta=+0.0938`, `z=1.91`, answer-logprob `+0.2163`, `z=16.67` at `n=192` | Exact-match underpowered but directionally strong. |
| AL | `delta=+0.1146`, `z=2.41`, answer-logprob `+0.2225`, `z=17.04`, corrupt cap-hit `0.5833` | Real positive, rejected by old corrupt-control health gate. |
| AM | `delta=+0.0938`, `z=4.35` at `n=1024`; post-hoc answer-logprob `+0.1784`, `z=36.76` | Strongest operational result. |
| AN | `delta=+0.0391`, `z=1.77`, answer-logprob `+0.1077`, `z=20.64`, answer-selection negative | Useful diagnostic, weaker than AM. |
| AO | `delta=-0.0322`, answer-logprob `-0.0069`; no corrupt-negative training, cache-mismatch objective | Clean rejection of that no-corrupt/cache-mismatch recipe. |
| PTC-020 | `acc_pause=0.6191`, `acc_nopause=0.5732`, `delta=+0.0459`, `z=2.12`, answer-logprob `+0.1110`, `z=23.76` | Current-stack AM retest is positive but weaker than AM; justifies ablations and stability follow-up. |

PTC-020 artifacts:

```text
/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T191133Z-configPTC-020-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl
experiments/opd_profile/autoresearch/scorecards/20260604T194427Z-PTC-020-step5-promote_retest.json
```

PTC-020 also validated the current stack:

- FlashQLA candidate rendering.
- `sync_method=p2p`.
- Parallel endpoint sync: `serial_endpoint_sync=false`.
- Step-5 sync success to both endpoints.
- No sampled-control or answer-logprob request failures.

Runtime reference:

- AM took about 56 minutes wall-clock to finish through step 5 on the old serial
  endpoint sync path with three sampled control arms.
- PTC-020 took about 27 minutes through step 5 on the current parallel P2P path.
- PTC-023/024/025 skip corrupt exact-match eval, so they should be at least in
  the PTC-020 runtime class, but the actual runtime is missing until one finishes.

## 5. Active Hypotheses

### H1: Operational AM-Style Prefill Performance

Arbitrary static filler tokens can become a useful learned pre-answer state when
the training objective applies answer-level pressure after the pause.

Evidence for:

- AH, AI, AL, AM, and PTC-020 all point in this direction.
- AM is strong at `n=1024`.
- PTC-020 reproduced the direction on the current stack.

Evidence against or caveats:

- The effect was weaker in PTC-020 than in AM.
- AM-style positives are not yet proven stable past the first large control.
- Answer-selection remains negative.

Next evidence needed:

- Component ablations PTC-023/024/025.
- Stability retest PTC-021.
- Batch-size retest PTC-022 if stability holds.

### H2: Hidden Matching May Be Unnecessary

AM used `opd_hidden_match_coef=2.0`, but prior no-hidden results do not isolate
AM-minus-hidden. PTC-023 is the clean ablation.

Prediction:

- If hidden matching is unnecessary, PTC-023 should retain a positive step-5
  pause-vs-no-pause delta with answer-logprob support.

What would falsify it:

- PTC-023 cleanly rejects while PTC-020 remains positive and reproducible.

### H3: Corrupt-Negative Training May Be Unnecessary Or Misleading

AM used corrupt buffer and corrupt-answer penalties. The user does not care
about the corrupt arm as an end goal, and the corrupt control has known
boundary/cap artifacts.

PTC-024 tests a practical no-corrupt replacement:

- keep hidden matching;
- set corrupt weights to zero;
- use `opd_positive_answer_weight=0.125` so the answer path still receives
  explicit non-corrupt pressure.

PTC-025 tests the minimal no-hidden/no-corrupt version with the same positive
answer pressure.

Caveat:

- This is not a perfect mathematical deletion of the corrupt arm. If corrupt
  answer contrast is removed without any positive answer-side replacement, the
  run may no longer test the same answer-causal idea. PTC-024/025 are therefore
  "replace corrupt negative with positive answer pressure" ablations.

### H4: Six-Step Runs Are Screens, Not Stability Proofs

AM showed operational signs of life only at its first scheduled large control,
step 5. Therefore six-step candidates are useful for fast AM-like screening.

They are not sufficient to prove:

- persistence through step 10 or later;
- resistance to optimizer drift;
- scale-readiness;
- batch-size robustness.

If PTC-023/024/025 are positive, add or launch corresponding stability
follow-ups. For a true step-10 readout, do not let the autopilot stop the run
after a positive step-5 verdict.

### H5: Prompt-Specific Mechanism Remains Open

AM may improve formatting, confidence, calibration, or a generic
answer-producing mode. It may also encode prompt-specific state, but that is not
proven.

Required evidence:

- boundary-identical corrupt or shuffle controls;
- positive correct-vs-distractor answer-selection behavior;
- generated-memory or cache-mismatch controls that preserve the visible prompt.

Do not spend the next run on this unless the operational branch stabilizes.

### H6: Generated-Memory / RiM Needs Different Credit Assignment

Generated 100-token memory runs produced early positives, but supervised weight
sweeps did not stabilize them.

Evidence:

- PTC-011/012/014 had early pause-vs-no-pause positives.
- Those signals decayed by later controls, and answer-logprob was often
  negative.
- PTC-019 was the first sampled-token PG/K1 run with old-logprob plumbing, but
  it was infra-invalid.

Next useful generated-memory run:

- a clean sampled-token PG/RiM retest or a true task-reward objective;
- not another supervised answer-weight interpolation.

## 6. Current Queue

Queue source:

```bash
experiments/opd_profile/autoresearch/ideas.yaml
```

Current active order:

| ID | Priority | Candidate | Purpose |
| --- | ---: | --- | --- |
| PTC-023 | 92 | `candidates/PTC-023.yaml` | AM minus hidden matching. |
| PTC-024 | 91 | `candidates/PTC-024.yaml` | AM minus corrupt negatives, hidden kept. |
| PTC-025 | 90 | `candidates/PTC-025.yaml` | AM minus hidden and corrupt negatives. |
| PTC-021 | 89 | `candidates/PTC-021.yaml` | Composite-AM stability retest to step 10. |
| PTC-022 | 88 | `candidates/PTC-022.yaml` | Composite-AM batch-size scale-up. |

Candidate details:

| ID | Key changes | Control schedule | Interpret as |
| --- | --- | --- | --- |
| PTC-023 | `opd_hidden_match_coef=0.0`; corrupt buffer/answer contrast kept | step-5 1k | Does AM need hidden matching? |
| PTC-024 | hidden kept; corrupt weights zero; `opd_positive_answer_weight=0.125` | step-5 1k | Does AM need corrupt-negative training? |
| PTC-025 | hidden zero; corrupt weights zero; `opd_positive_answer_weight=0.125` | step-5 1k | Is a minimal positive-answer static-filler recipe enough? |
| PTC-021 | same as PTC-020; `default_num_steps=11` | step 5 and step 10 if not stopped early | Does composite AM persist? |
| PTC-022 | same as PTC-020 but `default_prompts_per_step=256` | step-5 1k | Does larger per-step batch reduce variance? |

All active AM-style candidates use:

- `base_config: AM`
- `gdn_backend: flashqla`
- `score_mode: pause_vs_nopause`
- `score_min_control_n: 1000`
- `eval_num_problems: 1024`
- `eval_answer_logprob_batch_size: 64`
- `eval_answer_logprob_max_concurrency: 2`
- `opd_pipeline_rl: false`
- `sampler_quiesce_before_sync: true`
- `sync_method: p2p`
- `serial_endpoint_sync: false`

PTC-023/024/025 additionally set:

```yaml
client_args:
  eval_corrupt_pause_control: false
```

That makes them cheaper operational screens. Corrupt exact-match is not part of
their scoring gate.

## 7. What Information Is Missing

Missing science results:

- PTC-023/024/025 outcomes.
- A true step-10 stability row for the composite AM recipe.
- Step-10 stability rows for any positive AM ablation.
- A batch-size retest result for composite AM or a winning ablation.
- A clean sampled-token PG/RiM generated-memory result.
- A prompt-specific mechanism proof with boundary-identical corrupt/shuffle and
  positive answer-selection behavior.

Missing operational data:

- Actual runtime of PTC-023/024/025 after corrupt free-generation eval is
  disabled.
- Whether the autopilot should gain a "terminal only after final control" option
  for multi-control stability runs. Right now, positive terminal verdicts can
  stop or return after the first step-5 control unless configured carefully.
- Whether PTC-024/025 positive-answer replacement is the best no-corrupt
  counterpart to AM. It is a pragmatic ablation, not a perfect objective
  identity.

Missing documentation after future runs:

- Scorecard paths for PTC-023/024/025.
- Exact control row metrics for every new completed run.
- Any failed launch/infra event from `runs.jsonl`.
- Any changed queue ordering after new evidence.

## 8. What Information Is Unnecessary Here

Keep out of this active runbook unless it changes a current decision:

- Full 235B per-step ledgers.
- Exhaustive A-X filler-surface tables.
- Raw W&B run IDs for old rejected runs.
- Long explanations of obsolete NCCL-broadcast or serialized endpoint paths,
  except as fallback notes.
- Detailed historical analyzer failures for runs whose conclusion is already
  captured.
- Repeated command variants for old configs.
- Filler-surface archaeology unless a new objective attaches to it.

Where to look if that detail is needed:

- `experiments/opd_profile/OPD_CONFIG_A_B_RUNBOOK_2026_06_02.md`
- `experiments/opd_profile/Q36_35B_OPD_REPROGRAMMABLE_SLOTS_RUNBOOK_2026_06_02.md`
- `experiments/opd_profile/K8S_REPROGRAMMABLE_SLOTS_RUNBOOK_2026_06_03.md`
- `experiments/opd_profile/PREFILL_TIME_COMPUTE_OPSD_RESEARCH_MEMO_2026_06_03.md`
- `experiments/opd_profile/PREFILL_TIME_COMPUTE_OPSD_RUNBOOK_2026_06_03.md`

## 9. Operating Procedure

### Check State

```bash
python "$GENERATOR" status
python "$CONTROLLER" next
```

If a run is launched, inspect the latest profile:

```bash
python "$CONTROLLER" monitor --profile latest --json
```

### Render A Candidate

```bash
python "$GENERATOR" render-control \
  --candidate "$CANDIDATES/PTC-023.yaml" \
  --num-steps 6 \
  --prompts-per-step 128 \
  --sampler-replicas 2 \
  --sampler-layout spare-teacher1 \
  --role trainer-head \
  --output /tmp/ptc-023-trainer-head.yaml
```

Check:

```bash
rg "XORL_GDN_BACKEND=flashqla|sync_method=p2p|eval_num_problems=1024|eval_control_start_step=5|opd_hidden_match_coef|opd_contrastive_corrupt|opd_positive_answer_weight" /tmp/ptc-023-trainer-head.yaml
```

### Launch The Next Candidate

Dry run:

```bash
python "$CONTROLLER" launch --dry-run
```

Actual launch:

```bash
python "$CONTROLLER" launch \
  --sampler-replicas 2 \
  --sampler-layout spare-teacher1
```

Explicit launch:

```bash
python "$CONTROLLER" launch \
  --id PTC-023 \
  --sampler-replicas 2 \
  --sampler-layout spare-teacher1
```

### Autopilot For Fast Screens

Use this for PTC-023/024/025 if you want it to launch the next queued idea after
a clean science rejection and stop on a positive screen:

```bash
python "$CONTROLLER" autopilot \
  --launch-next \
  --poll-seconds 300 \
  --sampler-replicas 2 \
  --sampler-layout spare-teacher1 \
  --terminal-action stop_advance \
  --launch-grace-seconds 600
```

This is appropriate because PTC-023/024/025 are six-step screens. A positive
screen should stop for review and follow-up candidate creation.

### Autopilot Caution For Stability Runs

PTC-021 has `default_num_steps=11` and `eval_control_start_step=5`. That means
it can emit a positive verdict at step 5 before the step-10 stability row exists.

For a true step-10 stability run, do one of the following:

- monitor manually and do not run the autopilot with
  `--terminal-action stop_advance`; or
- launch a dedicated stability candidate with `eval_control_start_step: 10`; or
- update the controller to support "terminal only after final scheduled control"
  before relying on unattended multi-control stability.

Do not claim a stability result from PTC-021 unless the profile actually contains
the later control row.

### Score And Advance Manually

```bash
python "$CONTROLLER" score --profile latest --idea-id PTC-023 --json
python "$CONTROLLER" advance --id PTC-023 --profile latest
```

Scorecards are written under:

```bash
experiments/opd_profile/autoresearch/scorecards/
```

### Stop Wedged Trainer Roles

```bash
python "$GENERATOR" stop-trainer-control --remove-run
python "$GENERATOR" status
```

Normal science iteration should relaunch trainer roles only; do not restart
student inference unless endpoint health or routing is actually bad.

## 10. Infrastructure Invariants

- Every GPU pod must carry `team: turbo`.
- Use `--sampler-layout spare-teacher1` until a dedicated second student sampler
  is intentionally scheduled and healthy.
- Keep each SGLang sampler internally serialized with `--max-running-requests 1`.
  Batched SGLang decoding previously caused repeated-suffix correctness
  failures.
- Use `sync_method=p2p` with `serial_endpoint_sync=false` for current promoted
  paths.
- Keep serialized P2P and NCCL broadcast as fallbacks only.
- Keep `opd_pipeline_rl=false` and `sampler_quiesce_before_sync=true` until a
  pipelined path has its own freshness proof.
- Keep answer-logprob scoring chunked:
  `eval_answer_logprob_batch_size=64`,
  `eval_answer_logprob_max_concurrency=2`.
- Treat corrupt fixed-token exact-match controls as diagnostics, not gate
  vetoes, unless the corrupt boundary/cap health is known clean.

## 11. Candidate Authoring Rules

For AM-style operational candidates:

```yaml
base_config: AM
gdn_backend: flashqla
score_mode: pause_vs_nopause
score_min_control_n: 1000
student_prefill_text: " ! | ~ _ * ^ # @ "
student_prefill_count: 1
student_prefill_suffix: "Answer: "
student_stop_sequences: '["\n"]'
default_prompts_per_step: 128
eval_num_problems: 1024
eval_accuracy_every: 5
eval_control_start_step: 5
eval_answer_logprob_control: true
eval_answer_logprob_batch_size: 64
eval_answer_logprob_max_concurrency: 2
opd_pipeline_rl: false
sampler_quiesce_before_sync: true
sync_method: p2p
serial_endpoint_sync: false
request_timeout: 1200
weight_sync_timeout: 900
opd_loss_max_clamp: 5.0
```

For fast ablation screens, use:

```yaml
default_num_steps: 6
client_args:
  eval_corrupt_pause_control: false
```

For stability follow-ups, prefer a dedicated candidate whose control schedule
cannot be mistaken for a step-5-only screen. If using an 11-step run, verify the
actual profile contains the step-10 row before advancing the science conclusion.

For generated-memory candidates, only use lower eval sizes when generated-memory
sampling makes full 1k controls too expensive:

```yaml
eval_num_problems: 256
score_min_control_n: 250
client_args:
  student_generated_memory_tokens: 100
  eval_generated_memory_control: true
  eval_generated_memory_tokens: 100
```

Do not add another generated-memory supervised weight sweep unless there is a new
mechanistic reason.

## 12. Quick Verification Commands

Static checks after controller or generator edits:

```bash
python -m py_compile "$CONTROLLER" "$GENERATOR"
uv run ruff check "$CONTROLLER" "$GENERATOR"
```

Queue and render checks:

```bash
python "$CONTROLLER" next
python "$CONTROLLER" launch --id PTC-023 --dry-run

python "$GENERATOR" render-control \
  --candidate "$CANDIDATES/PTC-023.yaml" \
  --num-steps 6 \
  --prompts-per-step 128 \
  --sampler-replicas 2 \
  --sampler-layout spare-teacher1 \
  --role trainer-head \
  --output /tmp/ptc-023-check.yaml

rg "config=PTC-023|base_config=AM|XORL_GDN_BACKEND=flashqla|sync_method=p2p|serial_endpoint_sync=false" /tmp/ptc-023-check.yaml
```

Profile summary snippet:

```bash
python - <<'PY'
import glob, json, os
root = "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots"
paths = glob.glob(root + "/*/opd_profile.jsonl")
path = max(paths, key=os.path.getmtime)
rows = [json.loads(line) for line in open(path) if line.strip()]
controls = [r for r in rows if "eval/acc_pause" in r]
print(path)
print("rows", len(rows), "last_step", rows[-1].get("step"))
if controls:
    r = controls[-1]
    for key in [
        "step",
        "eval/control_n",
        "eval/acc_pause",
        "eval/acc_nopause",
        "eval/buffer_delta",
        "eval/buffer_delta_z",
        "eval/answer_logprob_margin",
        "eval/answer_logprob_margin_z",
    ]:
        print(key, r.get(key))
PY
```

## 13. Current Next Action

Launch PTC-023 when ready:

```bash
python "$CONTROLLER" launch \
  --id PTC-023 \
  --sampler-replicas 2 \
  --sampler-layout spare-teacher1
```

Then either:

- run the fast-screen autopilot command above; or
- wait for the step-5 control row, score it, and advance manually.

Interpretation after PTC-023:

- Positive: hidden matching is likely not necessary for the fast AM-like
  operational effect. Add a no-hidden stability follow-up.
- Negative: hidden matching may be important, or PTC-020 may have been a weak
  transient. Continue to PTC-024/025 only if the question is component
  attribution, not immediate scale-up.
```


### `experiments/opd_profile/OPD_CONFIG_A_B_RUNBOOK_2026_06_02.md`

```markdown
# OPD Config A/B Runbook and Handoff

Date: 2026-06-02

This document is the future-agent runbook for the Qwen3-235B OPD encoded-reasoning experiment after both Config A and Config B were run. It supersedes the live-run portions of `ENCODED_REASONING_MASTER_RUNBOOK_2026_06_02.md`, but that master runbook remains the source of truth for the original setup, OPD mechanics, prompt construction, and earlier negative controls.

## 0. TL;DR

Both Config A and Config B optimize the OPD training objective, but neither produces a stable encoded-reasoning improvement.

The important new result is that `eval/acc_pause` goes down over time in both configurations.

- Config A is the clean buffer-only test: answer masked, hidden/KL supervision on the NATO buffer. Hidden loss, KL, and total loss fall quickly, but `acc_pause` drops from 0.396 at step 0 to 0.333 at step 30. `acc_nopause` remains pinned at 0.010, so the buffer delta stays positive mostly because the no-pause baseline is dead.
- Config B is the answer-unmasked fallback: the answer path is supervised too. It proves direct answer learning is possible because `acc_nopause` jumps from 0.104 to 0.365 by step 10. But over the full 400-step run, the pause advantage collapses and the final controlled eval is `acc_pause=0.302`, `acc_nopause=0.313`, `buffer_delta=-0.010`.
- Combined verdict: do not rerun Config A or Config B as-is. The objective is learnable, but the learned behavior is not the desired pause-buffer reasoning channel. The strongest observation is not just "no improvement"; it is that optimizing this OPD target appears to degrade pause accuracy.

## 1. Source Documents

- `experiments/opd_profile/ENCODED_REASONING_MASTER_RUNBOOK_2026_06_02.md`
  - Section 3: OPD mechanics, prompt/teacher/student construction, Config A/B definitions.
  - Section 5.4: bring-up hazards.
  - Section 6: restart procedure.
- `experiments/opd_profile/HANDOFF_OPD_ENCODED_REASONING.md`
  - Older handoff. Useful for background, but parts are stale.
  - In particular, the warning that multi-token filler silently breaks K is outdated for the current code path; current OPD uses `k_filler=len(prefill_tokens)` and aligns the NATO buffer correctly.

## 2. Final Artifact Locations

Experiment family:

```bash
BASE=/shared/opd-coord/encoded_reasoning/results/qwen3_235b_self_distill/er-opd-235b-clean4d
```

Config A:

```bash
RUN_A=$BASE/20260602T055219Z-er-opd-235b-clean4d-trainer-head-zcmzv
PROFILE_A=$RUN_A/opd_profile.jsonl
WANDB_A=jzc4g717
```

Config B:

```bash
RUN_B=$BASE/20260602T065545Z-configb-er-opd-235b-clean4d-trainer-head-96lwh
PROFILE_B=$RUN_B/opd_profile.jsonl
WANDB_B=b8zejpa3
```

Manifest used for the 235B run:

```bash
experiments/opd_profile/k8s/generated/er-opd-235b-clean4d.yaml
```

The manifest was switched from Config A to Config B by changing:

```yaml
opd_supervise_buffer_only: false
save_name_prefix: opd-235b-clean4d-nato-configb
wandb_run_name: opd-235b-clean4d-nato-0shot-hm-configb
```

It retained:

```bash
XORL_TORCHRUN_RDZV_CONF=timeout=1800
save_every=0
opd_kl_backend=streaming
```

## 3. Config Definitions

Both configs used the viable 235B setup from the master runbook:

- Model: Qwen3-235B-A22B base.
- Task: clean 4-digit multiplication.
- Prompting: 0-shot.
- Buffer: NATO filler repeated 3 times, about 102 tokens.
- Packing/sequence length: full CoT path with packing length 1024.
- Hidden match coefficient: `opd_hidden_match_coef=1.0`.
- Teachers and dispatch: default pool.
- Trainer and Mooncake samplers: `node-group=nccl`.

Config A:

```yaml
opd_supervise_buffer_only: true
```

Meaning: mask answer tokens, supervise only the buffer hidden/KL path. This is the cleanest test of whether teacher CoT state can be distilled into the buffer.

Config B:

```yaml
opd_supervise_buffer_only: false
```

Meaning: unmask answer tokens. This is the fallback that allows direct answer supervision in addition to buffer supervision.

## 4. Results

### 4.1 Config A: Buffer-Only

`opd_profile.jsonl` has 31 rows, steps 0 through 30.

Controlled eval rows:

```text
step  acc_pause  acc_nopause  buffer_delta  hidden_loss  kl_loss   loss
0     0.395833   0.010417     0.385417      0.147964     0.319312  0.467275
5     0.375000   0.010417     0.364583      0.132743     0.280177  0.412920
10    0.354167   0.010417     0.343750      0.105457     0.130634  0.236091
15    0.343750   0.010417     0.333333      0.086055     0.116803  0.202858
20    0.322917   0.010417     0.312500      0.074220     0.106577  0.180797
25    0.354167   0.010417     0.343750      0.066588     0.105162  0.171750
30    0.333333   0.010417     0.322917      0.060300     0.099516  0.159816
```

Interpretation:

- The objective is being optimized: hidden loss, KL, and total loss fall strongly.
- Pause accuracy deteriorates from 0.396 to 0.333 over 30 steps.
- No-pause accuracy remains pinned at about 1/96.
- The positive `buffer_delta` is not evidence of improved encoded reasoning; it mostly reflects that no-pause remains dead while pause gets worse.

Config A is negative.

### 4.2 Config B: Answer-Unmasked Fallback

`opd_profile.jsonl` has 400 rows, steps 0 through 399. The trainer head completed cleanly.

Selected controlled eval rows:

```text
step  acc_pause  acc_nopause  buffer_delta  hidden_loss  kl_loss   loss
0     0.458333   0.104167     0.354167      0.139197     0.285339  0.424616
5     0.479167   0.291667     0.187500      0.120272     0.182158  0.303331
10    0.468750   0.364583     0.104167      0.101936     0.128065  0.229966
15    0.447917   0.385417     0.062500      0.084370     0.111642  0.196073
20    0.447917   0.385417     0.062500      0.074292     0.097694  0.172989
25    0.333333   0.270833     0.062500      0.066902     0.097709  0.164848
50    0.354167   0.281250     0.072917      0.058570     0.097771  0.156692
100   0.395833   0.385417     0.010417      0.050861     0.105290  0.156489
150   0.364583   0.322917     0.041667      0.047898     0.094566  0.142950
200   0.260417   0.187500     0.072917      0.044760     0.090619  0.135301
250   0.333333   0.343750    -0.010417      0.044112     0.098121  0.141942
300   0.281250   0.270833     0.010417      0.045112     0.099894  0.144077
350   0.291667   0.291667     0.000000      0.042054     0.081940  0.124075
375   0.343750   0.270833     0.072917      0.044752     0.079852  0.124498
380   0.375000   0.322917     0.052083      0.044326     0.079438  0.124132
385   0.385417   0.354167     0.031250      0.042140     0.085337  0.127461
390   0.354167   0.322917     0.031250      0.043429     0.076300  0.119737
395   0.302083   0.312500    -0.010417      0.042797     0.106779  0.149587
```

Final profile tail:

```text
388 hidden=0.043054 KL=0.082356 loss=0.125566
389 hidden=0.042049 KL=0.073804 loss=0.115958
390 acc_pause=0.354167 acc_nopause=0.322917 delta=0.031250 hidden=0.043429 KL=0.076300 loss=0.119737
391 hidden=0.041627 KL=0.071867 loss=0.113440
392 hidden=0.042627 KL=0.077904 loss=0.120480
393 hidden=0.041830 KL=0.080724 loss=0.123602
394 hidden=0.043798 KL=0.080174 loss=0.124017
395 acc_pause=0.302083 acc_nopause=0.312500 delta=-0.010417 hidden=0.042797 KL=0.106779 loss=0.149587
396 hidden=0.043201 KL=0.100874 loss=0.144006
397 hidden=0.044495 KL=0.090304 loss=0.134811
398 hidden=0.043318 KL=0.077833 loss=0.120783
399 hidden=0.040818 KL=0.069142 loss=0.110081
```

W&B final summary:

```text
eval/acc_pause=0.30208
eval/acc_nopause=0.31250
eval/buffer_delta=-0.01042
eval/accuracy=0.65625
```

Interpretation:

- Direct answer supervision works initially: no-pause accuracy rises sharply by step 10.
- The pause advantage collapses quickly.
- Over the full run, both pause and no-pause accuracy drift down or oscillate below early values.
- The final pause accuracy is worse than the starting pause accuracy, despite lower hidden/KL/loss.

Config B is negative.

## 5. Scientific Interpretation

The key finding is the mismatch between objective optimization and answer behavior.

The training losses say the student is becoming closer to the teacher targets under the OPD objective. The evals say that this does not translate into robust arithmetic accuracy, and the pause condition becomes worse over time.

Working hypotheses:

1. Teacher-target mismatch / representation poisoning.
   - The teacher hidden targets are conditioned on a full CoT before the NATO buffer.
   - The student sees only prompt plus NATO buffer.
   - Matching the teacher's post-CoT state may push the student into an unnatural representation that is close by hidden-space MSE/KL but not causally useful for answer generation.

2. The buffer target is not a causal scratchpad.
   - Native filler lift may be a runtime/prompting property of the pretrained model.
   - Weight updates that force hidden-state imitation may destroy the pretrained behavior that made the pause useful.

3. Answer-unmasked training learns direct-answer behavior, not encoded reasoning.
   - Config B's early no-pause jump shows answer supervision can train the answer path.
   - But this erases the pause/no-pause gap and does not preserve the original pause advantage.

4. Formatting and continuation artifacts may be contaminating the training signal.
   - Late samples observed during Config B included repeated `Answer:` segments, answer text glued near NATO filler, and other continuation artifacts.
   - The model may be learning to continue the packed training format rather than learning arithmetic through a buffer.

5. Eval noise exists but does not explain the full pattern.
   - Controlled eval uses `n=96`, so individual points have several percentage points of noise.
   - The Config B trend over 400 steps and the Config A monotonic early loss/accuracy mismatch are large enough to treat as real.

## 6. Do Not Repeat

Do not relaunch Config A as-is.

Do not relaunch Config B as-is.

Do not spend another long 400-step 235B run on the same settings without a diagnostic change that specifically targets the observed failure mode.

Do not interpret positive `buffer_delta` in Config A as success unless `acc_pause` itself improves against its own step-0 baseline. In Config A, `buffer_delta` is positive because `acc_nopause` is dead.

Do not trust objective loss alone. For this experiment, `opd_hidden_match_loss`, `opd_kl`, and `loss` can all improve while the behavior of interest deteriorates.

## 7. Recommended Next Work

Start with artifact analysis before launching another large run.

### 7.1 Analyze Existing Profiles

Plot these metrics for both Config A and Config B:

```bash
jq -r 'select(.["eval/acc_pause"] != null) |
  [.step,
   .["eval/acc_pause"],
   .["eval/acc_nopause"],
   .["eval/buffer_delta"],
   .opd_hidden_match_loss,
   .opd_kl,
   .loss,
   .["eval/lead_pause"],
   .["eval/lead_nopause"],
   .["eval/has_think_close_frac"]] | @tsv' "$PROFILE_B"
```

Focus on:

- `acc_pause` vs step.
- `acc_nopause` vs step.
- `buffer_delta` vs step.
- Hidden/KL/loss vs step.
- `has_think_close_frac`, `lead_pause`, and `lead_nopause` if populated.

### 7.2 Inspect Samples

Inspect early, middle, and late generated samples from W&B or local artifacts if available. Classify failures by:

- Wrong arithmetic with otherwise clean format.
- Repeated `Answer:`.
- NATO leakage into answer region.
- Missing or malformed final answer.
- CoT continuation after answer.
- Prompt or packing continuation artifacts.

This should be done before designing the next config because the failure class determines whether to attack formatting, objective design, LR, or eval.

### 7.3 Next Config Candidates

Use short diagnostic runs first: 30 to 60 steps, controlled eval every 5 steps. Only extend to hundreds of steps if `acc_pause` improves against step 0 and the pause/no-pause gap remains meaningful.

Candidate C: KL-only, answer masked.

```yaml
opd_supervise_buffer_only: true
opd_hidden_match_coef: 0.0
```

Purpose: isolate whether hidden-state MSE is causing representation poisoning. If pause degradation disappears, hidden-match is the likely problem. If degradation remains, the issue is broader than hidden MSE.

Candidate D: hidden-only or reduced KL diagnostic.

Purpose: isolate whether KL on the buffer logits is responsible for drift. This is lower priority than KL-only because Config A already included both hidden and KL.

Candidate E: answer-only supervised control.

Purpose: quantify how much of Config B is ordinary direct answer learning and whether pause degradation occurs without the OPD buffer losses.

Candidate F: teacher without CoT.

Teacher path should be prompt plus NATO plus answer, not prompt plus CoT plus NATO plus answer.

Purpose: test whether the post-CoT teacher state is fundamentally unreachable from the student context.

Candidate G: lower LR Config B.

Purpose: test whether the long-run decline is generic optimizer damage. Use only after sample inspection, because Config B already failed the pause-buffer objective.

Candidate H: larger control eval.

Increase controlled eval N from 96 to about 400 for short runs if throughput permits. This reduces uncertainty when deciding whether a 5 to 10 point change is real.

## 8. Operational State

Config B completed, but at last inspection several pods were still running. Preserve artifacts before cleanup if needed.

Last known pod state:

```text
er-opd-235b-clean4d-dispatch              1/1 Running
er-opd-235b-clean4d-sglang-0              1/1 Running
er-opd-235b-clean4d-sglang-1              1/1 Running
er-opd-235b-clean4d-teacher-sglang-0      1/1 Running
er-opd-235b-clean4d-teacher-sglang-1      1/1 Running
er-opd-235b-clean4d-teacher-smg           1/1 Running
er-opd-235b-clean4d-trainer-head-96lwh    0/1 Completed
er-opd-235b-clean4d-trainer-worker-1..7   1/1 Running
```

If the next agent is not immediately relaunching on the same teachers, free everything:

```bash
kubectl delete job -n apanda er-opd-235b-clean4d-trainer-head --wait=true --timeout=300s || true
kubectl delete pod -n apanda \
  er-opd-235b-clean4d-trainer-worker-1 \
  er-opd-235b-clean4d-trainer-worker-2 \
  er-opd-235b-clean4d-trainer-worker-3 \
  er-opd-235b-clean4d-trainer-worker-4 \
  er-opd-235b-clean4d-trainer-worker-5 \
  er-opd-235b-clean4d-trainer-worker-6 \
  er-opd-235b-clean4d-trainer-worker-7 \
  er-opd-235b-clean4d-sglang-0 \
  er-opd-235b-clean4d-sglang-1 \
  er-opd-235b-clean4d-dispatch \
  er-opd-235b-clean4d-teacher-sglang-0 \
  er-opd-235b-clean4d-teacher-sglang-1 \
  er-opd-235b-clean4d-teacher-smg \
  --wait=true --timeout=300s || true
```

If immediately relaunching with the same teacher setup, keep teachers warm and delete only trainer, workers, samplers, and dispatch:

```bash
kubectl delete job -n apanda er-opd-235b-clean4d-trainer-head --wait=true --timeout=300s || true
kubectl delete pod -n apanda \
  er-opd-235b-clean4d-trainer-worker-1 \
  er-opd-235b-clean4d-trainer-worker-2 \
  er-opd-235b-clean4d-trainer-worker-3 \
  er-opd-235b-clean4d-trainer-worker-4 \
  er-opd-235b-clean4d-trainer-worker-5 \
  er-opd-235b-clean4d-trainer-worker-6 \
  er-opd-235b-clean4d-trainer-worker-7 \
  er-opd-235b-clean4d-sglang-0 \
  er-opd-235b-clean4d-sglang-1 \
  er-opd-235b-clean4d-dispatch \
  --wait=true --timeout=300s || true
```

Poll until old objects are gone:

```bash
kubectl get pods -n apanda | grep -E 'er-opd-235b-clean4d-(trainer|sglang|dispatch)' || true
```

## 9. Relaunch Procedure

1. Edit `experiments/opd_profile/k8s/generated/er-opd-235b-clean4d.yaml`.
2. Give the new run a unique config name in `RUN_ID`, `save_name_prefix`, and `wandb_run_name`.
3. Keep trainer and Mooncake samplers on the `nccl` pool.
4. Keep teachers and dispatch off known-bad default nodes.
5. Retain the longer rendezvous timeout unless using Volcano gang scheduling:

```bash
XORL_TORCHRUN_RDZV_CONF=timeout=1800
```

6. Dry-run server-side:

```bash
kubectl apply -n apanda --dry-run=server -f experiments/opd_profile/k8s/generated/er-opd-235b-clean4d.yaml
```

7. Apply:

```bash
kubectl apply -n apanda -f experiments/opd_profile/k8s/generated/er-opd-235b-clean4d.yaml
```

8. Watch the pod bring-up:

```bash
kubectl get pods -n apanda | awk '/er-opd-235b-clean4d/ {print $1,$2,$3,$4}'
```

9. Confirm the run reaches step 0. The historical failure mode was that an 8-node trainer gang started partially, some workers scheduled late, the torch rendezvous hit the old 901s timeout, and the job cascaded. Do not judge a new science config until it reaches step 0.

Cluster note: all GPU pods must have `team: turbo` on the pod template labels. Do not manually override Volcano-injected scheduler fields unless there is a specific reason.

## 10. Monitoring Commands

Set the active run dir:

```bash
RUN_DIR=/shared/opd-coord/encoded_reasoning/results/qwen3_235b_self_distill/er-opd-235b-clean4d/<run-id>
PROFILE=$RUN_DIR/opd_profile.jsonl
```

Recent profile rows:

```bash
jq -r '[.step,
        .["eval/acc_pause"],
        .["eval/acc_nopause"],
        .["eval/buffer_delta"],
        .opd_hidden_match_loss,
        .opd_kl,
        .loss,
        .step_total_s] | @tsv' "$PROFILE" | tail -20
```

Controlled eval only:

```bash
jq -r 'select(.["eval/acc_pause"] != null) |
  [.step,
   .["eval/acc_pause"],
   .["eval/acc_nopause"],
   .["eval/buffer_delta"],
   .["eval/accuracy"],
   .["eval/lead_pause"],
   .["eval/lead_nopause"],
   .["eval/has_think_close_frac"]] | @tsv' "$PROFILE"
```

Trainer log summary:

```bash
tail -120 "$RUN_DIR/trainer_job.log" | grep -E 'control step|=== OPD step|Traceback|ERROR|OPD run completed'
```

Health check from the trainer head, while running:

```bash
kubectl exec -n apanda <trainer-head-pod> -- curl -fsS http://127.0.0.1:26050/health
```

Pod state:

```bash
kubectl get pods -n apanda | awk '/er-opd-235b-clean4d/ {print $1,$2,$3,$4}'
```

## 11. Bring-Up Hazards

The main operational hazard is not the science; it is synchronized 8-node 235B bring-up.

Observed historical failure:

- 8-node trainer gang on a contended 10-node `nccl` pool.
- Some workers scheduled late.
- Torch rendezvous hit the 901s timeout.
- The run cascaded through about six failures before a clean synchronized bring-up.

Mitigations:

- Free the whole `nccl` pool before relaunch when possible.
- Delete the old trainer job with `--wait=true`.
- Poll until old worker/sampler/dispatch pods are actually gone before applying the manifest.
- Keep `XORL_TORCHRUN_RDZV_CONF=timeout=1800`.
- Consider Volcano gang scheduling if repeated partial-start failures return.
- Keep trainer and Mooncake samplers off the uncurated default pool.
- Avoid known-bad default nodes for teachers/dispatch if the master runbook still lists them as bad.

## 12. Decision Criteria For Future Runs

A future config is promising only if all of these are true:

- It reaches step 0 cleanly.
- `acc_pause` improves against its own step-0 baseline, not just against `acc_nopause`.
- `buffer_delta` remains positive because pause improves, not because no-pause collapses.
- Formatting/sample inspection does not show repeated `Answer:`, NATO leakage, or packed-continuation artifacts dominating generations.
- Hidden/KL/loss improvement correlates with behavior, rather than moving in the opposite direction.

Stop early if:

- `acc_pause` drops by more than about 5 points by step 20 to 30.
- `buffer_delta` collapses to zero while loss keeps improving.
- Samples show format corruption becoming common.
- The run has not reached step 0 because of rendezvous/scheduling failure; fix infra first and do not count it as a science result.
```


### `experiments/opd_profile/Q36_35B_OPD_REPROGRAMMABLE_SLOTS_RUNBOOK_2026_06_02.md`

```markdown
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
```


### `experiments/opd_profile/K8S_REPROGRAMMABLE_SLOTS_RUNBOOK_2026_06_03.md`

```markdown
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
```


### `experiments/opd_profile/PREFILL_TIME_COMPUTE_OPSD_RESEARCH_MEMO_2026_06_03.md`

```markdown
# Prefill-Time-Compute OPSD Research Memo - 2026-06-03

## Position

The strongest near-term direction is not filler-token capability transfer or filler-type search. It is direct prefill-time-compute distillation:

- Teacher input: question, teacher chain of thought, student-visible filler, answer.
- Student input: question, student-visible filler, answer.
- Loss surface: KL and optional hidden matching only on the filler and answer continuation, not the question or teacher CoT prefix.
- Vocabulary: existing model tokens only. No new latent-token vocabulary is needed for the first signal.

This asks whether the student can learn to make an otherwise semantically meaningless filler span carry teacher prefill computation forward into the answer distribution. Compression and short filler length are secondary. A 1000-token filler that barely improves decoded-token efficiency would still be a valid first latent/encoded-reasoning signal if the effect is causal and robust.

## What The Existing Runs Say

The Config A-X family mostly falsified easy stories: raw hidden matching, answer masking tweaks, buffer-only supervision, buffer text choice, smaller learning rates, and superficial filler variants did not produce a robust causal pause benefit. These runs are still useful because they make "search the filler string" a low-priority axis.

The later Q36 35B runs sharpened the measurement rather than solving the problem. Z/AA/AC/AG were largely negative or ambiguous. AH produced a large-looking improvement, but subsequent controls exposed a corrupt-control artifact tied to continuation-boundary changes rather than a clean encoded-reasoning effect. AI/AL/AM kept weak positives alive but did not establish prompt-specific memory. AO then removed the signed corrupt-answer anti-target and focused on cache mismatch; it did not recover a robust signal.

AN is the best current evidence for a weak prefill-time-compute effect: final control accuracy was approximately `acc_pause=0.5596`, `acc_nopause=0.5205`, `acc_corrupt=0.4854`, `delta=+0.0391`, `z=1.77`, with a strong answer-logprob pause margin and positive pause-vs-corrupt margin. That is not enough for a claim, but it is enough to justify an autoresearch loop that can retest, perturb, and reject larger changes automatically.

## Prior Work Closest To The Proposed OPSD Variant

The exact proposed variant has not been cleanly run:

- The original 235B Config B is closest in spirit, because the teacher has CoT and the student carries a filler span, but the legacy loss was not isolated to the student filler plus answer positions with the current diagnostics and controls.
- Q36 AH-AN added answer-causal and corrupt-control machinery. These runs taught us about artifacts and answer-logprob diagnostics, but they were still dominated by contrastive corrupt arms and historical hidden-match settings.
- AO isolated a different cache-mismatch hypothesis and removed the signed corrupt-answer anti-target. It is not the pure positive OPSD objective.

So the right next experiment is not a small sweep of answer KL. It is a candidate-level objective change: construct positive OPSD examples where the teacher carries CoT before the visible filler and the student is trained only on the filler plus answer positions, with prompt positions masked out of the KL denominator.

## Autoresearch Implication

The loop should dequeue hypotheses, generate runnable candidates, launch on the reusable slots, monitor for infra validity, score completed profile rows, and write a scorecard before a human inspects traces. The first programme should bias toward decisive objective changes:

- Pure positive OPSD with zero prompt KL.
- Sampled-answer versus gold-answer teacher continuation.
- Optional hidden matching on the filler span after the pure KL baseline.
- Scale-up of batch/eval size only after the objective itself is valid.

The monitor should automatically reject stale W&B/profile mismatches, failed syncs, request failures, cap-hit-heavy evals, answer-logprob failures, and sampler imbalance. Manual inspection should start from scorecards, not raw logs.
```


### `experiments/opd_profile/PREFILL_TIME_COMPUTE_OPSD_RUNBOOK_2026_06_03.md`

```markdown
# Prefill-Time-Compute OPSD Runbook - 2026-06-03

## Render A Candidate

```bash
python experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py render-control \
  --candidate experiments/opd_profile/autoresearch/candidates/PTC-001.yaml \
  --num-steps 9 \
  --prompts-per-step 128 \
  --sampler-replicas 2 \
  --sampler-layout spare-teacher1 \
  --role trainer-head \
  --output /tmp/ptc-001-trainer-head.yaml
```

Check the rendered trainer command for:

- `opd_ptc_positive_buffer_kl_weight=1.0`
- `opd_mask_zero_weight_positions=true`
- `opd_teacher_answer_source=sampled` or `gold`
- `eval_num_problems=1000`
- candidate-specific `wandb_group=q36-ptc-autoresearch`

## Launch

Dry run:

```bash
python experiments/opd_profile/autoresearch/controller.py launch --id PTC-001 --dry-run
```

Actual trainer launch:

```bash
python experiments/opd_profile/autoresearch/controller.py launch \
  --id PTC-001 \
  --sampler-replicas 2 \
  --sampler-layout spare-teacher1
```

The controller calls `stop-trainer-control --remove-run` and then writes trainer control scripts for the candidate. It does not recreate the Kubernetes Pods.

## Monitor

```bash
python experiments/opd_profile/autoresearch/controller.py monitor --profile latest --json
```

This wraps the existing live monitor and appends the snapshot to the autoresearch event log.

## Score And Advance

```bash
python experiments/opd_profile/autoresearch/controller.py score --profile latest --idea-id PTC-001
python experiments/opd_profile/autoresearch/controller.py advance --id PTC-001 --profile latest
```

Scorecards are written under `experiments/opd_profile/autoresearch/scorecards/`. The queue status is advanced only by the `advance` command.

## Manual Analysis

For deeper completed-run analysis:

```bash
python experiments/opd_profile/analyze_slot_profile.py latest \
  --min-control-n 900 \
  --min-answer-logprob-margin 0.0 \
  --min-answer-logprob-z 10.0
```

For W&B/profile consistency:

```bash
python experiments/opd_profile/audit_wandb_profile.py latest --wandb-run <run-id>
```
```


### `experiments/opd_profile/ENCODED_REASONING_MASTER_RUNBOOK_2026_06_02.md`

```markdown
# Encoded-Reasoning / Filler-Token OPD — Master Runbook & Source of Truth (2026-06-02)

This is the single consolidated record of the encoded-reasoning investigation: the
scientific question, every eval/experiment run (this session + the parallel "tomi"
session), the methodology and its corrections, the OPD distillation experiment now in
flight, all infrastructure + bring-up hazards, and how to reproduce/continue. If you are
picking this up cold, read §0 then §6 (the live experiment) and §9 (how to run things).

---

## 0. TL;DR / Bottom line

**The question:** can a model's *filler/pause "thinking buffer"* be made to carry
distilled teacher reasoning — i.e., is the buffer a trainable *encoded-reasoning* channel
(OPD: distill a CoT-equipped teacher into a buffer-equipped student)?

**What we now know (high confidence):**
1. **The paper's filler-token effect is REAL and reproduces** — but only on **Qwen3-235B**
   (both `base` and `Instruct-2507`), and only when the direct-answer baseline sits in a
   **"responsive zone" (~0.3–0.7)**. The full **type-inversion** pattern (word-based fillers
   win at 0-shot, numerical/ellipsis win at 10-shot, `pi_digits`/`random_numbers`
   catastrophic at 0-shot) reproduces cell-for-cell.
2. **It does NOT extend down-scale.** Q3.5/Q3.6-35B-A3B, Q3-30B-Instruct-2507: under a
   *clean* prompt the filler is flat-to-negative on every task. Their paper-style "lift" is
   **format-recovery** (the filler partially undoes the `/no_think` soft-suppression; a
   properly-prompted baseline beats it).
3. **Q3.5-397B is saturated** (clean 4×4 baseline 0.98) — no headroom; its original
   "+20.1pp ellipsis" was format-recovery + an **anti-EOS** artifact (and an FP8-multimodal
   serving-garbage artifact in one session).
4. **The single viable cell = Q3-235B-A22B `base`, 4-digit mult.** Native filler shows a
   **compute-concentration signature** (pause **+0.10–0.13 pass@1 / −0.04 pass@8** under the
   clean `chat_hardoff` prompt) — pass@1↑ without pass@8↑ = *compute*, not *search*. This is
   the paper's own headline model+task.

**The load-bearing OPEN question (the experiment now running):** can that compute be
*trained into a buffer* via distillation? The only prior attempt (`nohm`) was **confounded**
(truncated CoT, suppressed baseline, no hidden-match) and went flat. The clean retest
(`er-opd-235b-clean4d`, config A) is in flight on Q3-235B base. **Prior is uncertain:** the
paper's *RL* on filler tokens yields pass@8 (search), not pass@1 (compute) — but distillation
targets the teacher's compute-concentrated buffer *directly*, a different objective, so it is
genuinely untested.

**Practical guidance:** only the **Qwen3-235B** setting is worth further compute. Everything
≤35B is a dead end for this premise under a clean prompt.

---

## 1. The eval findings (6 models, fully logged)

Two harnesses produced these; both live under `experiments/opd_profile/` and
`/old-data/apanda/tomi/outputs/`. Numbers are cross-validated between this session's runs and
the parallel session's independent samplings.

### 1.1 The three prompt regimes (this distinction is the crux of the whole investigation)
- **`raw_nothink`** (paper-faithful): V3 system prompt `/no_think\n<task>\n\nFormat:\nAnswer:
  [number]`, K-shot examples with filler in the assistant turn, **filler + `Answer:`
  prefilled**, model emits only the number. This is the paper's intended methodology — it
  *deliberately* soft-suppresses CoT so the filler effect is isolated.
- **`chat_hardoff`** (clean reference): same V3 block prompt but **`chat_template_kwargs=
  {"enable_thinking": false}`** (hard chat-template toggle, not the `/no_think` text). Higher,
  leak-free baseline = what the model can *actually* do with CoT genuinely off.
- **Diagnostic rule:** a lift that exists under `raw_nothink` but is **beaten by the
  `chat_hardoff` baseline** = **format-recovery** (not real capability). A lift that beats the
  `chat_hardoff` baseline = **genuine compute**.

### 1.2 pass@1 vs pass@8 (compute vs search)
- pass@1↑ **without** pass@8↑ = **compute concentration** (the buffer helps the model land the
  answer it could already reach — *distillable in principle*).
- pass@8↑ **without** pass@1↑ = **search expansion** (diversity; the paper's RL section's
  finding — *not* compute, *not* distillable into a deterministic forward pass).

### 1.3 Per-model verdicts (chat_hardoff 4×4 pass@1 = the clean-prompt test)

| Model | base | pause | Δp@1 | Δp@8 | baseline zone | verdict |
|---|---:|---:|---:|---:|---|---|
| Q3.5-35B-A3B | 0.85 | 0.78 | −0.07 | −0.01 | saturated | pause hurts |
| Q3.6-35B-A3B | 0.84 | 0.80 | −0.04 | −0.03 | saturated | flat |
| **Q3-235B-A22B base** | **0.46** | **0.59** | **+0.13** | **−0.04** | **responsive** | **distillable compute** ✦ |
| Q3-235B-A22B-Instruct-2507 | 0.73 | 0.71 | −0.02 | −0.03 | saturated-by-instruct | flat (pause); see §1.5 |
| Q3.5-397B-A17B BF16 | 0.98 | 0.99 | +0.01 | 0.00 | saturated | flat |

### 1.4 Format-recovery diagnostic (raw_nothink lift vs chat_hardoff baseline)

| Model | raw base | +pause | paper-style lift | chat_hardoff base | format-recovery only? |
|---|---:|---:|---:|---:|---|
| Q3.5-35B-A3B | 0.58 | 0.76 | +0.18 (=paper +14.9pp) | 0.85 | **yes** (clean > lifted) |
| Q3.6-35B-A3B | 0.60 | 0.69 | +0.09 (=paper +4.2pp) | 0.84 | **yes** |
| Q3-235B-A22B base | 0.59 | 0.67 | +0.08 | 0.46 | **NO — pause beats clean** ✦ |
| Q3-235B-Instruct-2507 | 0.75 | 0.71 | −0.04 | 0.73 | yes (no lift either; see §1.5) |
| Q3.5-397B BF16 | 0.30 | 0.30 | 0.00 (vs paper +20.1) | 0.98 | yes (unreproducible) |

### 1.5 Type-inversion — the paper's headline, reproduced on Q3-235B (17-filler V3 sweep, N=1000)
Single-filler probes (pause) are **insufficient** — pause is among the weakest fillers
(paper: |σ|≤2.0). The effect requires the *right* filler per regime. Full sweeps:

**Q3-235B-A22B-Instruct-2507** (paper's exact target variant; fs0 baseline 0.412, fs10 0.650):
- 0-shot winners (word-based): `random_tokens` **+9.7pp**, `counting` +8.0, `fruits` +7.2,
  `nato` +4.8; `pause` +2.0.
- 0-shot losers (numerical, catastrophic): `pi_digits` **−8.6**, `random_numbers` **−8.3**.
- 10-shot inversion: `random_numbers` **+2.1** (best), `fibonacci` +1.8, `ellipsis` +1.5;
  word-based go negative (`fruits` −3.8, `nato` −3.1).
- Every directional prediction of the paper reproduces.

**Q3-235B-A22B base** (fs0 baseline 0.123 — lower → larger magnitudes):
- 0-shot: `nato` **+13.8pp** (best), `states` +11.7, `lorem` +11.2, `counting` +5.1.
- 10-shot: `nato` **+9.5**, `random` +8.8, `fibonacci` +7.9.

### 1.6 The 397B anti-EOS mechanism (ellipsis, raw_nothink)
397B's raw-regime "lift" is **premature-EOS recovery**, not reasoning: ~33% of raw-regime
completions are *empty* (the model EOSes immediately after `Answer:`); ellipsis halves the
empty rate (4×4: 37.8% → 19.2%) and the pass@1 gain (0.25→0.46, +0.21) **equals** the
recovered empties. In `chat_hardoff` the 397B does 6-digit mult at 0.89 with 0% empty and
filler flat — the capability is fully there; the raw deficit is the EOS pathology.

### 1.7 Mechanistic conclusion for the eval side
Every "filler benefit" is a **generation/format artifact** — format-recovery (35B),
instruction-saturation (Instruct-2507 on pause), capability-saturation + anti-EOS (397B),
free-gen reasoning-leak (235B base under the old free-gen harness). The **one exception** is
the Q3-235B base compute-concentration cell (§1.3 ✦), which is the only genuine, non-saturated,
search-free signal — and exactly the paper's headline model+task.

---

## 2. The viable cell in detail — Q3-235B-A22B base, 4-digit mult

- **Native filler (untrained, inference-time):** clean `chat_hardoff` 4×4 pause **+0.13 pass@1
  / −0.04 pass@8** (other agent), **+0.10** at N=400 (this session's contested-cell re-run),
  +0.06 (this session's first N=100). Truth ≈ **+0.10–0.13**, robustly nonzero (~4 SE at N=400).
- **It beats the clean baseline** (raw+pause 0.66 > raw+base 0.59; and chat_hardoff pause 0.59
  > chat_hardoff base 0.46) → **not** format-recovery (unlike 35B).
- **pass@8 unmoved/down** → compute concentration, **not** search → *the signature compatible
  with distillation*.
- **Why this model and not 35B:** base-235B's V3 baseline (0.123) sits in the responsive zone;
  35Bs are saturated (0.84) or floored. Mechanism conjecture (unproven): per-shot output
  entropy is highest in the 0.3–0.7 zone, so extra forward-pass compute has the most marginal
  utility.

**Caveat on "distillable compute":** the label means the *signature* is compute-not-search
(so distillation is *not ruled out*). It does **not** mean distillation has succeeded — that
is the open question §6 tests.

---

## 3. The OPD distillation experiment (the load-bearing test)

### 3.1 What OPD self-distillation does here (teacher / student / masking / KL)
Self-distillation: teacher and student are the **same** base-235B weights; the teacher just
gets more context.
- **Teacher** = *frozen* base-235B (served by `teacher-sglang`). `teacher_cot_mode=insert`
  sequence: `prompt + CoT + nato_buffer + answer`. The per-prompt **CoT** (precomputed
  reasoning) is the teacher's "filler" (`teacher_cot_json` overrides `teacher_filler_text`).
  The teacher's hidden states *at the buffer positions are CoT-informed*.
- **Student** = base-235B *being trained* (the 8-node trainer). Sampled sequence:
  `prompt + nato_buffer + "Answer:" + answer`. **No CoT.**
- **Masking / KL** (config A, see §3.3): `supervise_student_cot=true` keeps the buffer in the
  teacher cache; `opd_supervise_buffer_only=true` → `mask_answer=true`. So **prompt, CoT, and
  answer are all masked; only the ~102 nato-buffer positions are supervised.** The loss = buffer
  logit-KL **+ hidden-match** (`opd_hidden_match_coef`) pushing the student's buffer hidden
  states toward the teacher's CoT-informed ones.
- **Key code refs** (`xorl-client-chat-completions/examples/on_policy_distillation.py`):
  insert/mask logic L170–303; `mask_answer=config.opd_supervise_buffer_only` L2196; `k_filler =
  len(prefill_tokens)` L1826/L1940 → **multi-token fillers (nato) are correctly aligned, no
  patch needed** (the old "multi-token NATO unsafe" memo is stale).

### 3.2 Why `nohm` (the only prior 235B distill run) is NOT a valid negative
`er-opd-235b-nohm` (2026-05-31, 39 steps; `experiments/encoded_reasoning/results/
qwen3_235b_self_distill/er-opd-235b-nohm/`): `buffer_delta` flat ~0.25–0.28. **Three confounds:**
1. **Suppressed baseline:** 0-shot + `</think>Answer:` cue → `acc_nopause` 0.13 (this is the
   real base-235B 0-shot capability, but the buffer "lift" is measured against it, and even
   `acc_pause` 0.40 stays *below* the model's true ability with a good prompt).
2. **Truncated CoT:** CoT median 518 tokens > `sample_packing_sequence_len=512` → the teacher
   saw only ~175 CoT tokens (often not reaching the answer) → garbage distillation target.
3. **No hidden-match:** `opd_hidden_match_coef=0.0` → buffer supervised by *logit-KL only*; the
   code itself documents that buffer-only "Pairs with `opd_hidden_match_coef>0`". The intended
   signal was off.
Plus the student reward-hacked (emit answer then ramble: "…buffer overflow attack…").

### 3.3 The clean retest — `er-opd-235b-clean4d` (config A)
Fixes all three confounds:

| dimension | nohm (confounded) | clean4d (this run) |
|---|---|---|
| shots | 0-shot + `</think>Answer:` | **0-shot** + clean `Answer:` cue (max headroom; nato strongest here) |
| CoT | truncated to ~175 tok (packing 512) | **full CoT** (`sample_packing_sequence_len=1024`, microbatch 64 → same memory) |
| buffer | ` pause`×100 (weakest filler) | **nato** ×3 ≈ **102 tokens** (base-235B's strongest filler) |
| supervision | buffer logit-KL only (`coef=0`) | **buffer logit-KL + hidden-match (`coef=1.0`)** |
| answer | masked (buffer-only) | masked (buffer-only) — **config A** |

**Config A vs B (decided A, with B as fallback):**
- **A** (`opd_supervise_buffer_only=true`, chosen): answer masked → **forces the buffer to be
  the only channel**. If accuracy rises it is *unambiguously* the buffer. Clean test.
- **B** (`opd_supervise_buffer_only=false`, fallback): supervise buffer **+ answer**. Gives the
  student an easy direct-answer path → it would learn input→answer in weights and ignore the
  buffer → `buffer_delta` flat = *false negative for the buffer*. Use only if A is inconclusive,
  attributing via `buffer_delta`.

**Success criteria (watch during training + a final pass@8 checkpoint eval):**
- `opd_hidden_match_loss` > 0 and decreasing → hidden-match engaging (sanity).
- **`buffer_delta` (acc_pause − acc_nopause) GROWS beyond the native ~+0.13, and/or pass@8
  rises** → the trained buffer carries distilled reasoning = **encoded reasoning works**.
- Flat `buffer_delta` → the buffer can't carry it even when forced (a *clean* negative this
  time — baseline de-suppressed, CoT full, hidden-match on). Then try B.

### 3.4 Live status (as of 2026-06-02, write time)
`er-opd-235b-clean4d` relaunching on the curated **nccl** pool (head + 2 workers + 2 samplers
Running, workers 3–7 scheduling into the freed nccl nodes; teachers + dispatch on `default`).
Watcher task `b37jt8qe2` armed (reports step-0 de-confound checks or any crash).

---

## 4. Methodology corrections (chronological honesty — do not repeat these)
1. **"No genuine lift cross-model" (early) — RETRACTED.** Was based on the RL trainer's
   *free-generation* prompt builder (`filler_tokens_rl.py`), miscategorized as the eval harness.
   Free-gen ≠ the paper's V3+prefill.
2. **V3 + `--prefill-answer` is the paper's *intended* method**, not a confound. Filler and
   no-filler both suppress CoT; the contrast is filler-vs-no-filler under matched suppression.
3. **`chat_hardoff` (hard `enable_thinking=False`) ≠ `/no_think` text.** The whole 0.82-vs-0.556
   "I can't reproduce it" saga was using the hard toggle while the paper uses the text-only soft
   suppression. Same model, that one knob flips the result.
4. **Single-filler (pause) probes are insufficient.** Pause is the weakest filler; "Instruct-2507
   kills the effect" was wrong — the full 17-filler sweep reproduces the type-inversion.
5. **0-shot 0.13 on base-235B is the REAL capability, not a bug.** Adding few-shot to "fix" it
   *reduces* the responsive-zone headroom; 0-shot is where nato is strongest.
6. **397B FP8 is BROKEN here** (multimodal `Qwen3_5MoeForConditionalGeneration`, emits `!`
   garbage). Use BF16 / TP=16. The "+20.1 ellipsis" was anti-EOS + possibly FP8 garbage.
7. **`nohm` "distillation says no" is NOT a valid negative** (§3.2). The distillation question is
   genuinely open → `clean4d`.

---

## 5. Infrastructure

### 5.1 Eval harness (built this session; reused by the parallel session)
- `experiments/opd_profile/k8s/build_eval_fleet.py` — generates **N sglang replicas + 1 SMG
  `--policy cache_aware` router** per model (cache_aware routes the shared prefix → KV reuse,
  ~3× per-GPU vs the old dispatch proxy).
- `experiments/opd_profile/eval_filler_fleet.py` — dual-regime (`chat_hardoff` + `raw_nothink`)
  × {base, pause/`--filler`} × {4×4,5×5,6×6} × pass@1/pass@8, **per-sample JSONL** to
  `results/filler_fleet/<run-id>/{samples.jsonl,summary.json}`. `--filler {pause,ellipsis,
  counting}`.
- Parallel session's wide sweeps: `/old-data/apanda/tomi/eval_chat_hardoff_filler_sweep.py`
  (17-filler chat_hardoff, N=1000) and `examples/run_fresh_filler0_eval_passk.py` +
  `run_task_all_fillers.sh` (paper-faithful V3 17-filler, N=1000). Driver default endpoint
  `research-common-08:12345` is NOT reachable from this cluster; the `LiteHF`/`LiteQwen3`
  renderers run it against any sglang.

### 5.2 OPD training stack (the 235B distill run)
~12 pods: 8-node **trainer** (head Job + 7 worker Pods, EP=8 intra-node + shard=64, adamw bf16,
p2p sync) + 2 **samplers** (sglang + Mooncake P2P weight-sync) + **dispatch** (SMG router) + 2
**teachers** (sglang prefill + CoT) + `teacher-smg`. Manifest: `experiments/opd_profile/k8s/
generated/er-opd-235b-clean4d.yaml`. Trainer config: `experiments/opd_profile/configs/
qwen3_235b_a22b_opd_8node_clean4d.yaml` (packing 1024). OPD client:
`xorl-client-chat-completions/examples/on_policy_distillation.py`.

### 5.3 Node-groups (LEARNED THE HARD WAY)
- **nccl** = 10 curated, clean-IB H100 nodes. The 8-node 235B trainer + 2 samplers (Mooncake
  IB) MUST run here; it's the only pool where the gang rendezvous + 235B init reliably succeeds
  (`nohm` proved it). All 10 needed → the whole pool.
- **default** = 42 nodes, but **uncurated** — has dead/flaky-IB nodes (`h100-113` is dead; both
  sessions' pods died on it). Teachers + dispatch (no Mooncake) run here fine. **Moving the
  8-node trainer here FAILED twice** (dead node, rank dropped mid-rendezvous). Do NOT put the
  trainer or Mooncake samplers on default.
- Known-bad nodes to exclude everywhere: **`h100-014`, `h100-050`, `h100-113`**.

### 5.4 Bring-up hazards + fixes (this session hit all of these)
1. **Dispatch on dead node `113`** → trainer-head hangs "Waiting for SMG router" → workers
   rendezvous-timeout. Fix: `nodeAffinity NotIn [113,014,050]` on the dispatch.
2. **Rendezvous timeout** ("6/8 clients joined, 901s"): workers scheduled late (nccl
   contention). Fix: free nodes so all 8 schedule fast; launch the gang **synchronized**.
3. **Engine-init timeout (1800s)** / **rank dropped mid-rendezvous**: gang fragility on
   slow/flaky nodes — use curated nccl, not default.
4. **`delete job … && apply` RACE** → the head Job silently isn't recreated (apply runs while
   old Job terminating). Fix: `kubectl delete job --wait=true`, poll until gone, *then* apply.
5. **Malformed `nodeAffinity` in the head *Job* template** → k8s "cannot unmarshal string into
   NodeSelectorTerm" → Job silently not created. Python `yaml.safe_load` accepts it but k8s
   rejects. Fix: don't hand-indent affinity into Job templates; the head is 1 node, run it
   without the exclusion + force-delete if it lands badly.
6. **Capacity:** the 235B stack needs all 10 nccl nodes; can't coexist with other nccl GPU
   work. Free idle fleets first.
7. **`pkill -f <pattern>` kills the calling shell** if the pattern is in its own argv (exit
   144). Use `pgrep … | guard $$` or just delete the backing pod.
8. **Instruct-2507 tokenizer bug:** snapshot ships only `merges.txt`; the base-tokenizer swap
   → `!`-garbage. Fix: `hf_hub_download` its own `tokenizer.json`/`vocab.json`.

### 5.5 Watcher
`/home/apanda/watch_clean4d.sh` (background): polls trainer-head for OOM/crash, then reports the
first OPD steps' `acc_nopause / acc_pause / buffer_delta / opd_kl` from `opd_profile.jsonl`.

---

## 6. How to run / continue

### 6.1 Eval a new model (filler sweep)
```
python experiments/opd_profile/k8s/build_eval_fleet.py --name eval-<m> \
  --model-path <snapshot> --served-name <name> --tp <N> --replicas 2 > /tmp/fleet.yaml
kubectl apply -n apanda -f /tmp/fleet.yaml
kubectl port-forward -n apanda svc/eval-<m>-router 18080:8080 &
python experiments/opd_profile/eval_filler_fleet.py --port 18080 --model <name> \
  --run-id <m> --filler pause --nprob 100 --ksamp 8
```
Protocol: calibrate `chat_hardoff` baseline first; if 0.3–0.7, sweep all fillers; if >0.8
saturated / <0.1 floored → wrong task (neither is "no effect").

### 6.2 The 235B OPD distill run
1. **Free the nccl pool** (the stack needs all 10): tear down any other nccl GPU pods.
2. `kubectl apply -n apanda -f experiments/opd_profile/k8s/generated/er-opd-235b-clean4d.yaml`
   (trainer+samplers → `node-group: nccl`; teachers+dispatch → `default`).
3. If a relaunch is needed: `kubectl delete job …trainer-head --wait=true`, poll until the Job
   is gone, force-delete trainer+sampler pods, **then** re-apply (avoid the race §5.4.4).
4. `bash /home/apanda/watch_clean4d.sh &` and watch `buffer_delta` + `opd_hidden_match_loss`.
5. If A's `buffer_delta` is flat after ~50+ steps → flip to **B**: set
   `opd_supervise_buffer_only=false` in the manifest's OPD-client args, relaunch the trainer.
6. Final verdict: serve the saved student checkpoint and run `eval_filler_fleet.py` pass@1 +
   pass@8, with vs without the nato buffer, against the native baseline.

### 6.3 Key knobs (OPD client args, in the manifest's trainer-head command)
`model_name=Qwen/Qwen3-235B-A22B` · `teacher_cot_json_path / prompts_json_path` ·
`student_prefill_text=<nato> student_prefill_count=3` · `student_prefill_suffix="Answer: "` ·
`teacher_cot_mode=insert supervise_student_cot=true` ·
`opd_supervise_buffer_only=true` (A) / `false` (B) · `opd_hidden_match_coef=1.0` ·
`num_steps=400 prompts_per_step=64 learning_rate=1e-5` · `sync_method=p2p`.

---

## 7. Open questions / next steps
1. **clean4d verdict** (in flight): does training move `buffer_delta` / pass@8 on Q3-235B base?
2. If A flat → **B** (unmask answer, attribute via `buffer_delta`).
3. **Clean-prompt 17-filler sweep on Q3-235B base** (only pause + V3 done) — close the
   responsive-zone loop on the clean prompt.
4. **Other tasks on Q3-235B** (arithmetic, var-counting) where the responsive zone may be wider
   than 4-digit mult.
5. The paper's **RL gives search (pass@8) not compute (pass@1)** — does distillation differ?
   That is exactly clean4d's bet.

---

## 8. File / data / config index
- **This master doc:** `experiments/opd_profile/ENCODED_REASONING_MASTER_RUNBOOK_2026_06_02.md`
- **My findings docs:** `experiments/opd_profile/OPD_FILLER_FLEET_FINDINGS_2026_06_01.md`,
  `OPD_ENCODED_REASONING_FINDINGS_2026_06_01.md`, `HANDOFF_OPD_ENCODED_REASONING.md`
- **Parallel session docs:** `/old-data/apanda/tomi/outputs/encoded_reasoning_final_
  reconciliation_20260601.md` (authoritative cross-model), `encoded_reasoning_final_summary_
  20260601.md`
- **Eval results:** `experiments/opd_profile/results/filler_fleet/<run>/` (samples.jsonl +
  summary.json); 17-filler sweeps under `/old-data/apanda/tomi/outputs/q3_235b*_v3_allfillers_*`,
  `q36_35b_chathardoff_*`, `q3_30b_instr_chathardoff_*`
- **235B OPD runs:** `experiments/encoded_reasoning/results/qwen3_235b_self_distill/{er-opd-235b-
  nohm,er-opd-235b-icl,er-opd-235b-clean4d}/` (opd_profile.jsonl + trainer logs)
- **Configs:** `experiments/opd_profile/configs/qwen3_235b_a22b_opd_8node{,_clean4d}.yaml`
- **Manifests:** `experiments/opd_profile/k8s/generated/er-opd-235b-clean4d.yaml`,
  `q3-235b-teach.yaml`, `q3-235bi-teach.yaml`
- **Eval scripts:** `eval_filler_fleet.py`, `/home/apanda/{paper_v3_prefill,passk_filler,
  tomi_raw_repro,raw_passk,dual_regime_passk}.py`, `build_fewshot_opd_data.py`
- **CoT data (Q3-235B, 4-digit):** `/shared/opd-coord/randnum_4digit_q3_235b_{cot,prompts}_
  filtered.json` (15979); fs5 variants `_fs5.json`
- **Models:** `/shared/huggingface/hub/models--Qwen--{Qwen3-235B-A22B (8efa617), Qwen3-235B-A22B-
  Instruct-2507 (ac9c66c, needs own tokenizer fetched), Qwen3.5-35B-A3B (59d61f3), Qwen3.6-35B-
  A3B (995ad96), Qwen3.5-397B-A17B (BF16; FP8 BROKEN)}`
- **SMG router:** `/home/apanda/smg-together-thunderagent-port/target/debug/smg`; runbook
  `experiments/opd_profile/runbooks/smg_router_swap.md`
- **Watcher:** `/home/apanda/watch_clean4d.sh`
```

