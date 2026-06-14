# STATUS: OPSD vs SFT on nested-expression arithmetic (Qwen3.6-35B-A3B)

**Date:** 2026-06-11 ~00:30 · **Branch:** `apanda-dev-prefill-time-compute` · **Stack:** `er-opd-q36-35b-slots` (8-node trainer, 2×TP=2 samplers, 2×TP=8 teachers)
**Predecessor:** `HANDOFF_ARITHMETIC_OPSD_2026_06_10.md` (work plan §2a–2f, executed below) · **Ops:** `autoresearch/CANONICAL_INFRA_RUNBOOK.md`

## 1. The question

On 4×4 multiplication, OPSD turned out to be unnecessary — plain SFT on gold answers hit greedy 0.948 in 15 steps (memory `project_opd_vs_sft_4x4_verdict`). The base model already had a single-pass algorithm that training merely repaired, so CoT distillation had nothing to add.

**Nested-expression arithmetic** (`/old-data/apanda/no_cot_math_public/arithmetic_problems.jsonl`, 100k problems like `((25 - ((-8 // -32) - -90)) + (-25 * -56)) * (55 + 38)`) inverts that profile: the base model is floored in a single forward pass but strong with chain-of-thought. The falsifiable claim under test: **on-policy distillation of the CoT-conditioned answer distribution can internalize serial computation into one forward pass better than answer-only SFT** (SFT ships ~8 digits of signal per example; OPD ships the whole distribution).

## 2. What was done (all of HANDOFF §2a–2e)

### 2a. Bucketed difficulty probe → band pick ✅
`autoresearch/arith_band_probe.py`, n=64/bucket, greedy, against the pristine teachers. Direct arm = `/no_think` + `Answer: ` prefill, single pass; CoT arm = thinking mode, 4096-token budget.

| bucket | direct | CoT | CoT trunc@4096 | gap |
|---|---|---|---|---|
| ops5 | 0.047 | 0.938 | 3.1% | +0.89 |
| **ops6 (chosen)** | **0.047** | **0.891** | 6.2% | **+0.84** |
| ops7 | 0.078 | 0.922 | 6.2% | +0.84 |
| ops5_d3 | 0.000 | 0.953 | 4.7% | +0.95 |
| ops6_d5p | 0.062 | 0.922 | 6.2% | +0.86 |
| ops7_d5p | 0.031 | 0.891 | 9.4% | +0.86 |

Surprise: difficulty is **flat** across op-count (the dataset only has 5/6/7-op expressions) — every band satisfies the selection rule (direct ≤0.1, CoT ≥0.8). Picked **ops6** (middle band, 33k problems) to avoid both the "too shallow" and "teacher too noisy" extremes. Shallow fallback (ops5/ops5_d3) documented if the bootstrap risk bites.

### 2b. Client scorer ✅
`eval_task="arithmetic"` added to `/home/apanda/xorl-client-chat-completions/examples/on_policy_distillation.py`:
- gold computed by AST-whitelist evaluation of the expression in the prompt (`+ - * // %` + unary minus — NB the dataset uses `%`, which the handoff omitted), never bare `eval`;
- **48.7% of answers are negative** — scoring parses the first signed integer (`-?\d[\d,]*`) and compares exactly (substring matching is unsafe: "13" ⊂ "-13"/"113").
Validated against all 100k recorded answers (0 mismatches) + unit tests.

### 2c. Data pools (contamination rule enforced) ✅
`autoresearch/arith_prep_pools.py` (seed 20260610), set-intersection asserts at every consumer:
- `/shared/opd-coord/arith_ops6_8192_prompts.json` — train pool (chat format, `/no_think` prefix)
- `/shared/opd-coord/arith_ops6_eval_1024.json` — disjoint in-loop eval (`client_args.eval_prompts_json_path`)
- `/shared/opd-coord/arith_ops6_fresh_1024.json` — second disjoint set for the endpoint fresh probe (`autoresearch/arith_fresh_probe.py`)

Trap discovered: the dataset's own `arithmetic_problems_heldout.jsonl` is **100% contained in the train file** — never use it as an eval source.

### 2d. Teacher CoT precompute ✅
`cot_precompute.py` (new `--enable-thinking true` flag pins `chat_template_kwargs` — PTC-118 lesson) + the "careful mathematician … end with `Answer: <number>`" system prompt, greedy, **max_tokens 8192** (Run-B lesson), split across both teachers, ~2h wall.
Filter (`autoresearch/arith_filter_cot.py`): **96.3% teacher-correct kept (7,892/8,192)**, truncation 1.7% (<2% threshold), median CoT 2,633 tokens. Output: `arith_ops6_nonempty_{prompts,cot}.json`, index-aligned, every entry ends `Answer: <gold>`. (Precompute config beats the probe's CoT arm — 0.96 vs 0.89 — thanks to the system prompt + bigger budget.)

### 2e. Candidates (all in `autoresearch/candidates/`, 101 steps × 128 prompts/step, lr 3e-6, eval every 5 steps on the disjoint set, step-100 greedy control n=1024)
- **ARITH-001-SFT** — SFT control on the full 8,192 pool (`sft_mode` + `opd_teacher_answer_source: gold`, no teacher). **DONE.**
- **ARITH-003-SFT-F** — same SFT on the 7,892 teacher-correct subset, so SFT and OPD are data-matched (user request). **RUNNING.**
- **ARITH-002-OPD** — reverse-KL no-filler OPD, `teacher_cot_mode: replace` with the precomputed CoTs. **QUEUED** (launches when 003 finishes).

## 3. Results so far

### ARITH-001-SFT (full pool) — COMPLETE
Held-out greedy `eval/accuracy` (n=256, disjoint):

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| acc | .047 | .094 | .117 | .152 | .156 | .188 | .191 | .203 | .230 | .230 | .215 |

- **Step-100 greedy control (n=1024): 0.229**
- **Fresh probe on never-trained problems (n=256): 0.230** → zero memorization gap; the lift is a genuine general improvement.
- W&B: `together-research/xorl-prefill-time-compute/runs/frs9krlp`

**Reading:** answer-only SFT grinds slowly and roughly linearly to ~0.23 — a real 5× lift over base but nowhere near the 0.89 teacher-CoT ceiling, and completely unlike 4×4 (0.90 by step 15). Composition is mostly NOT learnable from 8 digits/example at this scale. This is the ideal backdrop for the OPD test: **~0.66 of headroom between SFT and the teacher.**

### ARITH-003-SFT-F (filtered pool) — COMPLETE (2026-06-11 00:35)
Tracked ARITH-001 within noise at every checkpoint, as expected — dropping the 300 teacher-wrong problems doesn't change what SFT learns.
- **Step-100 greedy control (n=1024): 0.1904** (pause arm 0.1875) · in-loop held-out endpoint 0.2031
- **Fresh probe on never-trained problems (n=256): 0.180** → zero memorization gap, again.
- vs ARITH-001's 0.2295: Δ≈0.04 ≈ 2σ on n=1024 — treat as noise; the late trajectory wobbled (.2305@70 → .180@90 → .203@100). **The data-matched SFT baseline for the OPD comparison is ~0.19–0.23.**
- W&B: `together-research/xorl-prefill-time-compute/runs/pqg0aaqp`

## 4. What we are trying to do right now

1. ~~Finish ARITH-003-SFT-F~~ DONE (control 0.1904, fresh 0.180, recorded above).
2. **ARITH-002-OPD LAUNCHED 2026-06-11 00:45** on the identical filtered pool/eval — the headline run. Success shapes:
   - **OPD ≫ ~0.23** → CoT-conditioned distillation transfers serial computation answers can't — the OPSD claim, finally on a task where SFT can't shortcut it.
   - **OPD ≈ SFT ≈ 0.23** → 5×5-redux; escalate to curriculum (shallow band first) or OPRD.
   - Watch the bootstrap risk (HANDOFF §4.1): at direct≈0.05, on-policy samples are ~all wrong early — monitor `opd_kl_answer_wrong_mean` vs `_correct_` and teacher-entropy splits; peak-then-decay while loss falls is the failure signature.
3. Both runs still climbing at step 100 → a 321-step extension of the better arm is the natural follow-up. **Extend SFT too** (set `save_every` on that run): "SFT ≈ 0.2" is a budget point, not an asymptote — an OPD win at 101 steps that's only a *slope* difference would not survive a converged-SFT comparison.

### Queue (user-directed, 2026-06-11 ~01:30)

1. **ARITH-004-OPD-NPK** ~~RUNNING~~ **COMPLETE 03:01 — FAILED.** Step-100 control (n=1024) **0.0674** / pause 0.0664; fresh probe **0.066** (consistent, no memorization gap); in-loop held-out crept 0.043→0.05–0.067, ≈ base 0.047. **OPD ≪ SFT (0.19) even with the corrected answer-only KL.** Healthy mechanics throughout: no collapse (`empty_frac` 0, completions ~3.4 tok), loss 2.85→0.8, clamp 0.33→~0.10, `frac_answer_correct` rose ~2%→3–6% but never compounded. Reading: the dilution fix was necessary but NOT the bottleneck — distilling teacher conditionals at ~95%-wrong on-policy prefixes doesn't teach composition the way gold-answer CE does. The remaining suspects: clamp (being tested), bootstrap (ARITH-005 SFT-warm-start), and the objective-content gap itself.
2. **ARITH-006-OPD-CL10** COMPLETE 04:50: step-100 control (n=1024) **0.0986** / fresh probe **0.074**; clamp_frac collapsed 0.12→~0.02 (knob worked). **Best OPD arm yet (+0.03 over 004, ~2× base) but still ≈half of SFT (0.19).** Clamp was *a* constraint, not *the* bottleneck.
3. **PTC-122-NPK** COMPLETE 06:30 — **the dilution thesis CONFIRMED on 4×4**: held-out 0.684→**0.902@step15** (= SFT's ramp; anchored OPD needed ~100 steps for 0.905), peak **0.922@25**, then drift down to 0.82@90; step-100 control **0.8408** (n=1024), fresh-pair probe **0.879**. Two conclusions: (a) the historical "SFT ≫ OPD" *speed* gap on 4×4 was the prompt-KL bug; (b) the undiluted gradient also overfits/drifts faster — corrected OPD wants **early stopping (~step 25)**, where it peaks 0.92, still shy of SFT 0.948. The 4×4 bootstrap is benign (38–69% answer supervision at correct prefixes), confirming arithmetic's failure is the wrong-prefix bootstrap, not the machinery.
4. **ARITH-007-PAUSECOT** (staged, user-proposed objective ~05:10): pause×1024 prefill + on-policy answer + reverse-KL of pause-position-i against teacher CoT-position-i distribution (`match_cot` + `supervise_student_cot` + `hidden_match_coef=0` = pure logprob space; zero code change). Fixed K=1024 (min CoT 1409 ⇒ K≤C holds ∀; eval coherent: acc_pause = greedy with the trained buffer). 64 prompts/step.
5. **ARITH-008-PAUSECOT-HM** COMPLETE 12:15 — **hidden matching at buffer↔CoT FAILED**: step-100 control acc_pause **0.039** / acc_nopause **0.046** / held-out 0.043 ≈ base 0.047, `buffer_delta=−0.007` (buffer not load-bearing). Crucial mechanism read: the hidden-MSE **did optimize** (raw 10.3→4.85, halved) while accuracy never moved — *the student learned to imitate the teacher's CoT hidden states at pause positions without acquiring any of the computation they represent*. Also: the reverse-KL component was clamp-dead the whole run (`clamp_frac` 0.99→0.86, per-token KL ≥10 at buffer positions), so this was a clean hidden-only test. Run took ~6h (1024-token prefills make eval steps heavy — budget note for future buffer arms).

### 4b. Loss-geometry caveat on ARITH-002 (verified in code 2026-06-11)

In the no-filler recipe the KL is NOT answer-only: the no-remap student datum path masks nothing, and the teacher cache keeps `(p−1)+ans` rows, so **prompt positions carry full-weight KL**. The teacher's prefix at those positions excludes its CoT (inserted at the prompt→answer boundary), so the prompt term is a pure anchor-to-base regularizer — `diag_region_ids` is metrics-only (`opd_loss.py`: "They never touch the loss value") and there is no down-weighting knob. On 4×4 prompts were ~70% of supervised tokens; here (~54-token prompts, ~4-token answers) expect `opd_frac_prompt` ≈ 0.9. Two consequences: (i) loss is normalized by global valid tokens → effective per-answer-token gradient ~10× smaller than SFT's answer-only CE at the same lr; (ii) the anchor competes with answer learning as the model drifts. **Confirmed against the reference implementation (user pointer, 2026-06-11 ~01:00):** `~/OPD/verl/recipe/gkd/megatron_workers.py:255` builds `calc_kl_mask` that zeroes everything before the response — reference OPD computes KL at RESPONSE positions only, normalized over response tokens. Our prompt-position KL is a deviation, not a design choice. **Fix implemented:** `opd_mask_prompt_kl` Config flag in `on_policy_distillation.py` (default false; masks region-0 positions, same boundary as `opd_region_ids`/sft_mode's answer weights → with the flag, OPD and SFT supervise IDENTICAL positions). Unit-tested both datum paths (no-remap + K>0 remap); generator passthrough dry-rendered. Candidate: **`ARITH-004-OPD-NPK.yaml`** = ARITH-002 + the flag. **User decision (~01:10): the unmasked objective is simply wrong — `opd_mask_prompt_kl` now defaults TRUE in the client** (set `false` explicitly to reproduce any pre-2026-06-11 OPD run, incl. the 4×4 0.905). **ARITH-002 KILLED at step ~32** (held-out flat at base 0.043–0.051 through step 25, vs SFT ~0.13@25 — partial profile kept at `20260611T003235Z-configARITH-002-OPD-*` as the with-anchor data point); **ARITH-004 launched 01:15** as the headline faithful-objective run. Expect its `opd_frac_prompt`=0 / `valid_tokens`≈7k (vs 58k) at step 0 — that's the mask working. Follow-up after that: **ARITH-005 = OPD warm-started from an SFT checkpoint** (raises correct on-policy prefixes ~5%→~23% at step 0, directly attacking the bootstrap risk; needs `save_every` on an SFT run first — current SFT weights were hot-only and are gone once 002 syncs).

### 4c. ARITH-009-SFT-PAUSE-SAVE — COMPLETE 21:20 (run wall 75 min on the unthrottled stack)

- **Pause-format SFT works: step-100 control acc_pause 0.238 / acc_nopause 0.215 (n=1024)**, held-out trajectory 0.047→0.191@50→0.215@100, fresh probe (no-pause direct, never-trained) **0.184** — no memorization gap. The 1k-pause buffer in context does NOT impede answer-SFT (matches the 0.19–0.23 no-pause ladder), and the model now answers correctly in the pause format ~24% of the time vs **0.000 at base** — the zero-signal bootstrap that killed pause-OPD (ARITH-007F/008) is fixed.
- **DCP checkpoint saved** (34 s): `server_output/weights/student/opd-q36-35b-arith-sft-pause1k-save-configarith-009-sft-pause1k-save-ckpt-step101` (xorl URI `xorl://student/weights/opd-q36-35b-arith-sft-pause1k-save-...-step101`) → seeds **ARITH-005-OPD-WARM**.
- Timing on the new flags (infra runbook §7b): plain 36.6 s, eval steps 65.8 s, terminal control 267 s. Caveat: this run's logprob-battery diagnostics ran at n≈598/arm (HTTP 503 queue overflow at conc 16, observed live; `--max-queued-requests 2048` deployed right after this run — future runs score the full n=1024).

### 4d. The warm-start wave (user-directed 2026-06-11 ~23:30) — queue + measurement doctrine

**Premise (user):** none of the pause-objective falsifications are conclusive until rerun from a seed that can already answer in pause format — the cold runs gave the matching gradients no answer-side behavior to compose with. Each run is now ~75 min, so run EVERYTHING:

| # | arm | objective | status |
|---|---|---|---|
| A | ARITH-005-OPD-WARM | answer-only reverse-KL, pause×1024, warm009 | **GROUND-TRUTH COMPLETE 15:14Z (verified-clean stack, 7th launch):** held-out 0.188-seed → 0.10-0.12 band by step 10 → grind to 0.074@90; **terminal control 0.094 pause / 0.097 nopause (n=1024)** vs seed 0.238/0.215; buffer_delta −0.003 (inert). **Warm answer-only OPD trains its own seed DOWN** — wrong-prefix supervision (3-17% correct at T=1.0) erodes; warm-starting alone does not fix the bootstrap. Next levers: lower prepare temperature / correct-prefix filtering. NB all earlier 005 'verdicts' carried the corrupted client; this run supersedes them. |
| B | ARITH-010-PAUSECOT-FKL-WARM | 007F's forward-KL pause-i↔CoT-i, warm009 | **PREEMPTED @60 (user call, throughput work) — verdict CLEAR: buffer-KL 9.65→4.09 (plateau @40) while held-out fell 0.238-seed→0.109@0→0.043@60. Warm-start replication of the trace-matching falsification: the matching loss converges AND consumes the seed's skill. 3-way consistent (cold belief / cold latent / warm belief).** |
| C | ARITH-011-PAUSECOT-HM-WARM | 008's KL+100·hidden-MSE, warm009 | **PREEMPTED @~25 — verdict CLEAR and harsher than 010: hm_raw 11.0→5.2 while held-out 0.238-seed→0.109@0→0.012@10→0.004@20 (empty_frac 0, KL clamp-dead 0.92 = hidden-only). Latent-space trace imitation destroys the seed faster than belief-space. 4-way consistent falsification of trace matching.** |
| D1 | ARITH-012-SFT-RAND-SAVE | SFT seed with 1024 FIXED RANDOM token ids | **DONE — FAILED, with a confound: control acc_pause 0.000 / acc_nopause 0.013 (n=1024), 85% of pause-arm completions cap-ramble (no stop seq) — SFT didn't even format-adapt to the random buffer in 101 steps (pause-SFT got 0.238). CONFOUND: first muon run (4-node auto-remap, muon_lr 3e-5); nopause < base suggests optimizer may share blame. Deconfound = 012 rerun on adamw before burying the hypothesis. **DONE 16:55Z (ARITH-012A, adamw): control 0.218 rand / 0.225 no-buffer (n=1024), held-out 0.051→0.203@100, fresh probe 0.203 — textbook SFT ramp. 012's 0.000 was ENTIRELY the muon config (opdb_4node muon_lr 3e-5/mom 0.95 breaks SFT — calibrate before science use). Buffer content irrelevant under SFT (rand == pause == none ≈ 0.22-0.24, all passive). Ckpt saved; exact-ids held-out client fix validated live.** ALSO: in-loop held-out eval is broken under student_prefill_token_ids ('Request must have either text or input_ids', all requests, client bug — control eval unaffected).** |
| D2 | ARITH-013-OPD-RAND-WARM | answer-only OPD from D1's ckpt | **MOOT pending 012 deconfound — the seed has zero answer signal (the bootstrap problem 009 was built to fix).** |
| E | ARITH-014-OPRD-WARM | hidden-match at SHARED answer positions vs CoT-informed teacher | **STEP-0 INFRA FAILURE — masked OPRD remap emits GLOBAL teacher-cache rows where the packer rebases per-sample slices (`teacher_cache_indices` OOB on every rank; first-ever exercise of supervise_student_cot=false + OPRD). Root cause + fix spec: docs/notes/oprd_warm_cache_indices_rebase_bug.md (other agent, 5cbb094d). Rerun after the client-side localization fix.** **FULL CLEAN RERUN (014D, coef 1, standard substrate, fixed stack) COMPLETE 06-12 ~21:05Z: FIRST POSITIVE OPRD SIGNAL — terminal control 0.145/0.143 (n=1024) vs the answer-only null 0.094; held-out dips with the erosion dynamic (0.117@10) then STABILIZES and RECOVERS (0.129@90→0.141@100, rising at end). Coef-1 answer-position hidden match HALVES wrong-prefix erosion. Below the 0.238 seed → not standalone; next = lever-§7.1 × OPRD combo, run extension, or coef∈(1,100).** |
| F | sequence-level reward on the buffer path (S10d) | **code change** | build after E |

**Measurement doctrine (user, recorded):**
- The decisive comparison is **this model's buffer-conditioned accuracy vs the no-pause baseline model (~0.23 SFT / 0.238 pause-SFT seed)** — cross-run, not within-run. The corrupt-buffer arm is NOT decision-relevant: the buffer tokens are prompt-independent, so there is no prompt-specific content to corrupt; the hypothesis is generic extra serial compute, not encoding. (Corrupt arm disabled in all new candidates.)
- Never optimize or gate on `acc_pause − acc_nopause`: the cheapest way to move it is torpedoing the no-pause arm (the S4 confound).
- Correction to §4 item 5's reading: 007F's buffer-KL 16.6→3.9 ≈ perplexity ~50/position — substantial movement, NOT "student can predict the trace."
- Attention-map worry (motivates D): with a uniform ' pause'×1024 buffer the model may simply never attend into the buffer → ~zero gradient through it from answer-only supervision → a no-buffer-use local minimum even if a buffer-using solution exists. Distinct random ids at least give the attention map something to differentiate. Checking attention mass into the buffer on trained checkpoints is the diagnostic if D also lands at the no-pause ceiling.

## 5. Infra notes from tonight

- **Sampler control restamp collision (21:46:58):** another session rewrote dispatch+sglang-0/1 controls mid-run (deploying the sanctioned switch to **default flashinfer attention backend** — user decision ~19:00; runbook §11's "fa3" is stale). My run died at step 20 on weight-sync connection-refused; recovery = stop-trainer → wait for `/v1/models` on both samplers → relaunch (clean). Step-0 baseline reproduced exactly on flashinfer (0.0469). **Multiple agents operate this box — on any mid-run death, check `generator status` re-exec timestamps first.**
- SFT runs need no teacher: `teacher_cot_json_path: ''` works (`sft_mode` never prefills), so SFT launched while the precompute occupied the teachers; ~42 s/step at 128 prompts/step (vs ~115 s/step for OPD no-filler).
- Memory file: `project_arith_opsd_ops6_workstream.md`.

## 6. Overnight wave 2026-06-13 (rev4 queue: lever-1 × OPRD; AMDAHL-optimal trainer-4 + quack_linear)

Driven by `autoresearch/overnight_supervisor.sh` (back-to-back queue + failure recovery + keepalive). Substrate = warm009 (009 pause-SFT seed, control 0.238), adamw, **ce_mode quack_linear** (user directive; gate §9c opd_kl=0.533 healthy). Stack: trainer-4 (32 GPU) + 1 teacher direct + 2 samplers. Pre-flight: recovered a wedged perf-agent AMDAHL-005 launch, killed its monitors (user OK), recreated samplers clean. Lever-1 = lower on-policy *prepare* temperature (generator hardcoded temperature=1.0 — templated it, [[project_opd_generator_temperature_hardcoded]]).

Comparators: seed/pause-SFT 0.238 (held-out ~0.188); answer-only warm-OPD T=1.0 null 0.094 (ARITH-005-FULL); coef-1 OPRD T=1.0 = 0.145 (014D); SFT ~0.19–0.23; base 0.047.

- **R1 ARITH-015-WARM-T03 (answer-only reverse-KL, pause×1024, T=0.3) — COMPLETE 11:18 (run wall ~61 min).** Terminal control n=1024 **acc_pause 0.1035 / acc_nopause 0.1123**; held-out trajectory 0.184(seed)→0.113@10→~0.10 plateau (steps 15–90 in 0.08–0.14, noise band); opd_kl 0.50→0.08, empty_frac 0, clamp ~0 throughout. **Verdict: lever-1 alone does NOT rescue the answer-only bootstrap** — 0.104 is ~1pp above the T=1.0 null (0.094, within ~1σ) and far below the seed 0.238. Lowering prepare temperature 1.0→0.3 raised correct-prefix mass too little to matter; the wrong-prefix erosion still dominates. Clean negative for L1-alone → the value (if any) must come from OPRD's restorative gradient (R2) and/or correct-prefix filtering (code, attended).
- **R2 ARITH-016-OPRD-T03 (coef-1 OPRD × T=0.3) — COMPLETE 12:22 (run wall ~64 min).** Terminal control n=1024 **acc_pause 0.1084 / acc_nopause 0.1162**; held-out dipped to 0.090@15 then recovered/plateaued ~0.11–0.13 (014D-shape but lower ceiling); oprd_loss 0.006→0.0024 (optimizing), opd_kl ~0.09, frac_answer 1.0, empty_frac 0. **Verdict: 0.108 < 014D's 0.145** (same arm at T=1.0). Combined with R1, **lever-1 (low prepare temperature) is settled NEGATIVE** — inert for answer-only (0.104≈0.094 null), mildly harmful for OPRD (0.108<0.145). The OPRD hidden-match gradient is the real lever; it works best at T=1.0. None of L1/L1×OPRD clear the seed 0.238.
- **R3 ARITH-017-OPRD-C3 (OPRD coef 3, T=1.0) — COMPLETE 13:27 (~64 min).** Terminal control n=1024 **acc_pause 0.1016 / acc_nopause 0.1084**; held-out dipped then stayed flat ~0.10 (NO 014D-style recovery). **Verdict: coef 3 (0.102) < coef 1 (0.145)** — over-weighting the hidden match suppresses the answer-learning recovery.
- **R4 ARITH-018-OPRD-C10 (OPRD coef 10, T=1.0) — COMPLETE 14:32 (~64 min).** Terminal control n=1024 **acc_pause 0.1074 / acc_nopause 0.1113**; held-out flat ~0.10 (no recovery, dipped to 0.062@80).
- **OPRD coef dose-response COMPLETE (all T=1.0): c1=0.145 (014D, best) > c10=0.107 ≈ c3=0.102.** Optimum is at/below coef 1; higher coefs monotonically suppress the late recovery.
- **⚠️ INFRA: `ce_mode: quack_linear` is INCOMPATIBLE with SFT** (`sft_mode`/gold CE). E0a hard-failed at step ~0 (14:38): `Engine error: ce_mode='quack_linear' does not support return_per_token=True`. SFT's gold cross-entropy needs per-token returns; quack_linear (chunked CE) doesn't provide them. **OPRD/OPD-KL runs are fine on quack_linear** (R1–R4 + the gate all ran clean — they use the KL/hidden-MSE path, not return_per_token). Fix: SFT configs (`..._4node_adamw.yaml`) reverted to `ce_mode: compiled`; OPRD/KL configs (`..._4node_warm009.yaml`) keep quack_linear. Recorded in memory [[project_opd_generator_temperature_hardcoded]]-adjacent.
- **Wave-2 queue (swapped in 14:43 after R4):** the dose-response showed the optimum is ≤ coef 1 and 014D was still rising at step 100, so the supervisor queue was swapped to the two promising probes before the SFT anchor: **ARITH-019-OPRD-C05** (coef 0.5, T=1.0 — probe below the optimum) → **ARITH-020-OPRD-C1-LONG** (coef 1, 201 steps — extend the winner) → **E0A-SFT-LONG** (compiled). 019 launched 14:43 (after a stale-rendezvous init-hang from a too-fast relaunch, cleared by a clean unhurried stop+restart — ops note: never relaunch the trainer within ~30s of a prior launch).
- **R019 ARITH-019-OPRD-C05 (OPRD coef 0.5, T=1.0) — COMPLETE 15:51.** Terminal control n=1024 **acc_pause 0.0977 / acc_nopause 0.1035**; held-out flat ~0.09–0.10 (no recovery, unlike 014D's c1).
- **OPRD COEF SWEEP COMPLETE (all T=1.0, warm009 seed, n=1024 control):** **c0.5=0.098 · c1=0.145 (peak) · c3=0.102 · c10=0.107.** Coef 1 is a genuine but sharp local optimum — gentler (0.5) and stronger (3/10) both drop to ~0.10. **Even the peak (0.145) is far below the pause-SFT seed (0.238) and the SFT ceiling (~0.23).** 014D's 0.145 came from a late recovery the other coefs don't reproduce; treat it as the best-case OPRD point, still SFT-dominated.
- **R020 ARITH-020-OPRD-C1-LONG (coef 1, 201 steps) — STOPPED at step ~193 (17:33) for stack handoff to the perf agent.** Held-out trajectory oscillated **~0.10–0.12 the entire back half** (100:0.121, 130:0.109, 150:0.105, 170:0.121, 190:~0.11) — **no sustained climb past 014D's 0.145**. Verdict (trajectory-determined; terminal control not reached): **extending the coef-1 winner to 2× steps does NOT reproduce 014D's 0.145** → confirms 0.145 was favorable variance, not a robust optimum. DCP checkpoint saved at step 99 (step-199 save not reached). 
- **WAVE WOUND DOWN 2026-06-13 17:33** — perf agent took the stack for throughput engineering (confirmed actively running: trainer loaded + samplers serving by ~17:44). Supervisor killed, trainer stopped (32 GPU freed), cron + monitor cleared; samplers/teacher left warm (perf agent restamped them to q35). Full overnight result + per-step throughput in science runbook §1e + infra runbook §7d-addendum.
- **CPF KNOB BUILT 2026-06-13 ~17:50 (offline, while the perf agent held the stack).** The §7.1 next-lever (correct-prefix filtering) no longer "needs a knob that doesn't exist" — `opd_correct_prefix_only` (default False) shipped in `xorl-client` `on_policy_distillation.py`: masks the whole target for any on-policy sample with `sample_ok != 1` (reuses the `-100` mask; correctness flag was already plumbed). Client-side only (no server change), 3 unit tests added (all 69 client tests pass), generator passes it via `client_args`, dry-render verified. **Ready candidate `ARITH-021-CPF-OPRD-WARM.yaml`** (CPF + OPRD c1 + warm009, T=0.6). **Launch protocol: §9c step-0 KL gate first; watch valid-tokens/step (CPF shrinks the batch to the correct fraction — raise prompts/step or temp if too sparse).** This is the program's first genuine OPD-beats-SFT test on the right manifold. **Promising UNTESTED directions for next session (documented, not run — clean queue preserved):** (1) **coef < 1** (e.g. 0.3/0.5 — gentler hidden-match may preserve more answer learning while keeping the restorative gradient; the optimum appears ≤1); (2) **extend the coef-1 winner past 101 steps** (014D was still *rising* at step 100: 0.117@10→0.141@100 — it may not have plateaued). Both target beating 0.145 toward the seed 0.238.
- **WAVE BOTTOM LINE so far:** OPRD's answer-position hidden-match (014D, coef 1, T=1.0 → 0.145) is the only positive lever vs the answer-only null (0.094). Prepare-temperature (lever-1) is inert/harmful (R1 0.104, R2 0.108). Coef>1 hurts (R3 0.102). **Nothing yet clears the pause-SFT seed (0.238) or the SFT ceiling (~0.23)** — i.e. on this floored task, no on-policy distillation variant beats answer-only SFT. The bootstrap (wrong-prefix supervision) remains the binding constraint; the real attack is correct-prefix filtering (code change, attended) or coef<1 / longer-c1 (cheap, queueable).
- **E0a E0A-SFT-LONG (321-step converged-SFT anchor, save_every 107)** — queue tail + keepalive.
