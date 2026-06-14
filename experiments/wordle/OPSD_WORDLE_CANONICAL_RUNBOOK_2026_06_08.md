# OPSD Wordle Gradient Baseline Canonical Runbook

Last refreshed: 2026-06-13 PM (HANDOFF — corrected science: mode-collapse not OPSD-dead; forward-KL + reason-first is the live hypothesis)

> **READ ORDER**: "🎯 HANDOFF — START HERE" immediately below is the single current
> source of truth. The dated "⭐⭐ FINDING" sections under it are the chronological ledger
> (note the THIRD FINDING's "OPSD structurally stuck" was CORRECTED — see START HERE).
> Infra/gate/monitor command sections lower down are still valid. Legacy 30B sections at
> the bottom are reference.

---
## 🆕 HANDOFF — START HERE (2026-06-13 EVENING, science-focused agent)

**TL;DR — two corrections + one new launch-ready experiment that attacks the root cause directly.**

**Correction 1 — the SFT-48 warm-start was a NEAR-NO-OP, so "SFT can't teach enumeration" is UNPROVEN.**
The headline SFT-on-Kimi-gold step-48 ran with **adamw lr=1e-6** (10-50× too low) — its training loss
barely moved (0.936→0.88 over 48 steps, grad_norm stuck ~5-6). Read the SFT-48 eval games: the model
emits a clean one-liner but **produces no `<think>` and does not enumerate** — it learned the no-think
shortcut. The git log already records this (`b6289968` "flat think-SFT applied yaml lr 1e-6→1e-5",
`c0a32ce4` "high-entropy think prose"). The Kimi gold itself is diluted: **418/1094 rows have
think_tokens=0** (empty think → teaches skip-thinking), only SOLVED games kept (291 easy-biased
targets), and Kimi's t=1.0 think rambles (median 353, max 10553 tok). The 676 think-bearing rows DO
contain real enumeration ("BATCH? MATCH? WATCH?... LATCH excluded... pick WATCH") — right signal, noisy
delivery. **A proper SFT (effective LR that drives the loss down, low-entropy think) was never run.**

**Correction 2 — the reason-first CoT cache teaches the WRONG thing.** `results/wordle_teacher_cot/
...cot_cache.jsonl` is median **25 words of post-hoc justification** ("DOERS narrows the remaining 220
possibilities") — it states the FORM and never enumerates. SFT/OPSD on it teaches confabulation, not
enumeration. The runbook's prior "live hypothesis" (forward-KL + reason-first candonly) is built on this
weak target — deprioritize. (reverse-KL collapse + guess-only inertia diagnoses below are CONFIRMED:
run `jbqm2` literally degenerates to "...a 'Public constraint summary' and a 'Public constraint
summary' and..." ×hundreds by step ~8.)

**NEW EXPERIMENT (built + validated offline, launch-ready) — ALGORITHMIC enumeration-think SFT.**
The whole bottleneck is generating the consistent-candidate list in-head (scaffold→0.97, none→0.22-0.47).
The **env computes that list + the exact best split for any state** — so we don't need a noisy LLM teacher.
`standalone/generate_wordle_algo_think_gold.py` (offline, no GPU) templates a clean, terminating,
fully-correct private think per turn: restate constraints → ENUMERATE the env's consistent words → pick
the lowest-expected-remaining public splitter → close `</think>` → public line. Dataset built:
`results/wordle_gold_sft/algo_think_v1_20260613T175144Z/gold.jsonl` (1973 turns, 512 targets, **full
coverage**, 0.98 env-policy solve, every row verified consistent, think mean ~135 tok vs Kimi's 353).
Drop-in for `build_gold_sft_rows` (objective=sft_gold). 34/34 prompt tests pass (3 new generator tests).
Selection is purely public (no target leak; target tiebreak off by default).

**Launch (collision-free — sample_eval OFF, so it NEVER touches the live `opsd-wordle-q36-sglang`
samplers the perf agent owns):**
```bash
kubectl create -n apanda -f experiments/zorl/k8s/qwen3-6-35b-a3b-opsd-wordle-algosft.yaml
```
Fresh full-weight SFT from BASE (config `..._fullft_ep8.yaml`, **muon_lr 0.002** — the real lever vs the
failed adamw 1e-6), 300 steps, save every 50, on its own clean 8-GPU node. **Immediate gate: step-1 loss
must DROP (the 1e-6 run was flat) — if it does, the under-training theory is confirmed.** Then floor-gate
each saved ckpt (50/100/.../300) on IDLE serving vs the 0.469 base+think floor: does exact climb AND does
the model actually emit enumeration think (read transcripts). If algo-SFT teaches enumeration, that's the
win and the warm-start for any follow-on OPSD/GRPO. If it confabulates the list (writes a plausible-but-
wrong consistent set), that's the real "distillation can't teach enumeration" result — and points to GRPO
(reward verifies the guess, can't be faked). See memory `project_opsd_wordle_enumeration_is_the_skill`,
`project_opsd_wordle_sft48_was_noop`.

### ⭐⭐ PROBE RESULT (2026-06-13 evening) — the bottleneck is constrained RETRIEVAL, not reasoning
Probed the LIVE base model (teacher endpoint, no training) with `standalone/probe_wordle_enumeration.py`:
asked it to enumerate the consistent words for real mid-game states. Think-mode aggregate: **validity
0.18** (82% of words it lists VIOLATE the constraints it was given), recall 0.33, sprays ~67 words.
The traces are decisive: for `divot` it derives the constraints **flawlessly** ("T not in 1,4; O not in
2; absent A,C,E,L,N,S; must contain I,O,T") then **cannot retrieve the words** — loops "TITAN? No.
TITAN? No…" and never finds BIGOT/DIVOT/IDIOT/PIVOT; for `manes` (33 answers cakes/canes/games/…) it
found 0. **The logic is intact; constrained vocabulary RETRIEVAL is the deficit.** This is exactly why
scaffold→0.97 (filter/pick works) but no-scaffold→0.47 (endgame retrieval fails). Implication: the
algo-think SFT (pairs exact constraints→exact consistent list) targets this retrieval gap precisely, and
it's a memory/association problem (SFT's strength) NOT a reasoning failure. GRPO-alone would face a
sparse-reward exploration problem (model rarely samples the right endgame word) → SFT-first is the order.

### NEW ASSETS + STATE (2026-06-13 wrap-up)
- **Generator** `standalone/generate_wordle_algo_think_gold.py` (offline, deterministic, no GPU/network)
  — env-computed enumeration think. Added `--exclude-eval-{seed,count}` so floor-eval targets are held
  out for a clean generalization test.
- **Broad gold** `results/wordle_gold_sft/algo_think_v2_broad_20260613T215318Z/gold.jsonl` — **15,986
  turns / 4,138 targets**, full word-list coverage, seed-777 floor-eval set excluded (0 leakage verified,
  0 bad rows, ~148 tok/completion). (v1 512-target = `algo_think_v1_20260613T175144Z`, superseded.)
- **Manifests**: `k8s/qwen3-6-35b-a3b-opsd-wordle-algosft.yaml` (8-GPU) and `…-algosft-4gpu.yaml`
  (4-GPU EP=4, fits fragmented capacity). Both: objective sft_gold, base model, muon_lr 0.002,
  ce_mode compiled (SFT-safe — NOT quack_linear), **SAMPLE_EVAL_INTERVAL=0 (collision-free, no sampler)**.
  NOTE the 4-GPU manifest needed the `--gold-data` passthrough added (older asym template lacked it).
- **🚧 OPEN: the SFT run HANGS at the first forward_backward+optimizer step.** Cold eval step=0 succeeds
  (loss **0.443** on held-out gold — already low, the structured think is predictable) but the first
  TRAIN step froze 33+ min with no progress (4-GPU EP=4 + fp8 + compiled CE; classic first-MoE-dispatch/
  backward hang — see `project_opd_mtp_moe_dispatch_oom` / `merge_opd_distributed_init_regressions`
  handoff). Stopped to free capacity for consolidation. **To resume:** retry with the documented
  workarounds (try `enable_fp8_training: false` and/or `ce_mode: compiled`→eager first; 60-min PG
  timeout; watch the first MoE alltoall). Then gate: step-1 loss must drop below 0.443; floor-eval saved
  ckpts (80/160/…/400) on the HELD-OUT targets to test whether retrieval generalizes vs memorizes.

---
## 🎯 HANDOFF — START HERE (2026-06-13 PM, for the next agent)

**Mission.** Get Qwen3.6-35B-A3B to *legitimately* play Wordle under the think contract
(think privately, then emit one public line `<reasoning>…</reasoning><guess>WORD</guess>`).
Capability ladder (64 games, seed 777, think, floor-protocol eval): base one-line **0.219**
→ base+think **0.469** → SFT-on-Kimi-gold+think **0.500** → candidates-hinted teacher **0.938**
→ candidates-SCAFFOLD ceiling **0.969**. The gap to close = the model doing candidate
enumeration *internally* (no scaffold).

**⚠️ Live runs are owned by a performance-focused agent** (handed off 2026-06-13 PM). Trainer
jobs as of handoff: `...mj4xb` (sampler-0, an old leak-free/inert guess-only holding run — may
be torn down) and `...rlp8p` (sampler-1, the perf agent's). **HARD RULE: two trainers must NOT
share a sampler** (each full-weight-syncs its own policy → crashes it). Coordinate before launching.
Serving stack (leave warm): samplers `opsd-wordle-q36-sglang-0/1` (TP=2), teacher
`opsd-wordle-q36-teacher-sglang`, gateway `opsd-wordle-q36-smg`.

**① THE foundational fix (in the working tree, tested — everything depends on it).** The
think-contract *training* rollout was dying at turn 1: the strict one-line `parse_turn_response`
rejected think output (the `<think>` block mentions the tags + trailing chatter), so `action_ok`
was always False → game broke at the opener → OPSD only ever trained the opener (where the
teacher has no candidate hint). Fix = `wordle.extract_action_text` (strip think → truncate after
first `</guess>` → drop preamble), now used by BOTH the OPSD rollout and the GRPO rollout. Games
now play turns 1-6. **31/31 prompt tests pass.**

**② Corrected science verdict — earlier "OPSD can't teach Wordle" was WRONG (I overclaimed).**
The three failures were one *fixable* pathology: **reverse-KL mode collapse**.
`OPD_LOSS_MODE=reverse_kl_full` is mode-SEEKING; training the long think tokens drove the policy
into a degenerate **repetition loop** ("…which is empty." ×hundreds) by step ~8. Signature:
`opd_kl` rose 0.042→0.060 then CRASHED to 0.019 with grad_norm 1.4→0.20. NOT a fundamental limit.
(The earlier "guess-only is inert" finding is real but orthogonal: scoring only the guess token,
conditioned on the student's own reasoning, gives ~0 gradient — the teacher's edge lives in the
REASONING, so you must train reasoning, with a loss that doesn't collapse.)

**③ The live hypothesis (what to actually run).** Two changes vs. the collapsing config:
(a) **forward-KL** — `OPD_LOSS_MODE=forward_kl_full` (mode-COVERING, the code default; counters
collapse). **Requires `OPD_KL_BACKEND=torch_compile`** (no streaming variant). (b) the user's
setup: a **candidate-scaffold-ONLY teacher** (the consistent-word list, NO computed best-guess)
that **generates its own CoT and grades the student's CoT** (`teacher_reason_first=1`, live CoT gen
via the teacher endpoint). The reason-first cache CoTs are clean + terminating ("DOERS narrows the
remaining 220 possibilities") — a good distillation target.

**Code changes in the working tree (uncommitted, per user policy):**
- `standalone/tasks/wordle.py` — `extract_action_text` (the ① fix; shared by eval + training).
- `standalone/train_opsd_baseline.py` — new teacher style **`public_candidates_only`** (the
  `candidates_only` branch in `_wordle_policy_hint_user_content`: candidate list, no answer, teacher
  reasons to pick); reason-first guard widened to accept it.
- `standalone/train_grpo_wordle.py` — think-aware rollout fix + **full-weight port** (`--full-weight`,
  `register_inference_endpoints`+`sync_weights_to_samplers`, gated `lora_path`).
- `standalone/test_wordle_opsd_prompts.py` — +3 `extract_action_text` regression tests, 3 updated
  for the public-reframe wording. (31/31 pass: `PYTHONPATH=. <repo>/../xorl-internal/.venv/bin/python
  -m pytest -q experiments/zorl/standalone/test_wordle_opsd_prompts.py`.)

**Experiment manifests (saved to `experiments/zorl/k8s/`):**
- `…-reasonfirst-candonly-fwdkl.yaml` — **THE experiment**: SFT-48 warm-start → reason-first OPSD,
  `public_candidates_only` teacher, forward-KL. Built + validated; **NOT yet launched** (was gated
  on the torch_compile check below).
- `…-fwdkl-control.yaml` — forward-KL control (answer-giving teacher) to isolate "does forward-KL
  alone stop the collapse?".
- `…-guessonly.yaml` — leak-free/inert holding config.
- `…-think-canonical.yaml` — the base canonical (reverse-KL + answer-giving teacher). **KNOWN TO
  COLLAPSE — do not run as-is; it's the negative control.**

**🚧 OPEN VALIDATION (gate before trusting forward-KL):** does `torch_compile` (forward-KL's required
backend) survive the variable-length think rows without a recompile crash/hang? I launched the
forward-KL control to test this but **did not confirm its first `forward_backward` completed**
before handoff (the pod was killed/rescheduled during a capacity event). **Verify a forward-KL run
reaches `[train step=1]` cleanly before trusting it.** If torch_compile thrashes on varlen, the
fallback is to keep reverse-KL but prevent collapse another way (much lower LR, KL-regularize /
trust-region toward the SFT-48 warm-start, or cap think length).

**④ GRPO — parallel path, ported + ready, blocked by one dep.** Reward-based (no teacher → no leak;
optimizes correct *termination* directly, which distillation can't supply). `train_grpo_wordle.py`
is full-weight-ported + rollout-fixed and the server supports its loss (`importance_sampling`/`n` in
`src/xorl/ops/loss/`). **Blocker:** it imports `xorl_client.rl.{advantages,datums}`, and the
installed `xorl_client` (git-pinned @2a3a60a7) lacks the `rl` submodule. To launch: install an
xorl-client build that has `rl` (CAUTION: shared venv at `xorl-internal/.venv` — verify it doesn't
break the running OPSD/sampler infra; consider a separate venv), then run `train_grpo_wordle.py
--full-weight` with the q36 sampler URLs + SFT-48 warm-start.

**Next-step decision tree:**
1. Confirm a forward-KL run reaches step 1 (torch_compile survives). 
2. Run the reason-first candidate-only experiment. **Gate at steps 8/16/24 (floor protocol, NOT
   the noisy in-trainer sample_eval): format must HOLD ~0.5 (no repetition collapse) and exact must
   climb off 0.** Read transcripts every gate (numbers hide collapse/confabulation).
3. If forward-KL still collapses → lower LR / trust-region toward warm-start / cap think length.
4. In parallel, unblock + launch GRPO (④).

**Throughput note (secondary):** the multi-turn fix made steps ~2.5× slower (~700-860s/step, 281k
tokens/step vs ~49k turn-1-only) — expected, it's training turns 2-6 now. The chunk16-vs-chunk64
sweep below was measured on the BUGGY turn-1-only code; treat it as unverified post-fix. Throughput
is NOT the bottleneck — the science (anti-collapse loss + teacher design) is.

**Key checkpoint:** SFT-on-Kimi-gold step-48 (the warm-start for all OPSD runs) =
`results/opsd_wordle_native_baseline/20260612T160347Z-…sft_gold/server_output/weights/default/step-000048`.

---
## ⭐ OVERNIGHT HANDOFF (2026-06-13) — your job: iterate the Wordle loop

**MISSION**: get a Qwen3.6-35B-A3B to *legitimately* play Wordle under the think
contract. Train, gate, read transcripts, decide the next move, repeat. The infra
is solved (below) — spend your tokens on the SCIENCE (closing the solve-rate gap),
not on relaunching for throughput.

> ## 🧭 CORRECTED VERDICT (2026-06-13, apanda) — earlier "OPSD can't work" was WRONG (overclaimed)
> The three failures I saw were all the SAME fixable training pathology, **not** "distillation
> can't teach Wordle". On re-reading the traces (user pushed back, correctly):
> 1. train reasoning (private `policy_hint` teacher) → leak/confab of the private reference.
> 2. guess-only → inert (teacher agrees with the guess given the student's own reasoning).
> 3. public-reframe (clean teacher, train reasoning+think) → **the "spiral" is actually a
>    DEGENERATE REPETITION LOOP** ("…which is empty." ×hundreds), not real enumeration.
>    `opd_kl` rose 0.042→0.060 then CRASHED to 0.019 w/ grad_norm 1.4→0.20 by step 7-8 =
>    textbook **reverse-KL mode collapse** (`OPD_LOSS_MODE=reverse_kl_full` is mode-SEEKING).
>
> What I had NOT tried (and what likely works): **forward-KL** (mode-covering, the code
> default — counters collapse; needs `opd_kl_backend=torch_compile`) + the user's setup:
> SFT-48 warm-start → reason-first OPSD with a **candidate-scaffold-ONLY teacher** (no
> computed answer) that **generates its own CoT and grades the student's CoT**. The
> reason-first cache CoTs are clean+terminating ("DOERS narrows the remaining 220").
>
> **LIVE EXPERIMENTS (2026-06-13 PM):**
> - **node B `s4c2s`** (sampler-1): forward-KL control — public-reframe teacher + `forward_kl_full`
>   + torch_compile. Tests "does forward-KL alone stop the collapse?" + validates torch_compile
>   survives varlen think rows.
> - **node A `<pending>`** (sampler-0): the USER'S full spec — `public_candidates_only` teacher
>   (new style: candidate list, NO best-guess) + `teacher_reason_first=1` (live CoT gen via teacher
>   endpoint) + `forward_kl_full` + torch_compile + warm-start SFT-48. New code: `candidates_only`
>   branch in `_wordle_policy_hint_user_content` (31/31 tests pass).
>
> GRPO remains a valid parallel path (ported, needs `xorl_client.rl` install) but is NOT the
> only option — getting OPSD to work with forward-KL + reason-first is the active goal.
> The **multi-turn rollout fix** (`extract_action_text`) is the reusable win underneath all of this.

### ⭐⭐ ROOT CAUSE FOUND (2026-06-13 ~10:30Z) — training rollout died at turn 1
**The "KL falls, play stays flat" pattern was a rollout bug, not a science dead end.**
Under the think contract the *training* multi-turn rollout never advanced past turn 1:
every game broke after the opener. Evidence: all 304 training rows in `...tb5t8...`
were `turn=1`, 0 solved; 324/336 had `valid_guess=False` with errors `[guess_tag_count,
reasoning_tag_count, not_exact_one_line_response, extra_line_or_trailing_text]`. Cause:
`build_multiturn_asymmetric_opsd_rows` judged turn validity (`action_ok`, which gates
`if not valid_guess: break` at train_opsd_baseline.py:1183) by running the strict
one-line `parse_turn_response` on the **raw full text** — but think output wraps a
`<think>` block that mentions the output tags and emits trailing chatter, so the parser
always failed. Consequence: OPSD trained **only the opener** — and turn 1 is exactly
where the teacher carries **no candidate hint** (candidates omitted on turn 1), so the
entire value of the candidate-hinted teacher (turns 2-6) was never sampled. Fully
explains every prior flat think-contract OPSD run.

**FIX (working tree, tested 31/31 + new regression tests)**: added
`wordle.extract_action_text` (strip think → truncate after first `</guess>` → drop
preamble) — the same think-aware extraction the held-out `rollout_completion` already
used; both eval and training rollouts now call it. Functional check: think-contract
turn-1 output parses `action_ok=True` (was False); constraint-violating guesses still
rejected. **VALIDATION GATE: confirm `generations.jsonl` now has turn≥2 rows** (was
100% turn-1). If solve-rate then climbs, the fix was the unlock.

**LIVE (2026-06-13 ~10:25Z)**: two fixed warm-start runs (two seeds) — `...jmfd6`
(S1, sampler-1, h100-099) and `...tk6k9` (S0, sampler-0, h100-092), chunk64+RB64
quack_linear from SFT-48. NOTE: launched at chunk64 (the throughput sweep below that
says chunk16 is best was measured on the BUGGY turn-1-only code; re-measure post-fix
at step 2 before trusting it). Throughput is secondary — the fix is the point.

**✅ FIX VALIDATED (2026-06-13 ~10:40Z)**: jmfd6 step-1 train rollout produced
**145 rows from 64 games** (turn dist 80/77/24/1 for turns 1/2/3/4) vs the pre-fix
**64 turn-1-only rows** — and **1 solve already** (`stool` @ turn 4, cold warm-start
policy). OPSD now trains turns 2-4 where the teacher's candidate hint lives. Cost
shifted: 281,855 tokens/step (was ~49k) → fb batches into ~3 calls @ RB64 → slower
steps + OOM watch. NEW lever now visible: many game-ending turns are think-spirals
past STUDENT_MAX_NEW_TOKENS=3072 (later-turn enumeration wants more budget) — raising
it lets games go deeper at higher token cost. The real test now = does sample_eval/
floor solve-rate climb over steps.

**THROUGHPUT VERDICT post-fix (step 1, both runs, 2026-06-13 ~10:56Z)**: dt≈1432-1477s
(was ~556s turn-1-only), fb≈1000-1082s, **fb_req=3** (RB64 on 145 rows), ~270-280k
tokens/step, **NO OOM** at chunk64. ~2.5× slower — the cost of training turns 2-6.
(step 1 includes compile warmup; step 2 is the steady number.) At ~1000-1400s/step the
overnight deliverable is the sample_eval trend at steps 8/16/24 (does `exact` climb off
0) + a floor-gate if step-32 checkpoint lands; NOT a long run. Throughput levers if a
future agent wants more steps: lower STUDENT_MAX_NEW_TOKENS (spirals dominate the 281k
tokens), lower train_size, or test chunk16 (prior "chunk16 2× faster" was pre-fix —
unverified post-fix). Decided NOT to thrash these tonight — the science signal comes first.

### ⭐⭐ SECOND FINDING (2026-06-13 ~12:15Z) — asymmetric-OPSD LEAK → guess-only pivot
The multi-turn fix unblocked training, and by **step 8 the policy DEGRADED**: sample_eval
format collapsed 0.47→0.20, turns 1.94→1.44, and transcripts showed the student
**confabulating the teacher's private context** — generating "Wait, the private info says:
'Private target: STARE'" and arguing with itself about a "private prompt" it never received
(student prompt is public-only). Pervasive in BOTH seeds, in BOTH training rollouts
(~10-13k "private" mentions / ~1.1k rows) and eval.

**Mechanism**: `asymmetric_opsd` scores the student's sampled tokens under the TEACHER
context, which contains the private reference block (`Reference next guess: STARE`,
candidate list). reverse-KL on the **think/reasoning** tokens pulls the student to emit
private-ref-consistent tokens → confabulation runaway → format collapse. The private ref
is REQUIRED for the teacher's informed guess but leaks into the reasoning. Reason-first
would NOT fix it (private ref stays in the scoring context).

**PIVOT (live ~12:21Z): GUESS-ONLY weights.** Set `WORDLE_REASONING_TOKEN_WEIGHT=0` +
`WORDLE_THINK_TOKEN_WEIGHT=0` (new manifest defaults; added the missing
`--wordle-think-token-weight` arg). Now ONLY the guess (action) token is distilled — the
teacher's private-ref edge informs the GUESS (exactly what we want), with no leak vector
in the trained tokens. The student still thinks freely (sampled, untrained). Runs:
`...t9szb` (S1/sampler-1/h100-099) + `...xjr6j` (S0/sampler-0/h100-092), chunk16 (reverted
by linter; it's the faster config per the sweep), warm-start SFT-48. **Validation:
training rollouts must stay leak-free (no "private" growth) AND sample_eval must not
collapse / solve-rate climb.** If guess-only plateaus (action signal too sparse to teach
enumeration), next lever = reason-first with the CLEAN public CoT cache (verified clean:
"...broad opener...maximize information gain"), accepting it may still partially leak.

### ⭐⭐ THIRD FINDING (2026-06-13 ~13:00Z) — guess-only INERT [⚠️ "OPSD stuck" CORRECTED — see START HERE: it was reverse-KL mode collapse, fixable with forward-KL + reason-first]
- **Guess-only result**: at step 8, format held ~0.50 (NO collapse — leak fixed ✅), but
  **opd_kl ≈ 0.0002, grad_norm ~0.02 → INERT** (t9szb 1/16, xjr6j 0/16 = n=16 noise). Why:
  the teacher scores the guess token *conditioned on the student's own reasoning*, so it
  agrees with whatever guess the reasoning leads to → ~zero gradient. **The teacher's
  enumeration edge flows through the REASONING, not the isolated guess.**
- **OPSD-with-private-teacher is structurally stuck**: train reasoning → LEAK; train
  guess-only → INERT. So I split the difference: **PUBLIC-REFRAME** — reworded
  `public_policy_hint` to drop ALL "Private reference / Reference next guess / Private
  teacher note / Do not reveal" framing → now "Public candidate analysis /
  `<candidate_analysis>` / Strongest guess: X / Why it narrows best". Candidate info is
  public (derivable from feedback), so training reasoning toward it teaches enumeration
  WITHOUT a private string to confabulate. Re-enabled reasoning+think weights (1.0).
  Tests: 31/31 (3 updated for new wording). **LIVE ~13:55Z: `...jbqm2` (S1/sampler-1) +
  `...tpprd` (S0/sampler-0), PUBLICREFRAME, chunk16, warm-start SFT-48.** Watch: does it
  teach enumeration (format stable + exact climbs) or just CONFABULATE THE FORM ("N
  candidates fit" without computing — the known distillation failure)? Either result
  closes the OPSD question.

### ⭐⭐ GRPO STATUS (2026-06-13) — ported + ready, BLOCKED by a missing dep
GRPO is the principled path (reward teaches enumeration SUBSTANCE, not form; no teacher →
no leak). **I fixed GRPO's identical turn-1 rollout bug** (think-aware truncation +
`extract_action_text`) **and ported it to full-weight** (added `--full-weight`,
register+sync via OPSD's funcs, gated `lora_path`; compiles). The server DOES support the
loss (`importance_sampling`/`n` in `src/xorl/ops/loss/`). **BLOCKER**: GRPO imports
`xorl_client.rl.{advantages,datums}` which is NOT installed (the git-pinned `xorl_client`
@2a3a60a7 lacks the `rl/` submodule; no checkout has it). To launch GRPO: install an
`xorl-client` build that has `xorl_client.rl` (CAUTION: shared venv — don't break live
runs; use a separate venv or verify compat), then run train_grpo_wordle.py with
`--full-weight` + the q36 sampler URLs + warm-start SFT-48. This is the clear high-EV
next step for SUPERVISED work.

### Where the science stands
Capability ladder (64 games, seed 777, think contract, floor protocol):
`base one-line 0.219 → base +think 0.469 → SFT-on-Kimi-gold +think 0.500 →
Kimi-K2.6 teacher (think,t1.0) 0.938 → candidates-scaffold ceiling 0.969`.
- SFT on 1094 Kimi gold turns transferred guess DISCIPLINE (invalid-guess deaths
  31→23) but not enumeration CAPABILITY (+3pts, noise). Checkpoint =
  `results/opsd_wordle_native_baseline/20260612T160347Z-...sft_gold/server_output/weights/default/step-000048`.
- The open question: does **OPSD with the candidates-hinted teacher** (warm-started
  from SFT-48) convert the teacher's 0.94 into student solves? That's the live run.

### CANONICAL INFRA (post-throughput-work — DO NOT re-litigate these)
- **Branch**: `apanda-dev-prefill-time-compute` (consolidated with the sibling OPD
  stack; pushed to shared remote — both envs work off it).
- **Launch (single 8-GPU trainer)**: `kubectl create -n apanda -f
  experiments/zorl/k8s/qwen3-6-35b-a3b-opsd-wordle-think-canonical.yaml`
  (warm-starts from SFT-48; config `configs/..._ep8_warmstart_sft48.yaml`).
- **THROUGHPUT IS ~0.13% MFU AND THE EASY LEVERS ARE EXHAUSTED (measured).** fb
  is ~91% of step (~106s) and is **memory-bandwidth-bound** (GPUs 75-94% util but
  only ~140W of 700W TDP = tiny memory-bound kernels, not compute). The
  request-batch / call-count lever was swept and is a DEAD END for our regime:
  | config | calls/step | MFU | fb-ms/1k-tok |
  |---|---|---|---|
  | **RB8 + chunk16 (CANONICAL, original)** | 8 | **0.13%** | ~2100 |
  | RB64 + chunk16 | 4 | 0.10% | ~3400 |
  | RB64 + chunk64 (1 call) | 1 | 0.054% | ~4187 |
  Fewer/bigger fb calls are **monotonically WORSE** (more per-call HBM traffic >
  the per-call FSDP cost saved). The sibling stack's "1.25MB→64MB chunk cap"
  fix (`ada302bc`) does NOT transfer — their regime was short completions
  (mnt=64); ours is long variable-length think (~3k tok/turn) + full-vocab KL.
  **So keep RB8 (manifest default) + chunk16.** Do NOT re-run the RB/chunk sweep.
- **Levers measured to NOT help (don't waste a run)**: request_batch/chunk size
  (above); quack vs triton MoE (sibling proved triton+deepep==quack+alltoall —
  kernels aren't the bottleneck); KL backend (`streaming` optimal; `torch_compile`
  WORSE on varlen); cheaper/top-k KL (KL is <2% of fb — measured via
  `--opd-profile-timings`: kl_compute ~1.8s/step). EP-dedup (`ff262108`) ON
  (gradient-identical; wall-neutral at single-EP-group but correct).
- **The ONE genuinely-untested lever worth trying for MFU**: `enable_gradient_
  checkpointing: false` (recompute_before_dispatch currently ON). The bottleneck
  is memory-bandwidth, and grad-ckpt recompute ADDS activation-read traffic; turning
  it off trades memory capacity for less traffic. A prior ckpt-off run FIT at step 1
  (g7w2f) before being killed for capacity — so it's memory-feasible at chunk16 and
  is the next thing to measure if chasing MFU. (Pure-training hits 16% MFU but with
  static 8k-packed big batches + adamw + compile — structurally unavailable to
  variable-length small-batch server OPSD. Do not expect to approach 16%.)
- **Serving** (leave warm): samplers `opsd-wordle-q36-sglang-0/1` (TP=2 each),
  teacher `opsd-wordle-q36-teacher-sglang`, gateway `opsd-wordle-q36-smg`. The
  canonical run talks DIRECT to one sampler (`SAMPLER_LOAD_URL`/`INFER_URL` =
  sglang-1, `EXPECTED_SMG_WORKERS=0`). **HARD RULE: two trainers must NOT share a
  sampler** (each full-weight-syncs its own policy → crashes it). One trainer per
  sampler.
- **Scale to 16 GPU (~2× more, optional)**: staged config
  `configs/..._fullft_2x8_warmstart_sft48.yaml` + gang manifest pattern
  `k8s/qwen3-6-35b-a3b-opsd-wordle-fullft-2x4-think-gang.yaml` (bump to 2×8). Needs
  TWO clean 8-GPU nodes. Compounds with RB64. Cross-node has NCCL fragility — only
  if a clean run is wanted.

### GATE PROTOCOL (how to judge — this is load-bearing)
- **Judge by the floor-protocol harness eval, NOT the in-trainer sample_eval**
  (3072-budget → think-spirals → noisy/flat, useless as a level).
  `cd experiments/zorl/standalone && PYTHONPATH=. python eval_wordle_sglang.py
  --base-url <idle-sampler-or-frozen-ckpt> --model Qwen/Qwen3.6-35B-A3B
  --num-games 64 --seed 777 --prompt-style public_reasoning_constraints_think
  --temperature 0.2 --max-new-tokens 12288 --no-ignore-eos --invalid-retries 2
  --batch-size 16` vs the **0.469** floor.
- **mnt ≤ 12288** vs pool samplers (`--max-total-tokens 16384` cap → 16384 400s).
- **NEVER eval against a LIVE trainer's sampler** — 64×12k-token thinks crash it
  and kill the trainer's weight sync. Use idle serving or throttle `--batch-size`.
- **READ THE TRANSCRIPTS every gate** (`games.jsonl`) — numbers hide confabulation
  / spirals. Standing rule.

### THE DECISION TREE (what to actually iterate tonight)
1. Run warm-start OPSD (the canonical manifest) ~50-100 steps; floor-gate every
   ~50 steps vs 0.469 (on idle serving, not the live sampler).
2. **If solve-rate climbs toward the teacher's 0.94** → keep going, that's the win.
3. **If flat** (KL falls, play doesn't — the from-base pattern), pull a lever:
   - **reason-first OPSD**: cache is BUILT + UNUSED —
     `results/wordle_teacher_cot/q36_cot_cache_v0_20260611T200929Z/cot_cache.jsonl`
     (1262 states); trainer flag `--teacher-reasoning-cache <path>` +
     `--teacher-reason-first`. The stated-preference design (teacher = prompt +
     candidates hint + teacher CoT). Note note: notes sometimes phrase the ref
     guess as "the student's guess" — read them.
   - **GRPO from SFT-48**: `standalone/train_grpo_wordle.py`; format-gate worry
     resolved (SFT gives 0.94 format). Dense reward (0.50 base solve).
   - **More/better gold (SFT v2)**: Kimi (TP=16, redeploy via
     `k8s/kimi-k26-gold-sglang-gang.yaml`) regenerate the 221 unsolved targets at
     eval-parity rules (~+800 turns). Kimi solves 0.94 under the think contract.

### CAPACITY OPS (cluster is contended — these are hard-won)
- Clean nccl-group nodes work for single-node trainers; **never tolerate
  `cuda-error`/`weka-error`/`unschedulable` taints** (those "free" 8-GPU nodes are
  broken). Verify a node: `kubectl get node <n> -o jsonpath='{.spec.taints}'`.
- **Relaunch = create-the-replacement-FIRST (let it camp), THEN delete the old** —
  delete-first opens a vulture window (we lost nodes that way repeatedly).
- Volcano gang needs explicit `schedulerName: volcano` (Kyverno's anchor can't
  override the API-server default).
- Pool pods have NO readinessProbe: `1/1 Running` ≠ serving; gate on `/v1/models`.

---
>
> **CURRENT STATE (2026-06-13 ~02:30Z)**
> - Goal: a 35B that legitimately plays Wordle (think contract). Not married
>   to OPSD vs GRPO; whichever converges.
> - **CAPABILITY LADDER (64 games, seed 777, think contract, floor protocol)**:
>   base one-line 0.219 → base +think 0.469 → **SFT-on-Kimi-gold +think 0.500**
>   → Kimi-K2.6 teacher (think, t1.0) 0.938 → candidates-scaffold ceiling 0.969.
> - **Kimi gold pipeline VALIDATED**: Kimi-K2.6 under think contract (t1.0,
>   mnt 16384, EOS-only) solves 0.938 with 0 illegal guesses. Gold set =
>   `results/wordle_gold_sft/kimi_k26_think_t10_20260612T003446Z/gold.jsonl`
>   (291/512 solved, 1094 think+answer turns, 5 leak-flagged). Traces show
>   real constraint algebra (quality-read OK).
> - **SFT-on-gold checkpoint**: 48 steps, gates 0.500 vs 0.469 floor —
>   +3pts (n=64 noise) on exact BUT real failure-mode shift (invalid-guess
>   deaths 31→23, valid_guess 0.877→0.904). Verdict: SFT transferred guess
>   DISCIPLINE, not enumeration CAPABILITY. Ckpt at
>   `.../20260612T160347Z-...sft_gold/server_output/weights/default/step-000048`.
> - **LIVE NOW**: (1) OPSD warm-start from SFT-48 (`8g-jlhjf` @ h100-005,
>   sampler-1, momentum-0, the plan-of-record run); (2) OPSD from-base
>   control (`8g-pc8pj` @ h100-049, sampler-0, step 130+ — DEFINITIVE
>   NEGATIVE: KL descends, sample_eval play flat through 128 steps).
> - **UNUSED LEVERS** ranked: reason-first OPSD (1262-state CoT cache built,
>   never run — stated-preference design); GRPO from SFT-48 (format 0.94 gate
>   resolves the format worry); SFT v2 on eval-parity-regenerated gold
>   (~+800 turns, needs Kimi redeploy).
> - **GATE PROTOCOL**: judge by 64-game floor-protocol harness eval vs 0.469,
>   NEVER the in-trainer sample_eval (3072-budget, think-spirals → flat/noisy)
>   and NEVER against a LIVE trainer's sampler (crashes it; throttle batch<=8,
>   mnt<=12288 for pool token cap, prefer frozen ckpt on idle serving).
> - **BRANCH CHANGED 2026-06-13**: consolidated with the prefill-time-compute
>   OPD stack. Now on **`apanda-dev-prefill-time-compute`** (was
>   `apanda-dev-wordle`) — a conflict-free merge of both branches, pushed to the
>   shared remote so both envs work off it. Our Wordle work lives in
>   `experiments/zorl/` (disjoint from their `experiments/opd_profile/`); we
>   inherited their `src/xorl` OPD improvements + reprogrammable-slots launcher +
>   throughput-tuner skill. Tests green post-merge (OPD runner 14/14, wordle 28/28).
> - Stale-reputation corrections: h100-005 IB is FINE (Mooncake sync works);
>   -087 mounts/sched FINE. Capacity ops: create replacement job FIRST (let
>   it camp), THEN delete old — delete-first opens a vulture window.

## 2026-06-10 Takeover: branch, probe results, candidate scaffold, Qwen3.6 retarget

All Wordle work now lives on branch `apanda-dev-wordle` (off `apanda-dev` + the
local OPSD server fixes; created from `opsd-wordle-apanda-dev-run-20260607`
HEAD). `experiments/zorl` code is committed there; `results/` stays gitignored.

New since the wind-down:

1. **Teacher-prompt probe harness** — `experiments/zorl/standalone/probe_wordle_teacher.py`.
   Scores paired responses under teacher vs student contexts via SGLang prompt
   logprobs (no trainer needed). Tests: public-vs-leaking rationale preference,
   reference-vs-alternative action signal, first-turn target invariance,
   on-policy per-token-class (tag/reasoning/guess) teacher-student gaps.
   Results on the 30B stack (`/tmp/probe_{public,legacy}_policy_hint_full.json`,
   24 mid states, 24 first-turn targets, 64 student samples):
   - `public_policy_hint`: public-minus-leak reasoning **+39.6** (fair), ref-minus-alt
     guess **+12.8** vs student −1.9 (strong usable advantage), ref-minus-repeat
     +16.9, first-turn target-minus-opener −26.7 (no oracle leak). On-policy gaps:
     guess −5.1/tok, reasoning −2.4/tok, tag −0.23/tok (signal flows to content).
   - legacy `policy_hint`: public-minus-leak reasoning **−26.5** — the oracle teacher
     actively preferred target-leaking rationales the student prompt forbids; and a
     weaker action signal (+4.1). Mechanistic explanation for the 304-step 0/16 run,
     on top of the decode corruption.
2. **Candidate-enumeration scaffold** — new student prompt style
   `public_reasoning_constraints_candidates` adds the public consistent-word list
   to the shared student/teacher prompt block (prefix-aligned; teacher block
   matches via `student_prompt_style` passthrough). A/B on 30B-Instruct, 64 games,
   seed 777, retry 2: exact **0.969 / format 1.0** with candidates vs
   **0.078 / 0.991** without. Candidate enumeration is essentially the whole
   bottleneck. Training implication: the no-candidates setting stays the OPSD
   science arm (teach internal enumeration); the scaffold is a curriculum/GRPO
   option and a capability ceiling proof.
3. **Qwen3.6-35B-A3B retarget** — goal model. Arch is `qwen3_5_moe`
   (40 layers, 3:1 GDN linear-attention:full-attention, 256 experts); weights at
   `Qwen/Qwen3.6-35B-A3B` in the shared HF cache. New parallel serving stack
   (30B pool left untouched):

   ```bash
   experiments/zorl/k8s/qwen3-6-35b-a3b-opsd-wordle-sglang-pool.yaml      # opsd-wordle-q36-sglang ×2 (TP=2, flashinfer, mem-frac 0.75, attn-only LoRA r16)
   experiments/zorl/k8s/qwen3-6-35b-a3b-opsd-wordle-teacher-sglang.yaml   # opsd-wordle-q36-teacher-sglang (TP=2, hidden cache)
   experiments/zorl/k8s/qwen3-6-35b-a3b-opsd-wordle-smg.yaml              # opsd-wordle-q36-smg
   experiments/zorl/k8s/qwen3-6-35b-a3b-opsd-wordle-baseline-4gpu.yaml    # 4-GPU EP=4 trainer job
   experiments/zorl/configs/qwen3_6_35b_a3b_opsd_wordle_ep4_deepep.yaml   # trainer config
   ```

   Chat template: `enable_thinking=False` renders a closed empty `<think></think>`
   (safe); never rely on the default (open `<think>` — PTC-118 class hazard).

4. **FULL-WEIGHT pivot (user-directed, 2026-06-10)** — no LoRA anywhere; plain
   gradient OPSD/GRPO; the focus is a working env, not the algorithm.
   - A LoRA smoke test (`make_smoke_lora_q36.py` no-op adapter) had exposed an
     sglang `qwen3_5` LoRA bug first: o_proj lora_A is not TP-sliced on load
     (buffer `[r,2048]` vs weight `[r,4096]` assertion → scheduler SIGQUIT).
     The model class lacks `get_hidden_dim`/per-arch LoRA hooks.
     **FIXED 2026-06-10** in xorl-sglang-internal `apanda-dev` @ `bc1ffb583`
     ("Fix LoRA serving on Qwen3.5/3.6-MoE hybrid models"): `_lora_pattern`
     override (attn projections sit directly on the decoder layer, so o_proj
     now gets LoRA-wrapped → `RowParallelLinearWithLoRA.slice_lora_a_weights`
     TP-slices lora_A) + per-arch `get_hidden_dim` (models `attn_output_gate`
     doubling the q section). Validated TP=2: zero-B adapter base-parity 3/3
     greedy. Pool must run a post-bc1ffb583 SGLang before any LoRA arm.
   - `train_opsd_baseline.py --full-weight`: creates the session without LoRA,
     registers each sampler shard via `POST /add_inference_endpoint`
     (host/port/world_size=TP), and replaces adapter export/load with
     `POST /api/v1/sync_inference_weights` (method `nccl_broadcast`, pause_mode
     retract, flush_cache true) on every `ensure_sampler` policy refresh.
     Generation omits `lora_path` — the synced base weights are the policy.
   - Trainer: single 8-GPU node, `qwen3_6_35b_a3b_opsd_wordle_fullft_ep8.yaml`
     (EP=8, shard=8, adamw + `optimizer_dtype: bf16`, grad ckpt on, triton MoE,
     `enable_lora: false`, `sync_inference_method: nccl_broadcast`) +
     `k8s/qwen3-6-35b-a3b-opsd-wordle-fullft-8gpu.yaml`. Mirrors the proven
     full-FT OPD recipe (see `apanda-dev-prefill-time-compute`
     `experiments/opd_profile/autoresearch/CANONICAL_RUNBOOK.md` §O3; their
     2-shard bf16 sync = 2.4s, and Mooncake env hazards apply only to the
     `p2p` method, not `nccl_broadcast`).
   - q36 sampler pool runs WITHOUT LoRA flags (full-weight receivers).

5. **Qwen3.6-35B-A3B env baselines (teacher endpoint, 64 games, seed 777,
   retry 2, temp 0.2)** — `results/wordle_eval/q36_base_64_*`:
   - `public_reasoning_constraints` (hard setting): exact **0.219**, format 0.979,
     valid-guess 0.927 — much stronger floor than 30B-Instruct (0.078) or
     Q3.5-35B (0.133).
   - `public_reasoning_constraints_candidates`: exact **0.969**, format 1.0 —
     same enumeration-given ceiling as 30B.
   - Teacher probe on Q3.6 (`/tmp/probe_q36_public_policy_hint.json`):
     public-beats-leak +17.9 (student −32.2), ref-beats-alt +5.1 (student ~0),
     ref-beats-repeat +8.5, first-turn target −21.1. Fair and informative.
   - Training headroom: 21.9% → ~97% ceiling; the KL signal exists and flows to
     guess/reasoning tokens.

6. **Capacity reality (2026-06-10 ~19:10Z)**: h100-040's 8 free GPUs were taken
   by the shared `zorl-ar-sglang` pool minutes after the 8-GPU job launched.
   The 8-GPU adamw job (`opsd-wordle-q36-fullft-8g-*`) stays QUEUED to catch a
   freed node. The live run is the 4-GPU variant
   (`opsd-wordle-q36-fullft-4g-*`, EP=4/shard=4): 35B full-FT fits on 4 GPUs
   only without optimizer state → muon with `muon_momentum: 0.0` (skips the
   momentum buffer entirely) + `muon_lr: 0.002`; embeddings/lm_head/norms fall
   back to AdamW groups internally (small).


This is the handoff for the gradient-based OPSD Wordle baseline currently running in Kubernetes. It is intentionally self-contained: a new agent should be able to continue monitoring, debug failures, or launch the next controlled run without using chat history.

## Current Status

The run was restarted on 2026-06-10 (user-directed) to fix the oracle-like teacher. The previous `policy_hint` run (`wt6wm`, W&B 9p9ophek) reached step 304 with `sample_eval exact=0/16` throughout; its `Reference next guess` was the private target after turn 1, so teacher behavior was not reproducible from the student's public context. The new run uses `WORDLE_TEACHER_PROMPT_STYLE=public_policy_hint`:

- The private reference block no longer contains the target word at all.
- `Reference next guess` is computed by a public candidate-splitting policy (minimize expected remaining candidates over the public candidate set; letter-coverage proxy above 200 candidates). The private target is used only to break near-ties (within 5% of the best split score).
- `Reference public reasoning` is informative: it states the public candidate count and why the guess narrows it (e.g. "54 candidates fit the clues; SABLE narrows them most.").
- Offline policy-rollout check: the public reference policy solves 64/64 sampled targets with tie-breaking (62/64 with the target fully ignored), avg 3.3 turns — a strong but public-reproducible teacher.

Code: `_wordle_public_policy_action` / `_wordle_policy_scores` in `train_opsd_baseline.py`; legacy `policy_hint` behavior is preserved for ablation.

Restart #2 (2026-06-10, user-directed): merged `origin/apanda-dev` into the checkout (branch `opsd-wordle-apanda-dev-run-20260607`, see Checkouts) and added `--attention-backend fa3` to the student sampler pool (teacher already had fa3). The first public_policy_hint run (`xv5xz`, W&B 1xnozd9o) ran to step 141 before this restart: KL declined 2.59→2.53, teacher diagnostics de-oracled (turns_used 3.375), and sample_eval produced the first-ever held-out solves (1/16 at steps 64 and 120, sporadic).

Restart #3 (2026-06-10 ~17:30Z, user-directed): refreshed the SGLang serving stack and refit the trainer to 4 GPUs.

- `xorl-sglang-internal` was already at origin/apanda-dev tip `5e9acb7db` ("MTP native multi-token decode + OPD/ZORL serving; fix overlap seq_lens desync") — no merge needed — but the running sampler pods had started at the previous tip `8bcf2653e` and were missing the **batched-decode corruption fix** (prepare_for_decode seq_lens desync under the overlap scheduler with concurrent load; corrupts batched decode on fa3 AND flashinfer). Our samplers run overlap-enabled at high concurrency, so student rollouts were exposed — prime suspect for why LoRA-OPD failed to learn while the user's full-weight training reproduced the behavior.
- Sampler pool, teacher, and SMG all restarted; samplers verified logging `git_head=5e9acb7db` + fa3.
- The 8-GPU trainer job (`46q8q`) sat Pending for 10+ hours — no schedulable node has 8 free GPUs (blockers are other tenants' 1-GPU pods on -058/-080; -087 has 8 free but is cordoned + known CSI-mount-bad). Deleted it and launched a **4-GPU / EP=4** variant instead (precedent: OPD trainer EP=4 fragmented-capacity fit). Expect roughly 2x step time vs the ~105s/step 8-GPU baseline.

New files (4-GPU variant; canonical 8-GPU files unchanged):

```bash
experiments/zorl/configs/qwen3_coder_30b_a3b_zorl_wordle_ep4_deepep.yaml
experiments/zorl/k8s/qwen3-coder-30b-a3b-opsd-wordle-baseline-4gpu.yaml
```

**WOUND DOWN 2026-06-10 (user-directed): no trainer is running. A new agent will take over the Wordle env/run.** The last trainer job (`4g-2nqn5`, W&B ypckknu8) was deliberately stopped at step 14 — not a failure. Its short trace on the fixed sampler stack: loss ~2.42-2.57, grad_norm ~0.32-0.37, ~108s/step on 4 GPUs, and notably `sample_eval format_rate` hit **0.97 (step 0) and 1.000 (step 8)** vs 0.57-0.94 on all pre-fix runs — evidence the sglang batched-decode corruption fix materially cleaned up student rollouts. Final log: `/tmp/opsd-wordle-4g-2nqn5-final.log`; run artifacts:

```bash
run_dir=experiments/zorl/results/opsd_wordle_native_baseline/20260610T173209Z-opsd-wordle-30b-native-asym-4g-2nqn5-9bfpj-wordle-r4-asymmetric_opsd
wandb=https://wandb.ai/together-research/zorl/runs/ypckknu8
```

Handoff state for the next agent:

- The dedicated serving stack was **left warm**: `opsd-wordle-sglang-0/1` (fa3, git_head `5e9acb7db`), `opsd-wordle-smg`, `opsd-wordle-30b-teacher-sglang` — all freshly restarted on the latest xorl-sglang-internal apanda-dev (includes the batched-decode seq_lens-desync fix). Scale these down only if the Wordle effort is abandoned.
- Trainer checkout is branch `opsd-wordle-apanda-dev-run-20260607` with origin/apanda-dev merged (see Checkouts).
- Launch manifests: 8-GPU canonical `qwen3-coder-30b-a3b-opsd-wordle-baseline.yaml` (could NOT schedule for 10+ h — no 8-free node) and the working 4-GPU variant `qwen3-coder-30b-a3b-opsd-wordle-baseline-4gpu.yaml` (+ `configs/..._ep4_deepep.yaml`, EP=4, schedules immediately, ~108s/step).
- Teacher prompt style is `public_policy_hint` (de-oracled; see Teacher Prompt section). The user has reproduced OPD behavior with full-weight training; the LoRA question (does rank-4 LoRA-OPD learn on the corruption-free stack?) is unresolved — the 14-step trace is too short to conclude anything beyond the format-rate improvement.

Science framing for this run: the user has reproduced the OPD learning behavior with full-weight training. This run is the LoRA (rank-4) reproduction attempt on the fixed serving stack (decode-corruption fix + fa3 + public_policy_hint teacher). If LoRA still fails to learn here, the next suspects are LoRA capacity/serving-path bugs rather than teacher signal.

Dedicated student sampler pool:

```bash
statefulset=opsd-wordle-sglang
replicas=2
pods=opsd-wordle-sglang-0,opsd-wordle-sglang-1
status=2/2 Running
restarts=0
service=opsd-wordle-sglang-headless
smg_service=opsd-wordle-smg.apanda.svc.cluster.local:8080
```

Current sampler/router health after the 2026-06-09 sampler repair:

```text
opsd-wordle-sglang-0  Ready  restart_count=0  recreated=2026-06-09T16:36:44Z
opsd-wordle-sglang-1  Ready  restart_count=0  created=2026-06-08T17:51:52Z
opsd-wordle-smg       Ready  rollout restarted at 2026-06-09T16:42Z
SMG /workers          total=2, both is_healthy=true, status=ready
SMG routing           recent logs show traffic to both sglang-0 and sglang-1
```

The trainer still logs `generate_endpoints=1` because it talks to one SMG endpoint. That is expected. Verify sampler fan-out through SMG `/workers` and SMG routing logs, not through the trainer's `generate_endpoints` line.

Dedicated teacher cache server:

```bash
deployment/pod=opsd-wordle-30b-teacher-sglang
service=opsd-wordle-30b-teacher-sglang:30000
endpoint=/teacher_hidden_cache
```

Current W&B:

```bash
project=zorl
run_name=OPSD-WORDLE-30B-R4-strict-opdmask-skipempty-lr1e-6-bs64-public-policy-hint
# previous (policy_hint, oracle teacher): run_id=9p9ophek
```

Current run directory: check the newest directory under

```bash
/home/apanda/xorl-opsd-wordle-apanda-dev-run-20260607/experiments/zorl/results/opsd_wordle_native_baseline/
# previous run: 20260609T162543Z-opsd-wordle-30b-native-asym-wt6wm-54ms5-wordle-r4-asymmetric_opsd
```

Primary artifacts:

```bash
run_config.json        # exact command/config resolved by the running job
metrics.jsonl          # train/eval/sample_eval/rollout/cache timing metrics
generations.jsonl      # per-turn student rollout traces used for OPSD rows
sample_eval.jsonl      # held-out multi-turn student eval examples
teacher_samples.jsonl  # diagnostic teacher-generated public-reasoning/guess rollouts on eval states
sampler_exports.jsonl  # native LoRA export/load events
job.log                # trainer stdout/stderr
server.log             # XORL server stdout/stderr
server_output/logs/orchestrator.log
```

Final metrics of the previous (`policy_hint`, oracle-teacher) run before the 2026-06-10 restart:

```text
train step 304  loss=1.956071  opd_kl=1.956071  grad_norm=0.7124  dt=115.0s
sample_eval step 304  exact=0/16  exact_match_rate=0.0000  reward_mean=-0.2423  format_rate=0.6875
teacher_sample (steps 0..120) exact 7-8/8, turns_used ~2.1 (oracle-like: STARE -> target)
```

Interpretation of that run: infrastructure healthy, KL nonzero, gradients fine, but the student never converted the KL signal into solves through 304 steps. Root cause (now fixed): the `policy_hint` teacher was oracle-like after turn 1 — the private reference block included the target and the reference action was usually the target itself, so the teacher policy was not reproducible from the student's public context. The 2026-06-10 `public_policy_hint` restart removes the target from the teacher prompt and anchors the reference action to a public candidate-splitting policy with informative public reasoning (see Teacher Prompt section).

What to expect from teacher diagnostics under `public_policy_hint`: teacher samples should now take ~3-4 turns instead of ~2 and may occasionally miss; `turns_used` near 2 with near-perfect exact would suggest the teacher is still leaking target knowledge and should be investigated.

## Hard Constraints

- Do not touch, scale down, recycle, or repurpose `zorl-ar-sglang`. That is a shared sampler pool and is not part of this run.
- This run uses the dedicated `opsd-wordle-sglang` pool through `opsd-wordle-smg`.
- All GPU workloads must have the pod-template label `team: turbo`.
- Do not reset, clean, or rebase `/home/apanda/xorl-sglang-internal` while this run is alive. The running SGLang pods depend on that dirty checkout.
- Do not clean untracked `experiments/zorl` files in `/home/apanda/xorl-opsd-wordle-apanda-dev-run-20260607`; this run uses them.

## Checkouts

Main OPSD Wordle checkout:

```bash
/home/apanda/xorl-opsd-wordle-apanda-dev-run-20260607
```

Branch state (2026-06-10): the checkout is on branch `opsd-wordle-apanda-dev-run-20260607` — formerly detached at `0acb8618`. The local server fixes (valid-label-filtered teacher hidden-cache rows, FSDP2-safe `no_grad` forward, wordle-python dep) are committed, and `origin/apanda-dev` (7 commits: quack gated-forward fix #355, OPD-loss restore #354, quack EP registration #353, OPD metric-key seeding #352, distributed-init + p2p repairs, lm-head TP) is merged in. One incoming test (`test_teacher_hidden_cache_trims_with_gathered_sp_labels`) was aligned to this branch's filtered hidden-cache semantics — reconcile that semantic (prefix-trim vs valid-label-filter) if upstreaming. `tests/server/runner/test_opd_runner.py` 13/13 green with `PYTHONPATH=<repo>/src`.

Important files:

```bash
experiments/zorl/standalone/train_opsd_baseline.py
experiments/zorl/standalone/tasks/wordle.py
experiments/zorl/standalone/test_wordle_opsd_prompts.py
experiments/zorl/configs/qwen3_coder_30b_a3b_zorl_wordle_ep8_deepep.yaml
experiments/zorl/k8s/qwen3-coder-30b-a3b-opsd-wordle-baseline.yaml
experiments/zorl/k8s/qwen3-coder-30b-a3b-opsd-wordle-sglang-pool.yaml
experiments/zorl/k8s/qwen3-coder-30b-a3b-opsd-wordle-smg.yaml
experiments/zorl/k8s/qwen3-coder-30b-a3b-opsd-wordle-teacher-sglang.yaml
```

SGLang checkout used by student samplers and teacher cache:

```bash
/home/apanda/xorl-sglang-internal
```

Important SGLang files:

```bash
python/sglang/srt/entrypoints/http_server.py
python/sglang/srt/entrypoints/teacher_hidden_cache.py
python/sglang/srt/managers/tokenizer_manager.py
python/sglang/srt/observability/scheduler_metrics_mixin.py
```

The SGLang checkout is heavily dirty from broader work. The Wordle run relies on at least these local behaviors:

- `/teacher_hidden_cache` route in `http_server.py`.
- `teacher_hidden_cache.py` materializes teacher hidden states and returns `cache_indices_by_sample`.
- `tokenizer_manager.py` handles scalar `customized_info` metadata as well as per-sample lists.
- `scheduler_metrics_mixin.py` clamps tiny negative KV usage instead of asserting in `/v1/loads?include=core`.

## What This Run Is Training

This is gradient-based native XORL OPSD, not ZORL/ES and not PEFT.

The student is a native XORL LoRA session:

```bash
base_model=Qwen/Qwen3-30B-A3B-Instruct-2507
lora_rank=4
lora_alpha=4
optimizer=adamw
optimizer_dtype=fp32
lr=1e-6
loss_fn=opd_loss
objective=asymmetric_opsd
```

The student samples multi-turn Wordle actions under a public prompt. For each turn:

1. Build a student prompt from only public state: previous guesses and G/Y/B feedback.
2. Sample the student continuation from the SGLang LoRA sampler.
3. Truncate the sampled continuation after the first complete `<guess>...</guess>` tag when possible.
4. Compute Wordle feedback locally from the private target.
5. Append `(guess, feedback)` to public history and continue up to 6 turns.
6. Build one OPSD training row per sampled turn.

The teacher does not generate the trajectory used for training. The teacher scores the exact student-sampled tokens under a richer teacher context. That is the OPSD shape: on-policy student rollouts, teacher-side scoring.

The current student output format is:

```text
<reasoning>PUBLIC_REASONING</reasoning><guess>WORD</guess>
```

This means the loss supervises both a short public rationale and the action. It is still on-policy: the optimized tokens are the student's own sampled tokens, not a teacher-generated trace.

The active run does not use training-time teacher reason-first:

```bash
TEACHER_REASON_FIRST=0
TEACHER_REASONING_TEMPERATURE=0.2
TEACHER_REASONING_MAX_NEW_TOKENS=64
```

The code still supports the OPSD-compatible `reason_first` shape, where a private `<teacher_reasoning>...</teacher_reasoning>` note is generated and prepended only to the teacher-side scoring context. That is not active in `wt6wm-54ms5`. In the live run, the teacher context is a deterministic `policy_hint` prompt: shared public prompt plus a private reference block and public-response transition.

The patched launch path emits diagnostic teacher-generated samples. These are observability only:

- `teacher_samples.jsonl` stores the teacher's own generated public `<reasoning>...</reasoning><guess>...</guess>` rollouts on the held-out eval subset.
- W&B logs numeric aggregates under `teacher_sample/*`.
- W&B logs a `teacher_sample/examples` table with target, pre-action history, teacher reasoning text, generated guess, feedback, validity, public-candidate audit fields, and score.
- These teacher-generated tokens are not used in the OPD loss.

## Student Prompt

Configured by:

```bash
WORDLE_PROMPT_STYLE=public_reasoning_constraints
```

System prompt:

```text
You are playing Wordle. Follow the public Wordle prompt exactly.
```

The user prompt is rebuilt from public state before each turn. It includes:

- the prior public transcript, e.g. `1. STARE -> XYXXY`;
- a computed public constraint summary;
- the already-guessed list;
- the output contract:

```text
Output exactly one line:
<reasoning>PUBLIC_REASONING</reasoning><guess>WORD</guess>
```

The prompt explicitly says:

- use only public transcript and feedback;
- do not claim to know a private target;
- reasoning must be one short sentence under 15 words;
- reasoning must not mention private targets, hints, hidden information, or oracle knowledge;
- `WORD` must be exactly five alphabetic letters, a common valid Wordle answer word, not repeated, and after feedback must fit all public constraints.

The student never sees the target, private reference, or full candidate list.

Relevant code:

```bash
experiments/zorl/standalone/tasks/wordle.py
  SHARED_PUBLIC_SYSTEM_PROMPT
  build_shared_public_prompt_content(...)
  _build_turn_messages(..., prompt_style="public_reasoning_constraints")

experiments/zorl/standalone/train_opsd_baseline.py
  _wordle_turn_prompt_ids(..., hinted=False, prompt_style="public_reasoning_constraints")
```

## Teacher Prompt

Configured by:

```bash
WORDLE_TEACHER_PROMPT_STYLE=public_policy_hint
```

(The previous oracle-like `policy_hint` style is preserved in the code for ablation; see "Legacy policy_hint" below.)

The teacher system prompt is intentionally the same public frame as the student:

```text
You are playing Wordle. Follow the public Wordle prompt exactly.
```

The teacher user prompt starts with the same shared public prompt as the student, then appends a private OPSD reference block and a public-response transition.

On turn 1, candidates are omitted and the reference opener is fixed:

```text
Private reference information for OPSD teacher only:
<private_reference>
Teacher policy: choose the public-valid candidate that best narrows the remaining candidates.
Remaining public candidate answers before this guess: omitted on turn 1.
Reference next guess: STARE
Reference public reasoning: Choose a broad opener with common letters.
</private_reference>
```

On later turns, the teacher gets pre-action public state plus a private reference computed by a PUBLIC policy — the target word never appears:

```text
Private reference information for OPSD teacher only:
<private_reference>
Teacher policy: choose the public-valid candidate that best narrows the remaining candidates.
The private target is intentionally not shown; rely on the public candidate analysis below.
Remaining public candidate answers before this guess:
count = {n}
{up to first 200 candidates computed only from prior feedback}
Reference next guess: {argmin over candidate guesses of expected remaining candidates; target may break near-ties within 5%}
Reference public reasoning: {informative public rationale, e.g. "54 candidates fit the clues; SABLE narrows them most."}
</private_reference>
```

Reference-policy implementation (`_wordle_public_policy_action`, `_wordle_policy_scores`):

- Exact expected-remaining partition scoring over the public candidate set when `count <= 200` (O(n^2) feedback partitions, all-green bucket counts as solved/zero); a letter-coverage proxy above 200.
- The private target is consulted only server-side and only to break near-ties (score within 5% of the best public guess). It never overrides a strictly better public guess and never appears in the prompt text.
- Offline rollout of this reference policy: 64/64 solved with tie-breaking, 62/64 with the target ignored entirely, avg 3.3 turns. The teacher's edge over the student is candidate enumeration + split quality — exactly the information gap identified as the eval bottleneck — and is in principle reproducible from public context.
- Scores are cached per candidate-set (`_POLICY_SCORE_CACHE`); prompt build cost measured at ~11-17 ms.

Legacy `policy_hint` (oracle-like, do not use for headline runs): private block contains `Target word: {TARGET}` and `Reference next guess` is the target whenever public-valid. This was the main scientific weakness of the runs up to 2026-06-10: teacher behavior was not reproducible from the student's public context, and the student stayed at 0/16 exact for 304 steps.

Then the teacher prompt says:

```text
Private teacher note:
The reference guess is {REFERENCE}. Prefer the public-facing rationale: "{REFERENCE_REASONING}" Do not reveal the private target or private reference.

After understanding the private reference, continue in exactly the same public response format:
<reasoning>PUBLIC_REASONING</reasoning><guess>WORD</guess>

Begin your response now.
```

The teacher prompt is pre-action. It must not contain the student's current guess feedback or any post-action candidate list. The teacher scores the same student-sampled continuation under this richer context.

Relevant tests:

```bash
test_policy_hint_first_turn_is_target_invariant
test_public_policy_hint_first_turn_is_target_invariant
test_public_policy_hint_never_shows_target_word
test_public_policy_hint_uses_pre_action_state_only
test_public_policy_action_prefers_information_over_target
test_public_policy_action_uses_target_only_on_near_ties
test_public_policy_action_single_candidate
test_policy_hint_uses_pre_action_state_only
test_public_candidates_handle_duplicate_letter_targets
test_teacher_wordle_diagnostics_logs_generated_reasoning
```

Relevant code:

```bash
experiments/zorl/standalone/train_opsd_baseline.py
  _wordle_policy_hint_user_content(...)
  _wordle_turn_prompt_ids(..., hinted=True, teacher_prompt_style="policy_hint")
```

Important nuance: `teacher_samples.jsonl` contains teacher-generated public responses for diagnostics, for example:

```text
<reasoning>Choose a broad opener with common letters.</reasoning><guess>STARE</guess>
<reasoning>Choose a common candidate consistent with the clues.</reasoning><guess>FETED</guess>
```

Those teacher samples are not the training trajectory. The OPSD loss is computed on student-sampled tokens only.

## Loss And Masking

The loss is XORL `opd_loss`, using a SGLang teacher hidden cache plus the teacher LM head:

```bash
TEACHER_CACHE_BACKEND=sglang
TEACHER_URL=http://opsd-wordle-30b-teacher-sglang:30000
OPD_LOSS_MODE=reverse_kl_full
OPD_KL_BACKEND=streaming
OPD_VOCAB_CHUNK_SIZE=32768
```

The training row has different contexts for student and teacher:

```text
student_input_ids = student_public_prefix + student_sampled_output
teacher_input_ids = teacher_policy_hint_prefix + same_student_sampled_output
```

Only the sampled output tokens are labeled. Prefix tokens are `IGNORE_INDEX`. Teacher cache indices are returned only for valid labeled positions and then aligned back onto the student row.

Token weights for Wordle output:

```bash
WORDLE_FIRST_TURN_WEIGHT=1.0
WORDLE_TAG_TOKEN_WEIGHT=0.0
WORDLE_REASONING_TOKEN_WEIGHT=1.0
```

The practical weighting behavior:

- `<reasoning>`, `</reasoning>`, `<guess>`, `</guess>`, whitespace, and other tag-ish tokens are masked out when detected.
- Reasoning content tokens inside `<reasoning>...</reasoning>` share `reasoning_token_weight`.
- Guess content tokens inside `<guess>WORD</guess>` share weight 1.0 across the guess content tokens.
- This normalizes guess words across tokenizer splits and avoids over-training easy tag tokens.

Relevant code:

```bash
experiments/zorl/standalone/train_opsd_baseline.py
  _wordle_output_weights(...)
  _truncate_output_ids_after_first_guess(...)
  build_multiturn_asymmetric_opsd_rows(...)
  materialize_teacher_cache(...)
  opd_loss_params(...)
```

Relevant tests:

```bash
test_wordle_output_weights_prioritize_guess_content_tokens
test_wordle_output_weights_include_reasoning_content_tokens
test_wordle_output_weights_include_unclosed_reasoning_content
test_truncate_output_ids_after_first_guess_removes_continuation
```

## What "Reward" Means Here

The training objective is OPD KL, not environment reward.

The `reward_mean` in `[sample_eval ...]` is a held-out multi-turn evaluation metric from `wordle.rollout_completion`, computed as:

```text
reward = 0.4 * solved + 0.3 * format_rate + 0.2 * info_gain + 0.1 * turn_bonus
```

Where:

- `solved` is exact target guessed within 6 turns.
- `format_rate` is the fraction of turns with a parseable five-letter `<guess>`.
- `info_gain` is a simple local heuristic over greens/yellows.
- `turn_bonus` is nonzero only if solved, larger for earlier solves.

Use `exact_match_rate` and sample generations to judge whether the model is actually learning Wordle. A falling or flat `reward_mean` with `exact_match_rate=0` means no task convergence yet.

## Current Throughput Shape

Current config:

```bash
TRAINER_GPUS=8
SAMPLER_REPLICAS=2
SGLANG_TP=2 per sampler
INFER_URL=http://opsd-wordle-smg.apanda.svc.cluster.local:8080
STUDENT_GENERATION_WORKERS=32
STUDENT_GENERATION_BATCH_SIZE=4
OPD_PIPELINE_CHUNK_SIZE=8
OPD_PIPELINE_PREFETCH_CHUNKS=2
SAMPLE_EVAL_INTERVAL=8
SAMPLE_EVAL_SIZE=32
```

The run is not obviously sampler-bound. Sampling is pipelined with teacher-cache materialization (`OPD_PIPELINE_CHUNK_SIZE=8`, `OPD_PIPELINE_PREFETCH_CHUNKS=2`). Recent rollout/cache timings after repairing SMG are:

```text
student rollout chunk: about 0.8-1.8s, with occasional 2-6s spikes
teacher cache chunk:   about 2.2-5.7s
trainer step:          about 103-116s after the sampler reboot window
```

The anomalous `train step=6 dt=288.2s` was caused by waiting through the `opsd-wordle-sglang-0` restart and LoRA load retries. Do not use that step as steady-state throughput.

SGLang sampler config:

```bash
--tp 2
--dtype bfloat16
--attention-backend fa3
--enable-lora
--lora-backend triton
--max-lora-rank 4
--lora-target-modules qkv_proj o_proj gate_proj up_proj down_proj
--lora-moe-format hybrid_shared
--experts-shared-outer-loras
--disable-custom-all-reduce
--cuda-graph-max-bs 64
--max-running-requests 1024
--max-queued-requests 512
--chunked-prefill-size 16384
--max-prefill-tokens 16384
```

CUDA graph is enabled. SGLang logs show decode batches with `cuda graph: True`.

SMG config:

```bash
policy=power_of_two
workers=opsd-wordle-sglang-0, opsd-wordle-sglang-1
health_check=/v1/models
request_timeout_secs=1800
```

## Monitoring Commands

Use absolute paths or `git -C`; the shell PWD can drift.

Check live Kubernetes state:

```bash
kubectl get jobs,pods,statefulsets,svc -n apanda | rg 'opsd-wordle|zorl-ar-sglang|NAME'
kubectl get pods -n apanda -o wide | rg 'opsd-wordle|zorl-ar-sglang|NAME'
kubectl describe job -n apanda opsd-wordle-30b-native-asym-wt6wm | sed -n '1,180p'
```

Check trainer log:

```bash
kubectl logs -n apanda pod/opsd-wordle-30b-native-asym-wt6wm-54ms5 --tail=400

kubectl logs -n apanda pod/opsd-wordle-30b-native-asym-wt6wm-54ms5 --tail=1200 \
  | rg '\[train step=|\[eval step=|\[sample_eval step=|\[pipeline train|\[rollout train_chunk|\[cache train_chunk|ERROR|Traceback'
```

Check SMG workers:

```bash
kubectl exec -i -n apanda opsd-wordle-30b-native-asym-wt6wm-54ms5 -- \
  /workspace/home/xorl-internal/.venv/bin/python - <<'PY'
import json, requests
r = requests.get('http://opsd-wordle-smg.apanda.svc.cluster.local:8080/workers', timeout=10)
print(json.dumps(r.json(), indent=2, sort_keys=True))
PY
```

Expected: `total=2`, both workers `is_healthy=true`, `status=ready`.

Check SMG routing:

```bash
kubectl logs -n apanda -l app=opsd-wordle-smg --since=20m \
  | rg -o 'opsd-wordle-sglang-[01]' | sort | uniq -c
```

Expected over a nontrivial window: both `opsd-wordle-sglang-0` and `opsd-wordle-sglang-1` appear.

Check sampler logs and CUDA graph:

```bash
kubectl logs -n apanda pod/opsd-wordle-sglang-0 --tail=300 \
  | rg 'cuda graph|LoRA adapter|Decode batch|Prefill batch|Traceback|ERROR'

kubectl logs -n apanda pod/opsd-wordle-sglang-1 --tail=300 \
  | rg 'cuda graph|LoRA adapter|Decode batch|Prefill batch|Traceback|ERROR'
```

Inspect run config:

```bash
run_dir=/home/apanda/xorl-opsd-wordle-apanda-dev-run-20260607/experiments/zorl/results/opsd_wordle_native_baseline/20260609T162543Z-opsd-wordle-30b-native-asym-wt6wm-54ms5-wordle-r4-asymmetric_opsd
jq . "$run_dir/run_config.json" | less
```

Summarize metrics:

```bash
run_dir=/home/apanda/xorl-opsd-wordle-apanda-dev-run-20260607/experiments/zorl/results/opsd_wordle_native_baseline/20260609T162543Z-opsd-wordle-30b-native-asym-wt6wm-54ms5-wordle-r4-asymmetric_opsd

tail -n 500 "$run_dir/metrics.jsonl" \
  | jq -c 'select(.event=="train" or .event=="sample_eval" or .event=="eval")
    | {event,step,loss,opd_kl,grad_norm,step_time_s,exact_match_rate,reward_mean,format_rate_mean,info_gain_mean,turns_used_mean}'
```

Inspect student training generations:

```bash
run_dir=/home/apanda/xorl-opsd-wordle-apanda-dev-run-20260607/experiments/zorl/results/opsd_wordle_native_baseline/20260609T162543Z-opsd-wordle-30b-native-asym-wt6wm-54ms5-wordle-r4-asymmetric_opsd

tail -n 80 "$run_dir/generations.jsonl" \
  | jq -r 'select(.event=="student_multiturn_generation")
    | [.project,.turn,.target,.guess,.feedback,.solved,.teacher_reason_first,.teacher_reasoning_context,.truncated_after_first_guess,.text] | @tsv'
```

Inspect held-out sample eval behavior:

```bash
run_dir=/home/apanda/xorl-opsd-wordle-apanda-dev-run-20260607/experiments/zorl/results/opsd_wordle_native_baseline/20260609T162543Z-opsd-wordle-30b-native-asym-wt6wm-54ms5-wordle-r4-asymmetric_opsd

tail -n 80 "$run_dir/sample_eval.jsonl" \
  | jq -r 'select(.event=="sample_eval_example")
    | [.step,.project,.target,.score.exact_match,.score.reward,.score.format_rate,.score.turns_used,.generated_text] | @tsv'
```

Inspect diagnostic teacher public-response samples:

```bash
run_dir=/home/apanda/xorl-opsd-wordle-apanda-dev-run-20260607/experiments/zorl/results/opsd_wordle_native_baseline/20260609T162543Z-opsd-wordle-30b-native-asym-wt6wm-54ms5-wordle-r4-asymmetric_opsd

tail -n 120 "$run_dir/teacher_samples.jsonl" \
  | jq -r 'select(.event=="teacher_sample_turn")
    | [.step,.project,.turn,.target,(.history_before|tostring),.teacher_guess,.feedback,.solved,.teacher_reasoning,.teacher_text] | @tsv'
```

## Launch And Relaunch Commands

Do not run these while the active job is healthy unless the user asks for a restart.

From the OPSD Wordle checkout:

```bash
cd /home/apanda/xorl-opsd-wordle-apanda-dev-run-20260607
```

Start or repair the dedicated sampler pool:

```bash
kubectl apply -f experiments/zorl/k8s/qwen3-coder-30b-a3b-opsd-wordle-sglang-pool.yaml
kubectl rollout status -n apanda statefulset/opsd-wordle-sglang --timeout=30m
```

Start or repair SMG:

```bash
kubectl apply -f experiments/zorl/k8s/qwen3-coder-30b-a3b-opsd-wordle-smg.yaml
kubectl rollout status -n apanda deploy/opsd-wordle-smg --timeout=10m
```

Start or repair the teacher cache server:

```bash
kubectl apply -f experiments/zorl/k8s/qwen3-coder-30b-a3b-opsd-wordle-teacher-sglang.yaml
kubectl rollout status -n apanda deploy/opsd-wordle-30b-teacher-sglang --timeout=30m
```

Launch a new baseline job:

```bash
kubectl create -f experiments/zorl/k8s/qwen3-coder-30b-a3b-opsd-wordle-baseline.yaml
```

If relaunching after an old run failed, leave old failed jobs alone unless they interfere. They are useful evidence. If cleanup is required, delete only the specific failed job after checking its logs:

```bash
kubectl logs -n apanda job/<job-name> > /tmp/<job-name>.log
kubectl delete job -n apanda <job-name>
```

## Validation Commands

Latest validation at 2026-06-10 (public_policy_hint restart):

```text
py_compile train_opsd_baseline.py wordle.py: passed
test_wordle_opsd_prompts.py: 19 collected, 19 passed
```

The previously stale assertions (old `Private target:` strings, `PUBLIC_REASONING_ONLY` wording, normalized content-token weights) were updated to current behavior on 2026-06-10, and six new `public_policy_hint` tests were added. The suite is a green gate again.

Prompt/masking leakage tests:

```bash
cd /home/apanda/xorl-opsd-wordle-apanda-dev-run-20260607
PYTHONPATH=/home/apanda/xorl-opsd-wordle-apanda-dev-run-20260607 \
  /home/apanda/xorl-internal/.venv/bin/python -m pytest -q \
  experiments/zorl/standalone/test_wordle_opsd_prompts.py
```

Python compile sanity:

```bash
cd /home/apanda/xorl-opsd-wordle-apanda-dev-run-20260607
PYTHONPATH=/home/apanda/xorl-opsd-wordle-apanda-dev-run-20260607:/home/apanda/xorl-opsd-wordle-apanda-dev-run-20260607/src \
  /home/apanda/xorl-internal/.venv/bin/python -m py_compile \
  experiments/zorl/standalone/train_opsd_baseline.py \
  experiments/zorl/standalone/tasks/wordle.py
```

SGLang compile sanity for the patched files:

```bash
cd /home/apanda/xorl-sglang-internal
/home/apanda/xorl-sglang-internal/.venv/bin/python -m py_compile \
  python/sglang/srt/entrypoints/teacher_hidden_cache.py \
  python/sglang/srt/managers/tokenizer_manager.py \
  python/sglang/srt/observability/scheduler_metrics_mixin.py \
  python/sglang/srt/entrypoints/http_server.py
```

## Failure Modes And Fixes Already Applied

### Degenerate KL at step 0

Old `opd_self_kl` used hinted teacher-forced rows for both teacher and student, so teacher and cold student saw effectively the same base+hint context and KL was near zero. Current `asymmetric_opsd` fixes this:

- student prompt is public/no-hint,
- teacher prompt is richer/private policy-hint,
- sampled tokens come from the student policy,
- teacher scores those same sampled tokens.

### Shared sampler pool disruption

The shared `zorl-ar-sglang` pool was previously touched and that was a mistake. Current run avoids it entirely. Use only:

```bash
opsd-wordle-sglang
opsd-wordle-smg
opsd-wordle-30b-teacher-sglang
```

### Stale SMG worker after sampler repair

On 2026-06-09, `opsd-wordle-sglang-0` was running at the Kubernetes level but SMG still had it marked failed, so all generation traffic went through `opsd-wordle-sglang-1`. The fix was:

```bash
kubectl delete pod -n apanda opsd-wordle-sglang-0
# Wait for SGLang startup, CUDA graph capture, /v1/models 200, and LoRA load success.
kubectl rollout restart deployment -n apanda opsd-wordle-smg
kubectl rollout status deployment -n apanda opsd-wordle-smg --timeout=300s
```

After this, SMG `/workers` reported both sampler URLs as `is_healthy=true`, `status=ready`, and routing logs showed traffic to both backends. If a sampler is recycled and SMG still reports it failed after the sampler is healthy, restart SMG; deleting only the sampler pod is not sufficient.

### SGLang `/v1/loads` crash

SGLang had an assertion path when token usage was a tiny negative value. The local patch in `scheduler_metrics_mixin.py` clamps `num_used_tokens`, `kv_token_usage`, and `token_usage` at zero. If samplers start restarting when SMG checks load, inspect that patch first.

### SMG/SGLang LoRA request shape

SMG rejects batched list-style `lora_path` with 422. `generate_batch_with_sglang()` sends scalar `lora_path`/adapter name for batched generation. If generation starts failing with 422, check this path first.

### Teacher cache consistency

Teacher cache keys are implicit in the full `teacher_input_ids` sent to `/teacher_hidden_cache`. The active rows include target-specific and transcript-specific teacher prompts, and `materialize_teacher_cache()` verifies:

- number of returned samples matches rows,
- number of cache indices matches labeled target tokens,
- returned cache path equals requested cache path.

### CUDA graph

CUDA graph is required for throughput. Student sampler manifest uses:

```bash
--cuda-graph-max-bs 64
--disable-custom-all-reduce
```

Do not add `--disable-cuda-graph`.

## Current Student Behavior

The student now usually emits the requested public `<reasoning>...</reasoning><guess>...</guess>` shape often enough to keep training rows nonempty, but held-out solve rate is still zero through step 120. The active run's sample-eval reward is noisy and mostly negative because invalid/failed games terminate early under the stricter parser.

The training path truncates after the first complete `<guess>...</guess>` when possible, so unrelated continuation after a valid guess does not always enter the OPSD row. The sample eval log still records full generated text and remains the main place to diagnose behavior.

Examples from recent `generations.jsonl`:

```text
<reasoning>Start with a broad opener to gather information.</reasoning><guess>CRANE</guess>
<reasoning>Eliminate C, R, A, N, E; choose new letters.</reasoning><guess>PLUMB</guess>
<reasoning>Letter A is in the word but not in position 3.</reasoning><guess>BLAST</guess>
```

Failure patterns:

- repeats guesses such as `PLUMB`,
- emits invalid placeholders like `R????` or `WORD`,
- sometimes outputs malformed tags such as missing `<guess>`,
- public reasoning can contradict feedback,
- exact solve rate is still zero.

This is why sample eval is required. Loss alone is not enough to judge success.

## What To Watch Overnight

The run has a 12 hour logical runtime:

```bash
MAX_RUNTIME_SECONDS=43200
activeDeadlineSeconds=46800
```

Watch these indicators:

- Trainer pod remains `Running`, `0` restarts.
- SMG `/workers` stays `total=2`, both ready.
- No new SGLang restarts.
- Train `opd_kl` remains nonzero and finite.
- `sample_eval/exact_match_rate` should eventually move above 0.
- `sample_eval/reward_mean` and `format_rate_mean` should not trend downward for many evals.
- `generations.jsonl` should show fewer invalid guesses, fewer repeats, and less unrelated chatter.

Current evidence has already crossed the earlier "step 64 still exact zero" warning threshold. The reason to leave this run alive is to collect a longer low-LR/batch-64 trace, not because it is currently promising. If exact remains `0/16` through several more sample evals and reward/format stay flat, the next agent should treat the configuration as scientifically failing even though the infrastructure is healthy.

## Likely Next Scientific Iterations

Do these only after the current run has enough evidence or fails.

1. Add stricter generation stopping.

   The model wastes tokens after a valid guess. A safer experiment is to stop on newline, not on `</guess>` unless SGLang is configured to include the stop string in output. Stopping on `</guess>` may remove the closing tag and hurt parsing.

2. Reconsider teacher reason-first as a controlled ablation.

   The active run has `teacher_reason_first=False`. A previous reason-first run increased teacher-side context but was not the current low-LR/batch-64 configuration. A clean comparison would keep `lr=1e-6`, `train_size=64`, `public_reasoning_constraints`, tag-token weight `0.0`, and only flip reason-first on.

3. Lower temperature for student sampling.

   Current temperature is `0.7`. A lower value may reduce malformed guesses and unrelated chatter, but may also reduce exploration.

4. Strengthen the public teacher policy. **DONE 2026-06-10** — `public_policy_hint` ranks candidate guesses by expected remaining candidates and lets the private target only break near-ties; the target no longer appears in the teacher prompt.

5. Try a stronger teacher model.

   The current teacher is the same Qwen 30B base under a privileged prompt. If this teacher cannot solve Wordle robustly without leaking target knowledge, use a stronger model or an external Wordle solver to provide teacher-side policy hints.

6. Consider explicit auxiliary penalties.

   The current optimizer is pure OPD KL. If exact/format does not improve, add explicit auxiliary terms for legal five-letter guess, no repeated guess, and candidate consistency. Keep this separate from the sample eval reward accounting.

7. Revisit max_new_tokens.

   Current `STUDENT_MAX_NEW_TOKENS=48` exists to allow a short reasoning trace. If chatter remains severe, try a lower cap with newline stopping.

## Reproduce The Active Config

The active run's exact `run_config.json` is the source of truth. Key values:

```bash
model=Qwen/Qwen3-30B-A3B-Instruct-2507
objective=asymmetric_opsd
lora_rank=4
lora_alpha=4
lr=1e-6
train_pool_size=512
train_size=64
eval_size=16
sample_eval_size=32
teacher_reason_first=false
teacher_reasoning_temperature=0.2
teacher_reasoning_max_new_tokens=64
teacher_sample_interval=8
teacher_sample_size=8
teacher_sample_temperature=0.2
teacher_sample_max_new_tokens=48
teacher_sample_workers=8
request_batch_size=8
eval_request_batch_size=8
student_temperature=0.7
student_max_new_tokens=48
student_generation_batch_size=4
student_generation_workers=32
student_ignore_eos=true
wordle_prompt_style=public_reasoning_constraints
wordle_teacher_prompt_style=public_policy_hint
wordle_first_turn_weight=1.0
wordle_tag_token_weight=0.0
wordle_reasoning_token_weight=1.0
teacher_cache_backend=sglang
opd_loss_mode=reverse_kl_full
opd_kl_backend=streaming
opd_pipeline_chunk_size=8
opd_pipeline_prefetch_chunks=2
max_runtime_seconds=43200
```

When in doubt, compare any proposed rerun to the newest `run_config.json` under:

```bash
/home/apanda/xorl-opsd-wordle-apanda-dev-run-20260607/experiments/zorl/results/opsd_wordle_native_baseline/
```

## 2026-06-10 late: granted capacity, Kimi-K2.6, gold SFT data, 2x4 trainer live

- **User granted**: er-q35-397b-bf16-sglang node0/node1 (h100-052/-100, 8 GPUs
  each) and the old 30B opsd-wordle stack (4 GPUs on -114, 2 on -001). All torn
  down. The two full nodes now run **Kimi-K2.6 2-node TP=16**
  (`k8s/kimi-k26-gold-sglang-2node.yaml`, service `kimi-k26-sglang:30000`,
  hostname-pinned via nodeSelector) for gold CoT generation.
- **Gold SFT v0 exists** (scaffold distillation, no Kimi needed):
  `results/wordle_gold_sft/q36_self_scaffold_20260610T221141Z/gold.jsonl` —
  384 train-pool targets, 311 solved (81%), **1247 turns**, 24 leak-flagged,
  5 min wall. Generator: `standalone/generate_wordle_gold_sft.py`
  (gen WITH candidates scaffold, rows recorded against the no-candidates
  student prompt; eval split excluded). Kimi slots in via `--base-url`.
- **2x4 multi-pod trainer LIVE** (head -077 + worker -116, muon momentum 0.95
  + block-FP8, EP=4/pod shard=8): cross-pod rendezvous OK, NCCL weight-sync
  group INITIALIZED after the IB env fix (GID_INDEX=0 + bad-HCA exclusion on
  trainer AND samplers) — step-0 broadcast processing FSDP modules.
- h100-047 still cordoned (driver-broken taint) — do not tolerate cuda-error
  taints; recheck later. h100-116 returned to the pool but was half-taken
  (charlie, 4 GPUs) during its reboot; our worker has the other 4.
- NEXT: SFT mode (CE on gold.jsonl rows through the same full-weight server +
  weight-sync path) → then OPSD/GRPO from the SFT checkpoint.

## 2026-06-10 ~23:15Z: muon+momentum+FP8 full-weight OPSD TRAINING (2x4, p2p sync)

`opsd-wordle-q36-fullft-2x4-head-lx27g` + worker: **train steps completing**
(step 2: loss/opd_kl 0.752, grad_norm 25.5, dt ~270s, tokens 9664) with
**p2p weight sync 2.9s/step to both shards** (runbook benchmark 2.4s).
muon momentum 0.95 (bf16 buffer) + enable_fp8_training (quack grouped
experts survived the historical step-2 assert point). Stack is fully
aligned with CANONICAL_INFRA_RUNBOOK §11.

Complete debug ledger to get here (each was one launch cycle):
1. `moe_grad_reduce_mode: bf16_a2a_fp32_sum` rejected — this branch's FSDP
   reduce_dtype is bf16 → drop the knob.
2. Full-weight server requires `model_id=default` (no multi-tenancy).
3. Full-weight server rejects per-session optimizer config → optimizer
   lives in the server YAML (`muon_lr` sets muon groups, NOT `lr`).
4. `add_inference_endpoint` must retry while samplers reload (SMG-ready ≠
   shard-ready).
5. `sync_inference_weights` must carry pod-IP `master_address` (server
   auto-detect uses the unresolvable pod hostname; samplers spin on
   gai -2 rendezvous retries — this also explains the first "collision"
   wedge).
6. nccl_broadcast trainer→sglang bootstrap fails (`wrong type 3 != 4`)
   even with full IB env, while trainer↔trainer NCCL works → use the
   proven `p2p` (Mooncake) method.
7. p2p receiver needs `--enable-rdma-weight-updates` on samplers.
8. RDMA transfers (ret=-1) need `rdma/infiniband: "1"` + `IPC_LOCK` on
   non-privileged pods (and trainers must NOT be privileged per infra
   runbook; no NCCL_IB_GID/HCA, no expandable_segments on Mooncake pods).
9. Kyverno injects `node-group: default` into nodeSelector when absent —
   set it explicitly to target nccl-group nodes. Kimi-style 1-replica GPU
   deployments need `strategy: Recreate` (rolling update deadlocks).

Kimi-K2.6 serving (TP=16, 2 nodes, snapshot-path model): READY at
`kimi-k26-sglang.apanda.svc.cluster.local:30000`. Baseline wordle eval
running; gold CoT generation next (generate_wordle_gold_sft.py --base-url).

## 2026-06-11: SFT verdict, the output-contract bottleneck, thinking-budget pivot

- **SFT-gold run** (2x4, muon+momentum+FP8): CE 0.78→0.34 over 32 steps,
  style fully transferred — but baseline-protocol eval of the step-32 weights
  = **0.203 exact / 0.967 format** vs base 0.219. NO solve lift. Checkpoint
  saved (`xorl://default/weights/step-000032`); run then crashed on the first
  forward after save_weights: `aten._fused_rms_norm got mixed Tensor/DTensor`
  (save→forward interaction, OPEN BUG — avoid mid-run saves until fixed).
- **Kimi-K2.6 evals**: hard setting 0.094/format 0.372; +scaffold 0.125/0.542.
  Kimi's binding constraint is the one-line output contract (it wants to
  think), not enumeration. Not a gold source under this contract.
- **Conclusion (consistent across 5 measurements)**: the under-15-word public
  rationale cannot carry candidate enumeration — it is serial compute that
  needs tokens. Scaffold (externalized enumeration) = 0.969; everything that
  must enumerate inside one line ≈ base.
- **Pivot**: `*_think` prompt styles (open-think chat rendering; public line
  after `</think>` is scored; `strip_think_prefix` applied in eval + rollout
  parsing). Loss-side: think tokens would be supervised by OPSD KL/SFT CE as
  sampled-output tokens — region weighting TBD after the base measurement.
- **Cross-check vs arithmetic stack's opd_mask_prompt_kl finding**: wordle
  OPSD rows already mask prompts AND zero-weight tags to IGNORE_INDEX —
  answer-only geometry confirmed; the prompt-KL deviation does NOT apply here.
- Wordle OPSD/SFT eval-rollout fix: action scored on first-guess-truncated
  text (serving newline stops unreliable; chatter previously ended games at
  turn 2 as invalid actions — eval measured serving, not play).

## 2026-06-11: THINKING GATE PASSED — base 0.422 with think budget (vs 0.219)

`public_reasoning_constraints_think`, 64 games, seed 777, mnt 4096:
exact **0.422** / format 0.963 / turns 4.56. Capability ladder:
one-line 0.219 → +think 0.422 → scaffold ceiling 0.969. The training
contract is now think-then-answer; target = closing the think→ceiling gap.

Next data: STaR self-distillation — think-mode generation WITHOUT scaffold
(traces perform real enumeration), keep solved games (~42% yield), SFT on
full think+answer tokens (`q36_star_think_*` dataset, in flight). Kimi
becomes a viable gold source under this contract (its failure was the
one-line format). Generator: think styles stop on `</guess>` (newline stop
would truncate thinking) and re-append the matched stop.

## Think-contract SFT launch recipe (fire when q36_star_think_v2 data lands)

```bash
kubectl create -n apanda -f <(... fullft-2x4.yaml with head overrides ...)
# Head env overrides vs the one-line SFT run:
#   BASELINE_OBJECTIVE=sft_gold
#   GOLD_DATA=${REPO_ROOT}/experiments/zorl/results/wordle_gold_sft/q36_star_think_v2_*/gold.jsonl
#   WORDLE_PROMPT_STYLE=public_reasoning_constraints_think   # sample_eval rollouts think
#   MAX_LENGTH=8192                  # think rows: ~600 prompt + up to ~4k think
#   SAMPLE_EVAL_MAX_NEW_TOKENS=3072  # was 48 — one-line sized
#   SAMPLE_EVAL_IGNORE_EOS=0         # MUST flip: with ignore_eos=1 every turn burns 3072 tokens past EOS
#   SAMPLE_EVAL_STOP=""              # no stop strings (think text mentions </guess>)
#   STEPS=<N> SAVE_INTERVAL=<N>      # final-step save only (post-save forward crashes — open DTensor bug)
#   WANDB_NAME=SFT-THINK-WORDLE-Q36-35B-2x4-...
# Gate: 64-game baseline-protocol eval (think style) vs base+think floor
# (0.422 measured with the buggy </guess> stop — re-measure with fixed stops
# for the true floor before judging).
# THEN: OPSD warm-start from this run's final checkpoint (load_checkpoint_path),
# student style = think+answer already learned; KL spends on decision quality;
# teacher private candidate block supervises think-token enumeration.

## 2026-06-11 (cont): one-line SFT checkpoint autopsy + STaR generation debugging

**Checkpoint transcript autopsy (one-line SFT, step 32)** — read the games, not
just the numbers (`results/wordle_eval/q36_SFT_step32_*/games.jsonl` vs the
base run on the same 64 targets):
- Style IS fully memorized: every game opens with the gold sentence verbatim
  ("Starting with a common word containing frequent letters..." + CRANE);
  rationales are gold templates; format near-perfect.
- But it confabulates: asserts "X fits the pattern and is a valid Wordle word"
  for invalid X. Best single diagnostic: target `yearn` T5 — the model starts
  genuinely deliberating INSIDE the 15-word reasoning tag, overflows, dies.
  v0 gold reasoning averaged 12.9 words (p90 17) — no room for computation.
- Conclusion: warm-start value of SFT = style; computation needs the think
  contract. (Standing rule: every checkpoint gate includes a transcript read.)

**STaR think-data generation debug ledger** (`generate_wordle_gold_sft.py`):
1. v1 (temp 0.7, stop "</guess>"): 1/224 solved — the model MENTIONS the
   output format (incl. `</guess>`) inside its think text; the stop truncated
   mid-think. → Think styles use NO stop strings, EOS only. (This also means
   the 0.422 base+think eval — measured with that stop — is likely an
   UNDERESTIMATE; re-measure with fixed stops before judging SFT gains.)
2. sglang reports EOS finishes with an INT `matched` token id — the
   re-append-matched-stop logic must guard `isinstance(matched, str)` (a
   leaked "248046" was breaking strict format parsing).
3. v2 (temp 0.7, EOS-only): 58/496 solved (11.7%), 183 turns. Trace quality
   verified by reading: real constraint algebra, pattern derivation,
   self-correction, then a terse compliant public line. Mean think 808 words.
4. v3 (temp 0.3 ×2 passes): ~2.5% solve — WORSE at lower temp. Probes show
   why: the model enumerates EXHAUSTIVELY at low temp ("Let's try ALINE...
   ALIVE... ALIBI...") and exhausts the 4096-token budget without ever
   closing `</think>` (all probes: 4096 tokens, no close). Passes killed.
5. In flight: 8192-token budget probe on the same hard state. Options if it
   completes: regenerate at temp 0.3/mnt 8192 (2x cost); add a think-budget
   steer to the prompt ("prefer a strong probe word over exhaustive
   enumeration; keep private thinking concise"); or both. MAX_LENGTH for SFT
   rows must cover prompt+think (manifest plumbs MAX_LENGTH; script default
   8192 — raise if mnt 8192 traces are kept).

**Ops gotchas added**: `pkill -f` self-matches the calling shell's command
line (exit 144) — kill via `ps | rg | awk` pid-file indirection; this box's
pgrep lacks `-q`. Background completion-watchers must not include the watched
pattern in their own command line.

## 2026-06-11 ~18:30Z: direction reset — Kimi-think gate, STaR abandoned

**Ledger correction**: the v3 (temp 0.3) STaR passes recorded above as
"killed" were NOT killed (the `pkill -f` self-match failure). Both ran
~4.7 h more to [256/496]: v3a solved=4, v3b solved=7 (~2% yield, 29 turns
total). They died/ended on their own by ~17:39Z. Net: the temp-0.3 arm is
confirmed worthless AND it burned the teacher pod for the afternoon. The
uncommitted think-budget steer in `tasks/wordle.py` (mtime 18:04Z) was never
exercised by any pass and has been reverted per the direction reset.

**User decisions (verbatim intent)**:
1. Goal = a 35B that legitimately plays Wordle well; OPSD-with-teacher-hint
   is the preferred bet but GRPO is acceptable — whichever converges.
2. Stop STaR self-distillation; stop constraining reasoning length/style
   entirely (no steer, no stop strings, generous token budgets, EOS-only).
3. Candidates scaffold is NOT legitimate in any student-visible place:
   not in the student prompt, not for generating student SFT data
   (rationalization-from-scaffold rejected — the model cannot know the
   candidate list while playing).
4. Candidates scaffold IS legitimate as the OPSD teacher's private hint:
   teacher context = shared public prompt + private candidates block +
   teacher's own CoT; this teacher KL-supervises the student's sampled
   think+answer tokens. This replaces answer-hinting (`policy_hint`) and
   upgrades `public_policy_hint` (reference-action hint) to a
   reasoning-capable hinted teacher. Teacher scoring think tokens it would
   not itself produce is accepted.
5. Kimi-K2.6 as gold-trace source: retry under the think contract with the
   CoT length restriction lifted entirely. (Prior "Kimi failed" verdict was
   under the one-line contract + 512-1024 mnt + `</guess>` stop, with
   Kimi's template thinking ON by default — it was being truncated
   mid-think; format 0.37 was a measurement artifact of the contract.)

**In flight (launched 18:28Z, logs /tmp, monitor armed)**:
- `kimi_k26_think_64_t02_20260611T182842Z` — Kimi, think style, temp 0.2,
  mnt 16384, --no-ignore-eos, EOS-only, 64 games seed 777, retry 2.
- `kimi_k26_think_64_t10_20260611T182842Z` — same at temp 1.0 (Kimi's
  native thinking regime; hedge against low-temp enumeration spirals).
- `q36_base_think_nostop_64_20260611T182842Z` — true base+think floor
  (0.422 was measured with the mid-think `</guess>` stop + mnt 4096).

**Gate logic**: if Kimi-think solves ≥~0.6 on the hard setting, Kimi is the
gold factory → generate think-SFT traces from Kimi (full think+answer kept,
no length constraints) → think-SFT q36 → eval gate vs the re-measured floor
→ OPSD (candidates-hinted teacher) or GRPO from that checkpoint. If
Kimi-think disappoints, skip SFT and go straight to OPSD-with-candidates-
hint / GRPO from base (floor 0.42+, dense on-policy signal exists).

## 2026-06-13 ~06:30Z: CHUNK-SIZE SWEEP — chunk16 is the sweet spot (~2x win), chunk32 no gain

CORRECTION to the ~05:30 "EP-dedup wall-neutral" entry: I measured too early
(killed dedup runs at step 2-3, BEFORE Triton kernels autotune over ~5-7 steps).
On a full warm run the chunk-size lever IS a real win:

| config | fb calls | rows/rank | warm fb_s | warm step_s | fb-ms/1k-tok |
|---|---|---|---|---|---|
| chunk2, NO-dedup (old baseline) | 32 | 2 (×8 dup) | ~226 | ~252 | ~4.3 |
| **chunk16 + dedup (SWEET SPOT)** | 4 | 2 | ~116 | ~149 | ~2.2 |
| chunk32 + dedup | 2 | 4 | ~105 | ~157 | ~2.25 |

- **chunk2→chunk16 (dedup-enabled): ~2x fb / ~1.7x step.** Fewer, bigger fb
  calls amortize per-call overhead (kernel launches, EP all-to-all setup,
  per-call defrag gc, dispatch). EP-dedup is load-bearing: it makes the bigger
  chunk gradient-correct + memory-feasible (distinct 1/8 slices). So dedup is
  NOT wall-neutral after all — paired with the chunk bump it's the ~2x.
- **chunk16→chunk32: NO gain** (token-normalized identical ~2.2 ms/1k-tok).
  chunk32 fits (no OOM, 4 rows/rank w/ streaming KL + defrag) but doubles
  memory for nothing — overhead already amortized at chunk16. chunk64 not tried
  (4x mem + kills pipeline overlap).
- **~2.2 fb-ms/1k-tok is the irreducible per-token full-vocab-KL floor at 8 GPU.**
  Below it: only more GPUs (parallelize per-token work, staged 2x8 config
  `..._fullft_2x8_warmstart_sft48.yaml`) or cheaper KL (science tradeoff).
- **CANONICAL throughput config: streaming + EP-dedup + chunk16.** (chunk32 run
  `8g-7qr94` left running — throughput-identical, healthy; use chunk16 on next
  launch for the memory headroom.)

## 2026-06-13 ~05:30Z: throughput levers exhausted (science-neutral); wall is full-vocab KL

Measured each lever on the live warm-start run (8 GPU, EP=8, think rows ~58k
labeled tok/step):
- **EP-dedup (ff262108): MERGED, KEPT, but WALL-NEUTRAL here.** valid/train
  tokens 7.89→0.99 (8x redundant FLOPs eliminated → real MFU/GPU-hour win), but
  fb wall UNCHANGED (226s pre = 226s post). The 8x duplication ran in PARALLEL
  across the 8 EP ranks (1 EP group at 8 GPU) — wasted GPU-hours, not wall-sec.
  Matters at multi-EP-group scale (32 GPU = 4 groups), not here. Keep it (free,
  correct). Rollback `XORL_SERVER_EP_DUPLICATE_BATCHES=1`.
- **KL backend streaming→torch_compile: WORSE (308s vs 226s steady).** The
  auto_chunker compile path loses on our VARIABLE-LENGTH packed think rows
  (recompile/specialization overhead > fusion gain). `streaming` was already the
  right kernel for variable shapes. Reverted to streaming.
- **fb is full-vocab reverse-KL bound**: lm_head projection to 151k vocab over
  ~58k think tokens ≈ 100x the MoE body's FLOPs; MFU ~6% incl. lm_head (memory-
  bandwidth-bound elementwise KL). This is INTRINSIC to the objective (full-vocab
  mode-covering KL over every think token — the science requirement).
- **vs sibling stack (amdahl doc):** OPPOSITE bottleneck. Their probe used
  mnt=64 (few labeled tok) → fb ~29s, teacher-prefill-bound, trainer idle. Ours
  uses mnt 3072 (~58k labeled tok) → fb 226s, trainer-bound. Same framework,
  inverted by workload. Their teacher-grow fixes do NOT apply to us.

**Remaining wall-throughput levers (all need a decision):**
1. **More trainer GPUs (science-neutral, ~linear).** fb is per-token-bound +
   embarrassingly parallel over the token dim; 8→16 GPU trainer ≈ 2x faster fb.
   CAPACITY-BLOCKED now (no free 8-GPU default node). This is the doc's balance
   rule inverted: grow the bottleneck (trainer), not the idle sampler/teacher.
2. **Fewer supervised tokens / top-k KL (SCIENCE tradeoff — user call).** The
   wall ∝ labeled think tokens × full vocab. Down-weighting think tokens or
   top-k KL would cut it hugely but changes the think-OPSD objective.
3. Accept current ~250s/step (streaming+dedup) and run the science.
Live run reverted to streaming: `8g-bl5tg` (warm-start from SFT-48).

## 2026-06-13 ~04:00Z: THROUGHPUT/MFU work — EP-dedup merge (user directive)

**MFU MEASURED (think-OPSD, 8-GPU EP=8, from-base run metrics):**
- step_time_s ~252, **forward_backward_s ~229 (91% of step)** → TRAINER-BOUND
  (prepare_wait only ~20s/8%). INVERSE of the sibling reprogrammable-slots
  stack (teacher-prefill bound, 32-GPU trainer idle) — see
  `~/xorl-apanda-dev-opd-port/docs/notes/amdahl_allocation_20260613.md`.
- **MFU ~0.06% useful** (6·3e9·59k tok / 229s / 8·989TF); ~0.46% counting the
  8x duplicated work. Even dedup-adjusted MFU is tiny → fb is NOT matmul-bound;
  overhead-bound (full-vocab streaming KL, EP all-to-all on small per-call
  batches, grad-ckpt recompute). Dedup necessary, not sufficient — RE-MEASURE.
- **8x EP-duplication CONFIRMED**: valid/train tokens = 7.89 ≈ ep_size 8. At
  8 GPUs EP=8 = 1 EP group; pre-fix all 8 ranks recompute the SAME batch.

**ACTION (done):** merged `origin/apanda-dev` (3 commits; clean; our
XORL_FB_PER_CALL_DEFRAG preserved; OPD runner tests 20/20):
- `ff262108` EP fb-dispatch dedup — per-rank-distinct slices, gradient-identical,
  rollback `XORL_SERVER_EP_DUPLICATE_BATCHES=1`. THE throughput lever.
- `82ece3a7` GDN cp-kernel torch.compile no-op fix; `2dddccee` build-info.
- Consolidated to ONE experiment (killed from-base control + old warm-start).
- Relaunched warm-start-from-SFT48 = `8g-pfh6j` with OPD_PIPELINE_CHUNK_SIZE 16
  (post-dedup ~2 distinct rows/rank = memory-neutral vs proven pre-dedup chunk2,
  8x useful throughput; commit advises >=32 packed rows/fb → bump to 32 if
  launch/comms-bound & memory allows).
- **Expected** fb ~229s → ~30-50s; then prepare_wait (~20s) may co-bind → grow
  teacher/sampler or raise prompts/step into the idle trainer.
- NEXT LEVERS after re-measure: grad-ckpt OFF (freed memory), MAX_LENGTH→16384,
  chunk→32.
## 2026-06-13 ~01:30Z: SFT GATE 0.500 (vs 0.469); OPSD WARM-START LAUNCHED

- **SFT step-48 gate** (`q36_SFT_kimigold_step48_*`, floor protocol but
  mnt 12288 due to pool token cap): exact **0.500** vs floor 0.469
  (+3pts, within n=64 noise). Failure profile DID move: invalid-guess
  deaths 31→23 (−26%), valid_guess_rate 0.877→0.904, max-turn deaths 3→9.
  Reading: 48 steps/~3.5% CE transferred guess discipline, not
  enumeration capability. Checkpoint:
  `.../20260612T160347Z-...sft_gold/server_output/weights/default/step-000048`
  (post-save forward worked — the DTensor save bug did NOT bite this path).
- **OPSD WARM-START from SFT48 launched** (`8g-r4zqw`, W&B
  ...THINK-WARMSTART-sft48...): warmstart config
  (`configs/..._ep8_warmstart_sft48.yaml` via CONFIG_PATH), sampler-1,
  chunk=2+defrag. Bound h100-005 (broken-IB reputation; single-node OK,
  Mooncake sync to sampler-1 is the live IB test — relaunch w/ exclusion
  if step-0 sync fails).
- OPSD v3 from-base continues @ -049/sampler-0 (longest trace; KL-down/
  play-flat through 40+ steps across 3 runs).
- Decision tree if warm-start OPSD also stays play-flat by ~step 50:
  (a) longer/hotter SFT from ckpt (load_checkpoint works), (b) regenerate
  gold at eval-parity rules (needs Kimi redeploy — gang manifest ready),
  (c) GRPO from SFT48 (format 0.94 = format-gate concern resolved).
## 2026-06-12 ~16:30Z: SFT flat-CE investigation closed — hard data, not breakage

- Attempts 4-7 ledger: (4) closed-think gold rows = format bug, FIXED
  (`enable_thinking = '</think>' in completion` in build_gold_sft_rows) —
  necessary for inference coherence but did NOT change CE; (5) Triton-OOM →
  rank cascade ("unspecified launch failure" = downstream symptom; also
  explains the -096 mystery) → OPD_PIPELINE_CHUNK_SIZE=2 fixed stability;
  (6/7) lr yaml 1e-6→1e-5 = NO effect (logged lr likely cosmetic; curves
  identical — but identical data order makes curve-matching non-diagnostic).
- **Resolution**: eval CE denominator includes padding to MAX_LENGTH
  (470048 tok / 55 rows = 8192/row exactly) → true eval CE ≈ 3x reported;
  true train CE ≈ 0.93/token everywhere. The 8x train/eval gap was
  normalization, not a bug. Kimi think prose = high-entropy cross-model
  text; one-line gold was templated (0.78→0.64 in 3 steps). Think-SFT
  descent is real but glacial → needs epochs, not steps.
- **DECISION: attempt 7 (8g-qr9fj @ -100) runs the full 48 steps (~13
  epochs, ~8h) untouched; judgment = final-step save + 64-game think gate
  vs 0.469, NOT CE.** OPSD v3 (8g-pc8pj @ -049) runs in parallel
  (from-base arm, 3rd reproduction).
## 2026-06-12 ~09:30Z overnight ledger: eval-vs-sampler crash, role swap, SFT memory hunt

- **LESSON (hard)**: do NOT run floor-protocol harness evals against a LIVE
  trainer's sampler. 64 concurrent 12k-token thinks on the TP=2 pool pod
  (mem-frac 0.75) + rollouts + weight-sync crashed sampler-0 at OPSD step 40
  → sync RuntimeError killed the trainer (no checkpoint: 8gpu SAVE_INTERVAL
  32 never fired a save line — 40 steps lost). Floor evals: throttle
  --batch-size <=8, or eval a frozen checkpoint on idle serving.
- OPSD-from-base trace before death (W&B p1s31gvq): train KL 0.085→0.064,
  held-out KL noisy-down 0.090→0.060→0.081 (eval games re-rolled per eval —
  composition noise), sample_eval 1/16+reward best at step 24, 0/16 at 32.
  Inconclusive-but-promising; needs a re-run.
- **Role swap**: floating SFT camper instantly took -125 when the OPSD pod
  died (SFT now Running there; sampler-1 unaffected). OPSD recreated as the
  floating camper (NotIn -105). -100 has 6/8 free (donglin 2-GPU squatter).
- Sampler pool cap discovered: --max-total-tokens 16384 → harness evals
  vs pool pods must use mnt<=12288 (16384 400s the request).
- SFT memory ledger: 16384 OOM (13.5GB logits ask) → 10240 OOM(frag, 7-10GB
  reserved-unallocated; defrag gate did NOT engage in sft_gold path —
  unverified insertion point, announce line added) → now 8192 + defrag, in
  flight on -125.
## 2026-06-12 ~04:30Z: KIMI NODES REPURPOSED — SFT-on-gold + single-node OPSD (user-directed)

- Gold done → Kimi STS + podgroup torn down; the two full 8-GPU nodes
  immediately rebound to pre-camped pinned jobs (zero-vulture swap):
  - **SFT-on-gold**: `opsd-wordle-q36-fullft-8g-6pdsb` @ h100-100.
    sft_gold + Kimi gold.jsonl (1094 turns), MAX_LENGTH=16384, STEPS=48,
    SAVE_INTERVAL=48 (final-save only, DTensor bug), think sample_eval
    (temp 0.2 / 3072 / EOS-only). W&B SFT-THINK-WORDLE-Q36-35B-8G-kimigold-*.
  - **think-OPSD**: `opsd-wordle-q36-fullft-8g-llmj4` @ h100-125. Same
    think contract as the 2x4 run, now single-node (no cross-node NCCL).
- **Sampler split (critical)**: two trainers must NOT share samplers
  (each full-weight-syncs its own policy). OPSD → sampler-0, SFT →
  sampler-1, direct URLs, SMG bypassed (`EXPECTED_SMG_WORKERS=0` skip
  gate added to the 8gpu manifest; GOLD_DATA+MAX_LENGTH passthrough
  ported too). SMG now serves nothing — do not route through it while
  both trainers live.
- 2x4 gang killed (attempt 4 was pre-step; -087/-049 released). The -096
  launch-failure and -087 cohabitation questions are MOOT on single-node
  full-node allocations.

## 2026-06-12 ~03:40Z: GOLD COMPLETE (1094 turns); trainer kernel-failure ledger

- **Kimi gold v1 COMPLETE**: 291/512 solved (56.8%), **1094 turns**, 5
  leak-flagged, 2.3h. `results/wordle_gold_sft/kimi_k26_think_t10_20260612T003446Z/`.
  Trace quality verified by reading: real constraint algebra + self-correction.
  Think words mean 2445 / p90 5920 / max 10553 → **SFT MAX_LENGTH=16384**
  (or drop top ~5%). Yield gap vs 0.9375 eval = generator's stricter
  constraint-conformity continuation rule (eval allows probe words) +
  retry-less deaths; relaxing to eval-parity is the lever if more data needed.
- **Trainer attempt 3 (-096/-049) died at step 3**: `CUDA error: unspecified
  launch failure` on head ranks mid-forward_backward (steps 1-2 clean:
  step1 896s/208k tokens, step2 414s, syncs 2.8s warm). Engine wedged 40min
  to NCCL abort. Open: data-dependent kernel bug (fp8/quack grouped GEMM
  under variable think-row shapes — prior history) vs -096 hardware.
- **-087 re-probed CLEAN** (4MiB/GPU, no processes) — the optim_step OOM's
  19GB foreign memory was transient cohabitation, not a zombie. Attempt 4
  (head BACK on -087 + worker -049) is the live discriminator for both
  questions. If launch-failure recurs on -087 too → flip enable_fp8_training
  off next relaunch.

## 2026-06-12 ~02:20Z: 2x4 think-OPSD assembled via gang; -087 optim_step OOM (zombie suspect)

- Kimi t0.2 arm final: **0.734** (vs t1.0 0.9375) — temp question closed.
- 2x4 gang (`...fullft-2x4-think-gang.yaml`, head startup-timeout 21600):
  worker camped ~40 min then bound -049; head ran on -087 through cross-pod
  init + 35.6s cold weight sync + cold eval + 15/16 fwd/bwd chunks at
  shard=8 — then **optim_step OOM with ~19GB FOREIGN memory on GPU 0**
  (device free 134MB vs our process 59.7GB on an 80GB part). Suspect:
  zombie GPU memory from -087's cordoned past (force-deleted pods leave
  allocations; known gotcha). -087 mounts/scheduling are fine but **treat
  -087 GPUs as dirty until nvidia-smi-probed**.
- Relaunch landed head on **-096** + worker -049 — **optim_step PASSED →
  -087 zombie CONFIRMED** (node-local; probe its GPUs before reuse).
- **TRAINING LIVE 2026-06-12 ~02:50Z**: step 1 = loss/opd_kl 0.0763,
  tokens 208,520, grad_norm 2.82, dt 896s (~15 min/step ≈ 96 steps/day;
  rollouts dominate). Full think-contract stack validated end-to-end:
  think rollouts → hinted teacher cache → shard=8 chunked fwd/bwd →
  muon mom0.95+FP8 optim step → 35s full-weight sampler sync.
  W&B OPSD-WORDLE-Q36-35B-FULLFT-2x4-THINK-hintonly-bs64-public-policy-hint.
  Gate protocol: in-trainer sample_eval (3072-budget yardstick, trend only)
  + periodic harness evals vs the 0.469 floor against synced samplers.
- Ops: deleting a failed gang job rebinds replacements in seconds if the
  blocks are still free — recycle immediately on head death.

## 2026-06-12 ~00:35Z: KIMI GATE PASSED 0.9375 — gold generation launched

- **Kimi-K2.6 think-contract eval (temp 1.0, mnt 16384, EOS-only, 64 games
  seed 777): exact 0.9375 (60/64), format 0.99, valid_guess_rate 1.00, 4.5
  turns, 92 min.** Within noise of the 0.969 scaffold ceiling WITHOUT the
  scaffold; zero invalid-guess deaths (vs 31/64 for q36 base). The prior
  "Kimi failed" verdict was 100% the one-line contract + serving truncation.
  (temp 0.2 arm still finishing; t1.0 is the generation regime regardless.)
- **Gold generation LIVE**: `generate_wordle_gold_sft.py` → 
  `results/wordle_gold_sft/kimi_k26_think_t10_20260612T003446Z/gold.jsonl`
  (512 train-pool targets, eval split excluded, temp 1.0, think style both
  gen and train rows, retries 2, log /tmp/kimi_gold_gen_t10.log). Expected
  ~12h, ~2k solved think+answer turns at ~94% keep rate.
- **Trainer OOM ledger (6 launches, ongoing)**: 4-GPU 35B full-FT + think
  rows OOMs at fwd/bwd. Established: NOT momentum (buffer materializes at
  step, never reached — momentum-0 still required for step-time fit), NOT
  chunk transients alone (chunk 1 still 256MB short after creep), NOT
  gc-able garbage (XORL_FB_PER_CALL_DEFRAG gate `daf662de` had no effect —
  retention is referenced memory). Signature: ~52GB post-load, ~70GB after
  first backward (grad shards), +1.5GB/pipeline creep → OOM in pipeline 1.
  EVAL_SIZE=1 discriminator OOM'd identically (796-token eval) → eval
  retention eliminated. **VERDICT: structurally impossible at shard=4.**
  Accounting closes with muon fp32 master weights (optimizer_dtype only
  governs the momentum buffer): params 17.5 + fp32 masters 17.5 + grads
  17.5 = 52.5GB/GPU before activations — matches 52GB post-load / 70GB
  post-backward exactly. The 2x4 (shard=8) fits because all three halve.
  **ARM PARKED 2026-06-12 ~01:00Z** (8 launches). h100-041's 4 GPUs left
  free as half of a future 2x4 SFT/OPSD block. Do NOT relaunch 4-GPU
  full-FT with think rows; LoRA or shard>=8 only.

## 2026-06-11 ~20:10Z: think-contract OPSD code landed; CoT cache generating

- **Think-contract OPSD support** (`7a67b78f`): hinted teacher prompts render
  open-think for `*_think` styles (were closed-think + guess_only response
  contract = CoT scored fully out-of-distribution); `public_reasoning_think`
  response contract; first-guess truncation ignores `<guess>` mentions inside
  think; NEW `--wordle-think-token-weight` (default 1.0 — think tokens were
  previously zero-weighted via tag_token_weight=0.0, i.e. the CoT contributed
  NOTHING to the KL). 28/28 prompt tests green.
- **Offline reason-first CoT cache** (`692ed728`, user Q2 directive):
  `generate_teacher_cot_cache.py` harvests pre-action states from
  student-policy think rollouts (all turns, not solved-only), pre-generates
  one hinted teacher note per unique state; trainer flag
  `--teacher-reasoning-cache` reads it, misses fall back online. v0 run IN
  FLIGHT on the restarted base-weight sampler pool (512 targets, temp 0.7,
  mnt 8192, `results/wordle_teacher_cot/q36_cot_cache_v0_20260611T200929Z/`,
  log /tmp/cot_cache_gen_v0.log). Known quality wrinkle: notes sometimes
  phrase the reference guess as "the student's guess" — iterate the
  teacher_reasoning contract before relying on it.
- **Sampler pool restarted to base weights** (user-approved) — the one-line
  SFT step-32 weights are gone from serving (DCP checkpoint retains them).
  NOTE: pool pods have no readinessProbe — `1/1 Running` ≠ serving; gate on
  `/v1/models` 200 (took ~25 min).
- Sequencing: hint-only OPSD-think first (no CoTs needed); reason_first arm
  reads the cache once it lands. Kimi gate (t0.2 + t1.0 arms) + q36 floor
  re-measure still in flight, ~2h/turn on Kimi.

## 2026-06-11 ~23:30Z results: TRUE think floor 0.469; CoT cache v0 done; Kimi gang queued

- **q36 base+think TRUE floor = 0.469** (`q36_base_think_nostop_64_*`,
  EOS-only, mnt 16384, temp 0.2, 64 games seed 777): format 0.954,
  turns 4.2, 3.9h wall on the TP=2 teacher. Ladder: one-line 0.219 →
  think 0.469 → scaffold 0.969. The capped 0.422 is superseded.
- **KEY failure mode**: 31/64 games ended invalid_or_unparseable (vs 3
  max-turns) — the binding failure is illegal/repeated/non-candidate
  guesses mid-game, NOT think depth. First target for the hinted teacher
  KL (or a legality-shaped reward): guess validity.
- **Teacher-CoT cache v0 COMPLETE**: 1262/1262 states with notes (mean 26
  words), 90 min total. Path:
  `results/wordle_teacher_cot/q36_cot_cache_v0_20260611T200929Z/cot_cache.jsonl`.
- **HINT-ONLY OPSD-THINK LAUNCHED**: job
  `opsd-wordle-q36-fullft-4g-wszcd` on h100-041 (node-group default override).
  Launch debug ledger (one cycle each): (1) `--wordle-prompt-style` choices
  predated think styles (`7d67798a`); (2) sample_eval scored 0/16 at step 0 —
  the model restates game state before the tag line; strict parser called it
  invalid_action while the 0.469 floor harness extracts leniently → action
  parse now slices preamble before the tag block + SAMPLE_EVAL_TEMPERATURE=0.2
  for protocol parity; (3) CUDA OOM at first train chunk — think rows ~5x
  one-line tokens → OPD_PIPELINE_CHUNK_SIZE 8→2, student mnt 4096→3072.
  W&B `OPSD-WORDLE-Q36-35B-FULLFT-4G-THINK-hintonly-bs64-public-policy-hint`.
  Overlay (`/tmp/opsd_think_hintonly_4g.yaml`) = fullft-4gpu manifest +
  think-contract env: prompt_style `*_think`, student mnt 4096 EOS-only
  (`__NONE__` stop sentinel, `c7458943`), sample_eval/teacher_sample mnt 3072
  EOS-only no-ignore-eos. Teacher = q36 hinted `public_policy_hint`
  (candidates block), reason_first OFF (cache ready for the flip). Expect
  slow steps (~10-20 min: 64 think games/step on 2 samplers). Gate:
  sample_eval exact vs 0.469 floor; watch invalid-guess rate first.
  `schedulerName: volcano` — Kyverno's anchor can't override the API-server
  default; without it the PodGroup is ignored). RDMA fix root cause: pods had
  no /dev/infiniband (no `rdma/infiniband` resource) → NCCL on TCP → 10 tok/s
  aggregate, GPUs 100% util at 130W. Pod-0 holds 8 GPUs on -100 in a benign
  watchdog-restart loop (cheap: crash precedes weight load); pod-1 queued for
  any 8-GPU default-group block. Kimi think evals relaunch when READY fires.
