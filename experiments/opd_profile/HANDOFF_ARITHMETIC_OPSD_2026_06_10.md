# HANDOFF: OPSD/SFT on the nested-arithmetic task (next battlefield)

**Date:** 2026-06-10 · **Branch:** `apanda-dev-prefill-time-compute` (all research lives here; `apanda-dev` carries the four upstreamed infra fixes #352/#353/#354/#355)
**Stack:** `er-opd-q36-35b-slots`, model **Qwen3.6-35B-A3B** (`--model q36` on EVERY generator command), all pods warm and healthy as of handoff.

## 1. Mission and why

The 4×4-multiplication line is **answered**: the exact OPD reproduction hit greedy 0.905, and the SFT ablation (`sft_mode`, plain CE on gold answers, no teacher) **beat it** — 0.948 control, **0.930 on fresh never-trained pairs**, 0.90 by step 15 (~2k examples). Base was greedy-0.66 with majority@8 ≈ greedy (no hidden reliable signal) and pass@8 0.78 < trained 0.93 ⇒ training *repaired* an existing single-pass algorithm; nothing for CoT distillation to add. Memorization ruled out by the fresh-pair probe. Full verdict: memory `project_opd_vs_sft_4x4_verdict.md`.

The new task inverts the profile. **Nested-expression arithmetic** (`/old-data/apanda/no_cot_math_public/arithmetic_problems.jsonl`, 100k problems):

```
{"problem": "Evaluate this Python expression. ((25 - ((-8 // -32) - -90)) + (-25 * -56)) * (55 + 38)", "answer": 124155}
```

Base Q3.6 difficulty probe (n=64, greedy):

| condition | accuracy |
|---|---|
| direct (`/no_think` + `Answer: ` prefill, single pass) | **0.031** |
| thinking mode (CoT, 2048-token budget) | **0.609** |

A **+0.58 serial-compute gap**: single-pass floored, CoT carries the capability. Here "internalize the CoT into one forward pass" is a real falsifiable claim, and OPSD-vs-SFT can genuinely separate (SFT gets ~8 digits of signal per example; OPD ships the whole CoT-conditioned answer distribution).

## 2. Work plan (ordered)

### 2a. Bucketed difficulty probe → pick the training band (DO FIRST, ~15 min)
Bucket the 100k problems by op-count (count operators in `problem`, or parse-tree depth) and measure base direct + CoT accuracy per bucket (reuse the parallel probe pattern below, teacher endpoint, n=64/bucket). Pick a band where **teacher-with-CoT ≥ ~0.8** and **direct ≤ ~0.1** — maximal transferable headroom with tolerable teacher noise. This decision gates everything (the CoT precompute commits hours to the chosen subset). Consider a 2-band curriculum (shallow first) given the bootstrap risk in §4.

Parallel probe pattern (teacher = pristine base, batching is safe post-KV-fix):
ThreadPoolExecutor(16) over `/v1/chat/completions` on `er-opd-q36-35b-slots-teacher-sglang-0:30000`;
direct arm = `messages=[user("/no_think " + problem), assistant("Answer: ")]`, `continue_final_message: true`,
`chat_template_kwargs: {"enable_thinking": false}`, `logprob_start_len: 0`, `stop: ["\n"]`, greedy;
CoT arm = `enable_thinking: true`, `max_tokens 2048+`, parse last integer after `</think>`.

### 2b. Client scorer: `eval_task="arithmetic"`
In `/home/apanda/xorl-client-chat-completions/examples/on_policy_distillation.py`:
- `_score_answer(prompt_text, completion, task)` and `_target_answer_text(prompt_text, task)` are task-keyed — add an `arithmetic` branch. Gold = evaluate the expression from the prompt text (it is plain int Python: `+ - * //` and parens; use `ast.literal_eval`-style safe eval via `ast` whitelist, NOT bare `eval`) — this keeps the scorer self-contained like the multiplication one (no answer file needed at eval time).
- **Answers can be NEGATIVE** — every number-parsing regex on this path must accept a leading `-` (`-?[\d][\d,]*`). The multiplication regexes don't. Check: `_score_answer`, `_target_answer_text`, the gold-replacement splice (`_replace_sampled_answers_with_gold` renders the gold as text — fine), and any probe scripts you copy.
- The answer-logprob/distractor eval tokenizes `_target_answer_text` output — works once the function exists.

### 2c. Data prep (data is FREE — never reuse train prompts for eval)
Pool-json format is a list of single-user-message chats:
`[[{"role": "user", "content": "/no_think Evaluate this Python expression. <expr>"}], ...]`
- Training pool: e.g. 8192 problems from the chosen band → `/shared/opd-coord/arith_<band>_8192_prompts.json`.
- **Disjoint eval set** (1024, same band, zero overlap) → `/shared/opd-coord/arith_<band>_eval_1024.json`; set via candidate `client_args: {eval_prompts_json_path: ...}`. This rule is absolute — the 4×4 tail-contamination measured **+0.08** the moment training wrapped onto eval prompts (science runbook entry 2026-06-10 18:00).
- A second disjoint set for an endpoint fresh-probe (adapt `experiments/opd_profile/autoresearch/fresh_pair_probe.py` — change the prompt builder + gold computation + `-?` regex).

### 2d. Teacher CoT precompute (hours — start early, run on the idle teachers)
Precedent: the `randnum_*_cot.json` files (index-aligned with the prompts json; entries carry the full CoT text ending in `Answer: <gold>`). Look at `experiments/opd_profile/scripts/` for the existing cot-precompute script (the 4×4/5×5 ones were built there) and adapt:
- thinking-mode greedy (or low-T) on each pool problem, **max_tokens ≥ 4096** — the Run-B lesson (memory `project_run_b_cot_mt8192`): a tight budget silently amputated 80% of CoTs once; check truncation rate and raise budget if >2%.
- **Filter to teacher-correct CoTs** (compare final answer to gold; teacher is only ~0.6–0.8 here, unlike 98.7% on 4×4) and rebuild the prompts json to the surviving subset so prompts/CoT stay index-aligned (the `_nonempty` convention).
- Store: `/shared/opd-coord/arith_<band>_{prompts,cot}.json` + a small README line in the json's sibling notes if conventions allow.

### 2e. Candidates
Clone the proven pair and re-point data/task knobs:
- `ARITH-001-SFT` from `PTC-118SFT.yaml` (`sft_mode: true`, `opd_teacher_answer_source: gold`) — the cheap control, run FIRST (no teacher needed, fastest signal on whether answer-only supervision can teach composition at all).
- `ARITH-002-OPD` from `PTC-118R.yaml` (reverse-KL, teacher CoT mode `replace` as in the no-filler recipe).
- Common edits: `prompts_json_path`/`teacher_cot_json_path`/`num_prompts`, `eval_task: arithmetic` (check it's plumbed — it's a top-level candidate key? if not, `client_args`), `client_args.eval_prompts_json_path`, `max_new_tokens: 64` (answers ≤ ~8 digits), keep `lr 3e-6`, `101 × 128`, `eval_accuracy_every: 5`, `eval_control_start_step: 100`, `eval_answer_logprob_every: 5` + distractor.
- Launch: `XORL_REPO worktree` is live; `$PY $GENERATOR --model q36 write-trainer-control --candidate ... --num-steps 101 --prompts-per-step 128 --sampler-replicas 2 --sampler-layout dedicated`. Full ops: `autoresearch/CANONICAL_INFRA_RUNBOOK.md`.

### 2f. Readouts
- Per-step: `eval/accuracy` (= held-out greedy, n=`eval_heldout_num_problems` 256, every 5 steps — the REAL eval since 2026-06-10; `eval/train_window_accuracy` is health-only and reads ~1.0 under sft/gold), gold-logprob + distractor margin.
- Endpoints: step-100 greedy control (n=1024, now on the disjoint set) + the fresh-probe on hot final weights (weights are NOT checkpointed unless you set `save_every` — probe before launching anything else, or add a save).
- Success shapes: SFT ≈ floor → composition not learnable from answers alone (interesting!); OPD ≫ SFT → CoT-conditioned distillation transfers serial computation (the headline OPSD claim); both ≈ floor → 5×5-redux, escalate to curriculum/OPRD.

## 3. Infrastructure state (inherited, all working)

- **All four 2026-06-09/10 bugs fixed**: quack interleave (PR #355), chat rendering + silent context fallback (client pins `chat_template_kwargs={"enable_thinking": false}` + `logprob_start_len=0`, fail-loud on empty `input_token_ids`), sglang scheduler KV-desync (`schedule_batch.py`, batching now safe at full capacity), CE-path kwargs leak (`target_tokens`/`weights` excluded). Post-mortems: `docs/notes/quack_fused_gated_moe_interleave_bug.md`, `docs/notes/sglang_batched_decode_corruption_handoff.md`.
- Samplers: unpinned attention backend, which SGLang auto-resolves to **fa3** on this build/model (same as every validated run — verified in ServerArgs). `--max-running-requests 256 --cuda-graph-max-bs 64`. If you ever pin a DIFFERENT backend explicitly, run the corruption handoff's trained-weights regression gate first.
- `sft_mode` (client) + server CE exclude-keys are committed on the branch; `_heldout_greedy_eval` likewise.

## 4. Pitfalls (tonight's scars, condensed)

1. **Bootstrap risk (the 5×5 lesson)**: at direct≈0.03, OPD's on-policy samples are ~all wrong → wrong-prefix supervision dominates and runs can peak-then-decay while loss falls (memory `project_opd_filler_mechanism_mismatch`, the 5×5 ledger). Watch `opd_kl_answer_wrong_mean` vs `_correct_` and `opd_teacher_entropy_answer_wrong_mean` (the 2026-06-09 diagnostic suite); use the shallow band / curriculum if it bites.
2. **Never trust sampled/in-loop metrics alone**; greedy held-out + serial probes are the arbiters. Loss is NEVER a success signal (EOS reward-hack + overfitting modes — science runbook §5).
3. **Contamination rule**: eval prompts must be provably disjoint; verify with a set-intersection assert like `fresh_pair_probe.py` does.
4. Teachers must NOT be quietly repurposed — they hold pristine base weights and are the reference for every "is it the model or the serving?" probe.
5. Ops: slot agents occasionally wedge on re-exec (re-write the control once, then pod-recreate; runbook §9). `--model q36` on every command — teacher-smg caches `/v1/models` at startup. Stop→wait→relaunch for trainer restarts; never rapid-fire.
6. wandb history ingestion had a service-side outage 2026-06-09 — `opd_profile.jsonl` on /shared is always the authoritative record; backfill pattern exists if it recurs.

## 5. Reference index

- Runbooks: `experiments/opd_profile/autoresearch/CANONICAL_INFRA_RUNBOOK.md` (ops), `CANONICAL_SCIENCE_RUNBOOK.md` (findings + eval methodology incl. the 2026-06-10 metric-semantics boundary).
- Key code: client `sft_mode` / `_heldout_greedy_eval` / `_chat_sampling_extras` / `_score_answer` in `examples/on_policy_distillation.py`; generator `experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py`; probes in `experiments/opd_profile/autoresearch/`.
- Memories: `project_opd_vs_sft_4x4_verdict`, `ptc118-root-cause-chat-rendering`, `quack-fused-gated-interleave-bug`, `opd-logging-overhaul-decay-diagnostics`.
- Task source: `/old-data/apanda/tomi/` (encoded-reasoning paper repo; `eval_chat_hardoff_filler_sweep.py` shows their arithmetic loading/scoring), dataset `/old-data/apanda/no_cot_math_public/arithmetic_problems.jsonl`.
