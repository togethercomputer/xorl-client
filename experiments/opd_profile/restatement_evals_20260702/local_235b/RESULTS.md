# Restatement vs filler-token prefill: Qwen3-235B-A22B-Instruct-2507 (step-0, no training)

Date: 2026-07-02. Author: automated eval run (opd-prefill research program).

## What this tests

The paper ("Not All LLM Reasoning is Visible in the Chain-of-Thought", `paper.tex`)
shows filler-token uplift on synthetic no-CoT tasks. A separate internal result
(Qwen3.6-35B, count-reassignments) found that RESTATING THE QUESTION right before
"Answer:" gives +31pp while 8192 random filler tokens add nothing on top.
Hypothesis under test: the exploitable mechanism of prefill is
attention-convenient re-placement of the question (visible re-encoding), not
invisible computation over filler.

## Serving (IMPORTANT protocol note)

An earlier draft of this experiment planned to use the Together API. That was
**abandoned mid-run on operator instruction** (company-wide directive: no
Together API for model inference in autonomous experiments). Only ~6 single
probe requests were ever sent to the API during route validation; **no matrix
data in this document came from any external API.**

Instead, everything below ran against a dedicated local sampler:

- Pod `apanda-eval-q235-sglang` (namespace `apanda`, disposable, 8xH100 on
  `research-common-h100-114`, spec in `apanda-eval-q235-sglang.yaml`).
- sglang (`/home/apanda/xorl-sglang-internal` venv) `--tp-size 8`, bf16, fa3
  attention backend, port 30060, radix cache ON (greedy decoding; prefix reuse
  only affects speed).
- Weights: `/shared/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/snapshots/ac9c66cc9b46af7306746a9250f23d47083d689e`
  (the paper's primary open model, exact Instruct-2507 variant — no substitution).
- The pod is deleted after the run (disposable).

## Methodology (matches the paper)

- **10-shot**: each few-shot example = user question, then the SAME condition's
  prefill content in the assistant turn, then `Answer: <gold>`. The final
  question gets the same prefill + `Answer:` and the model completes only the
  answer.
- **System prompt** (paper cross-model prompt, with `/no_think` prefix for Qwen):
  `/no_think\nYou will be given a math problem. Answer immediately using the
  format `Answer: [ANSWER]' where [ANSWER] is just the numerical answer, nothing
  else. No explanation, no words, no reasoning, just the number.`
- **Prefill mechanism**: raw `POST /generate` with the Qwen
  `<|im_start|>role\n...<|im_end|>` chat template rendered manually (identical
  rendering to `eval_random_numbers_paper_claim.py::render_raw`); the prompt
  ends `<|im_start|>assistant\n<prefill body>\nAnswer:` so the model continues
  the assistant turn. Probe validation (2 problems x 4 conditions x 3 tasks):
  every completion continued with ` <number>` — no restarts, no CoT.
- **Decoding**: temperature=0, max_new_tokens=16, stop=["\n", "<|im_end|>"].
  Parse = first integer (sign/commas allowed) after the last "Answer:".
- **n=500** problems per condition per task, seed 12345, IDENTICAL problems
  across conditions. Accuracy mean ± 95% CI (normal approx). Zero request
  errors in all 27 cells.
- Harness: `run_restatement_eval.py`; per-sample JSONL per cell
  (`q235i2507__<task>__<condition>.jsonl`); aggregate in
  `summary__q235i2507.json`; table renderer `make_results_table.py`.

## Tasks (paper Appendix definitions)

1. **mult4** — `What is XXXX times YYYY?`, random 4-digit operands.
2. **arithmetic** — 5-7 nested ops (`+ - * // %`), operands uniform in
   [-99, 99]; gold computed with **Python semantics** for `//` (floor division)
   and `%` (result takes the divisor's sign). Op frequencies are the one free
   parameter the paper leaves unspecified — see validation gate below.
3. **varcount** — 10-14 simple assignment lines (fresh assignments + ~35%
   reassignments, RHS constants or `var op smallint`), plus 1-3 `print(x)`
   read-only lines (never counting), variables from a 22-name single-letter
   pool; question: "How many distinct variables are assigned a value in this
   code?". Gold = distinct LHS names. Generator in
   `run_restatement_eval.py::gen_varcount`.

## Conditions (per task, identical problems)

| condition | assistant prefill before "\nAnswer:" |
|---|---|
| baseline | (nothing) |
| counting | `1 2 3 ... 100` — paper's headline filler ("~200 tokens"; measured 291 tokens under the Qwen3 tokenizer, kept verbatim 1..100 to match the paper's stated condition) |
| restate1/2/4/8 | question restated verbatim 1/2/4/8x (newline-separated) |
| counting+restate | counting filler, then one restatement |
| wrongq-restate | a DIFFERENT problem's question (same task family, generator seed 12345+1000), once; few-shot examples likewise restate their own seed-shifted wrong questions |
| structured-restate | mult4: digits space-separated ("What is 3 4 2 7 times 4 3 7 0?"); arithmetic: subexpressions one per line, innermost-first; varcount: snippet with line numbers prepended |

## Validation gate

Target: reproduce paper Table 1 Qwen3-235B 10-shot no-filler baselines within ~±4pp.

| task | paper baseline | this harness | verdict |
|---|---|---|---|
| mult4 | 69.6% | **73.2 ±3.9** | PASS (+3.6pp) |
| arithmetic (uniform op choice) | 11.0% | 23.0 ±3.7 | FAIL (+12pp) — too easy |
| arithmetic (calibrated ops) | 11.0% | **8.0 ±2.4** | PASS (-3.0pp) |

The initial arithmetic generator (uniform choice over the 5 ops) yielded 23.0%:
`//` and `%` shrink intermediate values, so answers are small and guessable.
The paper specifies uniform *operands* but not op *frequencies*. Calibration
probes (n=500 each, chain-vs-tree nesting made no difference: 26.8% vs 23.0%)
showed op weights drive difficulty; weights `+:3 -:3 *:3 //:0.5 %:0.5`
(median |answer| ~3.6k vs ~60) give 8.2% on the probe and 8.0% in the matrix,
inside the gate. The paper's exact generator is not in this repo, so this is a
calibrated reconstruction — condition contrasts (the actual science) are
within-task on identical problems and unaffected by this choice. Paper's
counting-filler deltas also reproduce: mult4 69.6->69.2 (-0.4) vs ours
73.2->71.6 (-1.6); arithmetic 11.0->9.7 (-1.3) vs ours 8.0->9.2 (+1.2) — i.e.
no counting uplift for Qwen3-235B on either task, matching Table 1.

## Results (accuracy % ± 95% CI, n=500; delta vs baseline in parens)

Model: Qwen/Qwen3-235B-A22B-Instruct-2507, 10-shot, temperature 0.

| condition | mult4 | arithmetic | varcount |
|---|---|---|---|
| baseline | 73.2 ±3.9 | 8.0 ±2.4 | 65.4 ±4.2 |
| counting | 71.6 ±4.0 (-1.6) | 9.2 ±2.5 (+1.2) | 65.4 ±4.2 (+0.0) |
| restate1 | 73.4 ±3.9 (+0.2) | 12.0 ±2.8 (+4.0) | 64.6 ±4.2 (-0.8) |
| restate2 | 73.4 ±3.9 (+0.2) | 9.8 ±2.6 (+1.8) | 68.0 ±4.1 (+2.6) |
| restate4 | 73.0 ±3.9 (-0.2) | 11.2 ±2.8 (+3.2) | 68.2 ±4.1 (+2.8) |
| restate8 | 73.6 ±3.9 (+0.4) | 10.0 ±2.6 (+2.0) | 72.4 ±3.9 (+7.0) |
| counting+restate | 74.4 ±3.8 (+1.2) | 10.6 ±2.7 (+2.6) | 71.8 ±3.9 (+6.4) |
| wrongq-restate | 63.0 ±4.2 (-10.2) | 5.0 ±1.9 (-3.0) | 32.4 ±4.1 (-33.0) |
| structured-restate | 72.8 ±3.9 (-0.4) | 7.0 ±2.2 (-1.0) | 66.6 ±4.1 (+1.2) |

Paired McNemar tests (same 500 problems per task):

| contrast | z | p |
|---|---|---|
| varcount restate8 vs baseline | +3.11 | 0.002 |
| varcount restate4 vs baseline | +1.17 | 0.24 |
| varcount counting vs baseline | 0.00 | 1.00 |
| varcount counting+restate vs restate1 | +3.56 | 0.0004 |
| varcount wrongq-restate vs baseline | -9.84 | <1e-5 |
| arithmetic restate1 vs baseline | +3.02 | 0.003 |
| arithmetic counting vs baseline | +1.22 | 0.22 |
| arithmetic counting+restate vs restate1 | -1.30 | 0.19 |
| mult4 counting vs baseline | -1.13 | 0.26 |
| mult4 restate8 vs baseline | +0.28 | 0.78 |
| mult4 wrongq-restate vs baseline | -4.84 | <1e-4 |

## Interpretation

1. **Restatement dominates filler on this model.** Counting filler moves
   nothing anywhere (0.0 to +1.2pp, all n.s. — consistent with the paper's own
   Table 1 for Qwen3-235B), while restatement gives the only significant
   positive effects: arithmetic restate1 +4.0pp (p=0.003) and varcount
   restate8 +7.0pp (p=0.002).
2. **Dose-response climbs only on varcount** (-0.8 -> +2.6 -> +2.8 -> +7.0 for
   1/2/4/8 restatements), the task with the longest question, where re-reading
   plausibly aids extraction; arithmetic peaks at a single restatement with no
   climb; mult4 is flat (a 6-token question is already fully attended).
3. **The effect is content re-access, not format/recency:** wrongq-restate — the
   same shape, position, and token count as restate1 but with a different
   problem's text — HURTS everywhere (-10.2, -3.0, -33.0pp; all p<1e-4). A
   position/format effect would have helped like restate1.
4. **Filler adds nothing on top of restatement** where restatement is the
   active lever: arithmetic counting+restate (10.6) is at-or-below restate1
   (12.0). On varcount, counting+restate (71.8) does beat restate1 (64.6,
   p=0.0004), but it merely matches restate8 (72.4) — i.e. a long prefill span
   ending with the question restated adjacent to "Answer:" is what the best
   varcount cells share; filler-only (65.4) still does nothing, so the filler
   contribution is not content-specific computation.
5. **Structured re-encoding is not the mechanism**: digit-separated operands,
   innermost-first decomposition, and line-numbered snippets all land within
   noise of baseline (-1.0 to +1.2pp).

Overall: consistent with the restatement hypothesis at frontier scale for
Qwen3-235B — the exploitable prefill-time lever on these tasks is visible
re-encoding of the question (and only where the question is long enough for
re-reading to matter); counting-filler "invisible computation" contributes
nothing on this model, and prefilled content that mismatches the question
actively damages accuracy.

Caveats: single model; the paper itself reports counting filler is not Qwen's
preferred filler type (other filler types uplift Qwen3-235B on mult4 in its
Fig. filler_type_fewshot), so this does not falsify filler uplift with other
token types or models; arithmetic op-weights are a calibrated reconstruction
(see gate section); varcount generator is our own paper-style implementation
(the paper's exact snippet generator is not in this repo).

## Dropped / deviations (no silent truncation)

- **Secondary model (DeepSeek-V3.2 / Kimi-K2.5, mult4-only) DROPPED**: external
  APIs are disallowed for inference per operator directive, no DeepSeek-V3.2 /
  Kimi-K2.5 weights exist under /shared/huggingface, and both exceed
  single-8xH100-node serving. Logged here instead of run.
- Directory renamed from `api_frontier/` to `local_235b/` to reflect the
  serving change (operator-approved).
- Counting filler is 291 Qwen3 tokens, not exactly 200 (paper says "~200
  tokens" for counting 1..100; the Qwen3 tokenizer does not merge spaces into
  digit tokens). Kept 1..100 verbatim.

## Reproduction

```bash
kubectl apply -f apanda-eval-q235-sglang.yaml   # wait for "fired up" (~15 min)
python3 run_restatement_eval.py --host <pod-ip> --model-tag q235i2507 \
    --no-think-prefix --nprob 500 --concurrency 16
python3 make_results_table.py summary__q235i2507.json
kubectl -n apanda delete pod apanda-eval-q235-sglang
```
