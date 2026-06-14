# OPD encoded-reasoning investigation — findings (2026-06-01, overnight)

## TL;DR
The OPD/encoded-reasoning premise — *distill a teacher's reasoning into a student's
filler "thinking buffer"* — does not hold up. The filler-token benefits documented in
the multimodel filler sweep are **format-specific generation effects, not distillable
reasoning**. Every distillation variant tried tonight was negative or moot.

## What ran
1. **Q3-235B-A22B buffer-only CoT-distillation** (teacher sees real CoT, student sees
   pause buffer, distill CoT-hiddens into the buffer; ±hidden-match).
   → `buffer_delta` plateaus ~0.22 over 100+ steps regardless of hidden-match. No
   encoded-reasoning emergence. (Infra win: 235B OPD works at scale, ~85s/step; clean-
   restart recipe for the 3 stale-state bring-up hazards is in memory.)
2. **ICL-distillation (teacher 10-shot, student 0-shot, both + filler)** on Q3-235B.
   → With the format instruction controlled in the system prompt, the 10 examples add
   **+0.025** (0.40→0.425). The apparent ICL benefit on 4dmult is *output-format*, not
   reasoning. Under `/no_think` the model is intuition-capped; ICL transfers nothing.
3. **Filler-benefit reproduction** on Qwen3.5-35B-A3B (the doc's cleanest cell:
   4dmult-10 pause +14.9, "digit-refinement"), in OPD-compatible formats.
   → pause **HURTS**: 0shot+pause 0.70 vs 0shot+nopause 0.80; 10shot+pause 0.775 vs
   10shot+nopause 0.875. Filler placed *after* "Answer:" (the doc's mechanism) → **0.000**
   (model keeps emitting pause, never reaches the number). Our `</think>Answer:` cue
   saturates the baseline to ~0.82 vs the doc's 55.6% regime where pause helps.

## Why
The doc's three filler mechanisms are all *generation* effects, none distillable as a
reasoning buffer:
- **Anti-EOS** (Q3.5-397B, ellipsis +20.1): filler prevents the premature-EOS the
  10-shot demos induce. A 0-shot student has no such pathology → nothing to transfer.
- **Digit-refinement** (Q3.5-35B-A3B, pause +14.9): 100 filler tokens *after* "Answer:"
  give extra forward-passes before the emit — only in the doc's ~55% baseline regime.
- **CoT-priming** (Q3-235B base): filler un-suppresses the base model's reasoning; the
  lifts are reasoning-leak-inflated (18-83% leak) → doc explicitly says avoid Q3-235B.

CoT itself reaches 93-99% on these tasks (the capability is there); fillers unlock only
a small, format-bound, non-distillable sliver.

## Blocker / next steps
- The doc's exact eval harness (V3 prompt + `--prefill-answer` + filler-after-cue
  placement) is **not in the accessible filesystem**. Reproducing the baseline filler
  benefit is a prerequisite to any distillation attempt — get the harness from tomi.
- Or pivot away from the filler/encoded-reasoning hypothesis.

## Cluster state
- All Q3-235B OPD stacks torn down. `q35-35b-teach` sglang (Qwen3.5-35B-A3B, 2 GPUs)
  left running for follow-up measurements. No active training runs.

## UPDATE (03:00): reproduced the doc's format — filler benefit is format-recovery, not capability
Found the harness at `/old-data/apanda/tomi/examples/filler_tokens_rl.py`. The doc's
format = system prompt with a filler *instruction* + few-shot examples demonstrating
`[filler]\nAnswer: [product]` + **free generation** (no forced prefill). 4-digit problems
are generated random pairs (same as ours). Reproduced on Qwen3.5-35B-A3B with the correct
Qwen toggle `chat_template_kwargs={"enable_thinking": false}`:
- clean v3 (direct answer): 0shot **0.75**, 10shot **0.85**
- counting filler: 0shot 0.00 (counts, never reaches Answer: in budget), 10shot **0.85** (= no-filler)

So **no filler lift in a clean setup**, and the model's clean ability (0.85) is *higher*
than the doc's best filler cell (0.705 = 55.6%+14.9). Conclusion: the doc's filler benefit
is the filler partially **rescuing a suppressed/suboptimal baseline**, not adding capability
beyond direct answering. Nothing to distill — using a good answer-format prompt beats any
filler. The encoded-reasoning / filler-distillation hypothesis does not have a real target
on these arithmetic tasks for instruction-capable MoE models.

## UPDATE (03:30): the grading itself rewards the filler convention
The doc's regrader (`parse_filler_response_permissive` in filler_tokens_rl.py) REQUIRES
the filler format: it (1) demands SOME filler before "Answer:", (2) rejects responses
that continue generating after "Answer: X", (3) rejects CoT. So the metric structurally
rewards the filler convention — a no-filler baseline that gets the right number but
doesn't follow the [filler]→Answer→stop shape is graded wrong. Under a plain capability
measure (exact-match on the product, our test), Qwen3.5-35B-A3B answers 4-digit mult at
0.85 with NO filler — equal-or-better than any filler condition. FINAL: the filler
"benefit" = suppressed-baseline format-recovery + grading-convention reward + a tiny
digit-refinement edge; it is not transferable capability. The encoded-reasoning /
filler-distillation program is a dead end for these arithmetic tasks on capable models.
The productive pivot (if desired) is standard CoT→direct-answer distillation on tasks the
model CANNOT already do directly — but that's ordinary distillation, not encoded reasoning.

## UPDATE (07:35): filler-uplift hunt — "find the setting where 10-shot filler helps"
User was confident the doc's 10-shot filler uplift reproduces with some prompt. Exhaustive hunt:
- **Difficulty matters but isn't sufficient.** Qwen3.5-35B-A3B (BF16, valid, TP=2), faithful doc
  format (system filler-instruction + 10-shot examples + free gen, enable_thinking=false, "What is A*B?"):
  filler HURTS/flat at every difficulty at n=200 — 4-digit v3=0.80 (ceiling), 5x4 v3=0.705 (−0.075),
  5x5 v3=0.185 (−0.06). An early 5-digit counting +4.9 was n=80 NOISE. My checkpoint is simply more
  capable at mult than the doc's (0.80 vs doc's 55.6%), so it computes in one pass — no filler-compute benefit.
- **Qwen3.5-397B-A17B-FP8 is BROKEN here.** arch=Qwen3_5MoeForConditionalGeneration (multimodal/vision);
  this sglang build mis-serves it → outputs '!' for everything (incl 0-shot+thinking). My earlier
  "40-50% empty / anti-EOS" numbers were the broken model, now invalidated. (Infra: TP-init hang fixed via
  --disable-custom-all-reduce; OOM/watchdog via small cuda-graph + low concurrency; then '!' garbage.)
- **Resolution:** the doc's +14.9/+20.1 is **format-recovery of a suppressed baseline** (the V3 prompt
  drives the baseline below the model's true ability — under-computing on weak checkpoints, premature-EOS
  on the 397B — and filler recovers it), NOT a genuine capability lift. Matches the parallel
  er-q35-35b-genuine investigation (no clean cross-model lift). The only untested cell is the doc's exact
  397B in **BF16 on TP=16/2 nodes** (188 shards, multimodal) — a confirmatory run of the same recovery mechanism.

## UPDATE (17:00): paper-faithful prefill + pass@k CLOSE the loop — answer to "why not a model where the paper holds"
Re-ran with the paper's EXACT method (V3 prompt + PREFILL filler+`Answer:`, model emits only the
number, 10-shot demos, exact-match grade) — the result is UNCHANGED from free-gen:
- Q3.5-35B-A3B 4dmult-10 prefill: baseline **0.823**, pause −0.040, ellipsis −0.033, counting −0.047.
- **pass@1 vs pass@8** (T=1.0, n120/cell), filler helps NEITHER metric on either servable model:
  - Q3.5-35B: 4x4 base .792/.933 vs pause .767/.942 ; 5x5 base .158/.333 vs pause .125/.275 (HURTS both).
  - Q3.6-35B: 4x4 base .758/.892 vs pause .725/.875 ; **6x6 base 0/0 vs pause .008/.008** — on a task the
    model can't one-shot (0%), 100 filler tokens give zero uplift. Cleanest negative.

**Root cause of the 0.82-vs-0.556 gap (answers the strategic question):** the tomi harness
`run_fresh_filler0_eval_passk.py` defaults to `research-common-08:12345` (UNREACHABLE here),
`DEFAULT_TOKENIZER_MODEL=.../8node-235b-ep/safetensors/step-114`, `DEFAULT_RUN_DIR=.../filler-rl-
multiplication_4digit-8edb7fe2`; `load_eval_pool` pulls eval problems from that RL run's config +
`samples/step_*.jsonl`. So the paper's 0.556→0.705 cell is an **RL-on-filler-finetuned checkpoint,
served on an inaccessible endpoint, evaluated on its own training distribution** — not stock
Qwen3.5-35B-A3B. I matched the harness mechanics; the gap is the checkpoint + problem set.

**Verdict:** (a) no accessible stock model shows any filler lift (pass@1 OR pass@8, 2 models × 3
difficulties × 2 methods); (b) the paper's positive cell is an inaccessible RL checkpoint on its
own train set; (c) the paper's own RL section concludes **filler-as-search (pass@8 not pass@1)**,
which is not distillable into a deterministic forward pass. The encoded-reasoning / distillable-
filler-buffer premise has no empirical support on accessible models. To revive: get network/file
access to the tinker endpoint + RL checkpoint, OR do a from-scratch "Let's Think Dot by Dot"
toy-task training repro with dense supervision (our OPD stack can host this, but the 235B buffer-
distill already plateaued at buffer_delta~0.22).
