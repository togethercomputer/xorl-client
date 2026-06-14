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

