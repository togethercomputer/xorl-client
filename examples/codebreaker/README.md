# Codebreaker: procedural Mastermind RL (wordle successor)

`codebreaker_env.py` and `codebreaker_scoring.py` are byte-faithful ports of
the codebreaker research program (see slime-snapshot
`examples/codebreaker_multiturn`; the scoring module differs only in its
import line). The env fixes wordle's measured weaknesses: procedural
per-episode alphabets (no word list to memorize, no universal opener),
count-only Mastermind feedback (cross-turn constraint intersection is
mandatory), a k^L difficulty dial, and a feedback-noise toggle that makes
value estimation load-bearing.

The harness mirrors `examples/wordle` (streaming complete-group GRPO through
the xorl tinker-compat API) with codebreaker semantics: burn-turn invalid
handling, candidate tracking for info-bits, the binary reward
(solved + all-turns-format, each exactly 0/1), and GRPO with the obo
zero-variance rescue — uniform all-failed groups train at -1/sqrt(n), uniform
all-solved at +1/sqrt(n), mixed zero-variance groups are dropped.

Run (same endpoint contract as the wordle example):

    python -m examples.codebreaker.train \
      --config examples/codebreaker/configs/cispo.yaml \
      --trainer-url http://127.0.0.1:8384 \
      --generation-url http://127.0.0.1:30004 --sync-url http://127.0.0.1:30004

Set `CODEBREAKER_ENABLE_THINKING=0` to disable thinking-mode chat templates.
Certification tests: `pytest tests/examples/codebreaker -q`.
