# ZORL Autoresearch Memo - 2026-06-03

## Position

ZORL needs an autoresearch loop that separates long-lived SGLang sampling
infrastructure from short-lived trainer or standalone-client candidates. The
immediate runnable programme should use the existing Countdown/24-puzzle
harness because it already exercises multi-LoRA SGLang sampling, ZORL
candidate scoring, parent updates, parent probes, and final sampler-LoRA eval.

ZORL-OPD now reuses the same substrate through the standalone SGLang client
and OPD multiplication task. ZORL-OPSD should follow the same pattern once its
driver lands. The SGLang side is the shared bottleneck, so the first setup work
is to keep stable, node-pinned SGLang controllers alive on known-free nodes.

## Infrastructure Stance

Available nodes `047`, `117`, and `001` are good SGLang hosts. The
autoresearch candidates render the existing Qwen3-Coder-30B-A3B SGLang
Service+Job into unique services:

- `zorl-ar-sglang-047`
- `zorl-ar-sglang-117`
- `zorl-ar-sglang-001`

Trainer and standalone-client candidates default to `zorl-ar-sglang-047`, but
`INFER_URL` can be overridden at render or launch time.

## First Runnable Programme

1. `ZORL-000`: synthetic-reward smoke gate for ZORL session/export plumbing.
2. `ZORL-001`: current Countdown Modal-style rollout baseline.
3. `ZORL-002`: more ES perturbation pairs with shallower rollout averaging.
4. `ZORL-003`: conservative SGD momentum retest.
5. `ZORL-004`: longer/deeper retest after a weak or promoted signal.
6. `ZORL-OPD-000`: standalone OPD multiplication smoke on 2-digit operands.
7. `ZORL-OPD-001`: standalone OPD multiplication 3-digit run gated by the smoke.

The first scoring target is parent-probe or final exact count on the 8-puzzle
Countdown eval. The default cold baseline is `1/8`; `2/8` is weak signal,
`3/8` promotes a retest, and `5/8` is treated as strong signal.
