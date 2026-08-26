"""Trajectory reward / correctness scoring for the Codebreaker recipe.

Mirrors ``wordle_scoring.score_trajectory``: recompute the environment reward from a finished
multi-turn trajectory's per-turn samples; the recipe picks ``cfg.reward_key`` out of the blob.
``binary_reward`` is the PRIMARY training reward for this env (the wordle binary phase showed
shaped rewards pre-solve the credit-assignment problem the estimators are supposed to
demonstrate): accuracy + format, each exactly 0 or 1, so the train objective coincides with
the eval metric and within-group variance among equally-(un)solved games is zero (obo keeps
uniform groups trainable). The legacy shaped-reward zoo is deliberately not ported; a sparse
solve(+speed) reward is kept for parity experiments.

Duck-typed trajectory (built by codebreaker_rollout): ``.samples`` / ``.solved`` /
``.stopped_reason``; each sample has ``.format_ok`` / ``.single_guess_tag`` / ``.valid_guess``
/ ``.guess`` (symbol tuple or None) / ``.feedback`` ((exact, close) or None) / ``.info_bits``
/ ``.constraint_consistent``.
"""

from __future__ import annotations

from . import codebreaker_env


# Turn-conditioned success grid (xorl aa05361): F(k) = 1{solved using <= k turns}, cumulative
# in k. Logged for BOTH training rollouts (rollout/solve_le_k, see rollout_metrics.py) and eval
# (eval/<name>/solve_le_k, see eval_metrics.py). solve_le_10 == exact_match whenever
# max_turns <= 10 — the -re series protocol — since a game can never use more turns than that.
SOLVE_TURN_KS = (4, 5, 6, 7, 8, 9, 10)


def score_trajectory(trajectory, *, code_length: int, max_turns: int = codebreaker_env.MAX_TURNS) -> dict[str, float]:
    samples = trajectory.samples
    turns_used = len(samples)
    format_hits = sum(1 for s in samples if s.format_ok)
    single_tag_hits = sum(1 for s in samples if s.single_guess_tag)
    valid_hits = sum(1 for s in samples if s.valid_guess)

    format_rate = format_hits / max(turns_used, 1)
    single_guess_tag_rate = single_tag_hits / max(turns_used, 1)
    valid_guess_rate = valid_hits / max(turns_used, 1)
    invalid_action = bool(any(not s.valid_guess for s in samples))

    # per-turn feedback richness (diagnostic only; never a training reward in the binary phase)
    info_scores = [
        codebreaker_env._info_score(s.feedback[0], s.feedback[1], code_length)
        for s in samples
        if s.valid_guess and s.feedback is not None
    ]
    info_gain = sum(info_scores) / max(len(info_scores), 1)
    info_bits_total = sum(float(s.info_bits) for s in samples if s.valid_guess and s.info_bits >= 0.0)

    # cross-turn information use: fraction of graded guesses that fit every prior feedback;
    # 1.0 when no turn was applicable (vacuous) — same convention as wordle
    consistency_checks = [s.constraint_consistent for s in samples if s.constraint_consistent is not None]
    constraint_consistent_rate = (
        sum(1.0 for c in consistency_checks if c) / len(consistency_checks) if consistency_checks else 1.0
    )

    binary_reward = float(trajectory.solved) + (1.0 if turns_used > 0 and format_hits == turns_used else 0.0)
    sparse_reward = codebreaker_env._sparse_reward(
        solved=trajectory.solved, turns_used=turns_used, max_turns=max_turns
    )

    score = {
        "binary_reward": float(binary_reward),
        "sparse_reward": float(sparse_reward),
        # reward: alias of the primary key so harness paths with a bare .get("reward") stay sane
        "reward": float(binary_reward),
        "exact_match": float(trajectory.solved),
        "format_rate": float(format_rate),
        "single_guess_tag_rate": float(single_guess_tag_rate),
        "valid_guess_rate": float(valid_guess_rate),
        "info_gain": float(info_gain),
        "info_bits_total": float(info_bits_total),
        "constraint_consistent_rate": float(constraint_consistent_rate),
        "turns_used": float(turns_used),
        "invalid_action": float(invalid_action),
    }
    # turn-conditioned success F(k) (see SOLVE_TURN_KS): a solved game's final sample is the
    # solving guess, so turns_used IS its turns-to-solve; burned/invalid turns count — F(k)
    # answers "solved within a budget of k turns", not "k valid guesses"
    for k in SOLVE_TURN_KS:
        score[f"solve_le_{k}"] = float(trajectory.solved and turns_used <= k)
    return score
