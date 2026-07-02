import pytest

from experiments.marin.standalone.length_penalty import LengthPenaltyConfig, length_penalty_reward, shaped_reward


def test_length_penalty_rewards_shorter_correct_completions() -> None:
    config = LengthPenaltyConfig(max_completion_tokens=100, target_length=20)
    short = length_penalty_reward(correct=True, completion_tokens=30, has_box=True, truncated=False, config=config)
    long = length_penalty_reward(correct=True, completion_tokens=90, has_box=True, truncated=False, config=config)
    assert short > long
    assert 0.0 <= long <= short <= 1.0


def test_length_penalty_matches_skyrl_wrong_and_truncated_regimes() -> None:
    config = LengthPenaltyConfig(max_completion_tokens=100)
    assert length_penalty_reward(correct=True, completion_tokens=100, has_box=True, truncated=True, config=config) == -2.0
    assert length_penalty_reward(correct=False, completion_tokens=10, has_box=False, truncated=False, config=config) == -1.0


def test_length_penalty_lpw_zero_is_legacy_plus_minus_one() -> None:
    config = LengthPenaltyConfig(max_completion_tokens=100, lpw=0.0)
    assert length_penalty_reward(correct=True, completion_tokens=5, has_box=True, truncated=True, config=config) == 1.0
    assert length_penalty_reward(correct=False, completion_tokens=5, has_box=False, truncated=False, config=config) == -1.0


def test_shaped_reward_uses_reference_reward_not_verifier_plus_bonus() -> None:
    config = LengthPenaltyConfig(max_completion_tokens=100, target_length=20, lpw=0.5)
    reward = shaped_reward(
        verifier_reward=1.0,
        correct=True,
        completion_tokens=50,
        has_box=True,
        truncated=False,
        config=config,
    )
    assert reward == pytest.approx(length_penalty_reward(correct=True, completion_tokens=50, has_box=True, truncated=False, config=config))
