import math

import pytest

from xorl_client.rl import compute_grpo_advantages


def test_grouped_advantages_preserve_order_and_use_population_std():
    values, stats = compute_grpo_advantages(
        [1.0, 10.0, 3.0, 14.0],
        group_ids=["a", "b", "a", "b"],
        return_stats=True,
    )
    assert values == pytest.approx([-1.0, -1.0, 1.0, 1.0])
    assert stats["a"].mean == 2.0
    assert stats["b"].std == 2.0


def test_zero_variance_groups_are_zero():
    assert compute_grpo_advantages([2.0, 2.0], group_ids=[0, 0]) == [0.0, 0.0]


def test_grouped_advantages_reject_bad_inputs():
    with pytest.raises(ValueError, match="cardinality"):
        compute_grpo_advantages([1.0], group_ids=[])
    with pytest.raises(ValueError, match="finite"):
        compute_grpo_advantages([math.nan], group_ids=[0])
