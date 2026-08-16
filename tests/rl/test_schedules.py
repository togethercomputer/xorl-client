import pytest

from xorl_client.rl import learning_rate_at_step


def test_cosine_schedule_warms_up_and_reaches_floor():
    values = [
        learning_rate_at_step(
            base_learning_rate=1e-4,
            step=step,
            total_steps=4,
            schedule="cosine",
            warmup_steps=1,
            min_learning_rate=1e-5,
        )
        for step in range(1, 5)
    ]
    assert values[0] == pytest.approx(1e-4)
    assert values[-1] == pytest.approx(1e-5)
    assert values == sorted(values, reverse=True)
