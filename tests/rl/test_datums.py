import pytest

from xorl_client.rl import IGNORE_INDEX, build_policy_datum, build_policy_loss_inputs


def test_policy_inputs_are_next_token_aligned_and_mask_prompt():
    values = build_policy_loss_inputs(
        [10, 11, 20, 21], prompt_len=2, old_logprobs=[-0.2, -0.3], advantage=1.5
    )
    assert values["target_tokens"] == [IGNORE_INDEX, 20, 21]
    assert values["logprobs"] == [0.0, -0.2, -0.3]
    assert values["advantages"] == [0.0, 1.5, 1.5]


def test_policy_datum_contains_shifted_model_input():
    datum = build_policy_datum(
        prompt_tokens=[10, 11],
        output_tokens=[20, 21],
        old_logprobs=[-0.2, -0.3],
        advantage=-1.0,
    )
    assert datum.model_input.to_ints() == [10, 11, 20]
    assert datum.loss_fn_inputs["target_tokens"].tolist() == [IGNORE_INDEX, 20, 21]


def test_policy_inputs_fail_closed_on_alignment_errors():
    with pytest.raises(ValueError, match="alignment"):
        build_policy_loss_inputs([10, 20], 1, [], 1.0)
    with pytest.raises(ValueError, match="generated token"):
        build_policy_loss_inputs([10], 1, [], 1.0)
