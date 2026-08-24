"""Value-model client helpers: datum dtypes, state_values output, GAE, EV."""

import math

import pytest

from xorl_client import compute_skip_observation_gae, explained_variance
from xorl_client.types.datum import Datum
from xorl_client.types.forward_backward_output import LossFnOutput
from xorl_client.types.model_input import ModelInput
from xorl_client.types.tensor_data import TensorData


def test_datum_infers_float32_for_value_fields():
    datum = Datum(
        model_input=ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            "target_tokens": [2, 3, 4],
            "weights": [1.0, 1.0, 0.0],
            "returns": [0.5, 0.25, 0.0],
            "old_values": [0.4, 0.2, 0.0],
        },
    )
    converted = datum.loss_fn_inputs
    assert converted["returns"].dtype == "float32"
    assert converted["old_values"].dtype == "float32"
    assert converted["target_tokens"].dtype == "int64"


def test_loss_fn_output_state_values_roundtrip():
    payload = {
        "state_values": {"data": [0.1, 0.2, 0.3], "dtype": "float32", "shape": [3]},
        "elementwise_loss": {"data": [1.0, 1.0, 1.0], "dtype": "float32", "shape": [3]},
    }
    output = LossFnOutput.from_dict(payload)
    assert isinstance(output.state_values, TensorData)
    assert output.state_values.data == [0.1, 0.2, 0.3]
    # Dict-like tinker-compat surface.
    assert "state_values" in output
    assert output["state_values"].data == [0.1, 0.2, 0.3]
    assert "state_values" in output.keys()
    assert output.logprobs is None
    assert "logprobs" not in output
    assert output.to_dict()["state_values"]["data"] == [0.1, 0.2, 0.3]


def test_skip_observation_gae_skips_observations():
    rewards = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    values = [0.5, 9.9, -9.9, 0.4, 9.9, 0.2]
    action_mask = [1, 0, 0, 1, 0, 1]
    adv, ret = compute_skip_observation_gae(rewards, values, action_mask, gamma=1.0, lam=1.0)
    # Action-token chain: bootstraps action-to-action across observation gaps.
    assert adv[5] == pytest.approx(1.0 - 0.2)
    assert adv[3] == pytest.approx((0.2 - 0.4) + adv[5])
    assert adv[0] == pytest.approx((0.4 - 0.5) + adv[3])
    for t in (1, 2, 4):
        assert adv[t] == 0.0 and ret[t] == 0.0
    for t in (0, 3, 5):
        assert ret[t] == pytest.approx(adv[t] + values[t])


def test_explained_variance_from_value_loss_metrics():
    returns = [0.0, 1.0, 2.0, 3.0]
    mean_r = 1.5
    sq_mean = sum(r * r for r in returns) / 4
    var_r = sq_mean - mean_r**2
    assert explained_variance(0.0, mean_r, sq_mean) == pytest.approx(1.0)
    assert explained_variance(var_r, mean_r, sq_mean) == pytest.approx(0.0)
    assert math.isnan(explained_variance(0.1, 1.0, 1.0))
