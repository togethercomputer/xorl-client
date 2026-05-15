import importlib.util
import sys
from pathlib import Path

import pytest


def _load_example():
    path = (
        Path(__file__).resolve().parents[1] / "examples" / "on_policy_distillation.py"
    )
    spec = importlib.util.spec_from_file_location("on_policy_distillation", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_inference_url_parsing_accepts_urls_and_node_suffixes():
    opd = _load_example()

    assert opd.get_inference_urls("http://student-a:30060,122", 30060) == [
        "http://student-a:30060",
        "http://research-common-122:30060",
    ]


def test_default_prompts_are_stable_and_disjoint():
    opd = _load_example()

    assert opd._default_prompts(2, 4) == [
        [1000, 1001, 1002, 1003],
        [5096, 5097, 5098, 5099],
    ]


def test_chunked_microbatches_preserves_prompt_order():
    opd = _load_example()

    assert opd._chunked([[1], [2], [3], [4], [5]], 2) == [
        [[1], [2]],
        [[3], [4]],
        [[5]],
    ]
    assert opd._chunked([[1], [2]], 0) == [[[1], [2]]]


def test_opd_loss_payload_aligns_cache_indices_with_shifted_tokens():
    opd = _load_example()

    data = opd._opd_loss_data([[10, 11, 12, 13]], [[20, 21, 22]])

    assert data == [
        {
            "model_input": {"input_ids": [10, 11, 12]},
            "loss_fn_inputs": {
                "target_tokens": [11, 12, 13],
                "teacher_ids": [0, 0, 0],
                "teacher_weights": [1.0, 1.0, 1.0],
                "teacher_cache_indices": [20, 21, 22],
            },
        }
    ]


def test_opd_loss_payload_rejects_unshifted_cache_indices():
    opd = _load_example()

    with pytest.raises(RuntimeError, match="cache index length"):
        opd._opd_loss_data([[10, 11, 12, 13]], [[20, 21, 22, 23]])


def test_loss_mean_accepts_tinker_tensor_data_loss_wire_format():
    opd = _load_example()

    output = opd.tomi.ForwardBackwardOutput.from_dict(
        {
            "loss_fn_outputs": [
                {"loss": {"data": [0.25], "dtype": "float32", "shape": [1]}},
                {"loss": {"data": [0.75], "dtype": "float32", "shape": [1]}},
            ],
            "metrics": {"loss:mean": 0.5},
        }
    )

    assert opd._loss_mean(output) == pytest.approx(0.5)
    assert opd._loss_mean_many([output]) == pytest.approx(0.5)


def test_valid_tokens_accepts_tinker_reduction_metric():
    opd = _load_example()

    output = opd.tomi.ForwardBackwardOutput(metrics={"valid_tokens:sum": 19})

    assert opd._valid_tokens(output) == 19
