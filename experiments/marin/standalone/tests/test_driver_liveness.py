from __future__ import annotations

import json
import types as py_types

import pytest
import requests

import experiments.marin.standalone.train_marin_grpo as driver_module
from experiments.marin.standalone.dataset import MathExample
from experiments.marin.standalone.train_marin_grpo import (
    _forward_backward_result,
    _optim_step_result,
    _score_sglang_response,
    _split_forward_backward_chunks,
    _wait_for_future_result,
    _wait_for_server_future,
)
from experiments.marin.standalone.length_penalty import LengthPenaltyConfig


class _FakeFuture:
    def __init__(self, *, result_value=None, exc: BaseException | None = None) -> None:
        self.result_value = result_value
        self.exc = exc
        self.timeout_seen = None
        self.cancelled = False

    def result(self, timeout=None):  # noqa: ANN001 - matches Future.result
        self.timeout_seen = timeout
        if self.exc is not None:
            raise self.exc
        return self.result_value

    def cancel(self) -> bool:
        self.cancelled = True
        return True


class _FakeTrainingClient:
    def __init__(self) -> None:
        self.forward_backward_calls = []
        self.optim_step_calls = []
        self.model_id = "default"

    def forward_backward(self, datums, loss_fn, loss_fn_params):  # noqa: ANN001 - xorl-client compatible fake
        self.forward_backward_calls.append(
            {
                "datums": list(datums),
                "loss_fn": loss_fn,
                "loss_fn_params": loss_fn_params,
            }
        )
        return _FakeFuture(result_value=_FakeForwardBackwardOutput(num_outputs=len(datums)))

    def optim_step(self, adam_params):  # noqa: ANN001 - xorl-client compatible fake
        self.optim_step_calls.append(adam_params)
        return _FakeFuture(result_value=_FakeOptimOutput())

    def _convert_datums(self, datums):  # noqa: ANN001 - xorl-client compatible fake
        return list(datums), []


class _FakeDirectTrainingClient(_FakeTrainingClient):
    def __init__(self) -> None:
        super().__init__()
        self.holder = py_types.SimpleNamespace(base_url="http://trainer")
        self._request_id_counter = 0
        self._turn_counter = 0
        self._turn_waiters = {}

    def _get_request_id(self) -> int:
        request_id = self._request_id_counter
        self._request_id_counter += 1
        return request_id


class _FakeForwardBackwardOutput:
    def __init__(self, *, num_outputs: int = 1) -> None:
        self.loss_fn_output_type = "drgrpo"
        self.loss_fn_outputs = [object()] * num_outputs
        self.metrics = {"valid_tokens:sum": float(num_outputs), "execution_time:sum": 1.0}


class _FakeOptimOutput:
    metrics = {"grad_norm": 0.25}


def _read_events(path):
    return [json.loads(line)["event"] for line in path.read_text(encoding="utf-8").splitlines()]


def test_wait_for_future_result_logs_success(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    future = _FakeFuture(result_value={"ok": True})

    result = _wait_for_future_result(
        future,
        label="optim_step",
        timeout_s=12.5,
        metrics_path=metrics_path,
        step=3,
        policy_step=4,
        event_prefix="train_future",
    )

    assert result == {"ok": True}
    assert future.timeout_seen == 12.5
    assert _read_events(metrics_path) == ["train_future_start", "train_future_done"]


def test_wait_for_future_result_logs_timeout_and_cancels(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    future = _FakeFuture(exc=TimeoutError("stalled"))

    with pytest.raises(TimeoutError, match="forward_backward timed out after 7.0s"):
        _wait_for_future_result(
            future,
            label="forward_backward",
            timeout_s=7.0,
            metrics_path=metrics_path,
            event_prefix="train_future",
        )

    assert future.cancelled is True
    assert _read_events(metrics_path) == ["train_future_start", "train_future_timeout"]


def test_forward_backward_wrapper_uses_xorl_client_future(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    training_client = _FakeTrainingClient()
    datums = [{"model_input": {"input_ids": [1, 2, 3]}, "loss_fn_inputs": {"labels": [2, 3, 4]}}]

    result = _forward_backward_result(
        training_client,
        datums,
        loss_fn="drgrpo",
        loss_fn_params={"beta": 0.0},
        sequential_chunks=False,
        timeout_s=30.0,
        metrics_path=metrics_path,
        step=5,
        policy_step=6,
    )

    assert isinstance(result, _FakeForwardBackwardOutput)
    assert training_client.forward_backward_calls == [
        {
            "datums": datums,
            "loss_fn": "drgrpo",
            "loss_fn_params": {"beta": 0.0},
        }
    ]
    assert _read_events(metrics_path) == [
        "train_future_xorl_client_forward_backward",
        "train_future_submit_done",
        "train_future_start",
        "train_future_done",
    ]


def test_split_forward_backward_chunks_merges_small_tail(monkeypatch) -> None:
    datums = [
        {"model_input": {"input_ids": [idx]}, "loss_fn_inputs": {"labels": [idx]}}
        for idx in range(10)
    ]
    monkeypatch.setattr(driver_module, "estimate_datum_bytes", lambda _datum: 10)

    chunks = _split_forward_backward_chunks(
        datums,
        max_chunk_len=4,
        max_chunk_bytes=40,
        min_tail_chunk_len=3,
    )

    assert [len(chunk) for chunk in chunks] == [4, 6]


def test_forward_backward_wrapper_can_send_sequential_chunks(tmp_path, monkeypatch) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    training_client = _FakeTrainingClient()
    datums = [
        {"model_input": {"input_ids": [1]}, "loss_fn_inputs": {"labels": [1]}},
        {"model_input": {"input_ids": [2]}, "loss_fn_inputs": {"labels": [2]}},
        {"model_input": {"input_ids": [3]}, "loss_fn_inputs": {"labels": [3]}},
    ]
    chunks = [datums[:2], datums[2:]]
    monkeypatch.setattr(
        driver_module,
        "_estimate_forward_backward_chunks",
        lambda _datums, **_kwargs: {
            "chunk_count": 2,
            "datum_count": 3,
            "datum_estimated_bytes_total": 300,
            "chunk_estimated_bytes_max": 200,
        },
    )
    monkeypatch.setattr(driver_module, "_split_forward_backward_chunks", lambda _datums, **_kwargs: chunks)

    result = _forward_backward_result(
        training_client,
        datums,
        loss_fn="drgrpo",
        loss_fn_params={"beta": 0.0},
        sequential_chunks=True,
        timeout_s=30.0,
        metrics_path=metrics_path,
        step=5,
        policy_step=6,
    )

    assert result.metrics["valid_tokens:sum"] == 3.0
    assert result.metrics["execution_time:sum"] == 2.0
    assert [call["datums"] for call in training_client.forward_backward_calls] == chunks
    assert _read_events(metrics_path) == [
        "train_future_xorl_client_forward_backward",
        "train_future_start",
        "train_future_chunk_submit_done",
        "train_future_chunk_start",
        "train_future_chunk_done",
        "train_future_chunk_submit_done",
        "train_future_chunk_start",
        "train_future_chunk_done",
        "train_future_done",
    ]


def test_forward_backward_wrapper_can_use_direct_server_futures(tmp_path, monkeypatch) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    training_client = _FakeDirectTrainingClient()
    datums = [
        {"model_input": {"input_ids": [1]}, "loss_fn_inputs": {"labels": [1]}},
        {"model_input": {"input_ids": [2]}, "loss_fn_inputs": {"labels": [2]}},
        {"model_input": {"input_ids": [3]}, "loss_fn_inputs": {"labels": [3]}},
    ]
    chunks = [datums[:2], datums[2:]]
    submitted_seq_ids = []
    retrieve_ids = []

    def fake_post_json(url, payload, *, timeout_s):  # noqa: ANN001 - test double
        if url.endswith("/api/v1/forward_backward"):
            submitted_seq_ids.append(payload["seq_id"])
            return {"request_id": f"future-{payload['seq_id']}"}
        if url.endswith("/api/v1/retrieve_future"):
            retrieve_ids.append(payload["request_id"])
            return {
                "loss_fn_output_type": "drgrpo",
                "loss_fn_outputs": [{}],
                "metrics": {"valid_tokens:sum": 1.0, "execution_time:sum": 1.0},
                "info": {},
            }
        raise AssertionError(url)

    monkeypatch.setattr(
        driver_module,
        "_estimate_forward_backward_chunks",
        lambda _datums, **_kwargs: {
            "chunk_count": 2,
            "datum_count": 3,
            "datum_estimated_bytes_total": 300,
            "chunk_estimated_bytes_max": 200,
        },
    )
    monkeypatch.setattr(driver_module, "_split_forward_backward_chunks", lambda _datums, **_kwargs: chunks)
    monkeypatch.setattr(driver_module, "_post_json", fake_post_json)

    result = _forward_backward_result(
        training_client,
        datums,
        loss_fn="drgrpo",
        loss_fn_params={"beta": 0.0},
        sequential_chunks=True,
        timeout_s=30.0,
        metrics_path=metrics_path,
        step=5,
        policy_step=6,
        direct_server_futures=True,
    )

    assert result.metrics["valid_tokens:sum"] == 2.0
    assert result.metrics["execution_time:sum"] == 2.0
    assert submitted_seq_ids == [1, 2]
    assert retrieve_ids == ["future-1", "future-2"]
    assert training_client._turn_counter == 2
    assert "train_future_chunk_submit_done" in _read_events(metrics_path)


def test_wait_for_server_future_retries_transient_retrieve_timeout(tmp_path, monkeypatch) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    calls = []

    def fake_post_json(url, payload, *, timeout_s):  # noqa: ANN001 - test double
        calls.append((url, payload, timeout_s))
        if len(calls) == 1:
            raise requests.ReadTimeout("stale long poll")
        return {"ok": True}

    monkeypatch.setattr(driver_module, "_post_json", fake_post_json)
    monkeypatch.setattr(driver_module.time, "sleep", lambda _seconds: None)

    result = _wait_for_server_future(
        train_url="http://trainer",
        server_request_id="future-1",
        timeout_s=30.0,
        metrics_path=metrics_path,
        label="forward_backward",
        step=5,
        policy_step=6,
        event_prefix="train_future_chunk",
    )

    assert result == {"ok": True}
    assert [call[1]["request_id"] for call in calls] == ["future-1", "future-1"]
    assert _read_events(metrics_path) == ["train_future_chunk_poll_retry", "train_future_chunk_done"]


def test_optim_step_wrapper_uses_xorl_client_future(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    training_client = _FakeTrainingClient()
    adam_params = object()

    result = _optim_step_result(
        training_client,
        adam_params,
        timeout_s=30.0,
        metrics_path=metrics_path,
        step=5,
        policy_step=6,
    )

    assert isinstance(result, _FakeOptimOutput)
    assert training_client.optim_step_calls == [adam_params]
    assert _read_events(metrics_path) == ["train_future_start", "train_future_done"]


def test_score_sglang_response_builds_rollout_payload() -> None:
    response = driver_module.types.SampleResponse(
        sequences=[
            driver_module.types.SampledSequence(
                tokens=[10, 11],
                logprobs=[-0.1, -0.2],
                text="Reasoning\nAnswer: 2",
                stop_reason="stop",
            )
        ],
        meta_info={},
    )
    example = MathExample(prompt="What is 1+1?", gold_answer="2", source="unit", example_id="unit-1")
    prefix = py_types.SimpleNamespace(token_ids=[1, 2, 3])

    scored = _score_sglang_response(
        response=response,
        example=example,
        prefix=prefix,
        client_index=0,
        sample_index=7,
        sampling_seed=123,
        request_max_tokens=16,
        length_config=LengthPenaltyConfig(min_response_length=0),
        step=5,
        cache_extra_key="policy-1",
    )

    assert len(scored["records"]) == 1
    assert scored["records"][0].reward == 1.0
    assert scored["correctness"] == [1.0]
    assert scored["prompt_correctness"] == [("unit-1", True)]
    assert scored["rollout_events"][0]["mean_neg_logprob"] == pytest.approx(0.15)
    assert scored["sample_text_events"][0]["predicted_answer"] == "2"
