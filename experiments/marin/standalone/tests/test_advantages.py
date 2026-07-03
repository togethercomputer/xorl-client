import sys
from types import SimpleNamespace

import pytest
import requests

from experiments.marin.standalone import train_marin_grpo as driver_module
from experiments.marin.standalone.advantages import RolloutRecord, build_drgrpo_datum, compute_group_advantages
from experiments.marin.standalone.dataset import MathExample
from experiments.marin.standalone.train_marin_grpo import (
    _behavior_k3_from_metrics,
    _client_index_for_sample,
    _configure_xorl_client_chunking,
    _derived_train_metrics,
    _effective_max_generate_tokens,
    _estimate_forward_backward_chunks,
    _filter_examples_by_prompt_length,
    _force_nonzero_advantages_for_smoke,
    _make_loss_params,
    _nonzero_advantage_pairs,
    _parse_inference_urls,
    _sampling_seed_for_sample,
    _wandb_log,
)


class _FakeGenerateResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def _sampling_params():
    return driver_module.types.SamplingParams(max_tokens=4, temperature=0.7, top_p=1.0, sampling_seed=3)


def _generate_payload() -> dict:
    return {
        "output_ids": [7, 8],
        "text": "ok",
        "meta_info": {
            "output_token_logprobs": [-0.5, -0.25],
            "finish_reason": {"type": "stop"},
        },
    }


def test_sample_sglang_generate_retries_connection_reset(monkeypatch) -> None:
    calls = []
    sleeps = []

    def fake_post(url, json, timeout):  # noqa: A002 - mirrors requests.post signature
        calls.append((url, json, timeout))
        if len(calls) == 1:
            raise requests.ConnectionError("reset")
        return _FakeGenerateResponse(200, _generate_payload())

    monkeypatch.setattr(driver_module.requests, "post", fake_post)
    monkeypatch.setattr(driver_module.time, "sleep", lambda delay: sleeps.append(delay))

    response = driver_module._sample_sglang_generate(
        base_url="http://sampling",
        input_ids=[1, 2],
        sampling_params=_sampling_params(),
        cache_extra_key="policy-0",
    )

    assert len(calls) == 2
    assert sleeps == [1.0]
    assert calls[-1][1]["extra_key"] == "policy-0"
    assert response.sequences[0].tokens == [7, 8]
    assert response.sequences[0].logprobs == [-0.5, -0.25]


def test_sample_sglang_generate_retries_transient_500(monkeypatch) -> None:
    calls = []
    sleeps = []

    def fake_post(url, json, timeout):  # noqa: A002 - mirrors requests.post signature
        calls.append(url)
        if len(calls) == 1:
            return _FakeGenerateResponse(500, text="backend unavailable")
        return _FakeGenerateResponse(200, _generate_payload())

    monkeypatch.setattr(driver_module.requests, "post", fake_post)
    monkeypatch.setattr(driver_module.time, "sleep", lambda delay: sleeps.append(delay))

    response = driver_module._sample_sglang_generate(
        base_url="http://sampling",
        input_ids=[1, 2],
        sampling_params=_sampling_params(),
        cache_extra_key=None,
    )

    assert len(calls) == 2
    assert sleeps == [1.0]
    assert response.sequences[0].tokens == [7, 8]


def test_compute_group_advantages_zeroes_constant_group() -> None:
    records = [
        RolloutRecord("p0", [1, 2], [3], [-0.1], 1.0),
        RolloutRecord("p0", [1, 2], [4], [-0.2], 1.0),
    ]
    assert compute_group_advantages(records) == [0.0, 0.0]


def test_compute_group_advantages_normalizes_per_prompt_group() -> None:
    records = [
        RolloutRecord("p0", [1, 2], [3], [-0.1], 1.0),
        RolloutRecord("p0", [1, 2], [4], [-0.2], 0.0),
        RolloutRecord("p1", [1, 2], [5], [-0.3], 4.0),
        RolloutRecord("p1", [1, 2], [6], [-0.4], 2.0),
    ]
    assert compute_group_advantages(records) == [1.0, -1.0, 1.0, -1.0]


def test_compute_group_advantages_sample_std_matches_skyrl_semantics() -> None:
    # SkyRL compute_grpo_outcome_advantage: sample std (N-1), always divide by (std + 1e-6).
    records = [
        RolloutRecord("p0", [1, 2], [3], [-0.1], 1.0),
        RolloutRecord("p0", [1, 2], [4], [-0.2], 0.0),
    ]
    advantages = compute_group_advantages(records, std_mode="sample")
    # mean 0.5, sample std = sqrt(((0.5)^2 + (0.5)^2) / 1) = 0.7071...
    expected = 0.5 / (0.7071067811865476 + 1e-6)
    assert advantages[0] == pytest.approx(expected)
    assert advantages[1] == pytest.approx(-expected)


def test_compute_group_advantages_sample_std_constant_group_is_zero() -> None:
    records = [
        RolloutRecord("p0", [1, 2], [3], [-0.1], 1.0),
        RolloutRecord("p0", [1, 2], [4], [-0.2], 1.0),
    ]
    assert compute_group_advantages(records, std_mode="sample") == [0.0, 0.0]


def test_compute_group_advantages_sample_std_singleton_group_keeps_raw_reward() -> None:
    # SkyRL singleton semantics: mean 0, std 1 -> advantage = reward / (1 + 1e-6).
    records = [RolloutRecord("p0", [1, 2], [3], [-0.1], 2.0)]
    advantages = compute_group_advantages(records, std_mode="sample")
    assert advantages[0] == pytest.approx(2.0 / (1.0 + 1e-6))


def test_force_nonzero_advantages_for_smoke_leaves_real_signal_unchanged() -> None:
    advantages, forced = _force_nonzero_advantages_for_smoke([1.0, -1.0])
    assert advantages == [1.0, -1.0]
    assert forced is False


def test_force_nonzero_advantages_for_smoke_balances_all_zero_vector() -> None:
    advantages, forced = _force_nonzero_advantages_for_smoke([0.0, 0.0, 0.0, 0.0])
    assert advantages == [1.0, -1.0, 1.0, -1.0]
    assert forced is True


def test_force_nonzero_advantages_for_smoke_keeps_odd_vector_zero_sum() -> None:
    advantages, forced = _force_nonzero_advantages_for_smoke([0.0, 0.0, 0.0])
    assert advantages == [1.0, -1.0, 0.0]
    assert forced is True


def test_nonzero_advantage_pairs_filters_zero_gradient_records() -> None:
    records = [
        RolloutRecord("p0", [1, 2], [3], [-0.1], 1.0),
        RolloutRecord("p0", [1, 2], [4], [-0.2], 1.0),
        RolloutRecord("p1", [1, 2], [5], [-0.3], 0.0),
    ]

    pairs = _nonzero_advantage_pairs(records, [1.0, 0.0, -2.0])

    assert pairs == [(records[0], 1.0), (records[2], -2.0)]


def test_nonzero_advantage_pairs_rejects_misaligned_inputs() -> None:
    records = [RolloutRecord("p0", [1, 2], [3], [-0.1], 1.0)]
    with pytest.raises(ValueError, match="length mismatch"):
        _nonzero_advantage_pairs(records, [])


def test_parse_inference_urls_accepts_comma_separated_replicas() -> None:
    assert _parse_inference_urls(" http://sglang-0:30000, http://sglang-1:30000/ ") == [
        "http://sglang-0:30000",
        "http://sglang-1:30000",
    ]


def test_parse_inference_urls_rejects_missing_port() -> None:
    with pytest.raises(ValueError, match="host and port"):
        _parse_inference_urls("http://sglang-0")


def test_client_index_for_sample_round_robins_samples_across_replicas() -> None:
    assert [_client_index_for_sample(0, i, 4) for i in range(8)] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert [_client_index_for_sample(1, i, 4) for i in range(4)] == [1, 2, 3, 0]


def test_sampling_seed_for_sample_is_step_stable() -> None:
    args = SimpleNamespace(sampling_seed_base=10, prompts_per_step=4, samples_per_prompt=16)
    assert _sampling_seed_for_sample(args, step=0, example_index=0, sample_index=0) == 10
    assert _sampling_seed_for_sample(args, step=0, example_index=1, sample_index=3) == 29
    assert _sampling_seed_for_sample(args, step=2, example_index=0, sample_index=0) == 138


def test_effective_max_generate_tokens_reserves_sglang_context_slot() -> None:
    args = SimpleNamespace(max_generate_tokens=3584, max_model_tokens=4096, context_token_reserve=1)
    assert _effective_max_generate_tokens(args, prompt_tokens=512) == 3583
    assert _effective_max_generate_tokens(args, prompt_tokens=128) == 3584


def test_filter_examples_by_prompt_length_keeps_fitting_pool() -> None:
    class Tokenizer:
        chat_template = None

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return list(range(len(text)))

    examples = [
        MathExample(prompt="Question: ok?", gold_answer="1", source="rlvr", example_id="ok"),
        MathExample(prompt="Question: " + ("x" * 200), gold_answer="2", source="rlvr", example_id="long"),
    ]
    args = SimpleNamespace(max_prompt_tokens=128, prompt_overlength_policy="skip")

    kept, summary = _filter_examples_by_prompt_length(Tokenizer(), examples, args)

    assert [example.example_id for example in kept] == ["ok"]
    assert summary["enabled"] is True
    assert summary["input_examples"] == 2
    assert summary["kept_examples"] == 1
    assert summary["skipped_examples"] == 1


def test_wandb_log_commits_scalars_and_updates_latest_summary() -> None:
    class DummyRun:
        def __init__(self) -> None:
            self.logs = []
            self.summary = {}

        def log(self, payload, *, commit: bool) -> None:
            self.logs.append((payload, commit))

    run = DummyRun()
    _wandb_log(run, {"rollout/records": 128, "ignored": None, "text": "ok"}, policy_step=3)
    assert run.logs == [({"rollout/records": 128, "text": "ok", "policy_step": 3}, True)]
    assert run.summary == {"latest/rollout/records": 128, "latest/policy_step": 3}


def test_maybe_init_wandb_uses_policy_step_axis(monkeypatch, tmp_path) -> None:
    class DummyRun:
        def __init__(self) -> None:
            self.defined_metrics = []

        def define_metric(self, *args, **kwargs) -> None:
            self.defined_metrics.append((args, kwargs))

    class DummyWandb:
        def __init__(self, run: DummyRun) -> None:
            self.run = run
            self.init_kwargs = None

        def init(self, **kwargs):
            self.init_kwargs = kwargs
            return self.run

    run = DummyRun()
    wandb_module = DummyWandb(run)
    monkeypatch.setitem(sys.modules, "wandb", wandb_module)

    args = SimpleNamespace(
        wandb_project="project",
        wandb_entity=None,
        wandb_name=None,
        wandb_group=None,
        wandb_tags="",
        wandb_mode=None,
    )
    initialized = driver_module._maybe_init_wandb(args, run_id="run-id", output_dir=tmp_path, run_config={})

    assert initialized is run
    assert wandb_module.init_kwargs["project"] == "project"
    assert run.defined_metrics == [
        (("policy_step",), {}),
        (("*",), {"step_metric": "policy_step"}),
    ]


def test_behavior_k3_from_logged_ratio_metrics() -> None:
    metrics = {
        "is_loss/ratio/mean:mean": 1.01,
        "is_loss/kl_policy/mean:mean": 0.02,
    }
    assert _behavior_k3_from_metrics(metrics) == pytest.approx(0.03)


def test_make_loss_params_defaults_logprob_temperature_to_sampling_temperature() -> None:
    args = SimpleNamespace(
        drgrpo_ratio_type="sequence",
        drgrpo_clip_low=0.2,
        drgrpo_clip_high=0.2,
        drgrpo_num_chunks=8,
        temperature=0.7,
        logprob_temperature=None,
    )

    assert _make_loss_params(args)["logprob_temperature"] == pytest.approx(0.7)


def test_make_loss_params_can_use_raw_logprob_temperature() -> None:
    args = SimpleNamespace(
        drgrpo_ratio_type="sequence",
        drgrpo_clip_low=0.2,
        drgrpo_clip_high=0.2,
        drgrpo_num_chunks=8,
        temperature=0.7,
        logprob_temperature=1.0,
    )

    assert _make_loss_params(args)["logprob_temperature"] == pytest.approx(1.0)


def test_make_loss_params_threads_drgrpo_num_chunks() -> None:
    args = SimpleNamespace(
        drgrpo_ratio_type="sequence",
        drgrpo_clip_low=0.2,
        drgrpo_clip_high=0.2,
        drgrpo_num_chunks=128,
        temperature=0.7,
        logprob_temperature=None,
    )

    assert _make_loss_params(args)["num_chunks"] == 128


def test_estimate_forward_backward_chunks_matches_client_limits() -> None:
    datums = [{"model_input": {"input_ids": [1]}, "loss_fn_inputs": {}} for _ in range(5)]

    estimate = _estimate_forward_backward_chunks(
        datums, max_chunk_len=2048, max_chunk_bytes=250, min_tail_chunk_len=0
    )

    assert estimate["chunk_count"] == 3
    assert estimate["chunk_datums_min"] == 1
    assert estimate["chunk_datums_max"] == 2
    assert estimate["datum_count"] == 5


def test_estimate_forward_backward_chunks_merges_short_tail() -> None:
    datums = [{"model_input": {"input_ids": [1]}, "loss_fn_inputs": {}} for _ in range(5)]

    estimate = _estimate_forward_backward_chunks(
        datums, max_chunk_len=2048, max_chunk_bytes=250, min_tail_chunk_len=2
    )

    assert estimate["chunk_count"] == 2
    assert estimate["chunk_datums_max"] == 3
    assert estimate["min_tail_chunk_len"] == 2


def test_configure_xorl_client_chunking_updates_training_client_constants() -> None:
    original_helpers_len = driver_module.xorl_chunked_helpers.MAX_CHUNK_LEN
    original_helpers_bytes = driver_module.xorl_chunked_helpers.MAX_CHUNK_BYTES_COUNT
    original_client_len = driver_module.xorl_training_client_module.MAX_CHUNK_LEN
    original_client_bytes = driver_module.xorl_training_client_module.MAX_CHUNK_BYTES_COUNT
    try:
        config = _configure_xorl_client_chunking(
            SimpleNamespace(
                xorl_client_max_chunk_len=4096,
                xorl_client_max_chunk_bytes=512_000_000,
                xorl_client_min_tail_chunk_len=256,
                xorl_client_sequential_chunks=True,
            )
        )

        assert config["max_chunk_len"] == 4096
        assert config["max_chunk_bytes"] == 512_000_000
        assert config["min_tail_chunk_len"] == 256
        assert driver_module.xorl_chunked_helpers.MAX_CHUNK_LEN == 4096
        assert driver_module.xorl_chunked_helpers.MAX_CHUNK_BYTES_COUNT == 512_000_000
        assert driver_module.xorl_training_client_module.MAX_CHUNK_LEN == 4096
        assert driver_module.xorl_training_client_module.MAX_CHUNK_BYTES_COUNT == 512_000_000
    finally:
        driver_module.xorl_chunked_helpers.MAX_CHUNK_LEN = original_helpers_len
        driver_module.xorl_chunked_helpers.MAX_CHUNK_BYTES_COUNT = original_helpers_bytes
        driver_module.xorl_training_client_module.MAX_CHUNK_LEN = original_client_len
        driver_module.xorl_training_client_module.MAX_CHUNK_BYTES_COUNT = original_client_bytes


def test_derived_train_metrics_reports_valid_tokens_per_second() -> None:
    metrics = _derived_train_metrics({"valid_tokens:sum": 128_000, "execution_time:sum": 2.0})

    assert metrics["valid_tokens_per_s"] == pytest.approx(64_000.0)
    assert metrics["valid_tokens"] == pytest.approx(128_000.0)
    assert metrics["forward_backward_execution_time_s"] == pytest.approx(2.0)


def test_require_sync_success_returns_payload_for_success() -> None:
    payload = driver_module._require_sync_success(
        {"success": True, "transfer_time": 1.25},
        event="initial_weight_sync",
        weight_version="run/policy-000000",
    )

    assert payload["transfer_time"] == pytest.approx(1.25)


def test_require_sync_success_raises_with_event_and_weight_version() -> None:
    with pytest.raises(RuntimeError, match="initial_weight_sync failed weight_version=run/policy-000000: nope"):
        driver_module._require_sync_success(
            {"success": False, "message": "nope"},
            event="initial_weight_sync",
            weight_version="run/policy-000000",
        )


def test_build_drgrpo_datum_masks_prefix_targets_and_aligns_fields() -> None:
    record = RolloutRecord("p0", [10, 11, 12], [20, 21], [-0.5, -0.25], 1.0)
    datum = build_drgrpo_datum(record, 0.75)
    payload = datum.to_dict()
    assert payload["model_input"]["input_ids"] == [10, 11, 12, 20]
    assert payload["loss_fn_inputs"]["target_tokens"]["data"] == [-100, -100, 20, 21]
    assert payload["loss_fn_inputs"]["advantages"]["data"] == [0.0, 0.0, 0.75, 0.75]
    assert payload["loss_fn_inputs"]["logprobs"]["data"] == [0.0, 0.0, -0.5, -0.25]


def test_build_drgrpo_datum_rejects_misaligned_logprobs() -> None:
    record = RolloutRecord("p0", [10, 11], [20, 21], [-0.5], 1.0)
    with pytest.raises(ValueError, match="same length"):
        build_drgrpo_datum(record, 1.0)
