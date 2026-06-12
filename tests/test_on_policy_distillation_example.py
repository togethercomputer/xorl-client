import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_sampler_metrics_urls_are_explicit_or_inferred():
    opd = _load_example()

    inference_urls = ["http://student-a:8080", "http://student-b:8080"]

    assert opd.get_sampler_metrics_urls("", inference_urls, 29000) == []
    assert opd.get_sampler_metrics_urls("http://metrics-a:29000", inference_urls, 29000) == [
        "http://metrics-a:29000"
    ]
    assert opd.get_sampler_metrics_urls("auto", inference_urls, 29000) == [
        "http://student-a:29000",
        "http://student-b:29000",
    ]


def test_old_logprobs_align_to_shifted_generated_token_positions():
    opd = _load_example()

    old_logprobs = opd._old_logprobs_for_generated_spans(
        sequence_len=10,
        spans=[
            (4, [-0.4, -0.5, -0.6]),
            (8, [-0.8, -0.9]),
        ],
    )

    assert old_logprobs == [0.0, 0.0, 0.0, -0.4, -0.5, -0.6, 0.0, -0.8, -0.9]


def test_sampler_metrics_delta_reports_worker_balance():
    opd = _load_example()

    before = {
        "snapshots": {
            "http://dispatch:29000": opd._parse_smg_metrics(
                """
                smg_worker_cb_outcomes_total{worker="http://sampler-0:30060",outcome="success"} 10
                smg_worker_cb_outcomes_total{worker="http://sampler-1:30000",outcome="success"} 12
                smg_worker_selection_total{policy="round_robin",model="m"} 22
                smg_router_requests_total{endpoint="chat"} 22
                smg_router_upstream_responses_total{status_code="200"} 22
                smg_http_responses_total{path="/v1/chat/completions",status_code="200"} 22
                smg_http_connections_active 0
                smg_http_inflight_request_age_count{gt="0",le="30"} 0
                smg_worker_pool_size{model="m"} 2
                smg_worker_health{worker="http://sampler-0:30060"} 1
                smg_worker_health{worker="http://sampler-1:30000"} 1
                """
            )
        },
        "errors": {},
    }
    after = {
        "snapshots": {
            "http://dispatch:29000": opd._parse_smg_metrics(
                """
                smg_worker_cb_outcomes_total{worker="http://sampler-0:30060",outcome="success"} 158
                smg_worker_cb_outcomes_total{worker="http://sampler-1:30000",outcome="success"} 160
                smg_worker_selection_total{policy="round_robin",model="m"} 318
                smg_router_requests_total{endpoint="chat"} 318
                smg_router_upstream_responses_total{status_code="200"} 318
                smg_http_responses_total{path="/v1/chat/completions",status_code="200"} 318
                smg_http_connections_active 0
                smg_http_inflight_request_age_count{gt="0",le="30"} 0
                smg_worker_pool_size{model="m"} 2
                smg_worker_health{worker="http://sampler-0:30060"} 1
                smg_worker_health{worker="http://sampler-1:30000"} 1
                """
            )
        },
        "errors": {},
    }

    metrics = opd._sampler_metrics_delta(before, after)

    assert metrics["sampler_metrics_available"] == pytest.approx(1.0)
    assert metrics["sampler_router_requests_delta"] == pytest.approx(296.0)
    assert metrics["sampler_router_upstream_responses_delta"] == pytest.approx(296.0)
    assert metrics["sampler_http_chat_responses_delta"] == pytest.approx(296.0)
    assert metrics["sampler_new_outstanding_requests"] == pytest.approx(0.0)
    assert metrics["sampler_http_connections_active"] == pytest.approx(0.0)
    assert metrics["sampler_http_inflight_request_age_count"] == pytest.approx(0.0)
    assert metrics["sampler_worker_selection_delta"] == pytest.approx(296.0)
    assert metrics["sampler_worker_count"] == 2
    assert metrics["sampler_worker_active_count"] == 2
    assert metrics["sampler_worker_success_delta_total"] == pytest.approx(296.0)
    assert metrics["sampler_worker_success_delta_min"] == pytest.approx(148.0)
    assert metrics["sampler_worker_success_delta_max"] == pytest.approx(148.0)
    assert metrics["sampler_worker_success_balance_ratio"] == pytest.approx(1.0)
    assert metrics["sampler_worker_health_min"] == pytest.approx(1.0)
    assert metrics["sampler_policy_round_robin_active"] == pytest.approx(1.0)


def test_sampler_quiescence_waits_for_new_requests_to_drain(monkeypatch):
    opd = _load_example()

    baseline = {
        "snapshots": {
            "http://dispatch:29000": {
                "router_requests_total": 100.0,
                "router_upstream_responses_total": 100.0,
                "http_connections_active": 0.0,
                "http_inflight_request_age_count": 0.0,
            }
        },
        "errors": {},
    }
    snapshots = [
        {
            "snapshots": {
                "http://dispatch:29000": {
                    "router_requests_total": 108.0,
                    "router_upstream_responses_total": 105.0,
                    "http_connections_active": 3.0,
                    "http_inflight_request_age_count": 3.0,
                }
            },
            "errors": {},
        },
        {
            "snapshots": {
                "http://dispatch:29000": {
                    "router_requests_total": 108.0,
                    "router_upstream_responses_total": 108.0,
                    "http_connections_active": 0.0,
                    "http_inflight_request_age_count": 0.0,
                }
            },
            "errors": {},
        },
    ]

    def fake_fetch(urls, timeout):
        assert urls == ["http://dispatch:29000"]
        assert timeout == pytest.approx(0.25)
        return snapshots.pop(0)

    monkeypatch.setattr(opd, "_fetch_sampler_metrics", fake_fetch)

    metrics, _ = opd._wait_for_sampler_quiescence(
        ["http://dispatch:29000"],
        baseline,
        metrics_timeout=0.25,
        timeout_s=1.0,
        poll_s=0.001,
        max_new_outstanding=0.0,
        max_connections_active=0.0,
        max_inflight=0.0,
    )

    assert metrics["sampler_quiesce_success"] == pytest.approx(1.0)
    assert metrics["sampler_quiesce_poll_count"] == pytest.approx(2.0)
    assert metrics["sampler_quiesce_new_requests"] == pytest.approx(8.0)
    assert metrics["sampler_quiesce_new_responses"] == pytest.approx(8.0)
    assert metrics["sampler_quiesce_new_outstanding"] == pytest.approx(0.0)
    assert metrics["sampler_quiesce_connections_active"] == pytest.approx(0.0)


def test_sampler_quiescence_fails_closed_when_connections_remain_active(monkeypatch):
    opd = _load_example()

    baseline = {
        "snapshots": {
            "http://dispatch:29000": {
                "router_requests_total": 100.0,
                "router_upstream_responses_total": 100.0,
                "http_connections_active": 0.0,
                "http_inflight_request_age_count": 0.0,
            }
        },
        "errors": {},
    }
    current = {
        "snapshots": {
            "http://dispatch:29000": {
                "router_requests_total": 108.0,
                "router_upstream_responses_total": 108.0,
                "http_connections_active": 1.0,
                "http_inflight_request_age_count": 0.0,
            }
        },
        "errors": {},
    }

    monkeypatch.setattr(opd, "_fetch_sampler_metrics", lambda urls, timeout: current)

    metrics, _ = opd._wait_for_sampler_quiescence(
        ["http://dispatch:29000"],
        baseline,
        metrics_timeout=0.25,
        timeout_s=0.0,
        poll_s=0.001,
        max_new_outstanding=0.0,
        max_connections_active=0.0,
        max_inflight=0.0,
    )

    assert metrics["sampler_quiesce_success"] == pytest.approx(0.0)
    assert metrics["sampler_quiesce_new_outstanding"] == pytest.approx(0.0)
    assert metrics["sampler_quiesce_connections_active"] == pytest.approx(1.0)


def test_default_prompts_are_stable_and_disjoint():
    opd = _load_example()

    assert opd._default_prompts(2, 4) == [
        [1000, 1001, 1002, 1003],
        [5096, 5097, 5098, 5099],
    ]


def test_chat_completions_defaults_to_message_prompts():
    opd = _load_example()

    config = opd.Config(inference_api_format="chat_completions", num_prompts=2)

    prompts = opd._load_prompts(config)

    assert prompts == [
        [
            {
                "role": "user",
                "content": "Write a concise answer to synthetic OPD prompt 0.",
            }
        ],
        [
            {
                "role": "user",
                "content": "Write a concise answer to synthetic OPD prompt 1.",
            }
        ],
    ]


def test_prompts_json_is_capped_by_num_prompts():
    opd = _load_example()

    config = opd.Config(
        prompts_json=json.dumps([[1], [2], [3]]),
        num_prompts=2,
    )

    assert opd._load_prompts(config) == [[1], [2]]


def test_prompts_json_path_is_capped_by_num_prompts(tmp_path):
    opd = _load_example()
    prompts_path = tmp_path / "prompts.json"
    prompts_path.write_text(json.dumps([[1], [2], [3], [4]]), encoding="utf-8")

    config = opd.Config(
        prompts_json_path=str(prompts_path),
        num_prompts=2,
    )

    assert opd._load_prompts(config) == [[1], [2]]


def test_chat_completions_trajectory_uses_returned_input_token_ids():
    opd = _load_example()

    sampled = opd.tomi.SampledSequence(tokens=[30, 31], prompt_tokens=[10, 11, 12])

    assert opd._sampled_sequence_tokens("hello", sampled) == [10, 11, 12, 30, 31]


def test_chat_completions_trajectory_requires_input_token_ids():
    opd = _load_example()

    sampled = opd.tomi.SampledSequence(tokens=[30, 31])

    with pytest.raises(RuntimeError, match="input_token_ids"):
        opd._sampled_sequence_tokens("hello", sampled)


def test_chat_completions_trajectory_can_use_tokenizer_fallback():
    opd = _load_example()

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert messages == [{"role": "user", "content": "hello"}]
            assert tokenize is True
            assert add_generation_prompt is True
            return [1, 2, 3]

    # None = backend did not offer ids at all -> tokenizer fallback allowed.
    sampled = opd.tomi.SampledSequence(tokens=[30, 31], prompt_tokens=None)

    assert opd._sampled_sequence_tokens("hello", sampled, FakeTokenizer()) == [
        1,
        2,
        3,
        30,
        31,
    ]

    # EMPTY ids = backend accepted the request but withheld its rendering
    # (logprob_start_len missing). Silent local re-render here caused the
    # 2026-06-10 train/sample context mismatch -> must raise.
    sampled_empty = opd.tomi.SampledSequence(tokens=[30, 31], prompt_tokens=[])
    with pytest.raises(RuntimeError, match="EMPTY input_token_ids"):
        opd._sampled_sequence_tokens("hello", sampled_empty, FakeTokenizer())


def test_chat_completions_tokenizer_fallback_accepts_single_batched_ids():
    opd = _load_example()

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return [[1, 2, 3]]

    assert opd._encode_chat_prompt("hello", FakeTokenizer()) == [1, 2, 3]


def test_chat_completions_tokenizer_fallback_accepts_input_ids_dict():
    opd = _load_example()

    class FakeTensor:
        def tolist(self):
            return [[1, 2, 3]]

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return {"input_ids": FakeTensor()}

    assert opd._encode_chat_prompt("hello", FakeTokenizer()) == [1, 2, 3]


def test_chat_completions_tokenizer_fallback_accepts_input_ids_attr():
    opd = _load_example()

    class FakeBatchEncoding:
        input_ids = [1, 2, 3]

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return FakeBatchEncoding()

    assert opd._encode_chat_prompt("hello", FakeTokenizer()) == [1, 2, 3]


def test_token_id_list_parser_accepts_json_and_csv():
    opd = _load_example()

    assert opd._parse_token_id_list("[1, 2, 3]") == [1, 2, 3]
    assert opd._parse_token_id_list("4, 5 6") == [4, 5, 6]
    assert opd._parse_token_id_list("") == []


def test_stop_sequence_parser_accepts_json_and_pipe_separator():
    opd = _load_example()

    assert opd._parse_stop_sequences('["\\n", "</answer>"]') == ["\n", "</answer>"]
    assert opd._parse_stop_sequences("END|</answer>") == ["END", "</answer>"]
    assert opd._parse_stop_sequences("") == []


def test_sample_student_batch_uses_exact_prefill_token_ids():
    opd = _load_example()

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert messages == [{"role": "user", "content": "hello"}]
            assert tokenize is True
            assert add_generation_prompt is True
            return [10, 11, 12]

        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            assert text == "Answer: "
            return [50, 51]

    class FakeClient:
        def __init__(self):
            self.calls = []

        def sample(self, prompt, sampling_params, num_samples=1, return_logprobs=False):
            self.calls.append(
                {
                    "prompt": prompt,
                    "max_tokens": sampling_params.max_tokens,
                    "continue_final_message": sampling_params.chat_continue_final_message,
                    "stop": sampling_params.stop,
                    "num_samples": num_samples,
                    "return_logprobs": return_logprobs,
                }
            )

            async def _coro():
                return SimpleNamespace(
                    sequences=[
                        SimpleNamespace(tokens=[90, 91], prompt_tokens=None, text="ok")
                    ]
                )

            return _coro()

    client = FakeClient()
    sequences, prompt_lens, k, completions, group_size, _old_lps, _k_per_sample = asyncio.run(
        opd._sample_student_batch(
            [client],
            [[{"role": "user", "content": "hello"}]],
            max_new_tokens=8,
            temperature=1.0,
            chat_tokenizer=FakeTokenizer(),
            student_prefill_token_ids="1001,1002",
            student_prefill_suffix="Answer: ",
            student_stop_sequences='["\\n"]',
        )
    )

    assert client.calls[0]["prompt"].to_ints() == [10, 11, 12, 1001, 1002, 50, 51]
    assert client.calls[0]["continue_final_message"] is False
    assert client.calls[0]["stop"] == ["\n"]
    assert sequences == [[10, 11, 12, 1001, 1002, 50, 51, 90, 91]]
    assert prompt_lens == [3]
    assert k == 4
    assert completions == [[90, 91]]
    assert group_size == 1


def test_sample_student_batch_times_out_waiting_for_sampler():
    opd = _load_example()

    class HangingClient:
        def sample(self, **kwargs):
            return asyncio.get_event_loop().create_future()

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(
            opd._sample_student_batch(
                [HangingClient()],
                [[1, 2, 3]],
                max_new_tokens=8,
                temperature=1.0,
                request_timeout=0.01,
            )
        )


def test_buffer_control_eval_uses_exact_prefill_token_ids():
    opd = _load_example()

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert messages == [{"role": "user", "content": "hello"}]
            assert tokenize is True
            assert add_generation_prompt is True
            return [10, 11, 12]

        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            assert text == "Answer: "
            return [50, 51]

    class FakeClient:
        def __init__(self):
            self.calls = []
            self.await_call_counts = []

        def sample(self, prompt, sampling_params, num_samples=1):
            self.calls.append(
                {
                    "prompt": prompt,
                    "max_tokens": sampling_params.max_tokens,
                    "continue_final_message": sampling_params.chat_continue_final_message,
                    "stop": sampling_params.stop,
                    "num_samples": num_samples,
                }
            )

            async def _coro():
                self.await_call_counts.append(len(self.calls))
                return SimpleNamespace(
                    sequences=[
                        SimpleNamespace(tokens=[90], prompt_tokens=None, text="42")
                    ]
                )

            return _coro()

    client = FakeClient()
    config = opd.Config(
        student_prefill_token_ids="[1001, 1002]",
        student_prefill_suffix="Answer: ",
        student_stop_sequences='["\\n"]',
        max_new_tokens=32,
        eval_max_new_tokens=64,
        eval_task="none",
    )
    metrics, _rows = asyncio.run(
        opd._buffer_control_eval(
            config,
            [client],
            [[{"role": "user", "content": "hello"}]],
            FakeTokenizer(),
        )
    )

    assert [call["prompt"].to_ints() for call in client.calls] == [
        [10, 11, 12, 1001, 1002, 50, 51],
        [10, 11, 12, 50, 51],
        [10, 11, 12, 1002, 1001, 50, 51],
    ]
    assert [call["continue_final_message"] for call in client.calls] == [False, False, False]
    assert [call["stop"] for call in client.calls] == [["\n"], ["\n"], ["\n"]]
    assert [call["max_tokens"] for call in client.calls] == [64, 64, 64]
    assert client.await_call_counts == [3, 3, 3]
    assert metrics["eval/control_total_requests"] == pytest.approx(3.0)
    assert metrics["eval/control_num_arms"] == pytest.approx(3.0)
    assert metrics["eval/control_sampler_clients"] == pytest.approx(1.0)
    assert metrics["eval/control_arms_concurrent"] == pytest.approx(1.0)
    assert metrics["eval/control_max_concurrency"] == pytest.approx(3.0)
    assert metrics["eval/control_configured_max_concurrency"] == pytest.approx(0.0)
    assert metrics["eval/control_bounded_concurrency_active"] == pytest.approx(0.0)
    assert metrics["eval/control_corrupt_pause_active"] == pytest.approx(1.0)
    assert metrics["eval/pause_request_latency_mean_s"] >= 0.0
    assert metrics["eval/nopause_request_latency_p95_s"] >= 0.0
    assert metrics["eval/corrupt_pause_request_latency_max_s"] >= 0.0
    assert metrics["eval/control_request_latency_max_s"] >= 0.0
    assert metrics["eval/control_client_queue_latency_max_s"] >= 0.0
    assert metrics["eval/control_service_latency_max_s"] >= 0.0


def test_buffer_control_eval_respects_max_concurrency():
    opd = _load_example()

    class SlowClient:
        def __init__(self):
            self.calls = 0
            self.active = 0
            self.max_active = 0

        def sample(self, prompt, sampling_params, num_samples=1):
            del prompt, sampling_params, num_samples
            self.calls += 1

            async def _coro():
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                try:
                    await asyncio.sleep(0.01)
                    return SimpleNamespace(
                        sequences=[
                            SimpleNamespace(tokens=[90], prompt_tokens=None, text="42")
                        ]
                    )
                finally:
                    self.active -= 1

            return _coro()

    client = SlowClient()
    progress_events = []
    config = opd.Config(
        student_prefill_text="pause ",
        student_prefill_count=1,
        student_prefill_suffix="Answer: ",
        eval_control_max_concurrency=2,
        eval_task="none",
        max_new_tokens=4,
    )
    prompts = [[{"role": "user", "content": f"prompt {idx}"}] for idx in range(4)]
    metrics, _rows = asyncio.run(
        opd._buffer_control_eval(
            config,
            [client],
            prompts,
            None,
            progress_callback=progress_events.append,
        )
    )

    assert client.calls == 12
    assert client.max_active <= 2
    assert progress_events[0]["eval/control_progress_completed_requests"] == pytest.approx(0.0)
    assert progress_events[-1]["eval/control_progress_completed_requests"] == pytest.approx(12.0)
    assert progress_events[-1]["eval/control_progress_completion_frac"] == pytest.approx(1.0)
    assert metrics["eval/control_total_requests"] == pytest.approx(12.0)
    assert metrics["eval/control_max_concurrency"] == pytest.approx(2.0)
    assert metrics["eval/control_configured_max_concurrency"] == pytest.approx(2.0)
    assert metrics["eval/control_bounded_concurrency_active"] == pytest.approx(1.0)
    assert metrics["eval/control_client_queue_latency_max_s"] > 0.0
    assert metrics["eval/control_service_latency_mean_s"] >= 0.0


def test_buffer_control_eval_scores_answer_logprob_margin_with_exact_prefill():
    opd = _load_example()

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is True
            assert add_generation_prompt is True
            assert messages == [{"role": "user", "content": "Calculate: 2 * 21"}]
            return [10, 11, 12]

        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            mapping = {
                "Answer: ": [50, 51],
                "42": [4, 2],
            }
            return mapping[text]

        def decode(self, token_ids, skip_special_tokens=False):
            assert skip_special_tokens is False
            return " ".join(str(token_id) for token_id in token_ids)

    class FakeClient:
        def sample(self, prompt, sampling_params, num_samples=1):
            async def _coro():
                return SimpleNamespace(
                    sequences=[
                        SimpleNamespace(tokens=[4, 2], prompt_tokens=None, text="42")
                    ]
                )

            return _coro()

        async def score_prompt_logprobs_batch_async(self, input_ids_batch, logprob_start_lens):
            outputs = []
            for seq, start in zip(input_ids_batch, logprob_start_lens):
                prefix = seq[:-2]
                assert start == len(prefix) - 1
                if prefix == [10, 11, 12, 1001, 1002, 50, 51]:
                    answer_lps = [-0.1, -0.2]
                elif prefix == [10, 11, 12, 1002, 1001, 50, 51]:
                    answer_lps = [-0.6, -0.8]
                else:
                    assert prefix == [10, 11, 12, 50, 51]
                    answer_lps = [-1.1, -1.2]
                outputs.append([None] * len(prefix) + answer_lps)
            return outputs

    config = opd.Config(
        student_prefill_token_ids="[1001, 1002]",
        student_prefill_suffix="Answer: ",
        eval_task="multiplication",
        eval_answer_logprob_control=True,
        eval_max_new_tokens=4,
    )
    metrics, _rows = asyncio.run(
        opd._buffer_control_eval(
            config,
            [FakeClient()],
            [[{"role": "user", "content": "Calculate: 2 * 21"}]],
            FakeTokenizer(),
        )
    )

    assert metrics["eval/answer_logprob_control_active"] == pytest.approx(1.0)
    assert metrics["eval/answer_logprob_control_available"] == pytest.approx(1.0)
    assert metrics["eval/answer_logprob_total_requests"] == pytest.approx(3.0)
    assert metrics["eval/answer_logprob_scored_pause"] == pytest.approx(1.0)
    assert metrics["eval/answer_logprob_scored_nopause"] == pytest.approx(1.0)
    assert metrics["eval/answer_logprob_scored_corrupt_pause"] == pytest.approx(1.0)
    assert metrics["eval/answer_logprob_mean_pause"] == pytest.approx(-0.15)
    assert metrics["eval/answer_logprob_mean_nopause"] == pytest.approx(-1.15)
    assert metrics["eval/answer_logprob_mean_corrupt_pause"] == pytest.approx(-0.7)
    assert metrics["eval/answer_logprob_margin"] == pytest.approx(1.0)
    assert metrics["eval/answer_logprob_vs_corrupt_margin"] == pytest.approx(0.55)
    assert metrics["eval/answer_logprob_causal_margin"] == pytest.approx(0.55)
    assert metrics["eval/answer_logprob_request_failure_frac"] == pytest.approx(0.0)
    assert metrics["eval/answer_logprob_group_latency_mean_s"] >= 0.0
    assert metrics["eval/answer_logprob_group_latency_p95_s"] >= 0.0
    assert metrics["eval/answer_logprob_group_latency_max_s"] >= 0.0


def test_answer_logprob_eval_accepts_sglang_start_offset_shape():
    opd = _load_example()

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            mapping = {
                "42": [4, 2],
                " pause": [1001],
                "Answer: ": [50, 51],
            }
            return mapping[text]

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is True
            assert add_generation_prompt is True
            return [10, 11, 12]

    class FakeSglangClient:
        async def score_prompt_logprobs_batch_async(self, input_ids_batch, logprob_start_lens):
            outputs = []
            for seq, start in zip(input_ids_batch, logprob_start_lens):
                prefix = seq[:-2]
                assert start == len(prefix) - 1
                if prefix == [10, 11, 12, 1001, 50, 51]:
                    outputs.append([None, -0.1, -0.2])
                elif prefix == [10, 11, 12, 50, 51]:
                    outputs.append([None, -1.1, -1.2])
                else:
                    outputs.append([None, -0.6, -0.8])
            return outputs

    config = opd.Config(eval_task="multiplication", eval_answer_logprob_control=True)
    metrics = asyncio.run(
        opd._answer_logprob_control_eval(
            config,
            [FakeSglangClient()],
            [[{"role": "user", "content": "Calculate: 2 * 21"}]],
            FakeTokenizer(),
            (
                ("pause", [1001, 50, 51]),
                ("nopause", [50, 51]),
                ("corrupt_pause", [2001, 50, 51]),
            ),
        )
    )

    assert metrics["eval/answer_logprob_request_failure_frac"] == pytest.approx(0.0)
    assert metrics["eval/answer_logprob_margin"] == pytest.approx(1.0)
    assert metrics["eval/answer_logprob_vs_corrupt_margin"] == pytest.approx(0.55)


def test_answer_logprob_eval_chunks_batches_and_bounds_concurrency():
    opd = _load_example()

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            if text == "Answer: ":
                return [50]
            return [100 + int(ch) for ch in text]

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is True
            assert add_generation_prompt is True
            prompt_text = messages[-1]["content"]
            prompt_idx = int(prompt_text.split("prompt ")[1].split(":")[0])
            return [10, prompt_idx]

    class ChunkRecordingClient:
        def __init__(self):
            self.batch_sizes = []
            self.active = 0
            self.max_active = 0

        async def score_prompt_logprobs_batch_async(self, input_ids_batch, logprob_start_lens):
            self.batch_sizes.append(len(input_ids_batch))
            assert len(input_ids_batch) <= 2
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await asyncio.sleep(0.01)
                outputs = []
                for seq, start in zip(input_ids_batch, logprob_start_lens):
                    answer_len = 2
                    prefix = seq[:-answer_len]
                    assert start == len(prefix) - 1
                    if 101 in prefix:
                        answer_lps = [-0.1, -0.2]
                    elif 201 in prefix:
                        answer_lps = [-0.6, -0.8]
                    else:
                        answer_lps = [-1.1, -1.2]
                    outputs.append([None] * len(prefix) + answer_lps)
                return outputs
            finally:
                self.active -= 1

    client = ChunkRecordingClient()
    progress_events = []
    prompts = [
        [{"role": "user", "content": f"prompt {idx}: Calculate: 2 * 21"}]
        for idx in range(4)
    ]
    config = opd.Config(
        eval_task="multiplication",
        eval_answer_logprob_control=True,
        eval_answer_logprob_batch_size=2,
        eval_answer_logprob_max_concurrency=2,
    )
    metrics = asyncio.run(
        opd._answer_logprob_control_eval(
            config,
            [client],
            prompts,
            FakeTokenizer(),
            (
                ("pause", [101, 50]),
                ("nopause", [50]),
                ("corrupt_pause", [201, 50]),
            ),
            progress_callback=progress_events.append,
        )
    )

    assert client.batch_sizes == [2, 2, 2, 2, 2, 2]
    assert client.max_active <= 2
    assert progress_events[0]["eval/answer_logprob_progress_completed_requests"] == pytest.approx(0.0)
    assert progress_events[-1]["eval/answer_logprob_progress_completed_requests"] == pytest.approx(12.0)
    assert progress_events[-1]["eval/answer_logprob_progress_completed_chunks"] == pytest.approx(6.0)
    assert progress_events[-1]["eval/answer_logprob_progress_completion_frac"] == pytest.approx(1.0)
    assert metrics["eval/answer_logprob_total_requests"] == pytest.approx(12.0)
    assert metrics["eval/answer_logprob_configured_batch_size"] == pytest.approx(2.0)
    assert metrics["eval/answer_logprob_batch_size"] == pytest.approx(2.0)
    assert metrics["eval/answer_logprob_chunk_count"] == pytest.approx(6.0)
    assert metrics["eval/answer_logprob_configured_max_concurrency"] == pytest.approx(2.0)
    assert metrics["eval/answer_logprob_max_concurrency"] == pytest.approx(2.0)
    assert metrics["eval/answer_logprob_bounded_concurrency_active"] == pytest.approx(1.0)
    assert metrics["eval/answer_logprob_request_failure_frac"] == pytest.approx(0.0)
    assert metrics["eval/answer_logprob_margin"] == pytest.approx(1.0)
    assert metrics["eval/answer_logprob_vs_corrupt_margin"] == pytest.approx(0.55)
    assert metrics["eval/answer_logprob_client_queue_latency_max_s"] > 0.0


def test_answer_logprob_eval_scores_distractor_answer_selection():
    opd = _load_example()

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            mapping = {
                "42": [4, 2],
                "30": [3, 0],
                "Answer: ": [50],
            }
            return mapping[text]

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is True
            assert add_generation_prompt is True
            text = messages[-1]["content"]
            if "2 * 21" in text:
                return [10]
            if "3 * 10" in text:
                return [20]
            raise AssertionError(text)

    class FakeClient:
        async def score_prompt_logprobs_batch_async(self, input_ids_batch, logprob_start_lens):
            outputs = []
            for seq, start in zip(input_ids_batch, logprob_start_lens):
                target = seq[-2:]
                prefix = seq[:-2]
                assert start == len(prefix) - 1
                prompt_token = prefix[0]
                correct = [4, 2] if prompt_token == 10 else [3, 0]
                is_correct = target == correct
                if 101 in prefix:
                    answer_lps = [-0.1, -0.1] if is_correct else [-1.1, -1.1]
                elif 201 in prefix:
                    answer_lps = [-0.3, -0.3] if is_correct else [-1.0, -1.0]
                else:
                    answer_lps = [-0.5, -0.5] if is_correct else [-0.9, -0.9]
                outputs.append([None] * len(prefix) + answer_lps)
            return outputs

    config = opd.Config(
        eval_task="multiplication",
        eval_answer_logprob_control=True,
        eval_answer_logprob_distractor_control=True,
        eval_answer_logprob_distractor_offset=1,
    )
    metrics = asyncio.run(
        opd._answer_logprob_control_eval(
            config,
            [FakeClient()],
            [
                [{"role": "user", "content": "Calculate: 2 * 21"}],
                [{"role": "user", "content": "Calculate: 3 * 10"}],
            ],
            FakeTokenizer(),
            (
                ("pause", [101, 50]),
                ("nopause", [50]),
                ("corrupt_pause", [201, 50]),
            ),
        )
    )

    assert metrics["eval/answer_logprob_total_requests"] == pytest.approx(12.0)
    assert metrics["eval/answer_logprob_distractor_control_active"] == pytest.approx(1.0)
    assert metrics["eval/answer_logprob_distractor_pairs"] == pytest.approx(2.0)
    assert metrics["eval/answer_logprob_distractor_skipped"] == pytest.approx(0.0)
    assert metrics["eval/answer_logprob_distractor_scored_pause"] == pytest.approx(2.0)
    assert metrics["eval/answer_logprob_select_margin_pause"] == pytest.approx(1.0)
    assert metrics["eval/answer_logprob_select_margin_nopause"] == pytest.approx(0.4)
    assert metrics["eval/answer_logprob_select_margin_corrupt_pause"] == pytest.approx(0.7)
    assert metrics["eval/answer_logprob_select_delta"] == pytest.approx(0.6)
    assert metrics["eval/answer_logprob_select_vs_corrupt_delta"] == pytest.approx(0.3)
    assert metrics["eval/answer_logprob_select_causal_delta"] == pytest.approx(0.3)


def test_corrupt_prefill_text_can_preserve_boundary_whitespace():
    opd = _load_example()

    text = " ! | ~ "

    assert opd._corrupt_prefill_text(text, mode="rotate") == "| ~ ! "
    assert opd._corrupt_prefill_text(text, mode="rotate_preserve_ws") == " | ~ ! "
    assert opd._corrupt_prefill_text(text, mode="rotate-preserve-ws") == " | ~ ! "
    assert opd._corrupt_token_ids([1, 2, 3], mode="rotate_preserve_ws") == [2, 3, 1]


def test_buffer_control_eval_reports_corrupt_prefill_boundary_metrics():
    opd = _load_example()

    class FakeClient:
        def sample(self, prompt, sampling_params, num_samples=1):
            async def _coro():
                return SimpleNamespace(
                    sequences=[
                        SimpleNamespace(tokens=[4, 2], prompt_tokens=None, text="42")
                    ]
                )

            return _coro()

    config = opd.Config(
        student_prefill_text=" ! | ~ ",
        student_prefill_count=1,
        student_prefill_suffix="Answer: ",
        eval_corrupt_pause_mode="rotate_preserve_ws",
        eval_task="none",
        max_new_tokens=4,
    )
    metrics, _rows = asyncio.run(
        opd._buffer_control_eval(
            config,
            [FakeClient()],
            [[{"role": "user", "content": "prompt"}]],
            None,
        )
    )

    assert metrics["eval/control_corrupt_pause_mode_preserve_boundary_ws"] == pytest.approx(1.0)
    assert metrics["eval/control_corrupt_pause_leading_ws_match"] == pytest.approx(1.0)
    assert metrics["eval/control_corrupt_pause_trailing_ws_match"] == pytest.approx(1.0)
    assert metrics["eval/control_corrupt_pause_len_delta_chars"] == pytest.approx(0.0)
    assert metrics["eval/control_corrupt_pause_changed_chars"] > 0.0
    assert metrics["eval/control_corrupt_pause_change_frac"] > 0.0


def test_generation_health_reports_artifact_metrics():
    opd = _load_example()

    class FakeTokenizer:
        def decode(self, toks, skip_special_tokens=False):
            assert skip_special_tokens is False
            mapping = {
                (1,): "! | ~ _ * ^ # @ Answer: 123 step-by-step\nextra",
                (2, 3, 4): "42</think>",
            }
            return mapping[tuple(toks)]

    metrics, rows = opd._generation_health(
        [[1], [2, 3, 4]],
        [
            "Calculate: 1 * 123",
            "Calculate: 2 * 21",
        ],
        FakeTokenizer(),
        "multiplication",
        max_new_tokens=3,
        filler_marker=" ! | ~ _ * ^ # @ ",
        answer_cue_marker="Answer: ",
        stop_sequences=["\n"],
    )

    assert metrics["eval/filler_leak_frac"] == pytest.approx(0.5)
    assert metrics["eval/answer_cue_leak_frac"] == pytest.approx(0.5)
    assert metrics["eval/stop_sequence_seen_frac"] == pytest.approx(0.5)
    assert metrics["eval/reasoning_phrase_frac"] == pytest.approx(0.5)
    assert metrics["eval/cap_hit_frac"] == pytest.approx(0.5)
    assert metrics["eval/has_think_close_frac"] == pytest.approx(0.5)
    # Renamed 2026-06-10: scored training samples are a health signal, not the
    # real eval (eval/accuracy is now the held-out greedy eval).
    assert metrics["eval/train_window_accuracy"] == pytest.approx(1.0)
    assert rows[0]["completion"].startswith("! | ~")


def test_buffer_control_eval_reports_sampler_artifact_metrics():
    opd = _load_example()

    class FakeClient:
        def __init__(self):
            self.responses = iter(
                [
                    ("111 Answer:\n", [1, 1, 1]),
                    ("111 step-by-step", [1, 1, 1]),
                    ("42", [4, 2]),
                    ("43", [4, 3]),
                    ("44", [4, 4]),
                    ("45", [4, 5]),
                ]
            )

        def sample(self, prompt, sampling_params, num_samples=1):
            text, tokens = next(self.responses)

            async def _coro():
                return SimpleNamespace(sequences=[SimpleNamespace(tokens=tokens, text=text)])

            return _coro()

    config = opd.Config(
        student_prefill_text=" ! | ~ _ * ^ # @ ",
        student_prefill_count=1,
        student_prefill_suffix="Answer: ",
        student_stop_sequences='["\\n"]',
        eval_task="none",
    )
    metrics, _rows = asyncio.run(
        opd._buffer_control_eval(
            config,
            [FakeClient()],
            [
                [{"role": "user", "content": "Calculate: 1 * 111"}],
                [{"role": "user", "content": "Calculate: 1 * 112"}],
            ],
            None,
        )
    )

    assert metrics["eval/pause_repeated_numeric_frac"] == pytest.approx(1.0)
    assert metrics["eval/nopause_repeated_numeric_frac"] == pytest.approx(0.5)
    assert metrics["eval/pause_answer_cue_leak_frac"] == pytest.approx(0.5)
    assert metrics["eval/pause_stop_sequence_seen_frac"] == pytest.approx(0.5)
    assert metrics["eval/control_stop_sequence_seen_frac_max"] == pytest.approx(0.5)
    assert metrics["eval/pause_reasoning_phrase_frac"] == pytest.approx(0.5)
    assert metrics["eval/pause_request_failure_frac"] == pytest.approx(0.0)
    assert metrics["eval/nopause_request_failure_frac"] == pytest.approx(0.0)
    assert metrics["eval/corrupt_pause_repeated_numeric_frac"] == pytest.approx(0.5)
    assert metrics["eval/corrupt_pause_request_failure_frac"] == pytest.approx(0.0)
    assert metrics["eval/control_total_requests"] == pytest.approx(6.0)
    assert metrics["eval/control_num_arms"] == pytest.approx(3.0)
    assert metrics["eval/control_corrupt_pause_active"] == pytest.approx(1.0)
    assert metrics["eval/buffer_causal_margin"] == pytest.approx(0.0)
    assert metrics["eval/buffer_causal_z_min"] == pytest.approx(0.0)
    assert metrics["eval/control_cap_hit_frac_max"] == pytest.approx(0.0)
    assert metrics["eval/control_cap_hit_frac_mean"] == pytest.approx(0.0)
    assert metrics["eval/control_request_failure_frac_max"] == pytest.approx(0.0)
    assert metrics["eval/control_request_latency_mean_s"] >= 0.0
    assert metrics["eval/control_request_latency_p95_s"] >= 0.0
    assert metrics["eval/control_request_latency_max_s"] >= 0.0


def test_buffer_control_eval_reports_causal_margin_summary():
    opd = _load_example()

    class FakeClient:
        def __init__(self):
            self.responses = iter(
                [
                    "42",
                    "41",
                    "41",
                    "41",
                    "41",
                    "41",
                ]
            )

        def sample(self, prompt, sampling_params, num_samples=1):
            text = next(self.responses)

            async def _coro():
                return SimpleNamespace(sequences=[SimpleNamespace(tokens=[4, 2], text=text)])

            return _coro()

    config = opd.Config(
        student_prefill_text=" ! | ~ _ * ^ # @ ",
        student_prefill_count=1,
        student_prefill_suffix="Answer: ",
        eval_task="multiplication",
        eval_max_new_tokens=4,
    )
    metrics, _rows = asyncio.run(
        opd._buffer_control_eval(
            config,
            [FakeClient()],
            [
                [{"role": "user", "content": "Calculate: 2 * 21"}],
                [{"role": "user", "content": "Calculate: 3 * 14"}],
            ],
            None,
        )
    )

    assert metrics["eval/acc_pause"] == pytest.approx(0.5)
    assert metrics["eval/acc_nopause"] == pytest.approx(0.0)
    assert metrics["eval/acc_corrupt_pause"] == pytest.approx(0.0)
    assert metrics["eval/buffer_causal_margin"] == pytest.approx(0.5)
    assert metrics["eval/buffer_causal_z_min"] > 0.0
    assert metrics["eval/control_cap_hit_frac_max"] == pytest.approx(0.0)
    assert metrics["eval/buffer_delta_se"] > 0.0
    assert metrics["eval/buffer_vs_corrupt_delta_se"] > 0.0


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
                # No prompt_token_lens -> regions unattributed; no correctness flags.
                "opd_region_ids": [-1, -1, -1],
                "opd_sample_ok": [-1, -1, -1],
            },
        }
    ]


def test_opd_loss_payload_preserves_separate_hidden_match_weights():
    opd = _load_example()

    data = opd._opd_loss_data(
        [[10, 11, 12, 13]],
        [[20, 21, 22]],
        teacher_weights_by_sample=[[1.0, 0.0, 0.0]],
        hidden_match_weights_by_sample=[[0.0, 1.0, -0.5]],
    )

    assert data[0]["loss_fn_inputs"]["teacher_weights"] == [1.0, 0.0, 0.0]
    assert data[0]["loss_fn_inputs"]["hidden_match_weights"] == [0.0, 1.0, -0.5]
    assert data[0]["loss_fn_inputs"]["teacher_cache_indices"] == [20, 21, 22]


def test_buffer_position_weights_cover_only_masked_filler_region():
    opd = _load_example()

    # prompt length p=3, K=2. OPD's causal positions for the forced buffer are
    # target slots [p-1, p-1+K), not prompt or answer slots.
    assert opd._buffer_position_weights([10, 11, 12, 50, 51, 90, 91], 3, 2, -0.25) == [
        0.0,
        0.0,
        -0.25,
        -0.25,
        0.0,
        0.0,
    ]


def test_answer_position_weights_cover_only_answer_region():
    opd = _load_example()

    # prompt length p=3, K=2. Answer target slots start at p-1+K because the
    # labels are shifted by one relative to sequence token positions.
    assert opd._answer_position_weights([10, 11, 12, 50, 51, 90, 91], 3, 2, 0.125) == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.125,
        0.125,
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


def test_forward_backward_profile_metrics_are_aggregated():
    opd = _load_example()

    outputs = [
        opd.tomi.ForwardBackwardOutput(
            metrics={
                "opd_profile_forward_compute_s:mean": 1.5,
                "opd_profile_backward_compute_s:mean": 2.5,
                "opd_profile_hidden_fetch_ms:mean": 100.0,
                "opd_profile_kl_compute_ms:mean": 250.0,
                "opd_profile_model_forward_ms:mean": 1100.0,
                "opd_profile_loss_compute_ms:mean": 400.0,
                "opd_profile_input_transfer_s:mean": 0.05,
                "opd_profile_sp_grad_sync_s:mean": 0.02,
                "opd_profile_metric_finalize_s:mean": 0.03,
                "opd_profile_final_synchronize_s:mean": 0.04,
                "opd_profile_forward_loop_total_s:mean": 5.0,
            }
        ),
        opd.tomi.ForwardBackwardOutput(
            metrics={
                "opd_profile_forward_compute_s:mean": 3.0,
                "opd_profile_backward_compute_s:mean": 4.0,
                "opd_profile_hidden_fetch_ms:mean": 50.0,
                "opd_profile_kl_compute_ms:mean": 125.0,
                "opd_profile_clear_gradients_ms:mean": 10.0,
                "opd_profile_model_forward_ms:mean": 900.0,
                "opd_profile_loss_compute_ms:mean": 600.0,
                "opd_profile_input_transfer_s:mean": 0.05,
                "opd_profile_sp_grad_sync_s:mean": 0.02,
                "opd_profile_metric_finalize_s:mean": 0.03,
                "opd_profile_final_synchronize_s:mean": 0.04,
                "opd_profile_forward_loop_total_s:mean": 7.0,
            }
        ),
    ]

    metrics = opd._aggregate_forward_backward_profile_metrics(outputs)

    assert metrics["opd_profile_forward_compute_s"] == pytest.approx(4.5)
    assert metrics["opd_profile_backward_compute_s"] == pytest.approx(6.5)
    assert metrics["opd_profile_hidden_fetch_s"] == pytest.approx(0.15)
    assert metrics["opd_profile_kl_compute_s"] == pytest.approx(0.375)
    assert metrics["opd_profile_clear_gradients_s"] == pytest.approx(0.01)
    assert metrics["opd_profile_model_forward_s"] == pytest.approx(2.0)
    assert metrics["opd_profile_loss_compute_s"] == pytest.approx(1.0)
    assert metrics["opd_profile_input_transfer_s"] == pytest.approx(0.10)
    assert metrics["opd_profile_sp_grad_sync_s"] == pytest.approx(0.04)
    assert metrics["opd_profile_metric_finalize_s"] == pytest.approx(0.06)
    assert metrics["opd_profile_final_synchronize_s"] == pytest.approx(0.08)
    assert metrics["opd_profile_forward_loop_total_s"] == pytest.approx(12.0)


def test_opd_loss_metrics_are_token_weighted_aggregated():
    """Per-microbatch OPDLossMetrics fields are token-weighted averaged per step."""

    opd = _load_example()

    # microbatch A: 100 valid tokens, kl=0.5, entropy=2.0, top1=0.6
    # microbatch B: 300 valid tokens, kl=0.9, entropy=4.0, top1=0.8
    # token-weighted mean: kl = (0.5*100 + 0.9*300) / 400 = 0.80
    #                    entropy = (2*100 + 4*300) / 400 = 3.5
    #                    top1 = (0.6*100 + 0.8*300) / 400 = 0.75
    outputs = [
        opd.tomi.ForwardBackwardOutput(
            metrics={
                "valid_tokens:sum": 100.0,
                "opd_kl:mean": 0.5,
                "opd_teacher_entropy:mean": 2.0,
                "opd_student_entropy:mean": 1.8,
                "opd_top1_agreement:mean": 0.6,
                "opd_loss_min:mean": 0.1,
                "opd_loss_max:mean": 1.2,
                "opd_pg_clipfrac:mean": 0.0,
                "opd_ppo_kl:mean": 0.0,
                "opd_pg_clipfrac_lower:mean": 0.0,
                "opd_num_teachers:mean": 1.0,
            }
        ),
        opd.tomi.ForwardBackwardOutput(
            metrics={
                "valid_tokens:sum": 300.0,
                "opd_kl:mean": 0.9,
                "opd_teacher_entropy:mean": 4.0,
                "opd_student_entropy:mean": 3.5,
                "opd_top1_agreement:mean": 0.8,
                "opd_loss_min:mean": 0.05,
                "opd_loss_max:mean": 1.5,
                "opd_pg_clipfrac:mean": 0.0,
                "opd_ppo_kl:mean": 0.0,
                "opd_pg_clipfrac_lower:mean": 0.0,
                "opd_num_teachers:mean": 1.0,
            }
        ),
    ]

    metrics = opd._aggregate_opd_loss_metrics(outputs)

    assert metrics["opd_kl"] == pytest.approx(0.80)
    assert metrics["opd_teacher_entropy"] == pytest.approx(3.5)
    assert metrics["opd_student_entropy"] == pytest.approx(3.075)
    assert metrics["opd_top1_agreement"] == pytest.approx(0.75)
    assert metrics["opd_loss_min"] == pytest.approx(0.0625)
    assert metrics["opd_loss_max"] == pytest.approx(1.425)
    assert metrics["opd_num_teachers"] == pytest.approx(1.0)


def test_opd_loss_metrics_skips_zero_valid_token_microbatches():
    """Microbatches with valid_tokens=0 (dummy-only ranks) don't contribute noise."""

    opd = _load_example()

    outputs = [
        opd.tomi.ForwardBackwardOutput(
            metrics={
                "valid_tokens:sum": 0.0,
                "opd_kl:mean": 999.0,  # nonsense — must be ignored
                "opd_teacher_entropy:mean": 999.0,
            }
        ),
        opd.tomi.ForwardBackwardOutput(
            metrics={
                "valid_tokens:sum": 50.0,
                "opd_kl:mean": 0.42,
                "opd_teacher_entropy:mean": 1.5,
            }
        ),
    ]

    metrics = opd._aggregate_opd_loss_metrics(outputs)
    assert metrics["opd_kl"] == pytest.approx(0.42)
    assert metrics["opd_teacher_entropy"] == pytest.approx(1.5)


def test_opd_loss_metrics_returns_empty_when_all_microbatches_have_zero_valid():
    opd = _load_example()

    outputs = [
        opd.tomi.ForwardBackwardOutput(metrics={"valid_tokens:sum": 0.0}),
        opd.tomi.ForwardBackwardOutput(metrics={"valid_tokens:sum": 0.0}),
    ]

    assert opd._aggregate_opd_loss_metrics(outputs) == {}


def test_full_vocab_diagnostic_metrics_explain_backend_availability():
    opd = _load_example()

    streaming_requested = opd._full_vocab_diagnostic_metrics(
        opd.Config(opd_emit_full_vocab_diagnostics=True, opd_kl_backend="streaming")
    )
    assert streaming_requested["opd_full_vocab_diag_requested"] == pytest.approx(1.0)
    # streaming gained diagnostics via xorl's no-grad streaming pass (2026-06-09).
    assert streaming_requested["opd_full_vocab_diag_active_expected"] == pytest.approx(1.0)
    assert streaming_requested["opd_full_vocab_diag_unavailable_expected"] == pytest.approx(0.0)

    compile_requested = opd._full_vocab_diagnostic_metrics(
        opd.Config(opd_emit_full_vocab_diagnostics=True, opd_kl_backend="torch_compile")
    )
    assert compile_requested["opd_full_vocab_diag_requested"] == pytest.approx(1.0)
    assert compile_requested["opd_full_vocab_diag_active_expected"] == pytest.approx(1.0)
    assert compile_requested["opd_full_vocab_diag_unavailable_expected"] == pytest.approx(0.0)

    disabled = opd._full_vocab_diagnostic_metrics(
        opd.Config(opd_emit_full_vocab_diagnostics=False, opd_kl_backend="torch_compile")
    )
    assert disabled["opd_full_vocab_diag_requested"] == pytest.approx(0.0)
    assert disabled["opd_full_vocab_diag_active_expected"] == pytest.approx(0.0)
    assert disabled["opd_full_vocab_diag_unavailable_expected"] == pytest.approx(0.0)


def test_sync_profile_metrics_report_serial_endpoint_sync():
    opd = _load_example()

    metrics = opd._sync_profile_metrics(
        SimpleNamespace(
            message="Serial endpoint sync succeeded to 2 endpoint(s) in 3.00s",
            endpoints_synced=[
                SimpleNamespace(host="sampler-0", port=30060, success=True, message=""),
                SimpleNamespace(host="sampler-1", port=30000, success=True, message=""),
            ],
            timing_breakdown={
                "serial_endpoint_sync": 1.0,
                "serial_endpoint_count": 2.0,
                "endpoint_0/transfer_s": 1.25,
                "endpoint_1/p2p_backend_max_transfer_s": 1.5,
                "ignored_detail_s": 99.0,
            },
            p2p_rank_summaries=[
                {"rank": 0, "endpoint_index": 0, "transfer_wall_s": 0.25},
                {"rank": 0, "endpoint_index": 1, "transfer_wall_s": 0.75},
            ],
        )
    )

    assert metrics["sync_endpoint_count"] == 2
    assert metrics["sync_endpoint_success_count"] == 2
    assert metrics["sync_endpoint_failure_count"] == 0
    assert metrics["sync_serial_endpoint_sync"] == pytest.approx(1.0)
    assert metrics["sync_p2p_rank_summary_count"] == 2
    assert metrics["sync_p2p_summary_endpoint_count"] == 2
    assert metrics["sync_p2p_transfer_wall_max_s"] == pytest.approx(0.75)
    assert metrics["sync_p2p_transfer_wall_mean_s"] == pytest.approx(0.5)
    assert metrics["sync_timing/serial_endpoint_count"] == pytest.approx(2.0)
    assert metrics["sync_timing/endpoint_0/transfer_s"] == pytest.approx(1.25)
    assert metrics["sync_timing/endpoint_1/p2p_backend_max_transfer_s"] == pytest.approx(1.5)
    assert "sync_timing/ignored_detail_s" not in metrics


def test_teacher_hidden_cache_data_per_sample_filler_replaces_student_filler():
    """Run B: each sample's teacher_filler is a different list, student_filler_count > 0.
    Teacher seq = student[:p] + per-sample CoT + student[p+K:], with -100 at
    CoT-predicting positions."""
    opd = _load_example()

    # Sample 0: student_seq = [P0, P1, U0, U1, A0, A1] (p=2, K=2, ans=2)
    # Sample 1: student_seq = [P0, P1, P2, U0, U1, A0]  (p=3, K=2, ans=1)
    sequences = [[100, 101, 200, 201, 300, 301], [110, 111, 112, 200, 201, 300]]
    prompt_lens = [2, 3]
    # Per-sample CoT: 3 tokens for sample 0, 4 tokens for sample 1.
    cots = [[900, 901, 902], [910, 911, 912, 913]]

    data = opd._teacher_hidden_cache_data(
        sequences,
        teacher_prefix_tokens=None,
        teacher_filler_tokens=cots,
        prompt_token_lens=prompt_lens,
        student_filler_count=2,
    )

    # Sample 0: teacher_seq = [P0,P1,C0,C1,C2,A0,A1] (replaces U0,U1 with cot[0])
    # input_ids = seq[:-1] = [P0,P1,C0,C1,C2,A0]
    # target_tokens = seq[1:] = [P1,C0,C1,C2,A0,A1]
    # Mask [p-1, p-1+C) = [1, 4) → positions 1,2,3
    assert data[0]["model_input"]["input_ids"] == [100, 101, 900, 901, 902, 300]
    assert data[0]["loss_fn_inputs"]["target_tokens"] == [101, -100, -100, -100, 300, 301]

    # Sample 1: teacher_seq = [P0,P1,P2,C0,C1,C2,C3,A0] (replaces U0,U1 with cot[1])
    # input_ids = [P0,P1,P2,C0,C1,C2,C3]
    # target_tokens = [P1,P2,C0,C1,C2,C3,A0]
    # Mask [2, 6) → positions 2,3,4,5
    assert data[1]["model_input"]["input_ids"] == [110, 111, 112, 910, 911, 912, 913]
    assert data[1]["loss_fn_inputs"]["target_tokens"] == [111, 112, -100, -100, -100, -100, 300]


def test_opd_loss_data_remaps_cache_indices_for_student_filler():
    """Run B: cache_indices length L_s, with -100 mask at K student-filler positions,
    and answer positions mapped to cache rows beyond the prompt block."""
    opd = _load_example()

    # student_seq = [P0,P1,P2,U0,U1,A0,A1,A2] (p=3, K=2, ans=3)
    # L_s = 7
    # Teacher cache rows = (p-1) + ans = 2 + 3 = 5 → server indices = [0,1,2,3,4]
    sequences = [[100, 101, 102, 200, 201, 300, 301, 302]]
    cache_indices = [[0, 1, 2, 3, 4]]

    data = opd._opd_loss_data(
        sequences,
        cache_indices,
        prompt_token_lens=[3],
        student_filler_count=2,
    )

    inputs = data[0]
    assert inputs["model_input"]["input_ids"] == [100, 101, 102, 200, 201, 300, 301]
    # Target tokens: seq[1:] = [P1,P2,U0,U1,A0,A1,A2], mask positions 2,3 (= [p-1, p+K-1))
    assert inputs["loss_fn_inputs"]["target_tokens"] == [101, 102, -100, -100, 300, 301, 302]
    # cache_indices: positions 0,1 → cache rows 0,1; positions 2,3 → placeholder; positions 4,5,6 → cache rows 2,3,4
    cache_idx = inputs["loss_fn_inputs"]["teacher_cache_indices"]
    assert len(cache_idx) == 7
    assert cache_idx[0] == 0
    assert cache_idx[1] == 1
    assert cache_idx[4] == 2
    assert cache_idx[5] == 3
    assert cache_idx[6] == 4


def test_opd_loss_data_masked_remap_emits_global_rows_and_base():
    """OPRD contract (oprd_warm_cache_indices_rebase_bug.md): the masked branches
    emit GLOBAL cache rows (the KL hidden-fetch indexes the global teacher cache)
    plus an explicit per-sample teacher_cache_base — the server packer derives the
    OPRD gather's local view from the base; the client must NOT localize rows."""
    opd = _load_example()

    # p=3, K=2, ans=3; teacher cache rows = (p-1)+ans = 5, global base 40.
    sequences = [[100, 101, 102, 200, 201, 300, 301, 302]]
    cache_indices = [[40, 41, 42, 43, 44]]

    data = opd._opd_loss_data(
        sequences,
        cache_indices,
        prompt_token_lens=[3],
        student_filler_count=2,
        teacher_input_ids_by_sample=[[1, 2, 3, 4, 5, 6, 7]],
        teacher_kept_indices_by_sample=[[0, 1, 3, 4, 5]],
    )

    inputs = data[0]["loss_fn_inputs"]
    # GLOBAL rows at supervised positions, 0-filled at the K masked positions.
    assert inputs["teacher_cache_indices"] == [40, 41, 0, 0, 42, 43, 44]
    assert inputs["target_tokens"] == [101, 102, -100, -100, 300, 301, 302]
    # Explicit re-base anchor = the sample's first global cache row.
    assert inputs["teacher_cache_base"] == 40
    assert inputs["teacher_input_ids"] == [1, 2, 3, 4, 5, 6, 7]
    assert inputs["teacher_kept_indices"] == [0, 1, 3, 4, 5]


def test_opd_loss_data_buffer_only_emits_global_rows_and_base():
    """Buffer-only masked branch (supervise_student_cot + mask_answer): prompt+buffer
    rows stay GLOBAL, answer positions are 0-filled and masked, base is emitted."""
    opd = _load_example()

    # p=3, K=2, ans=3; buffer-only teacher cache rows = (p-1)+K = 4, global base 40.
    sequences = [[100, 101, 102, 200, 201, 300, 301, 302]]
    cache_indices = [[40, 41, 42, 43]]

    data = opd._opd_loss_data(
        sequences,
        cache_indices,
        prompt_token_lens=[3],
        student_filler_count=2,
        supervise_student_cot=True,
        mask_answer=True,
        teacher_input_ids_by_sample=[[1, 2, 3, 4, 5, 6]],
        teacher_kept_indices_by_sample=[[0, 1, 2, 3]],
    )

    inputs = data[0]["loss_fn_inputs"]
    assert inputs["teacher_cache_indices"] == [40, 41, 42, 43, 0, 0, 0]
    assert inputs["target_tokens"] == [101, 102, 200, 201, -100, -100, -100]
    assert inputs["teacher_cache_base"] == 40


def test_opd_loss_data_run_a_path_unchanged():
    """Run A (student_filler_count=0) must behave exactly as before: cache_indices used as-is."""
    opd = _load_example()

    sequences = [[1, 2, 3, 4, 5]]
    cache_indices = [[0, 1, 2, 3]]

    data = opd._opd_loss_data(sequences, cache_indices)
    assert data[0]["model_input"]["input_ids"] == [1, 2, 3, 4]
    assert data[0]["loss_fn_inputs"]["target_tokens"] == [2, 3, 4, 5]
    assert data[0]["loss_fn_inputs"]["teacher_cache_indices"] == [0, 1, 2, 3]


def test_teacher_memory_pair_diagnostics_compare_cross_prompt_cache_rows(tmp_path):
    opd = _load_example()

    import torch  # noqa: PLC0415
    from safetensors.torch import save_file  # noqa: PLC0415

    cache_path = tmp_path / "teacher_hidden.safetensors"
    hidden_states = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    save_file({"hidden_states": hidden_states}, str(cache_path))

    metrics = opd._teacher_memory_pair_diagnostics(
        cache_path,
        cache_indices_by_sample=[
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ],
        prompt_token_lens=[2, 2],
        corrupt_span=2,
        group_size=1,
    )

    assert metrics["opd_teacher_memory_pair_diag_active"] == pytest.approx(1.0)
    assert metrics["opd_teacher_memory_pair_sample_count"] == pytest.approx(2.0)
    assert metrics["opd_teacher_memory_pair_cross_token_count"] == pytest.approx(4.0)
    assert metrics["opd_teacher_memory_pair_cross_cosine_similarity_mean"] == pytest.approx(0.0)
    assert metrics["opd_teacher_memory_pair_cross_cosine_distance_mean"] == pytest.approx(1.0)
    assert metrics["opd_teacher_memory_pair_within_adjacent_token_count"] == pytest.approx(2.0)
    assert metrics["opd_teacher_memory_pair_within_adjacent_distance_mean"] == pytest.approx(0.0)
    assert metrics["opd_teacher_memory_pair_cross_minus_within_distance"] == pytest.approx(1.0)


def test_aggregate_prepared_metrics_weights_teacher_memory_pair_diagnostics(tmp_path):
    opd = _load_example()

    batches = [
        opd.PreparedOpdBatch(
            sequences=[[1, 2]],
            data=[{}],
            cache_path=tmp_path / "a.safetensors",
            metrics={
                "student_sampling_s": 1.0,
                "student_sampling_output_tokens": 1,
                "teacher_prefill_s": 1.0,
                "teacher_prefill_tokens": 1,
                "prepare_s": 1.0,
                "num_samples": 1.0,
                "num_opd_datums": 1.0,
                "opd_teacher_memory_pair_diag_requested": 1.0,
                "opd_teacher_memory_pair_diag_active": 1.0,
                "opd_teacher_memory_pair_sample_count": 1.0,
                "opd_teacher_memory_pair_cross_token_count": 2.0,
                "opd_teacher_memory_pair_cross_cosine_similarity_mean": 0.25,
                "opd_teacher_memory_pair_cross_cosine_distance_mean": 0.75,
                "opd_teacher_memory_pair_cross_cosine_distance_min": 0.5,
                "opd_teacher_memory_pair_cross_cosine_distance_max": 1.0,
                "opd_teacher_memory_pair_within_adjacent_token_count": 1.0,
                "opd_teacher_memory_pair_within_adjacent_distance_mean": 0.1,
                "opd_cache_mismatch_memory_weight": 0.5,
                "opd_cache_mismatch_examples": 2.0,
                "opd_cache_mismatch_skipped_examples": 1.0,
                "opd_cache_mismatch_span_tokens": 2.0,
                "opd_cache_mismatch_span_token_total": 4.0,
                "opd_cache_mismatch_changed_cache_rows": 3.0,
                "opd_cache_mismatch_same_visible_input": 1.0,
                "opd_cache_mismatch_negative_answer_kl_weight": 0.0,
            },
        ),
        opd.PreparedOpdBatch(
            sequences=[[3, 4]],
            data=[{}],
            cache_path=tmp_path / "b.safetensors",
            metrics={
                "student_sampling_s": 1.0,
                "student_sampling_output_tokens": 1,
                "teacher_prefill_s": 1.0,
                "teacher_prefill_tokens": 1,
                "prepare_s": 1.0,
                "num_samples": 1.0,
                "num_opd_datums": 1.0,
                "opd_teacher_memory_pair_diag_requested": 1.0,
                "opd_teacher_memory_pair_diag_active": 1.0,
                "opd_teacher_memory_pair_sample_count": 1.0,
                "opd_teacher_memory_pair_cross_token_count": 6.0,
                "opd_teacher_memory_pair_cross_cosine_similarity_mean": 0.75,
                "opd_teacher_memory_pair_cross_cosine_distance_mean": 0.25,
                "opd_teacher_memory_pair_cross_cosine_distance_min": 0.0,
                "opd_teacher_memory_pair_cross_cosine_distance_max": 0.5,
                "opd_teacher_memory_pair_within_adjacent_token_count": 3.0,
                "opd_teacher_memory_pair_within_adjacent_distance_mean": 0.2,
                "opd_cache_mismatch_memory_weight": 0.5,
                "opd_cache_mismatch_examples": 1.0,
                "opd_cache_mismatch_skipped_examples": 0.0,
                "opd_cache_mismatch_span_tokens": 2.0,
                "opd_cache_mismatch_span_token_total": 2.0,
                "opd_cache_mismatch_changed_cache_rows": 1.0,
                "opd_cache_mismatch_same_visible_input": 1.0,
                "opd_cache_mismatch_negative_answer_kl_weight": 0.0,
            },
        ),
    ]

    metrics = opd._aggregate_prepared_metrics(batches)

    assert metrics["opd_teacher_memory_pair_diag_active"] == pytest.approx(1.0)
    assert metrics["opd_teacher_memory_pair_cross_token_count"] == pytest.approx(8.0)
    assert metrics["opd_teacher_memory_pair_cross_cosine_similarity_mean"] == pytest.approx(0.625)
    assert metrics["opd_teacher_memory_pair_cross_cosine_distance_mean"] == pytest.approx(0.375)
    assert metrics["opd_teacher_memory_pair_cross_cosine_distance_min"] == pytest.approx(0.0)
    assert metrics["opd_teacher_memory_pair_cross_cosine_distance_max"] == pytest.approx(1.0)
    assert metrics["opd_teacher_memory_pair_within_adjacent_token_count"] == pytest.approx(4.0)
    assert metrics["opd_teacher_memory_pair_within_adjacent_distance_mean"] == pytest.approx(0.175)
    assert metrics["opd_teacher_memory_pair_cross_minus_within_distance"] == pytest.approx(0.2)
    assert metrics["opd_cache_mismatch_memory_weight"] == pytest.approx(0.5)
    assert metrics["opd_cache_mismatch_examples"] == pytest.approx(3.0)
    assert metrics["opd_cache_mismatch_skipped_examples"] == pytest.approx(1.0)
    assert metrics["opd_cache_mismatch_span_tokens"] == pytest.approx(2.0)
    assert metrics["opd_cache_mismatch_span_token_total"] == pytest.approx(6.0)
    assert metrics["opd_cache_mismatch_changed_cache_rows"] == pytest.approx(4.0)
    assert metrics["opd_cache_mismatch_change_frac"] == pytest.approx(4.0 / 6.0)
    assert metrics["opd_cache_mismatch_same_visible_input"] == pytest.approx(1.0)
    assert metrics["opd_cache_mismatch_negative_answer_kl_weight"] == pytest.approx(0.0)


def test_concurrent_prepare_submits_all_completed_batches(monkeypatch, tmp_path):
    opd = _load_example()

    prompt_count = 4
    started = 0
    release = asyncio.Event()
    submitted_batches = []

    async def fake_prepare(
        config,
        sampling_clients,
        teacher_url,
        prompts,
        output_dir,
        step,
        chat_tokenizer=None,
        microbatch_idx=0,
        teacher_prefix_tokens=None,
        teacher_filler_tokens=None,
    ):
        nonlocal started
        started += 1
        if started == prompt_count:
            release.set()
        await release.wait()
        return opd.PreparedOpdBatch(
            sequences=[[microbatch_idx, microbatch_idx + 10]],
            data=[{"batch": microbatch_idx}],
            cache_path=tmp_path / f"teacher-{microbatch_idx}.safetensors",
            metrics={
                "student_sampling_s": 1.0,
                "student_sampling_output_tokens": 1,
                "teacher_prefill_s": 1.0,
                "teacher_prefill_tokens": 1,
                "prepare_s": 1.0,
            },
        )

    class FakeTrainingClient:
        def __init__(self, holder, model_id, base_model):
            pass

        async def forward_backward(self, data, loss_fn, loss_fn_params):
            submitted_batches.append(data)
            return SimpleNamespace(loss_fn_outputs=[], metrics={"valid_tokens:sum": 1})

    monkeypatch.setattr(opd, "_prepare_opd_batch", fake_prepare)
    monkeypatch.setattr(opd, "_load_prompts", lambda config: list(range(prompt_count)))
    monkeypatch.setattr(opd, "_load_chat_tokenizer", lambda config: None)
    monkeypatch.setattr(opd, "get_inference_urls", lambda urls, port: ["student"])
    monkeypatch.setattr(opd, "_wait_for_xorl", lambda *args, **kwargs: None)
    monkeypatch.setattr(opd, "_wait_for_sglang", lambda *args, **kwargs: None)
    monkeypatch.setattr(opd, "_ensure_xorl_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        opd.tomi, "ServiceClient", lambda *args, **kwargs: SimpleNamespace(holder=None)
    )
    monkeypatch.setattr(opd.tomi, "SamplingClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(opd, "TrainingClient", FakeTrainingClient)

    config = opd.Config(
        num_steps=1,
        num_prompts=prompt_count,
        opd_microbatch_size=1,
        opd_prepare_concurrency=prompt_count,
        skip_optim_step=True,
        teacher_head="teacher",
        output_dir=str(tmp_path),
        profile_output=str(tmp_path / "profile.jsonl"),
    )

    asyncio.run(opd.main(config))

    assert len(submitted_batches) == prompt_count
    assert sorted(batch[0]["batch"] for batch in submitted_batches) == list(
        range(prompt_count)
    )


def test_control_eval_start_step_skips_early_control_rows(monkeypatch, tmp_path):
    opd = _load_example()

    control_calls = []

    async def fake_prepare(
        config,
        sampling_clients,
        teacher_url,
        prompts,
        output_dir,
        step,
        chat_tokenizer=None,
        microbatch_idx=0,
        teacher_prefix_tokens=None,
        teacher_filler_tokens=None,
    ):
        del config, sampling_clients, teacher_url, prompts, step, chat_tokenizer
        del teacher_prefix_tokens, teacher_filler_tokens
        return opd.PreparedOpdBatch(
            sequences=[[microbatch_idx, microbatch_idx + 10]],
            data=[{"batch": microbatch_idx}],
            cache_path=output_dir / f"teacher-{microbatch_idx}.safetensors",
            metrics={
                "student_sampling_s": 1.0,
                "student_sampling_output_tokens": 1,
                "teacher_prefill_s": 1.0,
                "teacher_prefill_tokens": 1,
                "prepare_s": 1.0,
            },
        )

    async def fake_buffer_control_eval(
        config,
        sampling_clients,
        eval_prompts,
        chat_tokenizer,
        logprob_clients=None,
        progress_callback=None,
    ):
        del config, sampling_clients, chat_tokenizer, logprob_clients, progress_callback
        control_calls.append(list(eval_prompts))
        return (
            {
                "eval/acc_pause": 1.0,
                "eval/acc_nopause": 0.0,
                "eval/buffer_delta": 1.0,
                "eval/control_n": float(len(eval_prompts)),
            },
            [],
        )

    class FakeTrainingClient:
        def __init__(self, holder, model_id, base_model):
            pass

        async def forward_backward(self, data, loss_fn, loss_fn_params):
            return SimpleNamespace(loss_fn_outputs=[], metrics={"valid_tokens:sum": len(data)})

    monkeypatch.setattr(opd, "_prepare_opd_batch", fake_prepare)
    monkeypatch.setattr(opd, "_buffer_control_eval", fake_buffer_control_eval)
    monkeypatch.setattr(opd, "_load_prompts", lambda config: list(range(4)))
    monkeypatch.setattr(opd, "_load_chat_tokenizer", lambda config: None)
    monkeypatch.setattr(opd, "get_inference_urls", lambda urls, port: ["student"])
    monkeypatch.setattr(opd, "_wait_for_xorl", lambda *args, **kwargs: None)
    monkeypatch.setattr(opd, "_wait_for_sglang", lambda *args, **kwargs: None)
    monkeypatch.setattr(opd, "_ensure_xorl_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        opd.tomi, "ServiceClient", lambda *args, **kwargs: SimpleNamespace(holder=None)
    )
    monkeypatch.setattr(opd.tomi, "SamplingClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(opd, "TrainingClient", FakeTrainingClient)

    profile_path = tmp_path / "profile.jsonl"
    config = opd.Config(
        num_steps=3,
        num_prompts=4,
        opd_microbatch_size=1,
        opd_prepare_batch_size=1,
        opd_prepare_concurrency=1,
        skip_optim_step=True,
        teacher_head="teacher",
        output_dir=str(tmp_path),
        profile_output=str(profile_path),
        eval_health_every=0,
        eval_accuracy_every=1,
        eval_num_problems=2,
        eval_control_start_step=2,
        wandb_enabled=False,
    )

    asyncio.run(opd.main(config))

    rows = [json.loads(line) for line in profile_path.read_text().splitlines()]
    assert [row["step"] for row in rows] == [0, 1, 2]
    assert len(control_calls) == 1
    assert control_calls[0] == [2, 3]
    assert "eval/buffer_delta" not in rows[0]
    assert "eval/buffer_delta" not in rows[1]
    assert rows[2]["eval/buffer_delta"] == pytest.approx(1.0)
    assert rows[0]["eval/control_allowed_by_start_step"] == pytest.approx(0.0)
    assert rows[1]["eval/control_allowed_by_start_step"] == pytest.approx(0.0)
    assert rows[2]["eval/control_allowed_by_start_step"] == pytest.approx(1.0)
    assert rows[2]["eval/control_start_step"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Group sampling + group teacher (group_size G > 1).
# ---------------------------------------------------------------------------


class _FakeSampleResponse:
    def __init__(self, tokens, prompt_tokens=None):
        self.sequences = [
            SimpleNamespace(tokens=tokens, prompt_tokens=prompt_tokens, text=None)
        ]


class _FakeGroupClient:
    """Records every sample() call; returns a deterministic per-call completion so
    we can prove the G-fanout + prompt-major flattening."""

    def __init__(self):
        self.calls = []

    def sample(self, prompt, sampling_params, num_samples=1, return_logprobs=False):
        # token id encodes call order so the flattening is checkable.
        idx = len(self.calls)
        self.calls.append(
            {
                "prompt": prompt,
                "num_samples": num_samples,
                "seed": getattr(sampling_params, "sampling_seed", None),
            }
        )

        async def _coro():
            # bare token-id prompts → _sampled_sequence_tokens uses prompt+tokens
            return _FakeSampleResponse(tokens=[900 + idx])

        return _coro()


def test_group_sampling_fans_out_g_calls_per_prompt_prompt_major():
    """group_size=G issues G num_samples=1 calls per prompt with distinct seeds,
    flattened prompt-major (mirrors filler_tokens_rl _submit_problem)."""
    opd = _load_example()

    client = _FakeGroupClient()
    prompts = [[1, 2], [3, 4], [5, 6]]  # N=3 token-id prompts
    G = 4
    sequences, prompt_lens, k, completions, returned_G, _old_lps, _k_per_sample = asyncio.run(
        opd._sample_student_batch(
            [client],
            prompts,
            max_new_tokens=8,
            temperature=1.0,
            group_size=G,
        )
    )

    assert returned_G == G
    # N*G calls, G per prompt, distinct seeds (G>1 path seeds every call).
    assert len(client.calls) == len(prompts) * G
    assert len(sequences) == len(prompts) * G == len(prompt_lens) == len(completions)
    # Prompt-major: sample i belongs to prompt i // G — its prefix must be that
    # prompt's tokens.
    for i, seq in enumerate(sequences):
        assert seq[: len(prompts[i // G])] == prompts[i // G]
    # All seeds present and distinct (diversity across the group).
    seeds = [c["seed"] for c in client.calls]
    assert all(s is not None for s in seeds)
    assert len(set(seeds)) == len(seeds)


def test_group_sampling_g1_is_byte_for_byte_single_sample():
    """group_size=1 takes the original single-future-per-prompt path: one call per
    prompt, the shared unseeded params (sampling_seed is None)."""
    opd = _load_example()

    client = _FakeGroupClient()
    prompts = [[1, 2], [3, 4]]
    sequences, prompt_lens, k, completions, returned_G, _old_lps, _k_per_sample = asyncio.run(
        opd._sample_student_batch(
            [client], prompts, max_new_tokens=8, temperature=1.0, group_size=1
        )
    )
    assert returned_G == 1
    assert len(client.calls) == len(prompts)  # one call per prompt
    # G==1 uses the shared params object → no per-call seed.
    assert all(c["seed"] is None for c in client.calls)
    assert len(sequences) == len(prompts)


def test_group_teacher_merge_equals_independent_single_sample_cache(tmp_path):
    """ALIGNMENT PROOF (merge level): the merged per-sample group cache equals the
    independently-computed single-sample cache for each (prompt, answer_g).

    _merge_group_phase_caches merges ONE Phase-A prefix block per prompt with G
    Phase-B answer blocks. We construct synthetic Phase-A (N rows-of-rows) and
    Phase-B (N*G) hidden tensors with KNOWN distinct values, then assert: for every
    sample i, the merged rows == [promptA[i//G] rows] ++ [answerB[i] rows] in order
    — i.e. identical row COUNT, ORDER, and VALUES to the single-sample two-phase
    merge (_merge_phase_caches) you'd get by replicating prompt i//G's Phase A and
    pairing it 1:1 with sample i's Phase B.
    """
    import torch  # noqa: PLC0415
    from safetensors.torch import load_file, save_file  # noqa: PLC0415

    opd = _load_example()

    N, G, H = 2, 3, 4
    # Phase A: per prompt, a contiguous block of prefix rows. Prompt 0 → 2 rows,
    # prompt 1 → 3 rows. Give each a unique value so misorder is detectable.
    a_blocks = [
        torch.arange(2 * H, dtype=torch.float32).reshape(2, H) + 100.0,   # prompt 0
        torch.arange(3 * H, dtype=torch.float32).reshape(3, H) + 200.0,   # prompt 1
    ]
    a_tensor = torch.cat(a_blocks, dim=0)
    a_indices = [[0, 1], [2, 3, 4]]  # contiguous per-prompt prefix rows
    # Phase B: per sample (N*G), an answer block. Sample i → (i+1) rows, unique.
    b_blocks = []
    b_indices = []
    off = 0
    for i in range(N * G):
        rows = (i % 2) + 1  # 1 or 2 answer rows
        blk = torch.arange(rows * H, dtype=torch.float32).reshape(rows, H) + 1000.0 * (i + 1)
        b_blocks.append(blk)
        b_indices.append(list(range(off, off + rows)))
        off += rows
    b_tensor = torch.cat(b_blocks, dim=0)

    a_path = tmp_path / "a.safetensors"
    b_path = tmp_path / "b.safetensors"
    merged_path = tmp_path / "merged.safetensors"
    save_file({"hidden_states": a_tensor}, str(a_path))
    save_file({"hidden_states": b_tensor}, str(b_path))

    merged_indices = opd._merge_group_phase_caches(
        a_path, b_path, a_indices, b_indices, G, merged_path
    )

    merged = load_file(str(merged_path))["hidden_states"]
    assert len(merged_indices) == N * G
    for i in range(N * G):
        prompt_idx = i // G
        expected = torch.cat([a_blocks[prompt_idx], b_blocks[i]], dim=0)
        got = merged[torch.tensor(merged_indices[i], dtype=torch.long)]
        # Row count + order + values all identical to the single-sample cache.
        assert got.shape == expected.shape, f"sample {i} row-count mismatch"
        torch.testing.assert_close(got, expected)
        # Cosine ~1.0 on every row (the gate-3 alignment metric, exact here).
        cos = torch.cosine_similarity(got, expected, dim=-1)
        assert torch.all(cos > 0.999), f"sample {i} cosine {cos}"


def test_group_teacher_merge_rejects_bad_sample_count(tmp_path):
    import torch  # noqa: PLC0415
    from safetensors.torch import save_file  # noqa: PLC0415

    opd = _load_example()
    a_path = tmp_path / "a.safetensors"
    b_path = tmp_path / "b.safetensors"
    save_file({"hidden_states": torch.zeros(2, 4)}, str(a_path))
    save_file({"hidden_states": torch.zeros(5, 4)}, str(b_path))
    # 2 prompts × G=3 = 6 expected Phase-B samples, but only 5 given.
    with pytest.raises(RuntimeError, match="Phase-B samples"):
        opd._merge_group_phase_caches(
            a_path, b_path, [[0], [1]], [[0]] * 5, 3, tmp_path / "m.safetensors"
        )


def test_prepare_opd_batch_group_teacher_phase_counts_and_ng_datums(monkeypatch, tmp_path):
    """End-to-end (client-side) group prepare: with G>1 + the supervise recipe,
    _prepare_opd_batch samples N*G, fires Phase A ONCE per prompt + Phase B ONCE
    per sample, merges, and emits N*G OPD datums whose teacher_cache_indices come
    from the merged group cache. Proves the group-teacher wiring + per-(prompt,
    sample) alignment at the prepare level (no live teacher needed)."""
    import torch  # noqa: PLC0415
    from safetensors.torch import save_file  # noqa: PLC0415

    opd = _load_example()

    N, G, K, H = 2, 3, 2, 4  # 2 prompts, 3 samples each, K=2 forced-prefix, hidden 4
    prompts = [
        [{"role": "user", "content": f"p{n}"}] for n in range(N)
    ]
    cots = [[700 + n, 701 + n] for n in range(N)]  # per-prompt CoT (2 tokens)

    # Fake group sampling: N*G prompt-major sequences. Each prompt n has prompt_len
    # p=3; forced-prefix K=2; answer length = (sample within group)+1 so the answer
    # rows differ per sample (exercises variable Phase-B sizes).
    seqs = []
    plens = []
    comps = []
    for n in range(N):
        for g in range(G):
            p = 3
            prompt_toks = [10 * n + j for j in range(p)]
            prefix = [50, 51]  # K forced-prefix tokens
            ans = [80 + g + k for k in range(g + 1)]
            seqs.append(prompt_toks + prefix + ans)
            plens.append(p)
            comps.append(ans)

    async def fake_sample(*args, **kwargs):
        assert kwargs.get("group_size") == G
        return seqs, plens, K, comps, G, None, [K] * len(seqs)

    monkeypatch.setattr(opd, "_sample_student_batch", fake_sample)

    phase_calls = {"a": 0, "b": 0}

    def fake_post_phase(teacher_url, data, cache_path, timeout):
        # Identify phase by which kept positions: Phase A keeps prompt+pause
        # (target != -100 count = (p-1)+K); Phase B keeps ONLY answer rows.
        cache_indices = []
        off = 0
        rows_per_sample = []
        for row in data:
            tgt = row["loss_fn_inputs"]["target_tokens"]
            kept = sum(1 for t in tgt if t != -100)
            cache_indices.append(list(range(off, off + kept)))
            off += kept
            rows_per_sample.append(kept)
        # Heuristic: Phase A is the N-row prefix payload, Phase B is the N*G payload.
        if len(data) == N:
            phase_calls["a"] += 1
        elif len(data) == N * G:
            phase_calls["b"] += 1
        # Write a hidden tensor of `off` rows so the merge can load it.
        save_file({"hidden_states": torch.randn(max(off, 1), H)}, str(cache_path))
        return {
            "cache_indices_by_sample": cache_indices,
            "path": str(cache_path),
            "error": None,
        }

    monkeypatch.setattr(opd, "_post_teacher_phase_cache", fake_post_phase)

    config = opd.Config(
        teacher_backend="sglang",
        teacher_cot_mode="insert",
        supervise_student_cot=True,
        student_prefill_text=" pause",
        student_prefill_count=K,
        group_size=G,
        teacher_head="t",
        inference_api_format="chat_completions",
        chat_tokenizer_path="x",
    )

    prepared = asyncio.run(
        opd._prepare_opd_batch(
            config,
            sampling_clients=[object()],
            teacher_url="http://teacher",
            prompts=prompts,
            output_dir=tmp_path,
            step=0,
            chat_tokenizer=None,
            microbatch_idx=0,
            teacher_prefix_tokens=None,
            teacher_filler_tokens=cots,
        )
    )

    # Phase A fired exactly once (one shared prefill per prompt batch); Phase B once.
    assert phase_calls == {"a": 1, "b": 1}
    # N*G OPD datums (group teacher active).
    assert len(prepared.data) == N * G
    assert prepared.metrics["group_teacher_active"] == 1.0
    assert prepared.metrics["num_samples"] == float(N * G)
    # Each datum's cache_indices length == its student input length (supervise
    # remap keeps the pause rows -> identity-length cache for every sample).
    for i, datum in enumerate(prepared.data):
        L_s = len(seqs[i]) - 1
        assert len(datum["loss_fn_inputs"]["teacher_cache_indices"]) == L_s
    # prompt_texts re-grouped prompt-major (sample i -> prompt i//G).
    assert len(prepared.prompt_texts) == N * G


def test_prepare_opd_batch_adds_corrupt_buffer_contrastive_datums(monkeypatch, tmp_path):
    opd = _load_example()

    K = 2
    sequences = [
        [10, 11, 12, 50, 51, 90, 91],
        [20, 21, 22, 60, 61, 92, 93],
    ]
    prompt_lens = [3, 3]
    completions = [[90, 91], [92, 93]]

    async def fake_sample(*args, **kwargs):
        return sequences, prompt_lens, K, completions, 1, None, [K] * len(sequences)

    def fake_teacher_cache(*args, **kwargs):
        return {
            "cache_indices_by_sample": [[0, 1, 2, 3], [4, 5, 6, 7]],
            "metrics": {
                "teacher_prefill_forward_compute_s": 0.0,
                "teacher_hidden_cache_write_s": 0.0,
            },
        }

    monkeypatch.setattr(opd, "_sample_student_batch", fake_sample)
    monkeypatch.setattr(opd, "_teacher_cache_from_sglang", fake_teacher_cache)

    config = opd.Config(
        teacher_backend="sglang",
        teacher_cot_mode="insert",
        supervise_student_cot=True,
        opd_supervise_buffer_only=True,
        opd_contrastive_corrupt_buffer_weight=0.25,
        student_prefill_text=" pause",
        student_prefill_count=K,
        teacher_head="t",
        inference_api_format="chat_completions",
        chat_tokenizer_path="x",
    )

    prepared = asyncio.run(
        opd._prepare_opd_batch(
            config,
            sampling_clients=[object()],
            teacher_url="http://teacher",
            prompts=[[{"role": "user", "content": "p0"}], [{"role": "user", "content": "p1"}]],
            output_dir=tmp_path,
            step=0,
            chat_tokenizer=None,
            microbatch_idx=0,
            teacher_prefix_tokens=None,
            teacher_filler_tokens=[[700, 701], [710, 711]],
        )
    )

    assert len(prepared.completions) == 2
    assert len(prepared.data) == 4
    assert prepared.metrics["num_samples"] == pytest.approx(2.0)
    assert prepared.metrics["num_opd_datums"] == pytest.approx(4.0)
    assert prepared.metrics["opd_contrastive_corrupt_examples"] == pytest.approx(2.0)
    assert prepared.metrics["opd_contrastive_data_multiplier"] == pytest.approx(2.0)
    assert prepared.metrics["opd_contrastive_corrupt_changed_tokens"] == pytest.approx(4.0)
    assert prepared.metrics["opd_contrastive_corrupt_change_frac"] == pytest.approx(1.0)
    assert prepared.metrics["opd_contrastive_corrupt_noop_frac"] == pytest.approx(0.0)

    positive = prepared.data[0]
    negative = prepared.data[1]
    assert positive["model_input"]["input_ids"] == sequences[0][:-1]
    assert negative["model_input"]["input_ids"][:3] == sequences[0][:3]
    assert negative["model_input"]["input_ids"][3:5] != sequences[0][3:5]
    assert negative["model_input"]["input_ids"][5:] == sequences[0][5:-1]
    assert positive["loss_fn_inputs"]["teacher_weights"] == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    assert positive["loss_fn_inputs"]["hidden_match_weights"] == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    assert negative["loss_fn_inputs"]["teacher_weights"] == [0.0] * 6
    assert negative["loss_fn_inputs"]["hidden_match_weights"] == [0.0, 0.0, -0.25, -0.25, 0.0, 0.0]
    assert positive["loss_fn_inputs"]["target_tokens"][-2:] == [-100, -100]
    assert negative["loss_fn_inputs"]["target_tokens"][-2:] == [-100, -100]


def test_prepare_opd_batch_adds_answer_contrast_to_corrupt_buffer_pair(monkeypatch, tmp_path):
    opd = _load_example()

    K = 2
    sequences = [
        [10, 11, 12, 50, 51, 90, 91],
        [20, 21, 22, 60, 61, 92, 93],
    ]
    prompt_lens = [3, 3]
    completions = [[90, 91], [92, 93]]
    teacher_cache_mask_answer = None

    async def fake_sample(*args, **kwargs):
        return sequences, prompt_lens, K, completions, 1, None, [K] * len(sequences)

    def fake_teacher_cache(*args, **kwargs):
        nonlocal teacher_cache_mask_answer
        # mask_answer is the 11th positional of _teacher_cache_from_sglang
        # (per_sample_filler_count / capture_layer_indices / layers_cache_path follow).
        teacher_cache_mask_answer = args[10]
        return {
            "cache_indices_by_sample": [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]],
            "metrics": {
                "teacher_prefill_forward_compute_s": 0.0,
                "teacher_hidden_cache_write_s": 0.0,
            },
        }

    monkeypatch.setattr(opd, "_sample_student_batch", fake_sample)
    monkeypatch.setattr(opd, "_teacher_cache_from_sglang", fake_teacher_cache)

    config = opd.Config(
        teacher_backend="sglang",
        teacher_cot_mode="insert",
        supervise_student_cot=True,
        opd_supervise_buffer_only=False,
        opd_contrastive_corrupt_buffer_weight=1.0,
        opd_contrastive_corrupt_answer_weight=0.125,
        student_prefill_text=" pause",
        student_prefill_count=K,
        teacher_head="t",
        inference_api_format="chat_completions",
        chat_tokenizer_path="x",
    )

    prepared = asyncio.run(
        opd._prepare_opd_batch(
            config,
            sampling_clients=[object()],
            teacher_url="http://teacher",
            prompts=[[{"role": "user", "content": "p0"}], [{"role": "user", "content": "p1"}]],
            output_dir=tmp_path,
            step=0,
            chat_tokenizer=None,
            microbatch_idx=0,
            teacher_prefix_tokens=None,
            teacher_filler_tokens=[[700, 701], [710, 711]],
        )
    )

    assert teacher_cache_mask_answer is False
    assert len(prepared.data) == 4
    assert prepared.metrics["num_samples"] == pytest.approx(2.0)
    assert prepared.metrics["num_opd_datums"] == pytest.approx(4.0)
    assert prepared.metrics["opd_contrastive_corrupt_examples"] == pytest.approx(2.0)
    assert prepared.metrics["opd_contrastive_corrupt_answer_examples"] == pytest.approx(2.0)
    assert prepared.metrics["opd_contrastive_corrupt_answer_weight"] == pytest.approx(0.125)
    assert prepared.metrics["opd_contrastive_corrupt_changed_tokens"] == pytest.approx(4.0)
    assert prepared.metrics["opd_contrastive_corrupt_change_frac"] == pytest.approx(1.0)
    assert prepared.metrics["opd_contrastive_corrupt_noop_frac"] == pytest.approx(0.0)

    positive = prepared.data[0]
    negative = prepared.data[1]
    assert positive["loss_fn_inputs"]["teacher_weights"] == [0.0, 0.0, 1.0, 1.0, 0.125, 0.125]
    assert positive["loss_fn_inputs"]["hidden_match_weights"] == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    assert negative["loss_fn_inputs"]["teacher_weights"] == [0.0, 0.0, 0.0, 0.0, -0.125, -0.125]
    assert negative["loss_fn_inputs"]["hidden_match_weights"] == [0.0, 0.0, -1.0, -1.0, 0.0, 0.0]
    assert positive["loss_fn_inputs"]["target_tokens"][-2:] == [90, 91]
    assert negative["loss_fn_inputs"]["target_tokens"][-2:] == [90, 91]


def test_prepare_opd_batch_can_corrupt_memory_span_without_answer_cue(monkeypatch, tmp_path):
    opd = _load_example()

    # student sequence = prompt(3) + memory(2) + suffix/cue(1) + answer(2)
    K = 3
    sequences = [
        [10, 11, 12, 50, 51, 70, 90, 91],
        [20, 21, 22, 60, 61, 71, 92, 93],
    ]
    prompt_lens = [3, 3]
    completions = [[90, 91], [92, 93]]

    async def fake_sample(*args, **kwargs):
        return sequences, prompt_lens, K, completions, 1, None, [K] * len(sequences)

    def fake_teacher_cache(*args, **kwargs):
        return {
            "cache_indices_by_sample": [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13]],
            "metrics": {
                "teacher_prefill_forward_compute_s": 0.0,
                "teacher_hidden_cache_write_s": 0.0,
            },
        }

    monkeypatch.setattr(opd, "_sample_student_batch", fake_sample)
    monkeypatch.setattr(opd, "_teacher_cache_from_sglang", fake_teacher_cache)

    config = opd.Config(
        teacher_backend="sglang",
        teacher_cot_mode="insert",
        supervise_student_cot=True,
        opd_supervise_buffer_only=False,
        opd_contrastive_corrupt_buffer_weight=1.0,
        opd_contrastive_corrupt_answer_weight=0.125,
        opd_contrastive_corrupt_buffer_mode="rotate",
        opd_contrastive_corrupt_buffer_span="memory_only",
        student_prefill_text=" pause",
        student_prefill_count=2,
        student_prefill_suffix="Answer: ",
        teacher_head="t",
        inference_api_format="chat_completions",
        chat_tokenizer_path="x",
    )

    prepared = asyncio.run(
        opd._prepare_opd_batch(
            config,
            sampling_clients=[object()],
            teacher_url="http://teacher",
            prompts=[[{"role": "user", "content": "p0"}], [{"role": "user", "content": "p1"}]],
            output_dir=tmp_path,
            step=0,
            chat_tokenizer=None,
            microbatch_idx=0,
            teacher_prefix_tokens=None,
            teacher_filler_tokens=[[700, 701], [710, 711]],
        )
    )

    assert prepared.metrics["opd_contrastive_corrupt_span_tokens"] == pytest.approx(2.0)
    assert prepared.metrics["opd_contrastive_corrupt_span_token_total"] == pytest.approx(4.0)
    assert prepared.metrics["opd_contrastive_corrupt_changed_tokens"] == pytest.approx(4.0)
    assert prepared.metrics["opd_contrastive_corrupt_change_frac"] == pytest.approx(1.0)
    assert prepared.metrics["opd_contrastive_corrupt_noop_frac"] == pytest.approx(0.0)
    assert prepared.metrics["opd_contrastive_corrupt_memory_only"] == pytest.approx(1.0)

    positive = prepared.data[0]
    negative = prepared.data[1]
    assert positive["model_input"]["input_ids"] == [10, 11, 12, 50, 51, 70, 90]
    # The two memory tokens rotate, but the answer cue token 70 remains intact.
    assert negative["model_input"]["input_ids"] == [10, 11, 12, 51, 50, 70, 90]
    assert positive["loss_fn_inputs"]["teacher_weights"] == [0.0, 0.0, 1.0, 1.0, 0.0, 0.125, 0.125]
    assert positive["loss_fn_inputs"]["hidden_match_weights"] == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert negative["loss_fn_inputs"]["teacher_weights"] == [0.0, 0.0, 0.0, 0.0, 0.0, -0.125, -0.125]
    assert negative["loss_fn_inputs"]["hidden_match_weights"] == [0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0]
    assert negative["loss_fn_inputs"]["target_tokens"][-3:] == [70, 90, 91]


def test_prepare_opd_batch_adds_memory_cache_mismatch_without_answer_kl(monkeypatch, tmp_path):
    opd = _load_example()

    # student sequence = prompt(3) + memory(2) + suffix/cue(1) + answer(2)
    K = 3
    sequences = [
        [10, 11, 12, 50, 51, 70, 90, 91],
        [20, 21, 22, 60, 61, 71, 92, 93],
    ]
    prompt_lens = [3, 3]
    completions = [[90, 91], [92, 93]]

    async def fake_sample(*args, **kwargs):
        return sequences, prompt_lens, K, completions, 1, None, [K] * len(sequences)

    def fake_teacher_cache(*args, **kwargs):
        return {
            "cache_indices_by_sample": [
                [0, 1, 2, 3, 4, 5, 6],
                [10, 11, 12, 13, 14, 15, 16],
            ],
            "metrics": {
                "teacher_prefill_forward_compute_s": 0.0,
                "teacher_hidden_cache_write_s": 0.0,
            },
        }

    monkeypatch.setattr(opd, "_sample_student_batch", fake_sample)
    monkeypatch.setattr(opd, "_teacher_cache_from_sglang", fake_teacher_cache)

    config = opd.Config(
        teacher_backend="sglang",
        teacher_cot_mode="insert",
        supervise_student_cot=True,
        opd_supervise_buffer_only=False,
        opd_contrastive_corrupt_buffer_weight=1.0,
        opd_contrastive_corrupt_answer_weight=0.125,
        opd_contrastive_corrupt_buffer_mode="rotate",
        opd_contrastive_corrupt_buffer_span="memory_only",
        opd_cache_mismatch_memory_weight=0.5,
        student_prefill_text=" pause",
        student_prefill_count=2,
        student_prefill_suffix="Answer: ",
        teacher_head="t",
        inference_api_format="chat_completions",
        chat_tokenizer_path="x",
    )

    prepared = asyncio.run(
        opd._prepare_opd_batch(
            config,
            sampling_clients=[object()],
            teacher_url="http://teacher",
            prompts=[[{"role": "user", "content": "p0"}], [{"role": "user", "content": "p1"}]],
            output_dir=tmp_path,
            step=0,
            chat_tokenizer=None,
            microbatch_idx=0,
            teacher_prefix_tokens=None,
            teacher_filler_tokens=[[700, 701], [710, 711]],
        )
    )

    assert prepared.metrics["num_samples"] == pytest.approx(2.0)
    assert prepared.metrics["num_opd_datums"] == pytest.approx(6.0)
    assert prepared.metrics["opd_contrastive_corrupt_examples"] == pytest.approx(2.0)
    assert prepared.metrics["opd_cache_mismatch_examples"] == pytest.approx(2.0)
    assert prepared.metrics["opd_cache_mismatch_skipped_examples"] == pytest.approx(0.0)
    assert prepared.metrics["opd_cache_mismatch_span_token_total"] == pytest.approx(4.0)
    assert prepared.metrics["opd_cache_mismatch_changed_cache_rows"] == pytest.approx(4.0)
    assert prepared.metrics["opd_cache_mismatch_change_frac"] == pytest.approx(1.0)
    assert prepared.metrics["opd_cache_mismatch_same_visible_input"] == pytest.approx(1.0)
    assert prepared.metrics["opd_cache_mismatch_negative_answer_kl_weight"] == pytest.approx(0.0)
    assert prepared.metrics["opd_cache_mismatch_balance_positive_hidden"] == pytest.approx(0.0)
    assert prepared.metrics["opd_cache_mismatch_positive_hidden_boost"] == pytest.approx(0.0)
    assert prepared.metrics["opd_memory_hidden_weight_balance_per_token"] == pytest.approx(-0.5)

    positive = prepared.data[0]
    corrupt = prepared.data[1]
    mismatch = prepared.data[2]
    assert mismatch["model_input"] == positive["model_input"]
    assert corrupt["model_input"] != positive["model_input"]
    assert positive["loss_fn_inputs"]["teacher_cache_indices"] == [0, 1, 2, 3, 4, 5, 6]
    assert mismatch["loss_fn_inputs"]["teacher_cache_indices"] == [0, 1, 12, 13, 4, 5, 6]
    assert mismatch["loss_fn_inputs"]["teacher_weights"] == [0.0] * 7
    assert mismatch["loss_fn_inputs"]["hidden_match_weights"] == [0.0, 0.0, -0.5, -0.5, 0.0, 0.0, 0.0]
    assert mismatch["loss_fn_inputs"]["target_tokens"] == positive["loss_fn_inputs"]["target_tokens"]

    second_mismatch = prepared.data[5]
    assert second_mismatch["loss_fn_inputs"]["teacher_cache_indices"] == [10, 11, 2, 3, 14, 15, 16]
    assert second_mismatch["loss_fn_inputs"]["teacher_weights"] == [0.0] * 7

    balanced_config = opd.Config(
        teacher_backend="sglang",
        teacher_cot_mode="insert",
        supervise_student_cot=True,
        opd_supervise_buffer_only=False,
        opd_contrastive_corrupt_buffer_weight=1.0,
        opd_contrastive_corrupt_answer_weight=0.125,
        opd_contrastive_corrupt_buffer_mode="rotate",
        opd_contrastive_corrupt_buffer_span="memory_only",
        opd_cache_mismatch_memory_weight=0.5,
        opd_cache_mismatch_balance_positive_hidden=True,
        student_prefill_text=" pause",
        student_prefill_count=2,
        student_prefill_suffix="Answer: ",
        teacher_head="t",
        inference_api_format="chat_completions",
        chat_tokenizer_path="x",
    )

    balanced = asyncio.run(
        opd._prepare_opd_batch(
            balanced_config,
            sampling_clients=[object()],
            teacher_url="http://teacher",
            prompts=[[{"role": "user", "content": "p0"}], [{"role": "user", "content": "p1"}]],
            output_dir=tmp_path,
            step=0,
            chat_tokenizer=None,
            microbatch_idx=1,
            teacher_prefix_tokens=None,
            teacher_filler_tokens=[[700, 701], [710, 711]],
        )
    )

    balanced_positive = balanced.data[0]
    balanced_mismatch = balanced.data[2]
    assert balanced.metrics["opd_cache_mismatch_balance_positive_hidden"] == pytest.approx(1.0)
    assert balanced.metrics["opd_cache_mismatch_positive_hidden_boost"] == pytest.approx(0.5)
    assert balanced.metrics["opd_memory_hidden_weight_balance_per_token"] == pytest.approx(0.0)
    assert balanced_positive["loss_fn_inputs"]["teacher_weights"] == [0.0, 0.0, 1.0, 1.0, 0.0, 0.125, 0.125]
    assert balanced_positive["loss_fn_inputs"]["hidden_match_weights"] == [
        0.0,
        0.0,
        1.5,
        1.5,
        0.0,
        0.0,
        0.0,
    ]
    assert balanced_mismatch["loss_fn_inputs"]["teacher_weights"] == [0.0] * 7
    assert balanced_mismatch["loss_fn_inputs"]["hidden_match_weights"] == [
        0.0,
        0.0,
        -0.5,
        -0.5,
        0.0,
        0.0,
        0.0,
    ]


def test_prepare_opd_batch_can_use_positive_answer_weight_without_corrupt_answer_negative(
    monkeypatch,
    tmp_path,
):
    opd = _load_example()

    # student sequence = prompt(3) + memory(2) + suffix/cue(1) + answer(2)
    K = 3
    sequences = [
        [10, 11, 12, 50, 51, 70, 90, 91],
        [20, 21, 22, 60, 61, 71, 92, 93],
    ]
    prompt_lens = [3, 3]
    completions = [[90, 91], [92, 93]]

    async def fake_sample(*args, **kwargs):
        return sequences, prompt_lens, K, completions, 1, None, [K] * len(sequences)

    def fake_teacher_cache(*args, **kwargs):
        return {
            "cache_indices_by_sample": [
                [0, 1, 2, 3, 4, 5, 6],
                [10, 11, 12, 13, 14, 15, 16],
            ],
            "metrics": {
                "teacher_prefill_forward_compute_s": 0.0,
                "teacher_hidden_cache_write_s": 0.0,
            },
        }

    monkeypatch.setattr(opd, "_sample_student_batch", fake_sample)
    monkeypatch.setattr(opd, "_teacher_cache_from_sglang", fake_teacher_cache)

    config = opd.Config(
        teacher_backend="sglang",
        teacher_cot_mode="insert",
        supervise_student_cot=True,
        opd_supervise_buffer_only=False,
        opd_contrastive_corrupt_buffer_weight=0.0,
        opd_contrastive_corrupt_answer_weight=0.0,
        opd_positive_answer_weight=0.125,
        opd_contrastive_corrupt_buffer_mode="rotate_preserve_ws",
        opd_contrastive_corrupt_buffer_span="memory_only",
        opd_cache_mismatch_memory_weight=0.5,
        opd_cache_mismatch_balance_positive_hidden=True,
        student_prefill_text=" pause",
        student_prefill_count=2,
        student_prefill_suffix="Answer: ",
        teacher_head="t",
        inference_api_format="chat_completions",
        chat_tokenizer_path="x",
    )

    prepared = asyncio.run(
        opd._prepare_opd_batch(
            config,
            sampling_clients=[object()],
            teacher_url="http://teacher",
            prompts=[[{"role": "user", "content": "p0"}], [{"role": "user", "content": "p1"}]],
            output_dir=tmp_path,
            step=0,
            chat_tokenizer=None,
            microbatch_idx=0,
            teacher_prefix_tokens=None,
            teacher_filler_tokens=[[700, 701], [710, 711]],
        )
    )

    assert len(prepared.data) == 4
    assert prepared.metrics["opd_contrastive_corrupt_examples"] == pytest.approx(0.0)
    assert prepared.metrics["opd_contrastive_corrupt_answer_examples"] == pytest.approx(0.0)
    assert prepared.metrics["opd_positive_answer_examples"] == pytest.approx(2.0)
    assert prepared.metrics["opd_positive_answer_weight"] == pytest.approx(0.125)
    assert prepared.metrics["opd_cache_mismatch_examples"] == pytest.approx(2.0)
    assert prepared.metrics["opd_cache_mismatch_balance_positive_hidden"] == pytest.approx(1.0)
    assert prepared.metrics["opd_memory_hidden_weight_balance_per_token"] == pytest.approx(1.0)

    positive = prepared.data[0]
    mismatch = prepared.data[1]
    assert mismatch["model_input"] == positive["model_input"]
    assert positive["loss_fn_inputs"]["teacher_weights"] == [0.0, 0.0, 1.0, 1.0, 0.0, 0.125, 0.125]
    assert positive["loss_fn_inputs"]["hidden_match_weights"] == [0.0, 0.0, 1.5, 1.5, 0.0, 0.0, 0.0]
    assert mismatch["loss_fn_inputs"]["teacher_weights"] == [0.0] * 7
    assert mismatch["loss_fn_inputs"]["hidden_match_weights"] == [0.0, 0.0, -0.5, -0.5, 0.0, 0.0, 0.0]
    assert mismatch["loss_fn_inputs"]["target_tokens"] == positive["loss_fn_inputs"]["target_tokens"]

    aggregate = opd._aggregate_prepared_metrics([prepared])
    assert aggregate["opd_positive_answer_weight"] == pytest.approx(0.125)
    assert aggregate["opd_positive_answer_examples"] == pytest.approx(2.0)
    assert aggregate["opd_contrastive_corrupt_answer_weight"] == pytest.approx(0.0)
    assert aggregate["opd_contrastive_corrupt_answer_examples"] == pytest.approx(0.0)


def test_prepare_opd_batch_ptc_positive_masks_zero_weight_prompt_rows(
    monkeypatch,
    tmp_path,
):
    opd = _load_example()

    # student sequence = prompt(3) + memory/cue filler(3) + answer(2)
    k_filler = 3
    sequences = [
        [10, 11, 12, 50, 51, 70, 90, 91],
        [20, 21, 22, 60, 61, 71, 92, 93],
    ]
    prompt_lens = [3, 3]
    completions = [[90, 91], [92, 93]]

    async def fake_sample(*args, **kwargs):
        return sequences, prompt_lens, k_filler, completions, 1, None, [k_filler] * len(sequences)

    def fake_teacher_cache(*args, **kwargs):
        return {
            "cache_indices_by_sample": [
                [0, 1, 2, 3, 4, 5, 6],
                [10, 11, 12, 13, 14, 15, 16],
            ],
            "metrics": {
                "teacher_prefill_forward_compute_s": 0.0,
                "teacher_hidden_cache_write_s": 0.0,
            },
        }

    monkeypatch.setattr(opd, "_sample_student_batch", fake_sample)
    monkeypatch.setattr(opd, "_teacher_cache_from_sglang", fake_teacher_cache)

    config = opd.Config(
        teacher_backend="sglang",
        teacher_cot_mode="insert",
        supervise_student_cot=True,
        opd_supervise_buffer_only=False,
        opd_ptc_positive_buffer_kl_weight=1.0,
        opd_ptc_positive_answer_kl_weight=0.125,
        opd_ptc_positive_hidden_weight=0.0,
        opd_mask_zero_weight_positions=True,
        student_prefill_text=" pause",
        student_prefill_count=2,
        student_prefill_suffix="Answer: ",
        teacher_head="t",
        inference_api_format="chat_completions",
        chat_tokenizer_path="x",
    )

    prepared = asyncio.run(
        opd._prepare_opd_batch(
            config,
            sampling_clients=[object()],
            teacher_url="http://teacher",
            prompts=[[{"role": "user", "content": "p0"}], [{"role": "user", "content": "p1"}]],
            output_dir=tmp_path,
            step=0,
            chat_tokenizer=None,
            microbatch_idx=0,
            teacher_prefix_tokens=None,
            teacher_filler_tokens=[[700, 701], [710, 711]],
        )
    )

    assert len(prepared.data) == 2
    assert prepared.metrics["opd_ptc_positive_examples"] == pytest.approx(2.0)
    assert prepared.metrics["opd_ptc_positive_buffer_kl_weight"] == pytest.approx(1.0)
    assert prepared.metrics["opd_ptc_positive_answer_kl_weight"] == pytest.approx(0.125)
    assert prepared.metrics["opd_ptc_positive_hidden_weight"] == pytest.approx(0.0)
    assert prepared.metrics["opd_mask_zero_weight_positions"] == pytest.approx(1.0)
    assert prepared.metrics["opd_teacher_weight_prompt_mean"] == pytest.approx(0.0)
    assert prepared.metrics["opd_teacher_weight_buffer_mean"] == pytest.approx(1.0)
    assert prepared.metrics["opd_teacher_weight_answer_mean"] == pytest.approx(0.125)
    assert prepared.metrics["opd_hidden_weight_buffer_mean"] == pytest.approx(0.0)
    assert prepared.metrics["opd_contrastive_data_multiplier"] == pytest.approx(1.0)

    first = prepared.data[0]["loss_fn_inputs"]
    assert first["teacher_weights"] == [0.0, 0.0, 1.0, 1.0, 1.0, 0.125, 0.125]
    assert first["hidden_match_weights"] == [0.0] * 7
    assert first["target_tokens"] == [-100, -100, 50, 51, 70, 90, 91]

    aggregate = opd._aggregate_prepared_metrics([prepared])
    assert aggregate["opd_ptc_positive_examples"] == pytest.approx(2.0)
    assert aggregate["opd_ptc_positive_buffer_kl_weight"] == pytest.approx(1.0)
    assert aggregate["opd_ptc_positive_answer_kl_weight"] == pytest.approx(0.125)
    assert aggregate["opd_mask_zero_weight_positions"] == pytest.approx(1.0)


def test_opd_pipeline_rl_trains_each_step_on_its_own_prepare(monkeypatch, tmp_path):
    """opd_pipeline_rl overlap must keep (prompts, samples, teacher cache, fwd_bwd)
    aligned per step.

    With per-step windowing on, the prepare for step N reads window N's prompts.
    Under the pipeline, step N's prepare is launched during step N-1 and consumed
    at step N. This test fakes _prepare_opd_batch to stamp each prepared batch with
    the exact prompts it was built from, and the fake forward_backward records the
    (step, prompts) of every datum it trains on. The assertion proves step N trains
    ONLY on data prepared from window N's prompts — i.e. no cross-step windowing
    misalignment (the failure mode of the deprecated two-phase path).
    """
    opd = _load_example()

    prompt_count = 6
    prompts_per_step = 2
    num_steps = 3
    # Record, per training datum, the (step, prompts) it was prepared from.
    trained = []
    # Track the inference "weight version": +1 each sync. A prepare stamps the
    # weight version live at the time it actually samples, so we can assert the
    # 1-step staleness explicitly.
    weight_version = {"v": 0}

    async def fake_prepare(
        config,
        sampling_clients,
        teacher_url,
        prompts,
        output_dir,
        step,
        chat_tokenizer=None,
        microbatch_idx=0,
        teacher_prefix_tokens=None,
        teacher_filler_tokens=None,
    ):
        # Yield control so sampling overlaps the awaited train/sync of the prior
        # step (exercises the genuine async overlap, not a serialized fast path).
        await asyncio.sleep(0)
        sampled_at_version = weight_version["v"]
        return opd.PreparedOpdBatch(
            sequences=[[microbatch_idx, microbatch_idx + 10]],
            data=[
                {
                    "prep_step": step,
                    "prompts": list(prompts),
                    "sampled_at_version": sampled_at_version,
                }
            ],
            cache_path=output_dir / f"teacher_step{step}_mb{microbatch_idx}.safetensors",
            metrics={
                "student_sampling_s": 1.0,
                "student_sampling_output_tokens": 1,
                "teacher_prefill_s": 1.0,
                "teacher_prefill_tokens": 1,
                "prepare_s": 1.0,
            },
        )

    class FakeTrainingClient:
        def __init__(self, holder, model_id, base_model):
            pass

        def forward_backward(self, data, loss_fn, loss_fn_params):
            for datum in data:
                trained.append(datum)
            fut = asyncio.get_event_loop().create_future()
            fut.set_result(
                SimpleNamespace(loss_fn_outputs=[], metrics={"valid_tokens:sum": 1})
            )
            return fut

        def optim_step(self, params):
            fut = asyncio.get_event_loop().create_future()
            fut.set_result(SimpleNamespace())
            return fut

        async def sync_weights_to_inference(self, **kwargs):
            # A sync bumps the inference weight version: any prepare that samples
            # AFTER this point sees the new weights.
            weight_version["v"] += 1
            return SimpleNamespace(
                success=True,
                message="ok",
                transfer_time=0.0,
                total_bytes=0,
                num_buckets=0,
            )

    monkeypatch.setattr(opd, "_prepare_opd_batch", fake_prepare)
    monkeypatch.setattr(opd, "_load_prompts", lambda config: list(range(prompt_count)))
    monkeypatch.setattr(opd, "_load_chat_tokenizer", lambda config: None)
    monkeypatch.setattr(opd, "get_inference_urls", lambda urls, port: ["student"])
    monkeypatch.setattr(opd, "_wait_for_xorl", lambda *args, **kwargs: None)
    monkeypatch.setattr(opd, "_wait_for_sglang", lambda *args, **kwargs: None)
    monkeypatch.setattr(opd, "_ensure_xorl_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        opd.tomi, "ServiceClient", lambda *args, **kwargs: SimpleNamespace(holder=None)
    )
    monkeypatch.setattr(opd.tomi, "SamplingClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(opd, "TrainingClient", FakeTrainingClient)

    config = opd.Config(
        num_steps=num_steps,
        num_prompts=prompt_count,
        prompts_per_step=prompts_per_step,
        opd_microbatch_size=1,
        opd_prepare_batch_size=prompts_per_step,  # one prepare batch per step window
        opd_prepare_concurrency=1,
        opd_pipeline_rl=True,
        learning_rate=1e-4,
        teacher_head="teacher",
        output_dir=str(tmp_path),
        profile_output=str(tmp_path / "profile.jsonl"),
    )

    asyncio.run(opd.main(config))

    # Exactly one datum trained per step (one prepare batch per step window).
    assert len(trained) == num_steps
    for step, datum in enumerate(trained):
        # Each step trains on the data prepared FOR that step's window — the
        # prep_step stamped inside the prepared batch equals the consuming step,
        # and the prompts equal that step's wrap-around window. This is the
        # alignment guarantee: prompts == samples == teacher cache == fwd_bwd input.
        assert datum["prep_step"] == step, (
            f"step {step} trained on data prepared for step {datum['prep_step']}"
        )
        expected_window = [
            (step * prompts_per_step + i) % prompt_count
            for i in range(prompts_per_step)
        ]
        assert datum["prompts"] == expected_window, (
            f"step {step} window mismatch: {datum['prompts']} != {expected_window}"
        )

    # Bounded staleness. Step N's prepare is LAUNCHED during step N-1's train
    # window (the create_task(_prepare_step_batches(step+1)) call sits before the
    # weight-sync await), so its samples reflect weights no fresher than step N-1's
    # post-sync weights — never weights from a future step. The exact version a
    # sample sees is an inherent prepare/sync race (mid-flight sync mutates weights;
    # accepted, mirrors filler_tokens_rl). The contract is: monotonic + bounded by
    # the step's own pre-sync version, and step 0 fully fresh.
    versions = [d["sampled_at_version"] for d in trained]
    assert versions[0] == 0, versions
    assert versions == sorted(versions), f"staleness not monotonic: {versions}"
    for step in range(num_steps):
        # Step N trains then syncs (-> version becomes <=N+1 after its own sync);
        # its samples were generated before its own sync, so version <= N.
        assert 0 <= versions[step] <= step, (
            f"step {step} sampled at version {versions[step]} (must be in [0, {step}] "
            f"— never from the future); trajectory {versions}"
        )


def test_opd_main_writes_profile_row_before_sync_failure_abort(monkeypatch, tmp_path):
    opd = _load_example()

    async def fake_prepare(
        config,
        sampling_clients,
        teacher_url,
        prompts,
        output_dir,
        step,
        chat_tokenizer=None,
        microbatch_idx=0,
        teacher_prefix_tokens=None,
        teacher_filler_tokens=None,
    ):
        return opd.PreparedOpdBatch(
            sequences=[[1, 2, 3]],
            data=[{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}],
            cache_path=output_dir / f"teacher_step{step}_mb{microbatch_idx}.safetensors",
            metrics={
                "student_sampling_s": 1.0,
                "student_sampling_output_tokens": 1,
                "teacher_prefill_s": 1.0,
                "teacher_prefill_tokens": 1,
                "prepare_s": 1.0,
                "num_samples": 1.0,
                "num_opd_datums": 1.0,
            },
        )

    class FakeTrainingClient:
        def __init__(self, holder, model_id, base_model):
            pass

        def forward_backward(self, data, loss_fn, loss_fn_params):
            fut = asyncio.get_event_loop().create_future()
            fut.set_result(
                SimpleNamespace(loss_fn_outputs=[], metrics={"valid_tokens:sum": 1})
            )
            return fut

        def optim_step(self, params):
            fut = asyncio.get_event_loop().create_future()
            fut.set_result(SimpleNamespace())
            return fut

        async def sync_weights_to_inference(self, **kwargs):
            raise RuntimeError("p2p failed")

    profile_path = tmp_path / "profile-sync-failure.jsonl"

    monkeypatch.setattr(opd, "_prepare_opd_batch", fake_prepare)
    monkeypatch.setattr(opd, "_load_prompts", lambda config: [[1, 2, 3]])
    monkeypatch.setattr(opd, "_load_chat_tokenizer", lambda config: None)
    monkeypatch.setattr(opd, "get_inference_urls", lambda urls, port: ["student"])
    monkeypatch.setattr(opd, "_wait_for_xorl", lambda *args, **kwargs: None)
    monkeypatch.setattr(opd, "_wait_for_sglang", lambda *args, **kwargs: None)
    monkeypatch.setattr(opd, "_ensure_xorl_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        opd.tomi, "ServiceClient", lambda *args, **kwargs: SimpleNamespace(holder=None)
    )
    monkeypatch.setattr(opd.tomi, "SamplingClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(opd, "TrainingClient", FakeTrainingClient)

    config = opd.Config(
        num_steps=1,
        num_prompts=1,
        opd_microbatch_size=1,
        opd_prepare_batch_size=1,
        opd_prepare_concurrency=1,
        learning_rate=1e-4,
        teacher_head="teacher",
        output_dir=str(tmp_path),
        profile_output=str(profile_path),
        eval_health_every=0,
        eval_accuracy_every=0,
        wandb_enabled=False,
        request_timeout=1.0,
        weight_sync_timeout=1.0,
    )

    with pytest.raises(RuntimeError, match="OPD aborting at step 0: RuntimeError: p2p failed"):
        asyncio.run(opd.main(config))

    rows = [json.loads(line) for line in profile_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["step"] == 0
    assert rows[0]["sync_success"] is False
    assert rows[0]["sync_failure"] == "RuntimeError: p2p failed"
    assert rows[0]["sync_message"] == "RuntimeError: p2p failed"
    assert rows[0]["sync_endpoint_success_count"] == 0
