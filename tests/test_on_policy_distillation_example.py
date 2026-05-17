import importlib.util
import asyncio
import sys
from types import SimpleNamespace
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

    sampled = opd.tomi.SampledSequence(tokens=[30, 31], prompt_tokens=[])

    assert opd._sampled_sequence_tokens("hello", sampled, FakeTokenizer()) == [
        1,
        2,
        3,
        30,
        31,
    ]


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
