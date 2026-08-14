import asyncio
import math
from unittest.mock import patch

import httpx
import pytest

from xorl_client import SamplingClient, SamplingParams


def _response(status: int, json):
    response = httpx.Response(status, json=json)
    response._request = httpx.Request("POST", "http://sampler/generate")
    return response


def test_native_batch_preserves_per_row_seeds_metadata_and_logprobs():
    client = SamplingClient(base_url="http://sampler")
    payloads = []

    async def post(path, json):
        payloads.append(json)
        return _response(
            200,
            [
                {
                    "text": "a",
                    "output_ids": [31],
                    "meta_info": {
                        "output_token_logprobs": [[-0.1, 31]],
                        "worker": "a",
                    },
                },
                {
                    "text": "b",
                    "output_ids": [41, 42],
                    "meta_info": {
                        "output_token_logprobs": [[-0.2, 41], [-0.3, 42]],
                        "worker": "b",
                    },
                },
            ],
        )

    with patch("httpx.AsyncClient.post", side_effect=post):
        rows = asyncio.run(
            client.generate_batch_native_async(
                [[1, 2], [3, 4]],
                [
                    SamplingParams(max_tokens=2, sampling_seed=101),
                    SamplingParams(
                        max_tokens=2,
                        sampling_seed=102,
                        ignore_eos=True,
                        no_stop_trim=True,
                    ),
                ],
            )
        )
    assert [p["sampling_seed"] for p in payloads[0]["sampling_params"]] == [101, 102]
    assert payloads[0]["sampling_params"][1]["ignore_eos"] is True
    assert payloads[0]["sampling_params"][1]["no_stop_trim"] is True
    assert [row.meta_info["worker"] for row in rows] == ["a", "b"]
    assert rows[1].tokens == [41, 42]
    assert rows[1].logprobs == [-0.2, -0.3]


def test_native_batch_strict_cardinality_and_semantic_4xx_no_retry():
    client = SamplingClient(base_url="http://sampler", max_retries=2, retry_delay=0)
    calls = 0

    with patch("httpx.AsyncClient.post", return_value=_response(200, [])):
        with pytest.raises(ValueError, match="cardinality"):
            asyncio.run(
                client.generate_batch_native_async([[1]], SamplingParams(max_tokens=1))
            )

    async def bad(path, json):
        nonlocal calls
        calls += 1
        return _response(400, {"error": "bad request"})

    with patch("httpx.AsyncClient.post", side_effect=bad):
        with pytest.raises(RuntimeError, match="HTTP 400"):
            asyncio.run(
                client.generate_batch_native_async([[1]], SamplingParams(max_tokens=1))
            )
    assert calls == 1


def test_native_batch_retries_gateway_failure():
    client = SamplingClient(base_url="http://sampler", max_retries=1, retry_delay=0)
    replies = iter(
        [
            _response(503, {"error": "busy"}),
            _response(
                200,
                [
                    {
                        "output_ids": [2],
                        "meta_info": {"output_token_logprobs": [[-1.0, 2]]},
                    }
                ],
            ),
        ]
    )
    with patch("httpx.AsyncClient.post", side_effect=lambda *a, **k: next(replies)):
        rows = asyncio.run(
            client.generate_batch_native_async([[1]], SamplingParams(max_tokens=1))
        )
    assert rows[0].tokens == [2]


def test_native_batch_accepts_one_shared_parameter_object():
    client = SamplingClient(base_url="http://sampler")

    async def post(path, json):
        assert isinstance(json["sampling_params"], dict)
        return _response(
            200,
            [
                {
                    "output_ids": [2],
                    "meta_info": {"output_token_logprobs": [[-1.0, 2]]},
                },
                {
                    "output_ids": [3],
                    "meta_info": {"output_token_logprobs": [[-2.0, 3]]},
                },
            ],
        )

    with patch("httpx.AsyncClient.post", side_effect=post):
        rows = asyncio.run(
            client.generate_batch_native_async(
                [[1], [1]], SamplingParams(max_tokens=1, sampling_seed=7)
            )
        )
    assert [row.tokens for row in rows] == [[2], [3]]


def test_native_batch_rejects_invalid_behavior_logprobs():
    client = SamplingClient(base_url="http://sampler")
    malformed = (
        ({}, "missing output_token_logprobs"),
        ({"output_token_logprobs": [[math.nan, 2]]}, "non-finite logprob"),
        ({"output_token_logprobs": [[-1.0, 99]]}, "token ID does not match"),
    )
    for meta_info, message in malformed:
        with pytest.raises(ValueError, match=message):
            client._parse_sample_response(
                {"output_ids": [2], "meta_info": meta_info},
                return_logprobs=True,
            )


def test_native_batch_passes_binary_r3_controls_outside_sampling_params():
    client = SamplingClient(base_url="http://sampler")

    async def post(path, json):
        assert json["return_routed_experts"] is True
        assert json["return_expert_logits"] is True
        assert json["return_routed_experts_file"] is True
        assert json["routed_experts_start_len"] == [0, 17]
        assert "return_routed_experts_file" not in json["sampling_params"][0]
        return _response(
            200,
            [
                {"output_ids": [2], "meta_info": {"output_token_logprobs": [[-1.0, 2]]}},
                {"output_ids": [3], "meta_info": {"output_token_logprobs": [[-2.0, 3]]}},
            ],
        )

    params = [
        SamplingParams(
            max_tokens=1,
            return_routed_experts=True,
            return_expert_logits=True,
            return_routed_experts_file=True,
            routed_experts_start_len=start,
        )
        for start in (0, 17)
    ]
    with patch("httpx.AsyncClient.post", side_effect=post):
        asyncio.run(client.generate_batch_native_async([[1], [1]], params))
