import asyncio
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
                    SamplingParams(max_tokens=2, sampling_seed=102, ignore_eos=True),
                ],
            )
        )
    assert [p["sampling_seed"] for p in payloads[0]["sampling_params"]] == [101, 102]
    assert payloads[0]["sampling_params"][1]["ignore_eos"] is True
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
