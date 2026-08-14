import asyncio
import unittest
from unittest.mock import patch

import httpx

from xorl_client.client.sampling_client import SamplingClient
from xorl_client.types import ModelInput, SamplingParams


def _make_httpx_response(
    status_code: int, url: str = "http://localhost:30000/generate", **kwargs
) -> httpx.Response:
    response = httpx.Response(status_code, **kwargs)
    response._request = httpx.Request("POST", url)
    return response


class TestSamplingClientGeneratePayload(unittest.TestCase):
    def setUp(self) -> None:
        self.client = SamplingClient(base_url="http://localhost:30000")

    def test_sample_async_passes_custom_params_and_logprob_start_len(self) -> None:
        async def run_test() -> dict:
            mock_response = _make_httpx_response(
                200,
                json={
                    "text": "",
                    "output_ids": [],
                    "meta_info": {
                        "finish_reason": {"type": "stop"},
                        "input_token_logprobs": [],
                        "output_token_logprobs": [],
                    },
                },
            )
            with patch(
                "httpx.AsyncClient.post", return_value=mock_response
            ) as mock_post:
                await self.client.sample_async(
                    prompt=ModelInput.from_ints([11, 12, 13]),
                    sampling_params=SamplingParams(
                        max_tokens=0,
                        temperature=0.0,
                        custom_params={
                            "attention_mask": {
                                "block_query_range": [5, 8],
                                "block_kv_range": [1, 4],
                            }
                        },
                    ),
                    num_samples=1,
                    return_logprobs=True,
                    logprob_start_len=2,
                )
                return mock_post.call_args.kwargs["json"]

        payload = asyncio.run(run_test())
        self.assertEqual(payload["logprob_start_len"], 2)
        self.assertEqual(payload["sampling_params"]["max_new_tokens"], 0)
        self.assertEqual(
            payload["sampling_params"]["custom_params"],
            {
                "attention_mask": {
                    "block_query_range": [5, 8],
                    "block_kv_range": [1, 4],
                }
            },
        )

    def test_sampling_params_to_dict_omits_custom_params_when_unset(self) -> None:
        params = SamplingParams(max_tokens=5, temperature=0.7)
        payload = params.to_dict()
        self.assertNotIn("custom_params", payload)

    def test_score_prompt_logprobs_batch_async_preserves_existing_api(self) -> None:
        async def run_test() -> tuple[dict, list[list[float | None]]]:
            mock_response = _make_httpx_response(
                200,
                json={
                    "0": {
                        "text": "",
                        "output_ids": [],
                        "meta_info": {
                            "finish_reason": {"type": "stop"},
                            "input_token_logprobs": [
                                [None, 10],
                                [-0.4, 11],
                                [-0.2, 12],
                            ],
                        },
                    },
                    "1": {
                        "text": "",
                        "output_ids": [],
                        "meta_info": {
                            "finish_reason": {"type": "stop"},
                            "input_token_logprobs": [[None, 20], [-0.7, 21]],
                        },
                    },
                },
            )
            with patch(
                "httpx.AsyncClient.post", return_value=mock_response
            ) as mock_post:
                result = await self.client.score_prompt_logprobs_batch_async(
                    input_ids_batch=[[10, 11, 12], [20, 21]],
                    logprob_start_lens=[0, 1],
                )
                return mock_post.call_args.kwargs["json"], result

        payload, result = asyncio.run(run_test())
        self.assertEqual(payload["logprob_start_len"], [0, 1])
        self.assertEqual(payload["sampling_params"]["max_new_tokens"], 0)
        self.assertEqual(result, [[None, -0.4, -0.2], [None, -0.7]])

    def test_chat_completions_messages_payload_and_parse(self) -> None:
        client = SamplingClient(
            base_url="http://dispatch:8080",
            model="Qwen/Qwen3-32B",
            model_path="xorl://default/sampler_weights/step-000123",
            api_format="chat_completions",
        )

        async def run_test() -> tuple[str, dict, object]:
            mock_response = _make_httpx_response(
                200,
                url="http://dispatch:8080/v1/chat/completions",
                json={
                    "id": "chatcmpl-test",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "42"},
                            "logprobs": {
                                "content": [
                                    {
                                        "token": "4",
                                        "logprob": -0.1,
                                        "top_logprobs": [],
                                    },
                                    {
                                        "token": "2",
                                        "logprob": -0.2,
                                        "top_logprobs": [],
                                    },
                                ]
                            },
                            "finish_reason": "stop",
                            "token_ids": [40, 42],
                            "prompt_token_ids": [11, 12, 13],
                        }
                    ],
                    "metadata": {"weight_version": 7},
                    "sglext": {"routed_experts": "[[[1, 2]]]"},
                },
            )
            with patch(
                "httpx.AsyncClient.post", return_value=mock_response
            ) as mock_post:
                result = await client.sample_async(
                    prompt=[
                        {"role": "system", "content": "answer tersely"},
                        {"role": "user", "content": "what is 6 * 7?"},
                    ],
                    sampling_params=SamplingParams(
                        max_tokens=5,
                        temperature=0.7,
                        top_p=0.9,
                        stop=["</answer>"],
                        return_routed_experts=True,
                        sampling_seed=123,
                    ),
                    num_samples=1,
                    return_logprobs=True,
                    logprob_start_len=2,
                )
                return (
                    mock_post.call_args.args[0],
                    mock_post.call_args.kwargs["json"],
                    result,
                )

        path, payload, result = asyncio.run(run_test())
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(payload["model"], "Qwen/Qwen3-32B")
        self.assertEqual(
            payload["messages"],
            [
                {"role": "system", "content": "answer tersely"},
                {"role": "user", "content": "what is 6 * 7?"},
            ],
        )
        self.assertNotIn("input_ids", payload)
        self.assertEqual(payload["max_tokens"], 5)
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["top_p"], 0.9)
        self.assertEqual(payload["n"], 1)
        self.assertIs(payload["logprobs"], True)
        self.assertIs(payload["return_token_ids"], True)
        self.assertIs(payload["return_prompt_token_ids"], True)
        self.assertEqual(payload["logprob_start_len"], 2)
        self.assertEqual(payload["lora_path"], "step-000123")
        self.assertIs(payload["return_routed_experts"], True)
        self.assertEqual(payload["seed"], 123)
        self.assertEqual(payload["stop"], ["</answer>"])
        self.assertEqual(result.tokens, [40, 42])
        self.assertEqual(result.sequences[0].prompt_tokens, [11, 12, 13])
        self.assertEqual(result.logprobs, [-0.1, -0.2])
        self.assertEqual(result.text, "42")
        self.assertEqual(result.meta_info["weight_version"], 7)
        self.assertEqual(result.meta_info["routed_experts"], [[[1, 2]]])

    def test_chat_completions_rejects_model_input_prompt(self) -> None:
        client = SamplingClient(
            base_url="http://dispatch:8080", api_format="chat_completions"
        )

        async def run_test() -> None:
            with self.assertRaisesRegex(
                ValueError, "requires a string prompt or a list of chat messages"
            ):
                await client.sample_async(
                    prompt=ModelInput.from_ints([11, 12, 13]),
                    sampling_params=SamplingParams(max_tokens=5),
                )

        asyncio.run(run_test())

    def test_chat_completions_string_prompt_uses_user_message(self) -> None:
        client = SamplingClient(base_url="http://dispatch:8080", api_format="chat")

        async def run_test() -> dict:
            mock_response = _make_httpx_response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "hi"},
                            "logprobs": {"content": []},
                            "finish_reason": "length",
                            "token_ids": [],
                        }
                    ]
                },
            )
            with patch(
                "httpx.AsyncClient.post", return_value=mock_response
            ) as mock_post:
                result = await client.sample_async(
                    prompt="hello",
                    sampling_params=SamplingParams(max_tokens=3, temperature=0.0),
                    return_logprobs=True,
                )
                self.assertEqual(result.sequences[0].stop_reason, "length")
                return mock_post.call_args.kwargs["json"]

        payload = asyncio.run(run_test())
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hello"}])
        self.assertNotIn("input_ids", payload)
        self.assertEqual(payload["model"], "default")

    def test_chat_completions_parses_multiple_choices(self) -> None:
        client = SamplingClient(
            base_url="http://dispatch:8080", api_format="chat_completions"
        )

        async def run_test():
            mock_response = _make_httpx_response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "A"},
                            "logprobs": {"content": [{"token": "A", "logprob": -0.3}]},
                            "finish_reason": "stop",
                            "token_ids": [65],
                        },
                        {
                            "message": {"role": "assistant", "content": "B"},
                            "logprobs": {"content": [{"token": "B", "logprob": -0.4}]},
                            "finish_reason": "stop",
                            "token_ids": [66],
                        },
                    ]
                },
            )
            with patch(
                "httpx.AsyncClient.post", return_value=mock_response
            ) as mock_post:
                result = await client.sample_async(
                    prompt=[{"role": "user", "content": "pick A or B"}],
                    sampling_params=SamplingParams(max_tokens=2),
                    num_samples=2,
                    return_logprobs=True,
                )
                return mock_post.call_args.kwargs["json"], result

        payload, result = asyncio.run(run_test())
        self.assertEqual(payload["n"], 2)
        self.assertEqual([seq.text for seq in result.sequences], ["A", "B"])
        self.assertEqual([seq.tokens for seq in result.sequences], [[65], [66]])

    def test_chat_completions_rejects_invalid_behavior_data(self) -> None:
        client = SamplingClient(
            base_url="http://dispatch:8080", api_format="chat_completions"
        )

        valid_choice = {
            "message": {"role": "assistant", "content": "A"},
            "logprobs": {"content": [{"token": "A", "logprob": -0.3}]},
            "finish_reason": "stop",
            "token_ids": [65],
        }
        cases = {
            "missing logprob": {
                **valid_choice,
                "logprobs": {"content": [{"token": "A"}]},
            },
            "non-finite logprob": {
                **valid_choice,
                "logprobs": {"content": [{"token": "A", "logprob": "nan"}]},
            },
            "token ID mismatch": {
                **valid_choice,
                "logprobs": {
                    "content": [{"token": "A", "token_id": 66, "logprob": -0.3}]
                },
            },
            "wrong cardinality": None,
        }

        async def run_case(choice):
            choices = [] if choice is None else [choice]
            mock_response = _make_httpx_response(200, json={"choices": choices})
            with patch("httpx.AsyncClient.post", return_value=mock_response):
                await client.sample_async(
                    prompt="pick A",
                    sampling_params=SamplingParams(max_tokens=1),
                    return_logprobs=True,
                )

        for name, choice in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                asyncio.run(run_case(choice))

    def test_chat_completions_can_be_selected_with_env_var(self) -> None:
        with patch.dict(
            "os.environ", {"XORL_INFERENCE_API_FORMAT": "openai"}, clear=False
        ):
            client = SamplingClient(base_url="http://dispatch:8080")
        self.assertEqual(client.api_format, "chat_completions")

    def test_invalid_api_format_raises(self) -> None:
        with self.assertRaises(ValueError):
            SamplingClient(base_url="http://localhost:30000", api_format="invalid")

    def test_pooled_dispatch_http_settings_from_env(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "XORL_SAMPLING_HTTP_CLIENT_MODE": "pooled",
                "XORL_SAMPLING_HTTP_MAX_CONNECTIONS": "8192",
                "XORL_SAMPLING_HTTP_MAX_KEEPALIVE_CONNECTIONS": "0",
                "XORL_SAMPLING_HTTP_KEEPALIVE_EXPIRY": "300",
            },
            clear=False,
        ):
            client = SamplingClient(base_url="http://dispatch:8080")

        self.assertEqual(client.http_client_mode, "pooled")
        self.assertEqual(client.http_max_connections, 8192)
        self.assertEqual(client.http_max_keepalive_connections, 0)
        self.assertEqual(client.http_keepalive_expiry, 300.0)

    def test_pooled_mode_reuses_client_within_event_loop(self) -> None:
        client = SamplingClient(
            base_url="http://dispatch:8080",
            http_client_mode="pooled",
            http_max_connections=2048,
            http_max_keepalive_connections=0,
        )

        async def run_test() -> None:
            async with client._request_client() as first:
                async with client._request_client() as second:
                    self.assertIs(first, second)
            await client.close_async()

        asyncio.run(run_test())
        self.assertFalse(client._pooled_clients)


if __name__ == "__main__":
    unittest.main()
