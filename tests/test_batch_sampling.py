"""
Unit tests for batch sampling with straggler handling.
"""

import asyncio
import unittest
from unittest.mock import patch

import httpx

from xorl_client import types
from xorl_client.client.sampling_client import SamplingClient


def make_sample_response(text: str) -> types.SampleResponse:
    """Create a mock SampleResponse."""
    return types.SampleResponse(
        sequences=[
            types.SampledSequence(tokens=[1, 2, 3], logprobs=[0.1, 0.2, 0.3], text=text)
        ],
        meta_info=None,
    )


class TestBatchSampling(unittest.TestCase):
    def setUp(self):
        self.client = SamplingClient(base_url="http://localhost:30000")

    def test_basic_batch_all_complete(self):
        """All prompts complete when no timeout."""

        async def mock_sample(prompt, params, num_samples, return_logprobs):
            await asyncio.sleep(0.01)
            return make_sample_response(f"Response to: {prompt}")

        with patch.object(self.client, "_sample_async", side_effect=mock_sample):
            result = self.client.sample_batch(
                prompts=["A", "B", "C", "D"],
                timeout=10.0,
            ).result()

        self.assertEqual(len(result.completed), 4)
        self.assertEqual(len(result.failed), 0)
        self.assertEqual(len(result.cancelled), 0)
        self.assertEqual({idx for idx, _ in result.completed}, {0, 1, 2, 3})

    def test_timeout_cancels_stragglers(self):
        """Timeout causes slow requests to be cancelled."""

        async def mock_sample(prompt, params, num_samples, return_logprobs):
            if prompt == "slow":
                await asyncio.sleep(10.0)
            else:
                await asyncio.sleep(0.01)
            return make_sample_response(f"Response to: {prompt}")

        with patch.object(self.client, "_sample_async", side_effect=mock_sample):
            result = self.client.sample_batch(
                prompts=["fast1", "fast2", "slow"],
                timeout=0.5,
            ).result()

        self.assertEqual(len(result.completed), 2)
        self.assertEqual(len(result.cancelled), 1)
        self.assertIn(2, result.cancelled)

    def test_index_tracking_correct(self):
        """Indices correctly map to original prompts."""

        async def mock_sample(prompt, params, num_samples, return_logprobs):
            await asyncio.sleep(0.01)
            return make_sample_response(f"RESPONSE:{prompt}")

        prompts = ["ALPHA", "BETA", "GAMMA", "DELTA"]

        with patch.object(self.client, "_sample_async", side_effect=mock_sample):
            result = self.client.sample_batch(prompts=prompts, timeout=10.0).result()

        for idx, response in result.completed:
            self.assertIn(prompts[idx], response.sequences[0].text)

    def test_failed_requests_tracked(self):
        """Failed requests are tracked correctly."""

        async def mock_sample(prompt, params, num_samples, return_logprobs):
            await asyncio.sleep(0.01)
            if prompt == "fail":
                raise RuntimeError("Simulated failure")
            return make_sample_response(f"Response to: {prompt}")

        with patch.object(self.client, "_sample_async", side_effect=mock_sample):
            result = self.client.sample_batch(
                prompts=["ok1", "fail", "ok2"],
                timeout=10.0,
            ).result()

        self.assertEqual(len(result.completed), 2)
        self.assertEqual(len(result.failed), 1)
        failed_idx, exc = result.failed[0]
        self.assertEqual(failed_idx, 1)
        self.assertIsInstance(exc, RuntimeError)

    def test_async_interface(self):
        """Async interface works correctly."""

        async def mock_sample(prompt, params, num_samples, return_logprobs):
            await asyncio.sleep(0.01)
            return make_sample_response(f"Response to: {prompt}")

        async def run_test():
            with patch.object(self.client, "_sample_async", side_effect=mock_sample):
                return await self.client.sample_batch_async(
                    prompts=["A", "B", "C"],
                    timeout=10.0,
                )

        result = asyncio.run(run_test())
        self.assertEqual(len(result.completed), 3)

    def test_empty_prompts(self):
        """Empty prompt list returns empty result."""
        result = self.client.sample_batch(prompts=[], timeout=10.0).result()
        self.assertEqual(len(result.completed), 0)
        self.assertEqual(len(result.failed), 0)
        self.assertEqual(len(result.cancelled), 0)

    def test_no_timeout_waits_for_all(self):
        """Without timeout, all requests complete."""

        async def mock_sample(prompt, params, num_samples, return_logprobs):
            delay = 0.01 * (ord(prompt[0]) - ord("A") + 1)
            await asyncio.sleep(delay)
            return make_sample_response(f"Response to: {prompt}")

        with patch.object(self.client, "_sample_async", side_effect=mock_sample):
            result = self.client.sample_batch(
                prompts=["A", "B", "C", "D"],
                timeout=None,
            ).result()

        self.assertEqual(len(result.completed), 4)
        self.assertEqual(len(result.cancelled), 0)


def _make_httpx_response(status_code, **kwargs):
    """Create an httpx.Response with a request set, as required for raise_for_status."""
    import httpx

    resp = httpx.Response(status_code, **kwargs)
    resp._request = httpx.Request("POST", "http://localhost:30000/test")
    return resp


class TestPauseContinueGeneration(unittest.TestCase):
    """Tests for pause_generation_async / continue_generation_async."""

    def setUp(self):
        self.client = SamplingClient(base_url="http://localhost:30000")

    def test_pause_generation_sends_correct_payload(self):
        """pause_generation_async POSTs to /pause_generation with mode."""

        async def run_test():
            mock_response = _make_httpx_response(
                200, json={"status": "paused", "paused_count": 5}
            )
            with patch(
                "httpx.AsyncClient.post", return_value=mock_response
            ) as mock_post:
                result = await self.client.pause_generation_async(mode="in_place")
                mock_post.assert_called_once_with(
                    "/pause_generation",
                    json={"mode": "in_place"},
                    timeout=30.0,
                )
                self.assertEqual(result["status"], "paused")
                self.assertEqual(result["paused_count"], 5)

        asyncio.run(run_test())

    def test_continue_generation_sends_correct_payload(self):
        """continue_generation_async POSTs to /continue_generation."""

        async def run_test():
            mock_response = _make_httpx_response(
                200, json={"status": "resumed", "resumed_count": 5}
            )
            with patch(
                "httpx.AsyncClient.post", return_value=mock_response
            ) as mock_post:
                result = await self.client.continue_generation_async()
                mock_post.assert_called_once_with(
                    "/continue_generation",
                    json={"torch_empty_cache": True},
                    timeout=30.0,
                )
                self.assertEqual(result["status"], "resumed")

        asyncio.run(run_test())

    def test_lora_load_is_idempotent_only_for_same_path(self):
        async def run_test():
            same = _make_httpx_response(
                400,
                json={
                    "success": False,
                    "error_message": "adapter policy is already loaded",
                    "loaded_adapters": {"policy": "/weights/step-1"},
                },
            )
            with patch("httpx.AsyncClient.post", return_value=same):
                result = await self.client.load_lora_adapter_async(
                    lora_name="policy", lora_path="/weights/step-1"
                )
                self.assertTrue(result["already_loaded"])

            collision = _make_httpx_response(
                400,
                json={
                    "success": False,
                    "error_message": "adapter policy is already loaded",
                    "loaded_adapters": {"policy": "/weights/old"},
                },
            )
            with patch("httpx.AsyncClient.post", return_value=collision):
                with self.assertRaises(httpx.HTTPStatusError):
                    await self.client.load_lora_adapter_async(
                        lora_name="policy", lora_path="/weights/new"
                    )

        asyncio.run(run_test())

    def test_pause_generation_retract_mode(self):
        """pause_generation_async supports retract mode."""

        async def run_test():
            mock_response = _make_httpx_response(200, json={"status": "paused"})
            with patch(
                "httpx.AsyncClient.post", return_value=mock_response
            ) as mock_post:
                await self.client.pause_generation_async(mode="retract")
                mock_post.assert_called_once_with(
                    "/pause_generation",
                    json={"mode": "retract"},
                    timeout=30.0,
                )

        asyncio.run(run_test())

    def test_pause_generation_raises_on_http_error(self):
        """pause_generation_async raises on HTTP errors."""
        import httpx

        async def run_test():
            mock_response = _make_httpx_response(500, text="Internal Server Error")
            with patch("httpx.AsyncClient.post", return_value=mock_response):
                with self.assertRaises(httpx.HTTPStatusError):
                    await self.client.pause_generation_async()

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
