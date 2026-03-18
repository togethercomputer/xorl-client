"""
Unit tests for batch sampling with straggler handling.
"""

import asyncio
import unittest
from unittest.mock import patch

from xorl_client import types
from xorl_client.client.sampling_client import BatchSampleResult, SamplingClient


def make_sample_response(text: str) -> types.SampleResponse:
    """Create a mock SampleResponse."""
    return types.SampleResponse(
        sequences=[types.SampledSequence(tokens=[1, 2, 3], logprobs=[0.1, 0.2, 0.3], text=text)],
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


if __name__ == "__main__":
    unittest.main()
