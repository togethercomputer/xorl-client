"""
SamplingClient - Tinker-compatible sampling client using model_path.

This version abstracts away worker URLs and uses model_path for routing.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

from xorl_client import types
from xorl_client.client.client_holder import ClientHolder

logger = logging.getLogger(__name__)


class SamplingClient:
    """Client for sampling from inference workers using model_path.

    The model_path is an opaque identifier (e.g., "xorl://model-123/step-100")
    that the service uses to route requests to the correct inference workers.
    Users never see or manage worker URLs directly.

    Example:
        >>> service_client = ServiceClient(base_url="http://localhost:5555")
        >>> sampling_client = service_client.create_sampling_client(
        ...     model_path="xorl://model-123/step-100"
        ... )
        >>> response = sampling_client.sample(
        ...     prompt=types.ModelInput.from_str("The capital of France is"),
        ...     sampling_params=types.SamplingParams(max_tokens=20, temperature=0.8)
        ... )
        >>> print(response.text)
    """

    def __init__(
        self,
        holder: ClientHolder,
        model_path: str,
    ):
        """Initialize SamplingClient.

        Args:
            holder: ClientHolder for connection management
            model_path: Model path identifier (e.g., "xorl://model-123/step-100")
        """
        self.holder = holder
        self.model_path = model_path

        # Resolve model_path to worker URLs
        try:
            self.worker_urls = holder.get_workers_for_model(model_path)
            self._current_worker_idx = 0
            logger.info(f"SamplingClient initialized: model_path={model_path}, workers={len(self.worker_urls)}")
        except ValueError:
            # Model path not registered yet - this is okay for newly created models
            logger.warning(f"Model path not yet registered: {model_path}. Will use default routing.")
            self.worker_urls = []
            self._current_worker_idx = 0

    def _get_worker_url(self) -> str:
        """Get next worker URL (round-robin)."""
        if not self.worker_urls:
            raise RuntimeError(f"No workers registered for model_path: {self.model_path}")

        url = self.worker_urls[self._current_worker_idx]
        self._current_worker_idx = (self._current_worker_idx + 1) % len(self.worker_urls)
        return url

    def sample(
        self,
        prompt: Union[types.ModelInput, str],
        sampling_params: Optional[types.SamplingParams] = None,
        num_samples: int = 1,
        return_logprobs: bool = True,
    ) -> types.SampleResponse:
        """Sample from the model.

        Args:
            prompt: Text prompt (ModelInput or string)
            sampling_params: Sampling parameters
            num_samples: Number of samples to generate (default: 1)
            return_logprobs: Whether to return logprobs

        Returns:
            SampleResponse with generated text, tokens, and logprobs

        Example:
            >>> response = sampling_client.sample(
            ...     prompt="The capital of France is",
            ...     sampling_params=types.SamplingParams(max_tokens=20, temperature=0.8)
            ... )
            >>> print(response.text)
        """
        # Convert prompt to string if needed
        if isinstance(prompt, types.ModelInput):
            if prompt.text is not None:
                prompt_str = prompt.text
            elif prompt.tokens is not None:
                # TODO: Convert tokens back to text (needs tokenizer)
                raise NotImplementedError("Token-based prompts not yet supported")
            else:
                raise ValueError("ModelInput must have either text or tokens")
        else:
            prompt_str = prompt

        # Convert sampling params to dict
        if sampling_params is None:
            sampling_params = types.SamplingParams()
        sampling_params_dict = sampling_params.to_dict()

        # Get worker URL
        worker_url = self._get_worker_url()
        logger.debug(f"Sampling from {worker_url} (model_path={self.model_path})")

        # Make request
        import requests
        try:
            response = requests.post(
                f"{worker_url}/generate",
                json={
                    "text": prompt_str,
                    "sampling_params": sampling_params_dict,
                    "return_logprob": return_logprobs,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()

            # Parse response
            text = data.get("text", "")
            meta_info = data.get("meta_info", {})

            # Extract tokens and logprobs (handle both direct SGLang and router formats)
            tokens = data.get("tokens")  # Direct SGLang format
            if tokens is None:
                tokens = data.get("output_ids")  # Router format
            if tokens is None and "output_token_ids" in meta_info:
                tokens = meta_info["output_token_ids"]

            logprobs = data.get("logprobs")
            if logprobs is None and "output_token_logprobs" in meta_info:
                output_logprobs = meta_info["output_token_logprobs"]
                if isinstance(output_logprobs, list) and output_logprobs:
                    if isinstance(output_logprobs[0], (list, tuple)) and len(output_logprobs[0]) >= 1:
                        logprobs = [item[0] for item in output_logprobs]
                    else:
                        logprobs = output_logprobs

            # Warn if logprobs missing
            if return_logprobs and (tokens is None or logprobs is None):
                logger.warning(
                    f"Logprobs requested but not fully returned. "
                    f"Got tokens={tokens is not None}, logprobs={logprobs is not None}"
                )
                tokens = tokens or []
                logprobs = logprobs or []

            # Create SampledSequence
            sequence = types.SampledSequence(
                tokens=tokens or [],
                logprobs=logprobs or [],
                text=text,
            )

            return types.SampleResponse(sequences=[sequence])

        except requests.exceptions.RequestException as e:
            logger.error(f"Sampling failed from {worker_url}: {e}")
            raise RuntimeError(f"Sampling failed: {e}") from e
