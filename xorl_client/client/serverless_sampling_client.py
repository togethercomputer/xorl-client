"""
Serverless sampling client using XoRL's OpenAI-compatible API.

This client can run on a laptop without SSH tunnels or GPU access.
The adapter must already be uploaded to XoRL.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from xorl_client import types

logger = logging.getLogger(__name__)


class ServerlessSamplingClient:
    """
    Sampling client that uses XoRL serverless LoRA inference.

    Uses the OpenAI-compatible API, so the standard `openai` library works.
    This client can run from a laptop without SSH tunnels to GPU workers.

    Example:
        >>> client = ServerlessSamplingClient(
        ...     provider_api_key="xxx",
        ...     model_id="username/adapter-name",
        ... )
        >>> response = client.sample(
        ...     prompt="The capital of France is",
        ...     sampling_params=types.SamplingParams(max_tokens=20),
        ... )
        >>> print(response.text)
    """

    def __init__(
        self,
        provider_api_key: str,
        model_id: str,
    ):
        """
        Initialize serverless sampling client.

        Args:
            provider_api_key: API key
            model_id: model name (from update_serverless_weights)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required for serverless sampling. "
                "Install with: pip install xorl-client[serverless]"
            )

        self.model_id = model_id
        self._provider_api_key = provider_api_key
        self.client = OpenAI(
            api_key=provider_api_key,
            base_url="https://api.together.xyz/v1",
        )

        logger.info(f"ServerlessSamplingClient initialized with model: {model_id}")

    def set_model(self, model_id: str):
        """
        Update the model ID.

        Use this after uploading a new adapter to switch to the latest weights.

        Args:
            model_id: New model name
        """
        self.model_id = model_id
        logger.info(f"Model updated to: {model_id}")

    def sample(
        self,
        prompt: Union[types.ModelInput, str],
        sampling_params: Optional[types.SamplingParams] = None,
        return_logprobs: bool = True,
    ) -> types.SampleResponse:
        """
        Sample from the model via XoRL serverless API.

        Args:
            prompt: Text prompt (ModelInput or string)
            sampling_params: Sampling parameters
            return_logprobs: Whether to return logprobs

        Returns:
            SampleResponse with generated text and tokens
        """
        # Convert prompt
        if isinstance(prompt, types.ModelInput):
            if prompt.text is not None:
                prompt_str = prompt.text
            elif prompt.tokens is not None:
                raise NotImplementedError("Token-based prompts not yet supported for serverless")
            else:
                raise ValueError("ModelInput must have either text or tokens")
        else:
            prompt_str = prompt

        # Convert sampling params
        if sampling_params is None:
            sampling_params = types.SamplingParams()

        # Build request kwargs
        kwargs: Dict[str, Any] = {
            "model": self.model_id,
            "prompt": prompt_str,
            "max_tokens": sampling_params.max_tokens,
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
        }

        # Add logprobs if requested
        if return_logprobs:
            kwargs["logprobs"] = 1

        # Add stop sequences if provided
        if sampling_params.stop:
            kwargs["stop"] = sampling_params.stop

        logger.debug(f"Sampling with model {self.model_id}, max_tokens={sampling_params.max_tokens}")

        # Call XoRL via OpenAI client
        try:
            response = self.client.completions.create(**kwargs)
        except Exception as e:
            logger.error(f"Sampling failed: {e}")
            raise RuntimeError(f"Serverless sampling failed: {e}") from e

        # Parse response
        choice = response.choices[0]
        text = choice.text

        tokens: List[int] = []
        logprobs: List[float] = []

        if return_logprobs and choice.logprobs:
            # OpenAI response format: logprobs.tokens and logprobs.token_logprobs
            tokens_list = getattr(choice.logprobs, "tokens", None)
            logprobs_list = getattr(choice.logprobs, "token_logprobs", None)

            if tokens_list:
                # Tokens are strings, we need to keep them as strings or convert
                # For now, store as empty list since we have text
                pass
            if logprobs_list:
                logprobs = [lp for lp in logprobs_list if lp is not None]

        sequence = types.SampledSequence(
            tokens=tokens,
            logprobs=logprobs if logprobs else None,
            text=text,
        )

        return types.SampleResponse(sequences=[sequence])

    def chat_sample(
        self,
        messages: List[Dict[str, str]],
        sampling_params: Optional[types.SamplingParams] = None,
    ) -> types.SampleResponse:
        """
        Sample from the model using chat format.

        Args:
            messages: List of chat messages [{"role": "user", "content": "..."}]
            sampling_params: Sampling parameters

        Returns:
            SampleResponse with generated text

        Example:
            >>> response = client.chat_sample(
            ...     messages=[{"role": "user", "content": "What is 2+2?"}],
            ...     sampling_params=types.SamplingParams(max_tokens=50),
            ... )
            >>> print(response.text)
        """
        if sampling_params is None:
            sampling_params = types.SamplingParams()

        kwargs: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": sampling_params.max_tokens,
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
        }

        if sampling_params.stop:
            kwargs["stop"] = sampling_params.stop

        logger.debug(f"Chat sampling with model {self.model_id}")

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.error(f"Chat sampling failed: {e}")
            raise RuntimeError(f"Serverless chat sampling failed: {e}") from e

        # Parse response
        choice = response.choices[0]
        text = choice.message.content or ""

        sequence = types.SampledSequence(
            tokens=[],
            logprobs=None,
            text=text,
        )

        return types.SampleResponse(sequences=[sequence])
