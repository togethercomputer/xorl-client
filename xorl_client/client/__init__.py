"""
XoRL Client Library.

Provides client interfaces for interacting with XoRL training and inference services.
"""

from xorl_client.client.sampling_client import SamplingClient
from xorl_client.client.training_client import TrainingClient

__all__ = ["SamplingClient", "TrainingClient"]
