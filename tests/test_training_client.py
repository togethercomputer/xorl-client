"""Tests for xorl_client TrainingClient with mocked dependencies."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from concurrent.futures import Future

from xorl_client import types
from xorl_client.client.training_client import TrainingClient
from xorl_client.client.service_client import ServiceClient
from xorl_client.client.client_holder import ClientHolder


class MockTokenizer:
    """Mock tokenizer for testing without loading actual models."""

    def __init__(self):
        self.vocab = {
            "Hello": 100,
            " world": 101,
            "!": 102,
            "English": 200,
            ":": 201,
            " ": 202,
            "banana": 203,
            " split": 204,
            "\n": 205,
            "Pig": 206,
            " Latin": 207,
            " anana": 208,
            "-bay": 209,
            " plit": 210,
            "-say": 211,
            "<|begin_of_text|>": 1,
        }
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Simple mock encoding - split by known tokens."""
        tokens = []
        if add_special_tokens:
            tokens.append(1)  # BOS token

        # Simple tokenization for testing
        remaining = text
        while remaining:
            found = False
            for token_str, token_id in sorted(self.vocab.items(), key=lambda x: -len(x[0])):
                if remaining.startswith(token_str):
                    tokens.append(token_id)
                    remaining = remaining[len(token_str):]
                    found = True
                    break
            if not found:
                # Unknown character - just skip or use a default
                tokens.append(999)  # UNK token
                remaining = remaining[1:]

        return tokens

    def decode(self, tokens: list[int]) -> str:
        """Simple mock decoding."""
        return "".join(self.reverse_vocab.get(t, "<unk>") for t in tokens)


class TestTrainingClientInit:
    """Tests for TrainingClient initialization."""

    def test_init(self):
        """Test TrainingClient initialization."""
        mock_holder = Mock(spec=ClientHolder)
        client = TrainingClient(
            holder=mock_holder,
            model_id="test-model-123",
            base_model="Qwen/Qwen2.5-3B-Instruct",
        )

        assert client.model_id == "test-model-123"
        assert client.base_model == "Qwen/Qwen2.5-3B-Instruct"
        assert client._tokenizer is None

    def test_get_tokenizer_caching(self):
        """Test that get_tokenizer caches the tokenizer."""
        mock_holder = Mock(spec=ClientHolder)
        client = TrainingClient(
            holder=mock_holder,
            model_id="test-model-123",
            base_model="Qwen/Qwen2.5-3B-Instruct",
        )

        # Mock the AutoTokenizer at the transformers module level
        mock_tokenizer = MockTokenizer()
        with patch("transformers.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer

            # First call should load tokenizer
            tokenizer1 = client.get_tokenizer()
            assert mock_auto.from_pretrained.call_count == 1

            # Second call should use cached tokenizer
            tokenizer2 = client.get_tokenizer()
            assert mock_auto.from_pretrained.call_count == 1  # Not called again

            assert tokenizer1 is tokenizer2


class TestTrainingClientWithMockTokenizer:
    """Tests for TrainingClient with mocked tokenizer."""

    @pytest.fixture
    def mock_holder(self):
        """Create a mock ClientHolder."""
        holder = Mock(spec=ClientHolder)

        # Mock post_async to return a future
        def mock_post_async(endpoint, data, timeout=None):
            future = Future()
            if endpoint == "/api/v1/forward_backward":
                future.set_result({
                    "loss_fn_outputs": [{"loss": 0.5}],
                    "metrics": {"total_tokens": 100},
                })
            elif endpoint == "/api/v1/optim_step":
                future.set_result({
                    "metrics": {"grad_norm": 1.0, "step": 1},
                })
            return future

        holder.post_async = mock_post_async
        return holder

    @pytest.fixture
    def training_client(self, mock_holder):
        """Create a TrainingClient with mocked dependencies."""
        client = TrainingClient(
            holder=mock_holder,
            model_id="test-model-123",
            base_model="test-model",
        )
        # Inject mock tokenizer
        client._tokenizer = MockTokenizer()
        return client

    def test_process_example_workflow(self, training_client):
        """Test the full workflow of processing examples as in xorl_client_example1.py."""
        tokenizer = training_client.get_tokenizer()

        # Create a simple example
        example = {
            "input": "banana split",
            "output": "anana-bay plit-say"
        }

        # Process example (simplified version of xorl_client_example1.py logic)
        prompt = f"English: {example['input']}\nPig Latin:"
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_weights = [0] * len(prompt_tokens)

        completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)
        completion_weights = [1] * len(completion_tokens)

        tokens = prompt_tokens + completion_tokens
        weights = prompt_weights + completion_weights

        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        weights = weights[1:]

        # Create Datum
        datum = types.Datum(
            model_input=types.ModelInput.from_ints(tokens=input_tokens),
            loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
        )

        # Verify datum structure - use .tolist() since lists are auto-converted to TensorData
        assert datum.model_input.to_ints() == input_tokens
        assert datum.loss_fn_inputs["target_tokens"].tolist() == target_tokens
        assert datum.loss_fn_inputs["weights"].tolist() == weights
        assert len(datum.model_input.to_ints()) == len(datum.loss_fn_inputs["target_tokens"].tolist())
        assert len(datum.model_input.to_ints()) == len(datum.loss_fn_inputs["weights"].tolist())

    def test_datum_iteration(self, training_client):
        """Test iterating over datum components as in the visualization code."""
        tokenizer = training_client.get_tokenizer()

        # Create simple datum
        input_tokens = [1, 100, 101, 102]
        target_tokens = [100, 101, 102, 999]
        weights = [0, 1, 1, 1]

        datum = types.Datum(
            model_input=types.ModelInput.from_ints(tokens=input_tokens),
            loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
        )

        # Test iteration (as in xorl_client_example1.py visualization) - use .tolist()
        items = list(zip(
            datum.model_input.to_ints(),
            datum.loss_fn_inputs["target_tokens"].tolist(),
            datum.loss_fn_inputs["weights"].tolist()
        ))

        assert len(items) == 4
        assert items[0] == (1, 100, 0)
        assert items[1] == (100, 101, 1)
        assert items[2] == (101, 102, 1)
        assert items[3] == (102, 999, 1)

    def test_multiple_examples_processing(self, training_client):
        """Test processing multiple examples."""
        tokenizer = training_client.get_tokenizer()

        examples = [
            {"input": "Hello", "output": "world"},
            {"input": "banana", "output": "split"},
        ]

        def process_example(example):
            prompt = f"Input: {example['input']}\nOutput:"
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
            prompt_weights = [0] * len(prompt_tokens)

            completion_tokens = tokenizer.encode(f" {example['output']}", add_special_tokens=False)
            completion_weights = [1] * len(completion_tokens)

            tokens = prompt_tokens + completion_tokens
            weights = prompt_weights + completion_weights

            input_tokens = tokens[:-1]
            target_tokens = tokens[1:]
            weights = weights[1:]

            return types.Datum(
                model_input=types.ModelInput.from_ints(tokens=input_tokens),
                loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
            )

        processed = [process_example(ex) for ex in examples]

        assert len(processed) == 2
        for datum in processed:
            assert isinstance(datum, types.Datum)
            # Use .tolist() since lists are auto-converted to TensorData
            assert len(datum.model_input.to_ints()) == len(datum.loss_fn_inputs["target_tokens"].tolist())
            assert len(datum.model_input.to_ints()) == len(datum.loss_fn_inputs["weights"].tolist())


class TestServiceClientMocked:
    """Tests for ServiceClient with mocked HTTP calls."""

    @pytest.fixture
    def mock_client_holder(self):
        """Create a mock ClientHolder."""
        with patch("xorl_client.client.service_client.ClientHolder") as mock_holder_class:
            mock_holder = Mock(spec=ClientHolder)
            mock_holder.get_model_id.return_value = "test-model-id-123"
            mock_holder.post.return_value = {"model_id": "test-model-id-123", "status": "created"}
            mock_holder_class.return_value = mock_holder
            yield mock_holder_class

    def test_create_lora_training_client(self, mock_client_holder):
        """Test creating a LoRA training client."""
        service_client = ServiceClient(base_url="http://localhost:6000")
        training_client = service_client.create_lora_training_client(
            base_model="Qwen/Qwen2.5-3B-Instruct",
            rank=32,
        )

        assert training_client.model_id == "test-model-id-123"
        assert training_client.base_model == "Qwen/Qwen2.5-3B-Instruct"

    def test_create_lora_training_client_with_custom_config(self, mock_client_holder):
        """Test creating a LoRA training client with custom config."""
        service_client = ServiceClient(base_url="http://localhost:6000")
        training_client = service_client.create_lora_training_client(
            base_model="meta-llama/Llama-3-8B",
            rank=64,
            alpha=128,
            dropout=0.1,
        )

        assert training_client.base_model == "meta-llama/Llama-3-8B"


class TestDatumSerialization:
    """Tests for Datum serialization for API calls."""

    def test_datum_to_dict_for_api(self):
        """Test that Datum serializes correctly for API calls."""
        datum = types.Datum(
            model_input=types.ModelInput.from_ints([1, 2, 3, 4]),
            loss_fn_inputs={
                "target_tokens": [2, 3, 4, 5],
                "weights": [0.0, 1.0, 1.0, 1.0],
            }
        )

        d = datum.to_dict()

        # Check structure matches what API expects
        assert "model_input" in d
        assert "loss_fn_inputs" in d
        assert d["model_input"]["input_ids"] == [1, 2, 3, 4]
        # Lists are auto-converted to TensorData, so they serialize as dicts
        assert d["loss_fn_inputs"]["target_tokens"]["data"] == [2, 3, 4, 5]
        assert d["loss_fn_inputs"]["weights"]["data"] == [0.0, 1.0, 1.0, 1.0]

    def test_multiple_datums_serialization(self):
        """Test serializing multiple datums."""
        datums = [
            types.Datum(
                model_input=types.ModelInput.from_ints([1, 2, 3]),
                loss_fn_inputs={"target_tokens": [2, 3, 4], "weights": [1, 1, 1]}
            ),
            types.Datum(
                model_input=types.ModelInput.from_ints([5, 6, 7]),
                loss_fn_inputs={"target_tokens": [6, 7, 8], "weights": [1, 1, 1]}
            ),
        ]

        serialized = [d.to_dict() for d in datums]

        assert len(serialized) == 2
        assert serialized[0]["model_input"]["input_ids"] == [1, 2, 3]
        assert serialized[1]["model_input"]["input_ids"] == [5, 6, 7]
