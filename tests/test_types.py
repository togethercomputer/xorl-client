"""Tests for xorl_client types module."""

import pytest
import numpy as np

from xorl_client import types


class TestTensorData:
    """Tests for TensorData class."""

    def test_from_list(self):
        """Test creating TensorData from a list."""
        data = [1.0, 2.0, 3.0, 4.0]
        tensor_data = types.TensorData.from_list(data)

        assert tensor_data.data == data
        assert tensor_data.dtype == "float32"
        assert tensor_data.shape == [4]

    def test_from_numpy(self):
        """Test creating TensorData from a numpy array."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor_data = types.TensorData.from_numpy(arr)

        assert tensor_data.data == [1.0, 2.0, 3.0]
        assert tensor_data.shape == [3]

    def test_from_numpy_2d(self):
        """Test creating TensorData from a 2D numpy array."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor_data = types.TensorData.from_numpy(arr)

        assert tensor_data.data == [1.0, 2.0, 3.0, 4.0]
        assert tensor_data.shape == [2, 2]

    def test_to_dict(self):
        """Test TensorData serialization to dict."""
        tensor_data = types.TensorData(data=[1.0, 2.0], dtype="float32", shape=[2])
        d = tensor_data.to_dict()

        assert d["data"] == [1.0, 2.0]
        assert d["dtype"] == "float32"
        assert d["shape"] == [2]

    def test_from_dict(self):
        """Test TensorData deserialization from dict."""
        d = {"data": [1.0, 2.0, 3.0], "dtype": "float64", "shape": [3]}
        tensor_data = types.TensorData.from_dict(d)

        assert tensor_data.data == [1.0, 2.0, 3.0]
        assert tensor_data.dtype == "float64"
        assert tensor_data.shape == [3]

    def test_to_numpy(self):
        """Test TensorData conversion to numpy array."""
        tensor_data = types.TensorData(data=[1.0, 2.0, 3.0], dtype="float32", shape=[3])
        arr = tensor_data.to_numpy()

        assert arr.dtype == np.float32
        assert arr.shape == (3,)
        assert list(arr) == [1.0, 2.0, 3.0]

    def test_to_numpy_2d(self):
        """Test TensorData conversion to 2D numpy array."""
        tensor_data = types.TensorData(data=[1.0, 2.0, 3.0, 4.0], dtype="float32", shape=[2, 2])
        arr = tensor_data.to_numpy()

        assert arr.dtype == np.float32
        assert arr.shape == (2, 2)
        assert arr[0, 0] == 1.0
        assert arr[1, 1] == 4.0

    def test_to_numpy_int64(self):
        """Test TensorData conversion to numpy array with int64 dtype."""
        tensor_data = types.TensorData(data=[1, 2, 3], dtype="int64", shape=[3])
        arr = tensor_data.to_numpy()

        assert arr.dtype == np.int64
        assert list(arr) == [1, 2, 3]

    def test_tolist_1d(self):
        """Test tolist for 1D TensorData."""
        tensor_data = types.TensorData(data=[1.0, 2.0, 3.0], dtype="float32", shape=[3])
        result = tensor_data.tolist()

        assert result == [1.0, 2.0, 3.0]

    def test_tolist_2d(self):
        """Test tolist for 2D TensorData (respects shape)."""
        tensor_data = types.TensorData(data=[1.0, 2.0, 3.0, 4.0], dtype="float32", shape=[2, 2])
        result = tensor_data.tolist()

        assert result == [[1.0, 2.0], [3.0, 4.0]]


class TestModelInput:
    """Tests for ModelInput class."""

    def test_from_ints(self):
        """Test creating ModelInput from token IDs."""
        tokens = [1, 2, 3, 4, 5]
        model_input = types.ModelInput.from_ints(tokens)

        assert model_input.tokens == tokens
        assert model_input.text is None

    def test_from_str(self):
        """Test creating ModelInput from text."""
        text = "Hello, world!"
        model_input = types.ModelInput.from_str(text)

        assert model_input.text == text
        assert model_input.tokens is None

    def test_to_ints(self):
        """Test converting ModelInput to list of ints."""
        tokens = [10, 20, 30]
        model_input = types.ModelInput.from_ints(tokens)

        assert model_input.to_ints() == tokens

    def test_to_ints_raises_when_no_tokens(self):
        """Test that to_ints raises ValueError when only text is set."""
        model_input = types.ModelInput.from_str("Hello")

        with pytest.raises(ValueError, match="tokens are not set"):
            model_input.to_ints()

    def test_to_dict_with_tokens(self):
        """Test ModelInput serialization with tokens."""
        tokens = [1, 2, 3]
        model_input = types.ModelInput.from_ints(tokens)
        d = model_input.to_dict()

        assert d["input_ids"] == tokens
        assert "text" not in d

    def test_to_dict_with_text(self):
        """Test ModelInput serialization with text."""
        text = "Hello"
        model_input = types.ModelInput.from_str(text)
        d = model_input.to_dict()

        assert d["text"] == text
        assert "input_ids" not in d

    def test_length_with_tokens(self):
        """Test length property with tokens."""
        tokens = [1, 2, 3, 4]
        model_input = types.ModelInput.from_ints(tokens)

        assert model_input.length == 4

    def test_length_with_text(self):
        """Test length property with text."""
        text = "Hello"
        model_input = types.ModelInput.from_str(text)

        assert model_input.length == 5  # Character count


class TestDatum:
    """Tests for Datum class."""

    def test_create_datum(self):
        """Test creating a Datum with model_input and loss_fn_inputs."""
        model_input = types.ModelInput.from_ints([1, 2, 3])
        loss_fn_inputs = {
            "target_tokens": [2, 3, 4],
            "weights": [1.0, 1.0, 1.0],
        }

        datum = types.Datum(model_input=model_input, loss_fn_inputs=loss_fn_inputs)

        assert datum.model_input.to_ints() == [1, 2, 3]
        # Lists are auto-converted to TensorData, use .tolist() to get the data
        assert datum.loss_fn_inputs["target_tokens"].tolist() == [2, 3, 4]
        assert datum.loss_fn_inputs["weights"].tolist() == [1.0, 1.0, 1.0]

    def test_datum_to_dict_with_lists(self):
        """Test Datum serialization with plain lists (auto-converted to TensorData)."""
        model_input = types.ModelInput.from_ints([1, 2, 3])
        loss_fn_inputs = {
            "target_tokens": [2, 3, 4],
            "weights": [1.0, 1.0, 1.0],
        }

        datum = types.Datum(model_input=model_input, loss_fn_inputs=loss_fn_inputs)
        d = datum.to_dict()

        assert d["model_input"]["input_ids"] == [1, 2, 3]
        # Lists are auto-converted to TensorData, so they serialize as dicts
        assert d["loss_fn_inputs"]["target_tokens"]["data"] == [2, 3, 4]
        assert d["loss_fn_inputs"]["weights"]["data"] == [1.0, 1.0, 1.0]

    def test_datum_to_dict_with_tensor_data(self):
        """Test Datum serialization with TensorData."""
        model_input = types.ModelInput.from_ints([1, 2, 3])
        loss_fn_inputs = {
            "target_tokens": types.TensorData.from_list([2, 3, 4]),
            "weights": types.TensorData.from_list([1.0, 1.0, 1.0]),
        }

        datum = types.Datum(model_input=model_input, loss_fn_inputs=loss_fn_inputs)
        d = datum.to_dict()

        assert d["loss_fn_inputs"]["target_tokens"]["data"] == [2, 3, 4]
        assert d["loss_fn_inputs"]["weights"]["data"] == [1.0, 1.0, 1.0]

    def test_datum_to_dict_with_numpy(self):
        """Test Datum serialization with numpy arrays."""
        model_input = types.ModelInput.from_ints([1, 2, 3])
        loss_fn_inputs = {
            "target_tokens": np.array([2, 3, 4]),
            "weights": np.array([1.0, 1.0, 1.0]),
        }

        datum = types.Datum(model_input=model_input, loss_fn_inputs=loss_fn_inputs)
        d = datum.to_dict()

        # Numpy arrays should be converted to TensorData dicts
        assert d["loss_fn_inputs"]["target_tokens"]["data"] == [2, 3, 4]
        assert d["loss_fn_inputs"]["weights"]["data"] == [1.0, 1.0, 1.0]

    def test_datum_tolist_method(self):
        """Test that loss_fn_inputs values have .tolist() method."""
        model_input = types.ModelInput.from_ints([1, 2, 3])
        loss_fn_inputs = {
            "target_tokens": [2, 3, 4],
            "weights": [1.0, 1.0, 1.0],
        }

        datum = types.Datum(model_input=model_input, loss_fn_inputs=loss_fn_inputs)

        # Should be able to call .tolist() like on numpy arrays
        assert datum.loss_fn_inputs["target_tokens"].tolist() == [2, 3, 4]
        assert datum.loss_fn_inputs["weights"].tolist() == [1.0, 1.0, 1.0]


class TestAdamParams:
    """Tests for AdamParams class."""

    def test_default_values(self):
        """Test AdamParams default values."""
        params = types.AdamParams(learning_rate=1e-4)

        assert params.learning_rate == 1e-4
        assert params.beta1 == 0.9
        assert params.beta2 == 0.95
        assert params.eps == 1e-8

    def test_custom_values(self):
        """Test AdamParams with custom values."""
        params = types.AdamParams(
            learning_rate=1e-5,
            beta1=0.85,
            beta2=0.99,
            eps=1e-6,
        )

        assert params.learning_rate == 1e-5
        assert params.beta1 == 0.85
        assert params.beta2 == 0.99
        assert params.eps == 1e-6

    def test_to_dict(self):
        """Test AdamParams serialization."""
        params = types.AdamParams(learning_rate=1e-4)
        d = params.to_dict()

        assert d["learning_rate"] == 1e-4
        assert d["beta1"] == 0.9
        assert d["beta2"] == 0.95
        assert d["eps"] == 1e-8


class TestSamplingParams:
    """Tests for SamplingParams class."""

    def test_default_values(self):
        """Test SamplingParams default values."""
        params = types.SamplingParams()

        assert params.max_tokens == 128
        assert params.temperature == 1.0
        assert params.top_p == 1.0
        assert params.top_k == -1
        assert params.stop is None

    def test_custom_values(self):
        """Test SamplingParams with custom values."""
        params = types.SamplingParams(
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            stop=["</s>", "\n\n"],
        )

        assert params.max_tokens == 256
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.top_k == 50
        assert params.stop == ["</s>", "\n\n"]

    def test_to_dict(self):
        """Test SamplingParams serialization."""
        params = types.SamplingParams(max_tokens=100, stop=["end"])
        d = params.to_dict()

        assert d["max_new_tokens"] == 100
        assert d["stop"] == ["end"]


class TestLoraConfig:
    """Tests for LoraConfig class."""

    def test_default_values(self):
        """Test LoraConfig default values."""
        config = types.LoraConfig()

        assert config.rank == 32
        assert config.alpha is None
        assert config.dropout == 0.0
        assert config.target_modules is None

    def test_custom_values(self):
        """Test LoraConfig with custom values."""
        config = types.LoraConfig(
            rank=64,
            alpha=128,
            dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        )

        assert config.rank == 64
        assert config.alpha == 128
        assert config.dropout == 0.1
        assert config.target_modules == ["q_proj", "v_proj"]

    def test_to_dict(self):
        """Test LoraConfig serialization."""
        config = types.LoraConfig(rank=16, alpha=32)
        d = config.to_dict()

        assert d["rank"] == 16
        assert d["alpha"] == 32
        assert d["dropout"] == 0.0


class TestResponseTypes:
    """Tests for response type classes."""

    def test_forward_backward_output(self):
        """Test ForwardBackwardOutput."""
        output = types.ForwardBackwardOutput(
            loss_fn_outputs=[{"loss": 0.5}],
            metrics={"total_tokens": 100},
        )

        assert output.loss_fn_outputs[0]["loss"] == 0.5
        assert output.metrics["total_tokens"] == 100

    def test_optim_step_response(self):
        """Test OptimStepResponse."""
        response = types.OptimStepResponse(
            metrics={"grad_norm": 1.5},
            step=10,
        )

        assert response.metrics["grad_norm"] == 1.5
        assert response.step == 10

    def test_save_weights_response(self):
        """Test SaveWeightsResponse."""
        response = types.SaveWeightsResponse(path="/path/to/checkpoint")

        assert response.path == "/path/to/checkpoint"

    def test_sample_response(self):
        """Test SampleResponse."""
        seq = types.SampledSequence(
            tokens=[1, 2, 3],
            logprobs=[-0.1, -0.2, -0.3],
            text="hello",
        )
        response = types.SampleResponse(sequences=[seq])

        assert response.text == "hello"
        assert response.tokens == [1, 2, 3]
        assert response.logprobs == [-0.1, -0.2, -0.3]
