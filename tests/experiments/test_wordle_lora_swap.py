import pytest


pytest.importorskip("transformers")

from experiments.wordle.standalone import train_opsd_baseline


def test_load_sglang_lora_treats_http_400_already_loaded_same_path_as_success(
    monkeypatch,
) -> None:
    def fake_post(*args, **kwargs):
        raise RuntimeError(
            "SGLang POST /load_lora_adapter returned HTTP 400: already loaded at /tmp/adapter-v2"
        )

    monkeypatch.setattr(train_opsd_baseline, "_post_sglang", fake_post)

    train_opsd_baseline.load_sglang_lora(
        "http://sglang", lora_name="policy", lora_path="/tmp/adapter-v2"
    )


def test_load_sglang_lora_rejects_already_loaded_different_path(monkeypatch) -> None:
    def fake_post(*args, **kwargs):
        return {
            "success": False,
            "error_message": "adapter policy already loaded at /tmp/adapter-v1",
        }

    monkeypatch.setattr(train_opsd_baseline, "_post_sglang", fake_post)

    with pytest.raises(RuntimeError, match="already loaded"):
        train_opsd_baseline.load_sglang_lora(
            "http://sglang", lora_name="policy", lora_path="/tmp/adapter-v2"
        )


def test_load_sglang_lora_treats_success_false_already_loaded_same_path_as_success(
    monkeypatch,
) -> None:
    def fake_post(*args, **kwargs):
        return {
            "success": False,
            "error_message": "adapter policy already loaded at /tmp/adapter-v2",
        }

    monkeypatch.setattr(train_opsd_baseline, "_post_sglang", fake_post)

    train_opsd_baseline.load_sglang_lora(
        "http://sglang", lora_name="policy", lora_path="/tmp/adapter-v2"
    )


def test_unload_sglang_lora_flushes_cache_before_unload(monkeypatch) -> None:
    calls = []

    def fake_post(infer_url, path, payload, **kwargs):
        calls.append((infer_url, path, payload, kwargs))
        return {}

    monkeypatch.delenv("OPSD_SGLANG_FLUSH_BEFORE_LORA_SWAP", raising=False)
    monkeypatch.setattr(train_opsd_baseline, "_post_sglang", fake_post)

    train_opsd_baseline.unload_sglang_lora("http://sglang", "policy")

    assert [call[1] for call in calls] == ["/flush_cache", "/unload_lora_adapter"]
    assert calls[0][3]["allow_non_json_success"] is True


def test_unload_sglang_lora_flush_cache_can_be_disabled(monkeypatch) -> None:
    calls = []

    def fake_post(infer_url, path, payload, **kwargs):
        calls.append(path)
        return {}

    monkeypatch.setenv("OPSD_SGLANG_FLUSH_BEFORE_LORA_SWAP", "0")
    monkeypatch.setattr(train_opsd_baseline, "_post_sglang", fake_post)

    train_opsd_baseline.unload_sglang_lora("http://sglang", "policy")

    assert calls == ["/unload_lora_adapter"]
