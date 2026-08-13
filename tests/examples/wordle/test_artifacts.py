import json

import pytest

from examples.wordle.artifacts import ArtifactStore, redact_url, terminal_audit


def test_url_redaction_removes_credentials_and_query():
    assert (
        redact_url("http://user:secret@host:8000/path?token=abc")
        == "http://host:8000/path"
    )


def test_resume_fails_closed_on_session_step_and_model_mismatch(tmp_path):
    store = ArtifactStore(tmp_path)
    store.initialize(
        run_config={"run_metadata": {"session_id": "session-a"}},
        source_info={},
        resume=False,
    )
    store.write_step(1, {"step": 1, "optimizer_complete": True})
    base = {
        "step": 1,
        "path": "xorl://wordle/weights/one",
        "model": "model-a",
        "model_id": "wordle",
        "session_id": "session-a",
        "optimizer": True,
    }
    store.append_checkpoint(base)
    assert store.load_resume(model="model-a", model_id="wordle").step == 1
    with pytest.raises(ValueError, match="model"):
        store.load_resume(model="model-b", model_id="wordle")
    ledger = tmp_path / "checkpoints.jsonl"
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    records[-1]["session_id"] = "session-b"
    ledger.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    with pytest.raises(ValueError, match="session"):
        store.load_resume(model="model-a", model_id="wordle")

    records[-1]["session_id"] = "session-a"
    records[-1]["step"] = 2
    ledger.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    with pytest.raises(ValueError, match="step"):
        store.load_resume(model="model-a", model_id="wordle")


def test_terminal_audit_success_and_failure_paths():
    good = {
        "step": 1,
        "optimizer_complete": True,
        "final_sync": True,
        "finite_loss": True,
        "finite_gradient": True,
        "correctness_gates_passed": True,
        "checkpoint": "x",
    }
    assert terminal_audit(records=[good], expected_steps=1, checkpoint_every=1)[
        "success"
    ]
    assert not terminal_audit(
        records=[dict(good, final_sync=False)], expected_steps=1, checkpoint_every=1
    )["success"]
