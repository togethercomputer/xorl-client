import json

import pytest

from examples.wordle.artifacts import ArtifactStore


def test_resume_rejects_declared_step_without_matching_completed_artifact(tmp_path):
    store = ArtifactStore(tmp_path)
    store.initialize(
        run_config={"run_metadata": {"session_id": "session"}},
        source_info={},
        resume=False,
    )
    store.append_checkpoint(
        {
            "step": 3,
            "path": "xorl://wordle/weights/step-3",
            "model": "model",
            "model_id": "wordle",
            "session_id": "session",
            "optimizer": True,
        }
    )
    with pytest.raises(ValueError, match="completed step artifact"):
        store.load_resume(model="model", model_id="wordle")


def test_resume_rejects_checkpoint_without_optimizer_state(tmp_path):
    store = ArtifactStore(tmp_path)
    store.initialize(
        run_config={"run_metadata": {"session_id": "session"}},
        source_info={},
        resume=False,
    )
    store.write_step(1, {"step": 1})
    checkpoint = {
        "step": 1,
        "path": "xorl://wordle/weights/step-1",
        "model": "model",
        "model_id": "wordle",
        "session_id": "session",
        "optimizer": False,
    }
    (tmp_path / "checkpoints.jsonl").write_text(json.dumps(checkpoint) + "\n")
    with pytest.raises(ValueError, match="optimizer"):
        store.load_resume(model="model", model_id="wordle")
