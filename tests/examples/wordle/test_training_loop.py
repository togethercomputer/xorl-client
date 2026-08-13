import asyncio
from pathlib import Path
from types import SimpleNamespace
import pytest

from examples.wordle.artifacts import ArtifactStore
from examples.wordle.config import load_config
from examples.wordle.task import WordleTask
from examples.wordle.train import (
    ExperimentRunner,
    XorlTrainerBackend,
    _k3_max,
    _ratio_error_max,
)
from xorl_client import types


ROOT = Path(__file__).parents[3]


class Tokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [1, 2, len(messages)]


class Sampler:
    def __init__(self, *, fail_step2=False):
        self.fail_step2 = fail_step2
        self.seeds = []

    async def generate_batch_native_async(self, prompts, params, return_logprobs):
        seeds = [param.sampling_seed for param in params]
        self.seeds.extend(seeds)
        if self.fail_step2 and any(seed >= 2_000_000 for seed in seeds):
            raise RuntimeError("simulated interruption")
        return [
            types.SampleResponse(
                sequences=[
                    types.SampledSequence(
                        tokens=[10, 11],
                        logprobs=[-0.1, -0.2],
                        text="<guess>[ABIDE]</guess>",
                    )
                ],
                meta_info={},
            )
            for _ in prompts
        ]


class Trainer:
    def __init__(self):
        self.events = []
        self.restored = None

    async def forward_backward(self, datums):
        self.events.append(("forward_backward", len(datums)))
        return {"loss": 0.5, "k3": 0.0, "ratio_error": 0.0}

    async def optimizer_step(self):
        self.events.append(("optimizer", None))
        return {"grad_norm": 1.0}

    async def sync(self):
        self.events.append(("sync", None))
        return {"success": True, "total_bytes": 10}

    async def checkpoint(self, name):
        self.events.append(("checkpoint", name))
        return f"xorl://wordle/weights/{name}"

    async def restore(self, path):
        self.restored = path
        return path


def _config(output):
    config = load_config(ROOT / "examples/wordle/configs/importance_sampling.yaml")
    config.trainer = config.trainer.model_copy(
        update={"steps": 2, "checkpoint_every": 1}
    )
    config.wordle = config.wordle.model_copy(
        update={
            "train_targets": 4,
            "eval_targets": 2,
            "targets_per_step": 1,
            "group_size": 2,
        }
    )
    config.artifacts = config.artifacts.model_copy(update={"output_dir": str(output)})
    return config


def _task(config):
    return WordleTask(
        targets_path=config.wordle.targets_path,
        legal_guesses_path=config.wordle.legal_guesses_path,
        train_targets=config.wordle.train_targets,
        eval_targets=config.wordle.eval_targets,
        seed=config.wordle.target_seed,
    )


def test_two_steps_checkpoint_restore_final_sync_and_no_repeated_step(tmp_path):
    config = _config(tmp_path / "run")
    store = ArtifactStore(config.artifacts.output_dir)
    store.initialize(
        run_config={"run_metadata": {"session_id": "session-1"}},
        source_info={"test": True},
        resume=False,
    )
    first_trainer = Trainer()
    interrupted = ExperimentRunner(
        config=config,
        task=_task(config),
        tokenizer=Tokenizer(),
        sampler=Sampler(fail_step2=True),
        trainer=first_trainer,
        store=store,
        session_id="session-1",
    )
    with pytest.raises(RuntimeError, match="interruption"):
        asyncio.run(interrupted.run())
    assert (store.steps / "step-00000001.json").is_file()
    assert not (store.steps / "step-00000002.json").exists()
    state = store.load_resume(model=config.model.model, model_id=config.model.model_id)
    assert state.step == 1

    resumed_trainer = Trainer()
    asyncio.run(resumed_trainer.restore(state.checkpoint_path))
    resumed = ExperimentRunner(
        config=config,
        task=_task(config),
        tokenizer=Tokenizer(),
        sampler=Sampler(),
        trainer=resumed_trainer,
        store=store,
        session_id=state.session_id,
    )
    audit = asyncio.run(resumed.run(start_step=state.step + 1))
    assert audit["success"]
    assert sorted(path.name for path in store.steps.glob("step-*.json")) == [
        "step-00000001.json",
        "step-00000002.json",
    ]
    assert [event[0] for event in resumed_trainer.events].count("optimizer") == 1
    assert resumed_trainer.events[-2][0] == "sync"
    assert resumed_trainer.events[-1][0] == "checkpoint"
    for relative in (
        "run_config.json",
        "source_info.json",
        "metrics.jsonl",
        "checkpoints.jsonl",
        "terminal_audit.json",
    ):
        assert (store.root / relative).is_file()


def test_cispo_backend_passes_absolute_ratio_bounds():
    config = load_config(ROOT / "examples/wordle/configs/cispo.yaml")

    class Client:
        def __init__(self):
            self.call = None

        async def forward_backward(self, datums, loss_fn, loss_fn_params):
            self.call = (datums, loss_fn, loss_fn_params)
            return SimpleNamespace(loss_fn_outputs=[], metrics={"loss": 0.5})

    client = Client()
    backend = XorlTrainerBackend(client, config)
    asyncio.run(backend.forward_backward([object()]))
    assert client.call[1] == "cispo"
    assert client.call[2]["clip_low_threshold"] == 0.0
    assert client.call[2]["clip_high_threshold"] == 4.0


def test_zero_k3_metrics_use_k3_and_ratio_distance_from_one():
    metrics = [
        {
            "kl_sample_train_k3": 2e-7,
            "kl_k3_debug_max": 8e-7,
            "ratio_mean": 1.0,
            "ratio_min": 0.999_998,
            "ratio_max": 1.000_003,
        }
    ]
    assert _k3_max(metrics) == pytest.approx(8e-7)
    assert _ratio_error_max(metrics) == pytest.approx(3e-6)
