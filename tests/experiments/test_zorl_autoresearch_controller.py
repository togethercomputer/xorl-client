from types import SimpleNamespace

import pytest
import yaml

from experiments.zorl.autoresearch import controller
from experiments.zorl.standalone import wandb_log_forwarder, zorl_client
from experiments.zorl.standalone.tasks import gsm8k, opd_multiplication, wordle
from experiments.zorl.standalone.tasks.base import Example


class FakeTokenizer:
    def apply_chat_template(self, msgs, **kwargs):
        return list(range(len(msgs) * 5))

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 251 for ch in text]


def test_parse_standalone_zorl_log_scores_weak_signal(tmp_path):
    log_path = tmp_path / "zorl_client.log"
    log_path.write_text(
        "\n".join(
            [
                "[init] task=opd_multiplication infer_url=http://sglang:30000",
                "  cold: reward_mean=0.2500 exact_rate=0.2500 exact_count=2.0/8",
                (
                    "  step 1/4: reward_mean=0.3000 best_cand=0.5000 "
                    "update_norm=123.00 used_pairs=4 t_score=1.0s t_apply=0.1s, "
                    "probe_reward=0.3125 exact_rate=0.3125 [emitted_number=1.000]"
                ),
                (
                    "  step 2/4: reward_mean=0.3500 best_cand=0.6250 "
                    "update_norm=125.00 unclipped_update_norm=250.00 update_clip_scale=0.5000 "
                    "used_pairs=4 t_score=1.0s t_apply=0.1s, "
                    "probe_reward=0.3750 exact_rate=0.3750 [emitted_number=1.000]"
                ),
                "[done] best probe: reward_mean=0.2500 exact_count=2.0/8",
            ]
        ),
        encoding="utf-8",
    )
    candidate = {
        "score_source": "standalone_zorl_log",
        "score_gates": {
            "min_steps": 2,
            "expected_total": 8,
            "min_update_norm_positive_rows": 1,
            "weak_reward_gain": 0.10,
            "promote_reward_gain": 0.20,
            "strong_reward_gain": 0.30,
            "weak_exact_gain": 1,
            "promote_exact_gain": 3,
            "strong_exact_gain": 5,
        },
        "env": {
            "TASK": "opd_multiplication",
            "INFER_URL": "http://sglang:30000",
            "SEED": "1234",
            "LORA_RANK": "4",
        },
    }

    score = controller.score_result(log_path, candidate)

    assert score["score_source"] == "standalone_zorl_log"
    assert score["verdict"] == "weak_zorl_signal"
    assert score["metrics"]["completed_steps"] == 2
    assert score["metrics"]["cold_exact"] == 2.0
    assert score["metrics"]["best_exact"] == 3.0
    assert score["metrics"]["exact_gain"] == 1.0
    assert score["metrics"]["update_norm_positive_rows"] == 2
    assert score["metrics"]["task"] == "opd_multiplication"
    assert score["metrics"]["infer_url"] == "http://sglang:30000"
    assert score["metrics"]["seed"] == "1234"
    assert score["metrics"]["update_clip_scale_last"] == 0.5
    assert score["metrics"]["unclipped_update_norm_last"] == 250.0


def test_parse_standalone_zorl_log_waits_for_min_steps_before_update_gate(tmp_path):
    log_path = tmp_path / "zorl_client.log"
    log_path.write_text(
        "\n".join(
            [
                "recipe: rank=4 alpha=4 steps=6 pairs=8 sigma=0.05 lr=0.01 max_update_norm=20000 perturbation_mode=b_only",
                "  cold: reward_mean=0.0500 exact_rate=0.0000 exact_count=0.0/16",
                (
                    "  step 1/6: reward_mean=0.1155 best_cand=0.2537 "
                    "update_norm=2221.66 pair_delta_mean=-0.0003 pair_delta_std=0.0053 "
                    "unclipped_update_norm=2221.66 grad_norm=2221.66 update_clip_scale=1.0000 "
                    "zero_score_pairs=0 dropped_pairs=0 score_normalization=standard used_pairs=8 "
                    "t_score=18.4s t_apply=3.2s, probe_reward=0.2890 exact_rate=0.0000"
                ),
            ]
        ),
        encoding="utf-8",
    )
    candidate = {
        "score_source": "standalone_zorl_log",
        "score_gates": {
            "expected_total": 16,
            "min_steps": 3,
            "min_update_norm_positive_rows": 2,
            "weak_reward_gain": 0.03,
            "promote_reward_gain": 0.08,
            "strong_reward_gain": 0.15,
        },
    }

    score = controller.score_result(log_path, candidate)

    assert score["verdict"] == "incomplete"
    assert score["metrics"]["update_norm_positive_rows"] == 1
    assert score["metrics"]["pair_delta_std_last"] == 0.0053
    assert score["metrics"]["perturbation_mode"] == "b_only"


def test_parse_standalone_zorl_log_waits_for_planned_steps_before_terminal_update_gate(tmp_path):
    log_path = tmp_path / "zorl_client.log"
    log_path.write_text(
        "\n".join(
            [
                "recipe: rank=4 alpha=4 steps=8 pairs=8 sigma=0.05 lr=0.01 max_update_norm=20000 perturbation_mode=b_only",
                "  cold: reward_mean=0.2500 exact_rate=0.0938 exact_count=1.5/16",
                (
                    "  step 1/8: reward_mean=0.1000 best_cand=0.1001 "
                    "update_norm=0.00 pair_delta_mean=0.0000 pair_delta_std=0.0000 "
                    "unclipped_update_norm=0.00 update_clip_scale=1.0000 used_pairs=8 "
                    "t_score=13.0s t_apply=1.7s, probe_reward=0.1167 exact_rate=0.0156"
                ),
                (
                    "  step 2/8: reward_mean=0.1002 best_cand=0.1005 "
                    "update_norm=0.00 pair_delta_mean=0.0000 pair_delta_std=0.0000 "
                    "unclipped_update_norm=0.00 update_clip_scale=1.0000 used_pairs=8 "
                    "t_score=14.6s t_apply=1.7s, probe_reward=0.2450 exact_rate=0.1094"
                ),
                (
                    "  step 3/8: reward_mean=0.1766 best_cand=0.2679 "
                    "update_norm=2222.44 pair_delta_mean=0.0145 pair_delta_std=0.0356 "
                    "unclipped_update_norm=2222.44 update_clip_scale=1.0000 used_pairs=8 "
                    "t_score=11.1s t_apply=3.2s, probe_reward=0.1059 exact_rate=0.0000"
                ),
                (
                    "  step 4/8: reward_mean=0.1419 best_cand=0.2678 "
                    "update_norm=2206.77 pair_delta_mean=-0.0001 pair_delta_std=0.0001 "
                    "unclipped_update_norm=2206.77 update_clip_scale=1.0000 used_pairs=8 "
                    "t_score=13.3s t_apply=3.2s, probe_reward=0.1054 exact_rate=0.0000"
                ),
                (
                    "  step 5/8: reward_mean=0.1419 best_cand=0.2675 "
                    "update_norm=0.00 pair_delta_mean=0.0000 pair_delta_std=0.0000 "
                    "unclipped_update_norm=0.00 update_clip_scale=1.0000 used_pairs=8 "
                    "t_score=13.9s t_apply=1.7s, probe_reward=0.2640 exact_rate=0.1094"
                ),
            ]
        ),
        encoding="utf-8",
    )
    candidate = {
        "score_source": "standalone_zorl_log",
        "score_gates": {
            "expected_total": 16,
            "min_steps": 4,
            "min_update_norm_positive_rows": 3,
            "weak_reward_gain": 0.02,
        },
    }

    score = controller.score_result(log_path, candidate)

    assert score["verdict"] == "incomplete"
    assert score["metrics"]["completed_steps"] == 5
    assert score["metrics"]["planned_steps"] == 8
    assert score["metrics"]["run_complete"] is False
    assert score["metrics"]["update_norm_positive_rows"] == 2


def test_parse_standalone_zorl_log_can_require_best_probe_after_positive_update(tmp_path):
    log_path = tmp_path / "zorl_client.log"
    log_path.write_text(
        "\n".join(
            [
                "recipe: rank=4 alpha=4 steps=5 pairs=8 sigma=0.05 lr=0.003 max_update_norm=20000 perturbation_mode=b_only",
                "  cold: reward_mean=0.2000 exact_rate=0.0000 exact_count=0.0/16",
                (
                    "  step 1/5: reward_mean=0.1400 best_cand=0.2600 "
                    "update_norm=2221.80 pair_delta_mean=-0.0025 pair_delta_std=0.0049 "
                    "unclipped_update_norm=2221.80 update_clip_scale=1.0000 used_pairs=8 "
                    "t_score=13.0s t_apply=3.3s, probe_reward=0.1000 exact_rate=0.0000"
                ),
                (
                    "  step 2/5: reward_mean=0.1400 best_cand=0.2600 "
                    "update_norm=2220.02 pair_delta_mean=-0.0003 pair_delta_std=0.0008 "
                    "unclipped_update_norm=2220.02 update_clip_scale=1.0000 used_pairs=8 "
                    "t_score=13.0s t_apply=3.2s, probe_reward=0.0980 exact_rate=0.0000"
                ),
                (
                    "  step 3/5: reward_mean=0.1000 best_cand=0.1000 "
                    "update_norm=0.00 pair_delta_mean=0.0000 pair_delta_std=0.0000 "
                    "unclipped_update_norm=0.00 update_clip_scale=1.0000 used_pairs=8 "
                    "t_score=13.5s t_apply=1.8s, probe_reward=0.1440 exact_rate=0.0000"
                ),
                (
                    "  step 4/5: reward_mean=0.1000 best_cand=0.1000 "
                    "update_norm=0.00 pair_delta_mean=0.0000 pair_delta_std=0.0000 "
                    "unclipped_update_norm=0.00 update_clip_scale=1.0000 used_pairs=8 "
                    "t_score=13.5s t_apply=1.8s, probe_reward=0.1450 exact_rate=0.0000"
                ),
                (
                    "  step 5/5: reward_mean=0.1000 best_cand=0.1000 "
                    "update_norm=0.00 pair_delta_mean=0.0000 pair_delta_std=0.0000 "
                    "unclipped_update_norm=0.00 update_clip_scale=1.0000 used_pairs=8 "
                    "t_score=13.5s t_apply=1.8s, probe_reward=0.2660 exact_rate=0.1094"
                ),
                "[done] best probe: reward_mean=0.2660 exact_count=1.8/16",
            ]
        ),
        encoding="utf-8",
    )
    candidate = {
        "score_source": "standalone_zorl_log",
        "score_gates": {
            "expected_total": 16,
            "min_steps": 5,
            "min_update_norm_positive_rows": 2,
            "weak_reward_gain": 0.02,
            "require_best_probe_after_positive_update": True,
        },
    }

    score = controller.score_result(log_path, candidate)

    assert score["verdict"] == "science_reject"
    assert score["reason"] == "best_probe_after_positive_update=false"
    assert score["metrics"]["best_probe_step"] == 5
    assert score["metrics"]["best_probe_update_norm"] == 0.0
    assert score["metrics"]["best_probe_after_positive_update"] is False


def test_ready_ideas_skips_blocked_and_sorts_fractional_priorities():
    data = {
        "ideas": [
            {"id": "low", "status": "queued", "priority": 1},
            {"id": "draft", "status": "draft", "priority": 100},
            {"id": "blocked", "status": "queued", "priority": 99, "requires": ["missing"]},
            {"id": "done", "status": "complete", "priority": 10},
            {"id": "high", "status": "queued", "priority": 2.5},
        ]
    }

    assert [idea["id"] for idea in controller.ready_ideas(data)] == ["high", "low"]


def test_gsm8k_numeric_shaped_reward_keeps_exact_metric_binary(monkeypatch):
    ex = Example(project="gsm", prompt_ids=[], metadata={"gold": 100.0})

    monkeypatch.setenv("ZORL_GSM8K_REWARD_MODE", "numeric_shaped")
    shaped = gsm8k.score_completion(ex, "#### 90")
    assert 0.0 < shaped["reward"] < 1.0
    assert shaped["exact_match"] == 0.0
    assert shaped["numeric_closeness"] > 0.0

    monkeypatch.setenv("ZORL_GSM8K_REWARD_MODE", "exact")
    exact = gsm8k.score_completion(ex, "#### 90")
    assert exact["reward"] == 0.0
    assert exact["exact_match"] == 0.0


def test_opd_numeric_shaped_reward_keeps_exact_metric_binary(monkeypatch):
    ex = Example(project="opd", prompt_ids=[], metadata={"product": 100})

    monkeypatch.setenv("ZORL_OPD_REWARD_MODE", "numeric_shaped")
    shaped = opd_multiplication.score_completion(ex, "Answer: 90")
    assert 0.0 < shaped["reward"] < 1.0
    assert shaped["exact_match"] == 0.0
    assert shaped["numeric_closeness"] > 0.0
    assert shaped["relative_error"] == 0.1

    monkeypatch.setenv("ZORL_OPD_REWARD_MODE", "exact")
    exact = opd_multiplication.score_completion(ex, "Answer: 90")
    assert exact["reward"] == 0.0
    assert exact["exact_match"] == 0.0


def test_wordle_teacher_forced_trace_contains_hint_and_target_guess():
    ex = Example(project="wordle", prompt_ids=[], metadata={"target": "crate"})
    args = type("Args", (), {"wordle_teacher_trace_style": "hinted_cot"})()

    tf_ex = wordle.build_teacher_forced_example(FakeTokenizer(), ex, args=args)

    assert tf_ex.metadata["teacher_target_token_count"] > 0
    assert "CRATE" in tf_ex.metadata["teacher_target_text"]
    assert "<guess>CRATE</guess>" in tf_ex.metadata["teacher_target_text"]
    assert tf_ex.metadata["teacher_opener"] != "crate"
    assert len(tf_ex.prompt_ids) > tf_ex.metadata["teacher_target_token_count"]


def test_teacher_forced_score_uses_suffix_input_logprobs():
    result = {
        "meta_info": {
            "input_token_logprobs": [
                [-9.0, 1, ""],
                [-8.0, 2, ""],
                [-2.0, 3, ""],
                [-4.0, 4, ""],
            ]
        }
    }

    score = zorl_client._teacher_forced_score_from_result(result, target_token_count=2)

    assert score["teacher_forced_logprob"] == -3.0
    assert score["teacher_forced_prob"] == score["reward"]
    assert 0.0 < score["reward"] < 1.0
    assert score["teacher_forced_finite_rate"] == 1.0


def test_standalone_rank_update_strategy_marks_scores_pre_normalized():
    candidate_rewards = [
        {"candidate_id": "low", "reward_mean": 0.1, "num_rollouts": 1},
        {"candidate_id": "high", "reward_mean": 0.9, "num_rollouts": 1},
        {"candidate_id": "mid", "reward_mean": 0.4, "num_rollouts": 1},
    ]

    transformed, metadata = zorl_client.transform_rewards_for_update(candidate_rewards, strategy="rank")

    by_id = {item["candidate_id"]: item for item in transformed}
    assert metadata["update_strategy"] == "rank"
    assert {item["_zorl_score_normalization"] for item in transformed} == {"none"}
    assert by_id["high"]["reward_mean"] > by_id["mid"]["reward_mean"] > by_id["low"]["reward_mean"]
    assert sum(item["reward_mean"] for item in transformed) == pytest.approx(0.0)


def test_standalone_project_baseline_standardized_uses_parent_anchor():
    candidate_rewards = [
        {
            "candidate_id": "plus",
            "reward_mean": 0.8,
            "num_rollouts": 1,
            "project_metrics": {
                "a": {"reward": 0.8},
                "b": {"reward": 0.4},
            },
        },
        {
            "candidate_id": "minus",
            "reward_mean": 0.3,
            "num_rollouts": 1,
            "project_metrics": {
                "a": {"reward": 0.2},
                "b": {"reward": 0.4},
            },
        },
    ]
    parent = {
        "candidate_id": "__parent__",
        "reward_mean": 0.4,
        "num_rollouts": 1,
        "project_metrics": {
            "a": {"reward": 0.5},
            "b": {"reward": 0.4},
        },
    }

    transformed, metadata = zorl_client.transform_rewards_for_update(
        candidate_rewards,
        strategy="project_baseline_standardized",
        parent_baseline=parent,
    )

    by_id = {item["candidate_id"]: item for item in transformed}
    assert metadata["update_strategy"] == "project_baseline_standardized"
    assert metadata["parent_train_reward_mean"] == pytest.approx(0.45)
    assert {item["_zorl_score_normalization"] for item in transformed} == {"none"}
    assert by_id["plus"]["reward_mean"] > 0.0
    assert by_id["minus"]["reward_mean"] < 0.0


def test_parse_countdown_log_accepts_fractional_candidate_and_parent_fraction(tmp_path):
    log_path = tmp_path / "letter_count_test.log"
    log_path.write_text(
        "\n".join(
            [
                "      Generation 4/64: reward_mean=0.0142, update_norm=1183.67, "
                "pair_delta_mean=0.001, rollout_exact_rate=0.0142, "
                "best_candidate_exact=1.50/32, best_candidate_reward=0.0469",
                "      Generation 8/64: reward_mean=0.0293, update_norm=1183.39, "
                "pair_delta_mean=-0.021, rollout_exact_rate=0.0293, "
                "best_candidate_exact=2.50/32, best_candidate_reward=0.0781",
                "      Parent probe 8: exact=2/32, reward_mean=0.0312, "
                "tf_prob_mean=0.0000, rollout_exact_rate=0.0625",
            ]
        ),
        encoding="utf-8",
    )
    candidate = {
        "score_source": "countdown_log",
        "score_gates": {
            "baseline_exact": 1,
            "weak_exact": 2,
            "promote_exact": 3,
            "strong_exact": 5,
            "min_generations": 8,
        },
    }

    score = controller.score_result(log_path, candidate)

    assert score["verdict"] == "weak_zorl_signal"
    assert score["metrics"]["completed_generations"] == 8
    assert score["metrics"]["best_parent_exact"] == 2
    assert score["metrics"]["best_candidate_exact"] == 2.5
    assert score["metrics"]["best_exact"] == 2.5
    assert score["metrics"]["exact_gain"] == 1.5


def test_wait_eval_reasons_gate_progress_and_probe_rows():
    previous = {
        "verdict": "weak_zorl_signal",
        "metrics": {
            "completed_generations": 12,
            "planned_generations": 64,
            "parent_probe_rows": 2,
            "fatal_error": False,
        },
    }

    no_wake = {
        "verdict": "weak_zorl_signal",
        "metrics": {
            "completed_generations": 16,
            "planned_generations": 64,
            "parent_probe_rows": 2,
            "fatal_error": False,
        },
    }
    assert (
        controller.wait_eval_reasons(
            previous,
            no_wake,
            min_generation_delta=5,
            min_step_delta=5,
            min_record_delta=1,
            wake_on_probe=True,
        )
        == []
    )

    probe_wake = {
        "verdict": "weak_zorl_signal",
        "metrics": {
            "completed_generations": 16,
            "planned_generations": 64,
            "parent_probe_rows": 3,
            "fatal_error": False,
        },
    }
    assert controller.wait_eval_reasons(
        previous,
        probe_wake,
        min_generation_delta=5,
        min_step_delta=5,
        min_record_delta=1,
        wake_on_probe=True,
    ) == ["parent_probe_rows advanced 2->3"]

    generation_wake = {
        "verdict": "weak_zorl_signal",
        "metrics": {
            "completed_generations": 17,
            "planned_generations": 64,
            "parent_probe_rows": 2,
            "fatal_error": False,
        },
    }
    assert controller.wait_eval_reasons(
        previous,
        generation_wake,
        min_generation_delta=5,
        min_step_delta=5,
        min_record_delta=1,
        wake_on_probe=True,
    ) == ["completed_generations advanced 12->17"]

    failure_wake = {
        "verdict": "infra_invalid",
        "metrics": {
            "completed_generations": 12,
            "planned_generations": 64,
            "parent_probe_rows": 2,
            "fatal_error": True,
        },
    }
    assert controller.wait_eval_reasons(
        previous,
        failure_wake,
        min_generation_delta=5,
        min_step_delta=5,
        min_record_delta=1,
        wake_on_probe=True,
    ) == ["terminal_or_failed verdict=infra_invalid"]


def test_render_manifest_adds_team_turbo_to_gpu_template(tmp_path):
    candidate_path = controller.ROOT / "candidates" / "SGL-047.yaml"
    candidate = controller.load_candidate(candidate_path)
    rendered = controller.render_manifest(candidate_path, candidate, output=tmp_path / "rendered.yaml")
    docs = list(yaml.safe_load_all(rendered.read_text(encoding="utf-8")))
    templates = [controller.pod_template_for(doc) for doc in docs if doc]
    gpu_templates = [
        template
        for template in templates
        if template is not None and controller.pod_template_requests_gpu(template)
    ]

    assert gpu_templates
    for template in gpu_templates:
        labels = template["metadata"]["labels"]
        assert labels["team"] == "turbo"


def test_render_sharded_wordle_candidate_exposes_owner_routing_flags(tmp_path):
    candidate_path = controller.ROOT / "candidates" / "ZORL-WORDLE-011-opsd-tf-sharded-smoke.yaml"
    candidate = controller.load_candidate(candidate_path)
    rendered = controller.render_manifest(candidate_path, candidate, output=tmp_path / "rendered.yaml")
    docs = list(yaml.safe_load_all(rendered.read_text(encoding="utf-8")))
    job = next(doc for doc in docs if doc and doc.get("kind") == "Job")
    container = job["spec"]["template"]["spec"]["containers"][0]
    env = {entry["name"]: entry["value"] for entry in container["env"]}
    command_script = container["command"][-1]

    assert env["POPULATION_SHARDING"] == "pair_shard"
    assert env["CANDIDATE_ROUTING"] == "owner"
    assert "--population-sharding" in command_script
    assert "--candidate-routing" in command_script


def test_render_standalone_candidate_exposes_train_pool_resampling_flags(tmp_path):
    candidate_path = controller.ROOT / "candidates" / "ZORL-WORDLE-012-opsd-tf-sharded-pop256.yaml"
    candidate = controller.load_candidate(candidate_path)
    candidate["id"] = "ZORL-WORDLE-TEST-RESAMPLED"
    candidate["env"] = {
        **candidate.get("env", {}),
        "TRAIN_SIZE": "128",
        "TRAIN_POOL_SIZE": "512",
        "RESAMPLE_TRAIN_EACH_STEP": "1",
    }
    rendered = controller.render_manifest(candidate_path, candidate, output=tmp_path / "rendered.yaml")
    docs = list(yaml.safe_load_all(rendered.read_text(encoding="utf-8")))
    job = next(doc for doc in docs if doc and doc.get("kind") == "Job")
    container = job["spec"]["template"]["spec"]["containers"][0]
    env = {entry["name"]: entry["value"] for entry in container["env"]}
    command_script = container["command"][-1]

    assert env["TRAIN_POOL_SIZE"] == "512"
    assert env["RESAMPLE_TRAIN_EACH_STEP"] == "1"
    assert "--train-pool-size" in command_script
    assert "--resample-train-each-step" in command_script
    assert "--score-job-order" in command_script
    assert "--export-dir" in command_script
    assert "--export-interval" in command_script
    assert "resample_train_each_step=" in command_script
    assert "wandb_log_forwarder.py" in command_script


def test_select_train_examples_for_step_cycles_deterministically():
    pool = [Example(project=f"train_{idx}", prompt_ids=[], metadata={}) for idx in range(6)]

    fixed = zorl_client._select_train_examples_for_step(
        pool,
        train_size=3,
        seed=123,
        step=5,
        resample=False,
    )
    assert [example.project for example in fixed] == ["train_0", "train_1", "train_2"]

    step0 = zorl_client._select_train_examples_for_step(
        pool,
        train_size=3,
        seed=123,
        step=0,
        resample=True,
    )
    step0_repeat = zorl_client._select_train_examples_for_step(
        pool,
        train_size=3,
        seed=123,
        step=0,
        resample=True,
    )
    step1 = zorl_client._select_train_examples_for_step(
        pool,
        train_size=3,
        seed=123,
        step=1,
        resample=True,
    )

    assert [example.project for example in step0_repeat] == [example.project for example in step0]
    assert len({example.project for example in step0}) == 3
    assert {example.project for example in step0}.isdisjoint({example.project for example in step1})
    assert {example.project for example in step0} | {example.project for example in step1} == {
        example.project for example in pool
    }


def test_wandb_log_forwarder_parses_cold_step_and_batch(tmp_path):
    log_path = tmp_path / "zorl_client.log"
    log_path.write_text(
        "\n".join(
            [
                "run_id=run-1",
                "recipe: rank=4 steps=256 pairs=512 sigma=0.05",
                "data: train_size=128 train_pool_size=512 resample_train_each_step=1 eval_size=128",
                "  cold: reward_mean=0.0096 exact_rate=0.0000 exact_count=0.0/128",
                "  train_batch step=16: size=128 pool=512 preview=wordle_train_0097,wordle_train_0188",
                (
                    "  step 16/256: reward_mean=0.0141 best_cand=0.0199 update_norm=277.67 "
                    "pair_delta_mean=0.0001 pair_delta_std=0.0031 unclipped_update_norm=277.67 "
                    "grad_norm=277.67 update_clip_scale=1.0000 zero_score_pairs=0 dropped_pairs=0 "
                    "score_normalization=standard update_strategy=raw used_pairs=512 "
                    "t_score=534.9s t_apply=110.3s, probe_reward=0.0140 exact_rate=0.0000 "
                    "[teacher_forced_finite_rate=1.000 teacher_forced_logprob=-4.280 "
                    "teacher_forced_prob=0.014 teacher_forced_token_count=60.789]"
                ),
            ]
        ),
        encoding="utf-8",
    )
    state = {"logged_cold": False, "logged_steps": []}

    cold, step_events, done = wandb_log_forwarder._parse_log(log_path, state)
    config = wandb_log_forwarder._run_config(log_path.read_text(encoding="utf-8").splitlines())

    assert done is False
    assert config["recipe/pairs"] == 512
    assert config["data/train_pool_size"] == 512
    assert cold["eval/cold_reward_mean"] == 0.0096
    assert len(step_events) == 1
    step, metrics = step_events[0]
    assert step == 16
    assert metrics["progress/planned_steps"] == 256
    assert metrics["data/train_batch_size"] == 128
    assert metrics["data/train_pool_size"] == 512
    assert metrics["data/train_batch_preview"] == "wordle_train_0097,wordle_train_0188"
    assert metrics["train/reward_mean"] == 0.0141
    assert metrics["train/best_candidate_reward"] == 0.0199
    assert metrics["update/update_norm"] == 277.67
    assert metrics["population/dropped_pairs"] == 0
    assert metrics["population/used_pairs"] == 512
    assert metrics["timing/score_sec"] == 534.9
    assert metrics["timing/apply_sec"] == 110.3
    assert metrics["timing/step_sec"] == pytest.approx(645.2)
    assert metrics["eval/probe_reward"] == 0.014
    assert metrics["eval/teacher_forced_logprob"] == -4.28


def test_combine_sharded_generation_results_adds_owner_urls_and_validates_pairs():
    results_by_url = [
        (
            "http://worker-0:30000",
            {
                "generation_id": "gen-1",
                "num_pairs": 2,
                "candidates": [
                    {
                        "candidate_id": "gen-1-p0000+",
                        "perturbation_index": 0,
                        "direction": "positive",
                        "lora_name": "zorl/gen-1/gen-1-p0000+",
                    },
                    {
                        "candidate_id": "gen-1-p0000-",
                        "perturbation_index": 0,
                        "direction": "negative",
                        "lora_name": "zorl/gen-1/gen-1-p0000-",
                    },
                ],
            },
        ),
        (
            "http://worker-1:30000",
            {
                "generation_id": "gen-1",
                "num_pairs": 2,
                "candidates": [
                    {
                        "candidate_id": "gen-1-p0001+",
                        "perturbation_index": 1,
                        "direction": "positive",
                        "lora_name": "zorl/gen-1/gen-1-p0001+",
                    },
                    {
                        "candidate_id": "gen-1-p0001-",
                        "perturbation_index": 1,
                        "direction": "negative",
                        "lora_name": "zorl/gen-1/gen-1-p0001-",
                    },
                ],
            },
        ),
    ]

    combined = zorl_client._combine_sharded_generation_results(results_by_url, expected_num_pairs=2)

    assert combined["generation_id"] == "gen-1"
    assert combined["global_population"] == 4
    assert [candidate["candidate_id"] for candidate in combined["candidates"]] == [
        "gen-1-p0000+",
        "gen-1-p0000-",
        "gen-1-p0001+",
        "gen-1-p0001-",
    ]
    assert {candidate["owner_url"] for candidate in combined["candidates"][:2]} == {"http://worker-0:30000"}
    assert {candidate["owner_url"] for candidate in combined["candidates"][2:]} == {"http://worker-1:30000"}


def test_combine_sharded_generation_results_rejects_duplicate_candidate_ownership():
    duplicated = [
        (
            "http://worker-0:30000",
            {
                "generation_id": "gen-1",
                "candidates": [
                    {"candidate_id": "dup", "perturbation_index": 0, "direction": "positive"},
                    {"candidate_id": "gen-1-p0000-", "perturbation_index": 0, "direction": "negative"},
                ],
            },
        ),
        (
            "http://worker-1:30000",
            {
                "generation_id": "gen-1",
                "candidates": [
                    {"candidate_id": "dup", "perturbation_index": 1, "direction": "positive"},
                    {"candidate_id": "gen-1-p0001-", "perturbation_index": 1, "direction": "negative"},
                ],
            },
        ),
    ]

    with pytest.raises(RuntimeError, match="Duplicate ZORL candidate"):
        zorl_client._combine_sharded_generation_results(duplicated, expected_num_pairs=2)


def test_score_candidates_routes_to_candidate_owner_url(monkeypatch):
    calls = []

    def fake_input_logprobs(url, *, input_ids, lora_path, headers=None):
        calls.append((url, tuple(input_ids), lora_path, headers))
        return {"meta_info": {"input_token_logprobs": [{"logprob": -0.5}]}}

    class FakeTask:
        def build_teacher_forced_example(self, tokenizer, ex, args):
            return SimpleNamespace(prompt_ids=[1, 2, 3], metadata={"teacher_target_token_count": 1})

    monkeypatch.setattr(zorl_client, "input_logprobs_with_lora", fake_input_logprobs)

    rewards, _outputs = zorl_client.score_candidates(
        "http://smg:30000",
        candidates=[
            {"candidate_id": "cand-0", "lora_name": "zorl/cand-0", "owner_url": "http://worker-0:30000"},
            {"candidate_id": "cand-1", "lora_name": "zorl/cand-1", "owner_url": "http://worker-1:30000"},
        ],
        examples=[Example(project="p0", prompt_ids=[], metadata={})],
        task=FakeTask(),
        args=SimpleNamespace(
            task="fake",
            score_mode="teacher_forced",
            score_max_workers=2,
            rollouts_per_puzzle=1,
            candidate_routing="owner",
        ),
        tokenizer=None,
    )

    assert {call[0] for call in calls} == {"http://worker-0:30000", "http://worker-1:30000"}
    assert {call[2] for call in calls} == {"zorl/cand-0", "zorl/cand-1"}
    assert {call[3] for call in calls} == {None}
    assert {reward["candidate_id"] for reward in rewards} == {"cand-0", "cand-1"}


def test_score_candidates_routes_owner_via_smg_with_target_worker_header(monkeypatch):
    calls = []

    def fake_input_logprobs(url, *, input_ids, lora_path, headers=None):
        calls.append((url, tuple(input_ids), lora_path, headers))
        return {"meta_info": {"input_token_logprobs": [{"logprob": -0.5}]}}

    class FakeTask:
        def build_teacher_forced_example(self, tokenizer, ex, args):
            return SimpleNamespace(prompt_ids=[1, 2, 3], metadata={"teacher_target_token_count": 1})

    monkeypatch.setattr(zorl_client, "input_logprobs_with_lora", fake_input_logprobs)

    rewards, _outputs = zorl_client.score_candidates(
        ["http://smg-a:30000", "http://smg-b:30000"],
        candidates=[
            {"candidate_id": "cand-0", "lora_name": "zorl/cand-0", "owner_url": "http://worker-0:30000"},
            {"candidate_id": "cand-1", "lora_name": "zorl/cand-1", "owner_url": "http://worker-1:30000"},
        ],
        examples=[Example(project="p0", prompt_ids=[], metadata={})],
        task=FakeTask(),
        args=SimpleNamespace(
            task="fake",
            score_mode="teacher_forced",
            score_max_workers=2,
            rollouts_per_puzzle=1,
            candidate_routing="owner_via_smg",
            score_job_order="owner_round_robin",
        ),
        tokenizer=None,
    )

    by_lora = {call[2]: call for call in calls}
    assert by_lora["zorl/cand-0"][0] == "http://smg-a:30000"
    assert by_lora["zorl/cand-0"][3] == {"X-SMG-Target-Worker": "http://worker-0:30000"}
    assert by_lora["zorl/cand-1"][0] == "http://smg-b:30000"
    assert by_lora["zorl/cand-1"][3] == {"X-SMG-Target-Worker": "http://worker-1:30000"}
    assert {reward["candidate_id"] for reward in rewards} == {"cand-0", "cand-1"}


def test_score_candidates_batches_teacher_forced_owner_via_smg(monkeypatch):
    calls = []

    def fake_input_logprobs_batch(url, *, input_ids_batch, lora_path, headers=None):
        calls.append((url, tuple(tuple(input_ids) for input_ids in input_ids_batch), lora_path, headers))
        return [
            {"meta_info": {"input_token_logprobs": [{"logprob": -0.5}]}}
            for _input_ids in input_ids_batch
        ]

    class FakeTask:
        def build_teacher_forced_example(self, tokenizer, ex, args):
            return SimpleNamespace(
                prompt_ids=[ex.metadata["idx"], 99],
                metadata={"teacher_target_token_count": 1},
            )

    monkeypatch.setattr(zorl_client, "input_logprobs_batch_with_lora", fake_input_logprobs_batch)

    examples = [Example(project=f"p{idx}", prompt_ids=[], metadata={"idx": idx}) for idx in range(4)]
    rewards, _outputs = zorl_client.score_candidates(
        ["http://smg-a:30000", "http://smg-b:30000"],
        candidates=[
            {"candidate_id": "cand-0", "lora_name": "zorl/cand-0", "owner_url": "http://worker-0:30000"},
            {"candidate_id": "cand-1", "lora_name": "zorl/cand-1", "owner_url": "http://worker-1:30000"},
        ],
        examples=examples,
        task=FakeTask(),
        args=SimpleNamespace(
            task="fake",
            score_mode="teacher_forced",
            score_max_workers=1,
            rollouts_per_puzzle=1,
            candidate_routing="owner_via_smg",
            score_job_order="owner_round_robin",
            teacher_forced_batch_size=2,
        ),
        tokenizer=None,
    )

    assert calls == [
        (
            "http://smg-a:30000",
            ((0, 99), (1, 99)),
            "zorl/cand-0",
            {"X-SMG-Target-Worker": "http://worker-0:30000"},
        ),
        (
            "http://smg-b:30000",
            ((0, 99), (1, 99)),
            "zorl/cand-1",
            {"X-SMG-Target-Worker": "http://worker-1:30000"},
        ),
        (
            "http://smg-a:30000",
            ((2, 99), (3, 99)),
            "zorl/cand-0",
            {"X-SMG-Target-Worker": "http://worker-0:30000"},
        ),
        (
            "http://smg-b:30000",
            ((2, 99), (3, 99)),
            "zorl/cand-1",
            {"X-SMG-Target-Worker": "http://worker-1:30000"},
        ),
    ]
    assert {reward["candidate_id"] for reward in rewards} == {"cand-0", "cand-1"}
    assert {len(reward["project_metrics"]) for reward in rewards} == {4}


def test_ordered_score_pairs_round_robins_owner_routed_candidates():
    candidates = []
    for pair_idx in range(8):
        owner = f"http://worker-{pair_idx % 4}:30000"
        for direction in ("positive", "negative"):
            candidates.append(
                {
                    "candidate_id": f"p{pair_idx}-{direction}",
                    "lora_name": f"zorl/p{pair_idx}-{direction}",
                    "owner_url": owner,
                    "perturbation_index": pair_idx,
                    "direction": direction,
                }
            )
    examples = [Example(project=f"ex-{idx}", prompt_ids=[], metadata={}) for idx in range(2)]
    args = SimpleNamespace(candidate_routing="owner", score_job_order="owner_round_robin")

    pairs = zorl_client._ordered_score_pairs(candidates, examples, args=args)

    assert len(pairs) == len(candidates) * len(examples)
    assert [cand["owner_url"] for cand, _ex in pairs[:8]] == [
        "http://worker-0:30000",
        "http://worker-1:30000",
        "http://worker-2:30000",
        "http://worker-3:30000",
        "http://worker-0:30000",
        "http://worker-1:30000",
        "http://worker-2:30000",
        "http://worker-3:30000",
    ]
    assert {ex.project for _cand, ex in pairs[: len(candidates)]} == {"ex-0"}


def test_ordered_score_batches_round_robins_owner_routed_candidates():
    candidates = []
    for pair_idx in range(8):
        owner = f"http://worker-{pair_idx % 4}:30000"
        for direction in ("positive", "negative"):
            candidates.append(
                {
                    "candidate_id": f"p{pair_idx}-{direction}",
                    "lora_name": f"zorl/p{pair_idx}-{direction}",
                    "owner_url": owner,
                    "perturbation_index": pair_idx,
                    "direction": direction,
                }
            )
    examples = [Example(project=f"ex-{idx}", prompt_ids=[], metadata={}) for idx in range(16)]
    args = SimpleNamespace(candidate_routing="owner", score_job_order="owner_round_robin")

    batches = zorl_client._ordered_score_batches(candidates, examples, batch_size=8, args=args)

    assert len(batches) == len(candidates) * 2
    assert [cand["owner_url"] for cand, _ex_batch in batches[:8]] == [
        "http://worker-0:30000",
        "http://worker-1:30000",
        "http://worker-2:30000",
        "http://worker-3:30000",
        "http://worker-0:30000",
        "http://worker-1:30000",
        "http://worker-2:30000",
        "http://worker-3:30000",
    ]
    assert {tuple(ex.project for ex in ex_batch) for _cand, ex_batch in batches[:8]} == {
        tuple(f"ex-{idx}" for idx in range(8))
    }


def test_start_zorl_generation_pair_shard_passes_preload_candidates(monkeypatch):
    calls = []

    def fake_post(url, path, payload, timeout):
        calls.append((url, path, dict(payload)))
        pair_idx = int(payload["materialization"]["shard_index"])
        candidates = [
            {
                "candidate_id": f"pair-{pair_idx}-pos",
                "perturbation_index": pair_idx,
                "direction": "positive",
                "lora_name": f"zorl/pair-{pair_idx}-pos",
            },
            {
                "candidate_id": f"pair-{pair_idx}-neg",
                "perturbation_index": pair_idx,
                "direction": "negative",
                "lora_name": f"zorl/pair-{pair_idx}-neg",
            },
        ]
        return {
            "success": True,
            "generation_id": "gen-1",
            "global_num_pairs": 2,
            "global_population": 4,
            "local_num_pairs": 1,
            "candidates": candidates,
        }

    monkeypatch.setattr(zorl_client, "_post", fake_post)

    result = zorl_client.start_zorl_generation_all(
        ["http://worker-0:30000", "http://worker-1:30000"],
        session_id="session-1",
        preload_candidates=True,
        num_pairs=2,
        population_sharding="pair_shard",
        num_shards=2,
    )

    assert len(calls) == 2
    assert {call[2]["preload_candidates"] for call in calls} == {True}
    assert {call[2]["owner_url"] for call in calls} == {
        "http://worker-0:30000",
        "http://worker-1:30000",
    }
    assert len(result["candidates"]) == 4


def test_ordered_score_pairs_can_preserve_candidate_major_order():
    candidates = [
        {"candidate_id": "cand-0", "owner_url": "http://worker-0:30000"},
        {"candidate_id": "cand-1", "owner_url": "http://worker-1:30000"},
    ]
    examples = [Example(project=f"ex-{idx}", prompt_ids=[], metadata={}) for idx in range(2)]
    args = SimpleNamespace(candidate_routing="owner", score_job_order="candidate_major")

    pairs = zorl_client._ordered_score_pairs(candidates, examples, args=args)

    assert [(cand["candidate_id"], ex.project) for cand, ex in pairs] == [
        ("cand-0", "ex-0"),
        ("cand-0", "ex-1"),
        ("cand-1", "ex-0"),
        ("cand-1", "ex-1"),
    ]


def test_validate_apply_results_rejects_metric_disagreement():
    with pytest.raises(RuntimeError, match="update_norm"):
        zorl_client._validate_apply_results_agree(
            ["http://worker-0:30000", "http://worker-1:30000"],
            [
                {"used_pairs": 2, "dropped_pairs": 0, "metrics": {"update_norm": 1.0}},
                {"used_pairs": 2, "dropped_pairs": 0, "metrics": {"update_norm": 1.1}},
            ],
        )


def test_opd_multiplication_scores_last_integer_only():
    ex = Example(
        project="opd_test",
        prompt_ids=[],
        metadata={"a": 2824, "b": 1409, "product": 2824 * 1409},
    )

    assert opd_multiplication.score_completion(ex, "3,979,016")["reward"] == 1.0
    assert opd_multiplication.score_completion(ex, "3979016 then 7777777")["reward"] == 0.0
    assert opd_multiplication.score_completion(ex, "") == {
        "reward": 0.0,
        "exact_match": 0.0,
        "emitted_number": 0.0,
    }
