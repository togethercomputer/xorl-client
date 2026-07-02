"""Tests for Countdown ZORL update strategy helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[2] / "experiments" / "zorl" / "run_countdown_test.py"
_SPEC = importlib.util.spec_from_file_location("zorl_countdown_test", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


pytestmark = [pytest.mark.cpu]


def test_best_candidate_update_encodes_unit_candidate_delta():
    args = SimpleNamespace(
        zorl_update_strategy="best_candidate",
        zorl_b_sigma=0.1,
        zorl_best_candidate_step_scale=1.0,
    )
    generation = {"num_pairs": 1, "candidates": []}
    candidate_rewards = [
        {"candidate_id": "weak", "reward_mean": 0.25, "num_rollouts": 1},
        {"candidate_id": "best", "reward_mean": 0.5, "num_rollouts": 1},
    ]

    rewards_for_update, apply_lr, metadata = _MODULE._select_zorl_update_payload(
        generation,
        candidate_rewards,
        args,
        lr=0.01,
    )

    assert apply_lr == pytest.approx(0.1)
    assert metadata["score_normalization"] == "none"
    assert metadata["selected_candidate_id"] == "best"
    assert rewards_for_update == [
        {
            "candidate_id": "best",
            "reward_mean": 1.0,
            "num_rollouts": 1,
            "_zorl_score_normalization": "none",
        }
    ]


def test_candidate_sort_key_preserves_fractional_multi_rollout_exact_count():
    low = {"candidate_id": "low", "reward_mean": 1.0, "rollout_exact_count": 2.25}
    high = {"candidate_id": "high", "reward_mean": 0.5, "rollout_exact_count": 2.75}

    assert _MODULE._candidate_sort_key(high) > _MODULE._candidate_sort_key(low)


def test_rollout_value_score_requires_valid_expression_and_tracks_target_distance():
    assert _MODULE._rollout_value_score(eval_value=18.0, target=24, uses_each_once=True) == pytest.approx(0.75)
    assert _MODULE._rollout_value_score(eval_value=24.0, target=24, uses_each_once=False) == 0.0


def test_rollout_reward_mode_can_add_valid_and_value_shaping():
    args = SimpleNamespace(
        zorl_reward_mode="rollout",
        zorl_rollout_prefix_weight=0.0,
        zorl_rollout_char_weight=0.0,
        zorl_rollout_valid_weight=0.2,
        zorl_rollout_value_weight=0.4,
        zorl_rollout_exact_bonus=1.0,
    )

    reward = _MODULE._reward_for_mode(
        args,
        teacher_forced_logprob=-100.0,
        teacher_forced_suffix_logprob=-100.0,
        teacher_forced_first_error_logprob=-100.0,
        rollout_components={
            "prefix_ratio": 0.0,
            "char_match_ratio": 0.0,
            "uses_each_once_rate": 0.5,
            "value_score": 0.25,
            "exact_match": 0.0,
        },
    )

    assert reward == pytest.approx(0.2)


def test_hard_project_reward_multiplier_scales_dense_components_not_exact_bonus():
    args = SimpleNamespace(
        zorl_reward_mode="rollout",
        zorl_rollout_prefix_weight=0.1,
        zorl_rollout_char_weight=0.05,
        zorl_rollout_valid_weight=0.2,
        zorl_rollout_value_weight=0.4,
        zorl_rollout_exact_bonus=1.0,
    )
    rollout_components = {
        "prefix_ratio": 0.5,
        "char_match_ratio": 0.4,
        "uses_each_once_rate": 1.0,
        "value_score": 0.25,
        "exact_match": 1.0,
    }

    base = _MODULE._reward_for_mode(
        args,
        teacher_forced_logprob=-100.0,
        teacher_forced_suffix_logprob=-100.0,
        teacher_forced_first_error_logprob=-100.0,
        rollout_components=rollout_components,
        hard_project_reward_multiplier=1.0,
    )
    boosted = _MODULE._reward_for_mode(
        args,
        teacher_forced_logprob=-100.0,
        teacher_forced_suffix_logprob=-100.0,
        teacher_forced_first_error_logprob=-100.0,
        rollout_components=rollout_components,
        hard_project_reward_multiplier=4.0,
    )

    # base dense = 0.1*0.5 + 0.05*0.4 + 0.2*1.0 + 0.4*0.25 = 0.05 + 0.02 + 0.2 + 0.1 = 0.37
    # exact bonus = 1.0; so base total = 1.37, boosted total = 4*0.37 + 1.0 = 2.48.
    assert base == pytest.approx(0.37 + 1.0)
    assert boosted == pytest.approx(4 * 0.37 + 1.0)
    # The exact bonus must NOT be scaled — confirm by setting prefix/char/valid/value to 0.
    dense_zero = {**rollout_components, "prefix_ratio": 0.0, "char_match_ratio": 0.0, "uses_each_once_rate": 0.0, "value_score": 0.0}
    bonus_only = _MODULE._reward_for_mode(
        args,
        teacher_forced_logprob=-100.0,
        teacher_forced_suffix_logprob=-100.0,
        teacher_forced_first_error_logprob=-100.0,
        rollout_components=dense_zero,
        hard_project_reward_multiplier=10.0,
    )
    assert bonus_only == pytest.approx(1.0)


def test_resolve_hard_project_reward_set_gates_on_active_threshold():
    args = SimpleNamespace(
        zorl_hard_project_reward_multiplier=3.0,
        zorl_hard_project_reward_active_threshold=4,
    )
    # Below threshold → hard set populated.
    pbm_below = {
        "parent_advantage_active_project_names": "q_00,q_04,q_05",
        "parent_advantage_active_projects": 3,
    }
    hard_set, mult, meta = _MODULE._resolve_zorl_hard_project_reward_set(args, pbm_below)
    assert hard_set == {"q_00", "q_04", "q_05"}
    assert mult == pytest.approx(3.0)
    assert meta["hard_project_reward_reason"] == "active_projects_at_or_below_threshold"

    # Above threshold → no boost.
    pbm_above = {
        "parent_advantage_active_project_names": "q_00,q_01,q_02,q_04,q_05",
        "parent_advantage_active_projects": 5,
    }
    hard_set_off, mult_off, meta_off = _MODULE._resolve_zorl_hard_project_reward_set(args, pbm_above)
    assert hard_set_off == set()
    assert mult_off == pytest.approx(1.0)
    assert meta_off["hard_project_reward_reason"] == "active_projects_above_threshold"

    # Multiplier <= 1.0 → disabled regardless of active count.
    args_off = SimpleNamespace(
        zorl_hard_project_reward_multiplier=1.0,
        zorl_hard_project_reward_active_threshold=999,
    )
    hard_set_d, mult_d, meta_d = _MODULE._resolve_zorl_hard_project_reward_set(args_off, pbm_below)
    assert hard_set_d == set()
    assert mult_d == pytest.approx(1.0)
    assert meta_d["hard_project_reward_reason"] == "disabled"


def test_rotated_puzzle_subset_is_deterministic_and_distinct_across_generations():
    args = SimpleNamespace(zorl_active_puzzle_count=8, zorl_puzzle_rotation_seed=7)
    reward_data = [{"project": f"q_{i:02d}"} for i in range(32)]

    subset_a, meta_a = _MODULE._select_rotated_puzzle_subset(reward_data, args, generation_index=0)
    subset_a_again, _ = _MODULE._select_rotated_puzzle_subset(reward_data, args, generation_index=0)
    subset_b, meta_b = _MODULE._select_rotated_puzzle_subset(reward_data, args, generation_index=1)

    # Determinism for a given (seed, gen).
    assert [ex["project"] for ex in subset_a] == [ex["project"] for ex in subset_a_again]
    # Different generations give different subsets (extremely unlikely to collide at 8 of 32).
    assert [ex["project"] for ex in subset_a] != [ex["project"] for ex in subset_b]
    assert len(subset_a) == 8
    assert meta_a["puzzle_active_count"] == 8
    assert meta_a["puzzle_pool_size"] == 32
    assert meta_b["puzzle_rotation_reason"] == "rotated"


def test_rotated_puzzle_subset_disabled_returns_full_pool():
    args = SimpleNamespace(zorl_active_puzzle_count=0, zorl_puzzle_rotation_seed=0)
    reward_data = [{"project": f"q_{i:02d}"} for i in range(5)]
    subset, meta = _MODULE._select_rotated_puzzle_subset(reward_data, args, generation_index=3)
    assert [ex["project"] for ex in subset] == [ex["project"] for ex in reward_data]
    assert meta["puzzle_rotation_reason"] == "disabled"
    assert meta["puzzle_active_count"] == 5


def test_extract_expression_strips_think_block_and_picks_final_answer():
    """When --zorl-enable-thinking is on, Qwen3 emits <think>reasoning</think>final.
    The extractor must strip the think block; otherwise digits inside the
    reasoning (e.g. ``2 + 3 = 5``) would win the longest-substring search."""
    text = (
        "<think>\nLet me try (11 - 5) * (7 - 3) = 6 * 4 = 24, that works.\n"
        "But wait, what about 2 + 22? No, only 4 numbers.\n</think>\n"
        "(11 - 5) * (7 - 3)"
    )
    extracted = _MODULE._extract_expression(text)
    # Whichever the extractor returns, it must be a valid arithmetic expression
    # that evaluates to 24 — i.e. it found the FINAL answer, not an intermediate
    # one from inside the think block.
    value = _MODULE._safe_eval_expr(extracted)
    assert value == pytest.approx(24.0)


def test_extract_expression_handles_orphan_close_think_tag():
    """A truncated think block (no open tag, just a stray </think> partway through)
    should be handled gracefully: everything before the close tag is reasoning,
    everything after is the answer."""
    text = "Let me think 1 + 1 = 2.</think>(11 - 5) * (7 - 3)"
    extracted = _MODULE._extract_expression(text)
    value = _MODULE._safe_eval_expr(extracted)
    assert value == pytest.approx(24.0)


def test_extract_expression_no_think_block_unchanged():
    """Plain (non-thinking) answers should still extract correctly."""
    text = "(2 + 4) * (3 + 1)"
    extracted = _MODULE._extract_expression(text)
    value = _MODULE._safe_eval_expr(extracted)
    assert value == pytest.approx(24.0)


def test_filter_parent_advantage_metadata_lists_active_and_skipped_projects():
    """The filter helper must always populate active/skipped project names so the
    hard-reward shaping path can read them even when the filter itself is disabled
    via --zorl-parent-advantage-keep-solved-in-gradient."""
    reward_data = [{"project": f"q_{i:02d}"} for i in range(4)]
    parent_baseline = {
        "project_metrics": {
            "q_00": {"rollout_exact_rate": 1.0},
            "q_01": {"rollout_exact_rate": 0.5},
            "q_02": {"rollout_exact_rate": 1.0},
            "q_03": {"rollout_exact_rate": 0.0},
        }
    }
    active_data, metadata = _MODULE._filter_parent_advantage_reward_data(
        reward_data, parent_baseline, solved_exact_rate=1.0
    )
    # Project names regardless of whether caller drops them.
    assert metadata["parent_advantage_active_projects"] == 2
    assert metadata["parent_advantage_skipped_projects"] == 2
    assert metadata["parent_advantage_active_project_names"] == "q_01,q_03"
    assert metadata["parent_advantage_skipped_project_names"] == "q_00,q_02"
    # Returned data drops the solved projects (caller decides whether to actually use this).
    assert [ex["project"] for ex in active_data] == ["q_01", "q_03"]


def test_default_zorl_native_session_id_is_unique_and_path_safe(monkeypatch):
    values = iter([111, 222])
    monkeypatch.setattr(_MODULE.time, "time_ns", lambda: next(values))

    first = _MODULE._default_zorl_native_session_id("train/model id")
    second = _MODULE._default_zorl_native_session_id("train/model id")

    assert first == "train-model-id-111"
    assert second == "train-model-id-222"


def test_post_sglang_zorl_all_reports_http_error_body(monkeypatch):
    class _Response:
        status_code = 400
        text = '{"detail":"already active"}'
        reason = "Bad Request"

        def json(self):
            raise AssertionError("json should not be parsed after HTTP error")

    monkeypatch.setattr(_MODULE.requests, "post", lambda *_args, **_kwargs: _Response())

    with pytest.raises(RuntimeError, match='HTTP 400: \\{"detail":"already active"\\}'):
        _MODULE._post_sglang_zorl_all("http://sglang:30060", "/start_zorl_session", {})


def test_start_sglang_zorl_sessions_sends_perturbation_mode(monkeypatch):
    calls = []

    def fake_post_all(infer_urls, path, payload):
        calls.append((infer_urls, path, payload))
        return [{"success": True}]

    monkeypatch.setattr(_MODULE, "_post_sglang_zorl_all", fake_post_all)
    args = SimpleNamespace(
        zorl_b_sigma=0.2,
        zorl_num_pairs=4,
        zorl_seed=123,
        zorl_perturbation_mode="a_and_b",
    )

    _MODULE.start_sglang_zorl_sessions(
        ["http://sglang-a", "http://sglang-b"],
        session_id="session",
        parent_lora_name="parent",
        args=args,
    )

    assert calls == [
        (
            ["http://sglang-a", "http://sglang-b"],
            "/start_zorl_session",
            {
                "session_id": "session",
                "parent_lora_name": "parent",
                "b_sigma": 0.2,
                "num_pairs": 4,
                "seed": 123,
                "antithetic_sampling": True,
                "perturbation_mode": "a_and_b",
            },
        )
    ]


def test_apply_sglang_zorl_rewards_can_send_max_update_norm(monkeypatch):
    calls = []

    def fake_post_all(infer_urls, path, payload):
        calls.append((infer_urls, path, payload))
        return [{"success": True, "metrics": {"update_clip_scale": 0.5}}]

    monkeypatch.setattr(_MODULE, "_post_sglang_zorl_all", fake_post_all)
    monkeypatch.setattr(_MODULE, "flush_inference_cache", lambda _url: None)

    result = _MODULE.apply_sglang_zorl_rewards(
        ["http://sglang-a", "http://sglang-b"],
        session_id="session",
        generation_id="generation",
        candidate_rewards=[{"candidate_id": "c", "reward_mean": 1.0}],
        lr=0.1,
        max_update_norm=2500.0,
    )

    assert result["metrics"]["update_clip_scale"] == pytest.approx(0.5)
    assert calls == [
        (
            ["http://sglang-a", "http://sglang-b"],
            "/apply_zorl_rewards",
            {
                "session_id": "session",
                "generation_id": "generation",
                "candidate_rewards": [{"candidate_id": "c", "reward_mean": 1.0}],
                "learning_rate": 0.1,
                "max_update_norm": 2500.0,
            },
        )
    ]


def test_start_sglang_zorl_generation_can_override_num_pairs(monkeypatch):
    calls = []

    def fake_post_all(infer_urls, path, payload):
        calls.append((infer_urls, path, payload))
        return [
            {
                "success": True,
                "generation_id": "generation",
                "candidates": [{"candidate_id": "c0"}],
            }
        ]

    monkeypatch.setattr(_MODULE, "_post_sglang_zorl_all", fake_post_all)

    result = _MODULE.start_sglang_zorl_generation(
        ["http://sglang-a", "http://sglang-b"],
        session_id="session",
        preload_candidates=False,
        num_pairs=64,
    )

    assert result["generation_id"] == "generation"
    assert calls == [
        (
            ["http://sglang-a", "http://sglang-b"],
            "/start_zorl_generation",
            {
                "session_id": "session",
                "preload_candidates": False,
                "num_pairs": 64,
            },
        )
    ]


def test_generate_rollouts_can_send_stop_strings(monkeypatch):
    calls = []

    class _Response:
        def json(self):
            return [{"text": "(1 + 2)"}]

    def fake_post(infer_url, endpoint, payload):
        calls.append((infer_url, endpoint, payload))
        return _Response()

    monkeypatch.setattr(_MODULE, "_post_inference_with_retry", fake_post)

    result = _MODULE.generate_rollouts(
        "http://sglang",
        [[1, 2, 3]],
        lora_paths=["lora-a"],
        max_new_tokens=16,
        temperature=0.6,
        stop=["\n", "="],
    )

    assert result == [{"text": "(1 + 2)"}]
    assert calls == [
        (
            "http://sglang",
            "/generate",
            {
                "input_ids": [[1, 2, 3]],
                "sampling_params": {
                    "temperature": 0.6,
                    "max_new_tokens": 16,
                    "stop": ["\n", "="],
                },
                "return_logprob": False,
                "lora_path": ["lora-a"],
            },
        )
    ]


def test_rescore_top_zorl_candidates_scores_only_initial_top_k(monkeypatch):
    candidates = [
        {"candidate_id": "a", "lora_name": "a"},
        {"candidate_id": "b", "lora_name": "b"},
        {"candidate_id": "c", "lora_name": "c"},
    ]
    initial_rewards = [
        {"candidate_id": "a", "reward_mean": 0.5, "rollout_exact_count": 2.0},
        {"candidate_id": "b", "reward_mean": 0.4, "rollout_exact_count": 2.0},
        {"candidate_id": "c", "reward_mean": 1.0, "rollout_exact_count": 1.0},
    ]
    args = SimpleNamespace(
        zorl_update_strategy="best_candidate",
        zorl_rescore_top_k=2,
        zorl_rescore_rollouts=4,
        zorl_rollouts_per_puzzle=1,
        zorl_reward_mode="rollout",
    )
    calls = []

    def fake_score_zorl_candidates(infer_url, rescored_candidates, reward_data, score_args):
        calls.append(
            (
                infer_url,
                [candidate["candidate_id"] for candidate in rescored_candidates],
                reward_data,
                score_args.zorl_rollouts_per_puzzle,
            )
        )
        return [
            {"candidate_id": "a", "reward_mean": 0.25, "rollout_exact_count": 1.0},
            {"candidate_id": "b", "reward_mean": 0.75, "rollout_exact_count": 2.5},
        ], {"reward_mean": 0.5, "reward_min": 0.25, "reward_max": 0.75}

    monkeypatch.setattr(_MODULE, "score_zorl_candidates", fake_score_zorl_candidates)

    rescored_rewards, metadata = _MODULE._rescore_top_zorl_candidates(
        "http://sglang",
        candidates,
        initial_rewards,
        [{"project": "p0"}, {"project": "p1"}],
        args,
    )

    assert calls == [("http://sglang", ["a", "b"], [{"project": "p0"}, {"project": "p1"}], 4)]
    assert [reward["candidate_id"] for reward in rescored_rewards] == ["a", "b"]
    assert metadata["rescore_top_k"] == 2
    assert metadata["rescore_candidate_count"] == 2
    assert metadata["rescore_rollouts_per_puzzle"] == 4
    assert metadata["rescore_score_rollouts"] == 16
    assert metadata["rescore_initial_best_candidate_id"] == "a"
    assert metadata["rescore_best_candidate_id"] == "b"
    assert metadata["rescore_best_exact_count"] == pytest.approx(2.5)


def test_project_metric_sort_key_prefers_fractional_exact_rate():
    low = {"reward": 1.0, "rollout_exact_rate": 0.25, "rollout_prefix_ratio": 1.0}
    high = {"reward": 0.1, "rollout_exact_rate": 0.5, "rollout_prefix_ratio": 0.0}

    assert _MODULE._project_metric_sort_key(high) > _MODULE._project_metric_sort_key(low)


def test_rescore_top_zorl_candidates_scores_project_winner_union(monkeypatch):
    candidates = [
        {"candidate_id": "a", "lora_name": "a"},
        {"candidate_id": "b", "lora_name": "b"},
        {"candidate_id": "c", "lora_name": "c"},
    ]
    initial_rewards = [
        {
            "candidate_id": "a",
            "reward_mean": 0.5,
            "rollout_exact_count": 1.0,
            "project_metrics": {
                "p0": {"reward": 1.0, "rollout_exact_rate": 1.0},
                "p1": {"reward": 0.0, "rollout_exact_rate": 0.0},
            },
        },
        {
            "candidate_id": "b",
            "reward_mean": 0.5,
            "rollout_exact_count": 1.0,
            "project_metrics": {
                "p0": {"reward": 0.0, "rollout_exact_rate": 0.0},
                "p1": {"reward": 0.25, "rollout_exact_rate": 0.25},
            },
        },
        {
            "candidate_id": "c",
            "reward_mean": 0.25,
            "rollout_exact_count": 0.5,
            "project_metrics": {
                "p0": {"reward": 0.0, "rollout_exact_rate": 0.0},
                "p1": {"reward": 0.5, "rollout_exact_rate": 0.5},
            },
        },
    ]
    args = SimpleNamespace(
        zorl_update_strategy="project_winner_delta",
        zorl_rescore_top_k=1,
        zorl_rescore_rollouts=4,
        zorl_rollouts_per_puzzle=1,
        zorl_reward_mode="rollout",
    )
    calls = []

    def fake_score_zorl_candidates(infer_url, rescored_candidates, reward_data, score_args):
        calls.append([candidate["candidate_id"] for candidate in rescored_candidates])
        return [
            {
                "candidate_id": "a",
                "reward_mean": 0.25,
                "rollout_exact_count": 0.5,
                "project_metrics": {
                    "p0": {"reward": 0.5, "rollout_exact_rate": 0.5},
                    "p1": {"reward": 0.0, "rollout_exact_rate": 0.0},
                },
            },
            {
                "candidate_id": "c",
                "reward_mean": 0.75,
                "rollout_exact_count": 1.5,
                "project_metrics": {
                    "p0": {"reward": 0.0, "rollout_exact_rate": 0.0},
                    "p1": {"reward": 1.0, "rollout_exact_rate": 1.0},
                },
            },
        ], {"reward_mean": 0.5, "reward_min": 0.25, "reward_max": 0.75}

    monkeypatch.setattr(_MODULE, "score_zorl_candidates", fake_score_zorl_candidates)

    rescored_rewards, metadata = _MODULE._rescore_top_zorl_candidates(
        "http://sglang",
        candidates,
        initial_rewards,
        [{"project": "p0"}, {"project": "p1"}],
        args,
    )

    assert calls == [["a", "c"]]
    assert [reward["candidate_id"] for reward in rescored_rewards] == ["a", "c"]
    assert metadata["rescore_top_k"] == 1
    assert metadata["rescore_candidate_count"] == 2
    assert metadata["rescore_project_count"] == 2
    assert metadata["rescore_initial_project_exact_mean"] == pytest.approx(0.75)
    assert metadata["rescore_project_exact_mean"] == pytest.approx(0.75)
    assert metadata["rescore_score_rollouts"] == 16


def test_project_winner_delta_scales_scores_to_candidate_delta_size():
    args = SimpleNamespace(
        zorl_update_strategy="project_winner_delta",
        zorl_b_sigma=0.1,
        zorl_candidate_delta_step_scale=1.5,
    )
    generation = {
        "num_pairs": 2,
        "candidates": [
            {"candidate_id": "p0", "perturbation_index": 0, "direction": "positive"},
            {"candidate_id": "n0", "perturbation_index": 0, "direction": "negative"},
            {"candidate_id": "p1", "perturbation_index": 1, "direction": "positive"},
            {"candidate_id": "n1", "perturbation_index": 1, "direction": "negative"},
        ],
    }
    candidate_rewards = [
        {
            "candidate_id": "p0",
            "reward_mean": 0.25,
            "num_rollouts": 1,
            "project_metrics": {
                "a": {"reward": 1.0, "rollout_exact_match": True},
            },
        },
        {
            "candidate_id": "n0",
            "reward_mean": 0.5,
            "num_rollouts": 1,
            "project_metrics": {
                "b": {"reward": 2.0, "rollout_exact_match": True},
            },
        },
        {
            "candidate_id": "p1",
            "reward_mean": 0.25,
            "num_rollouts": 1,
            "project_metrics": {
                "c": {"reward": 1.0, "rollout_exact_match": True},
            },
        },
        {
            "candidate_id": "n1",
            "reward_mean": 0.0,
            "num_rollouts": 1,
            "project_metrics": {
                "a": {"reward": 0.0, "rollout_exact_match": False},
                "b": {"reward": 0.0, "rollout_exact_match": False},
                "c": {"reward": 0.0, "rollout_exact_match": False},
            },
        },
    ]

    rewards_for_update, apply_lr, metadata = _MODULE._select_zorl_update_payload(
        generation,
        candidate_rewards,
        args,
        lr=0.01,
    )

    reward_by_candidate = {item["candidate_id"]: item for item in rewards_for_update}
    assert apply_lr == pytest.approx(0.15)
    assert metadata["score_normalization"] == "none"
    assert metadata["selected_project_reward_sum"] == pytest.approx(4.0)
    assert metadata["selected_candidate_weight_max"] == pytest.approx(0.5)
    assert reward_by_candidate["p0"]["reward_mean"] == pytest.approx(0.5)
    assert reward_by_candidate["n0"]["reward_mean"] == pytest.approx(1.0)
    assert reward_by_candidate["p1"]["reward_mean"] == pytest.approx(0.5)
    assert {item["_zorl_score_normalization"] for item in rewards_for_update} == {"none"}


def test_project_winner_delta_clears_generation_without_update_when_all_scores_zero():
    args = SimpleNamespace(
        zorl_update_strategy="project_winner_delta",
        zorl_b_sigma=0.1,
        zorl_candidate_delta_step_scale=1.0,
    )
    generation = {
        "num_pairs": 1,
        "candidates": [
            {"candidate_id": "p0", "perturbation_index": 0, "direction": "positive"},
            {"candidate_id": "n0", "perturbation_index": 0, "direction": "negative"},
        ],
    }
    candidate_rewards = [
        {
            "candidate_id": "p0",
            "reward_mean": 0.0,
            "num_rollouts": 1,
            "project_metrics": {"a": {"reward": 0.0, "rollout_exact_match": False}},
        },
        {
            "candidate_id": "n0",
            "reward_mean": 0.0,
            "num_rollouts": 1,
            "project_metrics": {"a": {"reward": 0.0, "rollout_exact_match": False}},
        },
    ]

    rewards_for_update, apply_lr, metadata = _MODULE._select_zorl_update_payload(
        generation,
        candidate_rewards,
        args,
        lr=0.01,
    )

    assert apply_lr == pytest.approx(0.1)
    assert metadata["zero_signal_update"] is True
    assert rewards_for_update == [
        {
            "candidate_id": "p0",
            "reward_mean": 0.0,
            "num_rollouts": 1,
            "_zorl_score_normalization": "none",
        }
    ]


def test_project_parent_advantage_updates_only_improved_projects():
    args = SimpleNamespace(
        zorl_update_strategy="project_parent_advantage",
        zorl_b_sigma=0.1,
        zorl_candidate_delta_step_scale=2.0,
    )
    generation = {
        "num_pairs": 2,
        "candidates": [
            {"candidate_id": "p0", "perturbation_index": 0, "direction": "positive"},
            {"candidate_id": "n0", "perturbation_index": 0, "direction": "negative"},
            {"candidate_id": "p1", "perturbation_index": 1, "direction": "positive"},
            {"candidate_id": "n1", "perturbation_index": 1, "direction": "negative"},
        ],
    }
    candidate_rewards = [
        {
            "candidate_id": "p0",
            "reward_mean": 0.5,
            "num_rollouts": 1,
            "project_metrics": {
                "a": {"reward": 0.7, "rollout_exact_match": False},
                "b": {"reward": 0.4, "rollout_exact_match": False},
            },
        },
        {
            "candidate_id": "n0",
            "reward_mean": 0.2,
            "num_rollouts": 1,
            "project_metrics": {
                "a": {"reward": 0.2, "rollout_exact_match": False},
                "b": {"reward": 0.5, "rollout_exact_match": False},
            },
        },
        {
            "candidate_id": "p1",
            "reward_mean": 0.3,
            "num_rollouts": 1,
            "project_metrics": {
                "a": {"reward": 0.6, "rollout_exact_match": False},
                "b": {"reward": 0.3, "rollout_exact_match": False},
            },
        },
    ]
    parent_baseline = {
        "project_metrics": {
            "a": {"reward": 0.6},
            "b": {"reward": 0.6},
        }
    }

    rewards_for_update, apply_lr, metadata = _MODULE._select_zorl_update_payload(
        generation,
        candidate_rewards,
        args,
        lr=0.01,
        parent_baseline=parent_baseline,
    )

    assert apply_lr == pytest.approx(0.2)
    assert metadata["score_normalization"] == "none"
    assert metadata["selected_project_improved"] == 1
    assert metadata["selected_project_advantage_sum"] == pytest.approx(0.1)
    assert metadata["selected_project_parent_reward_mean"] == pytest.approx(0.6)
    assert rewards_for_update == [
        {
            "candidate_id": "p0",
            "reward_mean": pytest.approx(1.0),
            "num_rollouts": 1,
            "_zorl_score_normalization": "none",
        }
    ]


def test_project_parent_advantage_skips_when_parent_is_not_beaten():
    args = SimpleNamespace(
        zorl_update_strategy="project_parent_advantage",
        zorl_b_sigma=0.1,
        zorl_candidate_delta_step_scale=1.0,
    )
    generation = {
        "num_pairs": 1,
        "candidates": [
            {"candidate_id": "p0", "perturbation_index": 0, "direction": "positive"},
            {"candidate_id": "n0", "perturbation_index": 0, "direction": "negative"},
        ],
    }
    candidate_rewards = [
        {
            "candidate_id": "p0",
            "reward_mean": 0.5,
            "num_rollouts": 1,
            "project_metrics": {"a": {"reward": 0.5, "rollout_exact_match": False}},
        },
        {
            "candidate_id": "n0",
            "reward_mean": 0.4,
            "num_rollouts": 1,
            "project_metrics": {"a": {"reward": 0.4, "rollout_exact_match": False}},
        },
    ]

    rewards_for_update, _apply_lr, metadata = _MODULE._select_zorl_update_payload(
        generation,
        candidate_rewards,
        args,
        lr=0.01,
        parent_baseline={"project_metrics": {"a": {"reward": 0.6}}},
    )

    assert metadata["zero_signal_update"] is True
    assert metadata["selected_project_improved"] == 0
    assert rewards_for_update == [
        {
            "candidate_id": "p0",
            "reward_mean": 0.0,
            "num_rollouts": 1,
            "_zorl_score_normalization": "none",
        }
    ]


def test_project_baseline_es_standardizes_all_candidates_against_parent_anchor():
    args = SimpleNamespace(
        zorl_update_strategy="project_baseline_es",
        zorl_b_sigma=0.1,
        zorl_candidate_delta_step_scale=2.0,
    )
    generation = {
        "num_pairs": 1,
        "candidates": [
            {"candidate_id": "p0", "perturbation_index": 0, "direction": "positive"},
            {"candidate_id": "n0", "perturbation_index": 0, "direction": "negative"},
        ],
    }
    candidate_rewards = [
        {
            "candidate_id": "p0",
            "reward_mean": 1.0,
            "num_rollouts": 1,
            "project_metrics": {"a": {"reward": 1.0, "rollout_exact_match": True}},
        },
        {
            "candidate_id": "n0",
            "reward_mean": 0.0,
            "num_rollouts": 1,
            "project_metrics": {"a": {"reward": 0.0, "rollout_exact_match": False}},
        },
    ]

    rewards_for_update, apply_lr, metadata = _MODULE._select_zorl_update_payload(
        generation,
        candidate_rewards,
        args,
        lr=0.01,
        parent_baseline={"project_metrics": {"a": {"reward": 0.5}}},
    )

    reward_by_candidate = {item["candidate_id"]: item for item in rewards_for_update}
    assert apply_lr == pytest.approx(0.2)
    assert metadata["score_normalization"] == "none"
    assert metadata["selected_pairs"] == 1
    assert metadata["selected_projects"] == 1
    assert metadata["selected_project_parent_reward_mean"] == pytest.approx(0.5)
    assert metadata["selected_project_advantage_mean"] == pytest.approx(0.5)
    assert metadata["selected_project_score_std_mean"] == pytest.approx(0.35355339059)
    assert reward_by_candidate["p0"]["reward_mean"] == pytest.approx(1.41421356237)
    assert reward_by_candidate["n0"]["reward_mean"] == pytest.approx(-1.41421356237)
    assert {item["_zorl_score_normalization"] for item in rewards_for_update} == {"none"}


def test_adaptive_update_norm_scales_with_parent_advantage_signal():
    args = SimpleNamespace(
        zorl_max_update_norm=3500.0,
        zorl_adaptive_update_norm=True,
        zorl_adaptive_update_norm_reference=0.25,
        zorl_adaptive_update_norm_min_scale=0.15,
        zorl_adaptive_update_norm_max_scale=1.0,
    )

    norm, metadata = _MODULE._resolve_zorl_max_update_norm(
        args,
        {"selected_project_advantage_mean": 0.05},
    )

    assert norm == pytest.approx(700.0)
    assert metadata["adaptive_update_norm_base"] == pytest.approx(3500.0)
    assert metadata["adaptive_update_norm_signal"] == pytest.approx(0.05)
    assert metadata["adaptive_update_norm_reference"] == pytest.approx(0.25)
    assert metadata["adaptive_update_norm_raw_scale"] == pytest.approx(0.2)
    assert metadata["adaptive_update_norm_scale"] == pytest.approx(0.2)
    assert metadata["adaptive_update_norm_effective"] == pytest.approx(700.0)


def test_adaptive_update_norm_clamps_to_minimum_scale():
    args = SimpleNamespace(
        zorl_max_update_norm=3500.0,
        zorl_adaptive_update_norm=True,
        zorl_adaptive_update_norm_reference=0.25,
        zorl_adaptive_update_norm_min_scale=0.15,
        zorl_adaptive_update_norm_max_scale=1.0,
    )

    norm, metadata = _MODULE._resolve_zorl_max_update_norm(
        args,
        {"selected_project_advantage_mean": 0.01},
    )

    assert norm == pytest.approx(525.0)
    assert metadata["adaptive_update_norm_raw_scale"] == pytest.approx(0.04)
    assert metadata["adaptive_update_norm_scale"] == pytest.approx(0.15)
    assert metadata["adaptive_update_norm_effective"] == pytest.approx(525.0)


def test_adaptive_update_norm_falls_back_when_signal_is_missing():
    args = SimpleNamespace(
        zorl_max_update_norm=3500.0,
        zorl_adaptive_update_norm=True,
        zorl_adaptive_update_norm_reference=0.25,
        zorl_adaptive_update_norm_min_scale=0.15,
        zorl_adaptive_update_norm_max_scale=1.0,
    )

    norm, metadata = _MODULE._resolve_zorl_max_update_norm(args, {})

    assert norm == pytest.approx(3500.0)
    assert metadata["adaptive_update_norm_skipped"] is True
    assert metadata["adaptive_update_norm_reason"] == "missing_selected_project_advantage_mean"


def test_adaptive_num_pairs_uses_base_before_parent_baseline_signal():
    args = SimpleNamespace(
        zorl_num_pairs=32,
        zorl_adaptive_num_pairs_multiplier=2.0,
        zorl_adaptive_num_pairs_active_threshold=4,
        zorl_adaptive_num_pairs_max=0,
    )

    num_pairs, metadata = _MODULE._resolve_zorl_generation_num_pairs(args)

    assert num_pairs == 32
    assert metadata["adaptive_num_pairs"] == 32
    assert metadata["adaptive_num_pairs_reason"] == "no_previous_active_projects"


def test_adaptive_num_pairs_grows_when_previous_active_projects_are_low():
    args = SimpleNamespace(
        zorl_num_pairs=32,
        zorl_adaptive_num_pairs_multiplier=2.5,
        zorl_adaptive_num_pairs_active_threshold=4,
        zorl_adaptive_num_pairs_max=96,
        _zorl_previous_active_projects=4,
    )

    num_pairs, metadata = _MODULE._resolve_zorl_generation_num_pairs(args)

    assert num_pairs == 80
    assert metadata["adaptive_num_pairs"] == 80
    assert metadata["adaptive_previous_active_projects"] == 4
    assert metadata["adaptive_num_pairs_reason"] == "active_projects_at_or_below_threshold"


def test_adaptive_num_pairs_keeps_base_when_previous_active_projects_are_high():
    args = SimpleNamespace(
        zorl_num_pairs=32,
        zorl_adaptive_num_pairs_multiplier=2.0,
        zorl_adaptive_num_pairs_active_threshold=4,
        zorl_adaptive_num_pairs_max=0,
        _zorl_previous_active_projects=5,
    )

    num_pairs, metadata = _MODULE._resolve_zorl_generation_num_pairs(args)

    assert num_pairs == 32
    assert metadata["adaptive_num_pairs"] == 32
    assert metadata["adaptive_previous_active_projects"] == 5
    assert metadata["adaptive_num_pairs_reason"] == "active_projects_above_threshold"


def test_hard_project_rollouts_grow_when_active_projects_are_low():
    args = SimpleNamespace(
        zorl_rollouts_per_puzzle=4,
        zorl_hard_project_rollout_multiplier=2.5,
        zorl_hard_project_rollout_active_threshold=4,
        zorl_hard_project_rollout_max=12,
    )

    score_args, metadata = _MODULE._resolve_zorl_hard_project_score_args(
        args,
        {
            "parent_advantage_active_projects": 3,
            "parent_advantage_total_projects": 8,
        },
    )

    assert score_args is not args
    assert score_args.zorl_rollouts_per_puzzle == 10
    assert metadata["hard_project_base_rollouts_per_puzzle"] == 4
    assert metadata["hard_project_rollouts_per_puzzle"] == 10
    assert metadata["hard_project_rollout_active_projects"] == 3
    assert metadata["hard_project_rollout_total_projects"] == 8
    assert metadata["hard_project_rollout_reason"] == "active_projects_at_or_below_threshold"


def test_hard_project_rollouts_keep_base_when_active_projects_are_high():
    args = SimpleNamespace(
        zorl_rollouts_per_puzzle=4,
        zorl_hard_project_rollout_multiplier=2.0,
        zorl_hard_project_rollout_active_threshold=4,
        zorl_hard_project_rollout_max=0,
    )

    score_args, metadata = _MODULE._resolve_zorl_hard_project_score_args(
        args,
        {"parent_advantage_active_projects": 5},
    )

    assert score_args is args
    assert metadata["hard_project_rollouts_per_puzzle"] == 4
    assert metadata["hard_project_rollout_active_projects"] == 5
    assert metadata["hard_project_rollout_reason"] == "active_projects_above_threshold"


def test_hard_project_rollouts_keep_base_without_active_project_signal():
    args = SimpleNamespace(
        zorl_rollouts_per_puzzle=4,
        zorl_hard_project_rollout_multiplier=2.0,
        zorl_hard_project_rollout_active_threshold=4,
        zorl_hard_project_rollout_max=0,
    )

    score_args, metadata = _MODULE._resolve_zorl_hard_project_score_args(args, {})

    assert score_args is args
    assert metadata["hard_project_rollouts_per_puzzle"] == 4
    assert metadata["hard_project_rollout_reason"] == "no_active_project_signal"


def test_parent_advantage_filter_keeps_only_unsolved_projects():
    reward_data = [
        {"project": "q0"},
        {"project": "q1"},
        {"project": "q2"},
    ]
    parent_baseline = {
        "project_metrics": {
            "q0": {"rollout_exact_rate": 1.0},
            "q1": {"rollout_exact_rate": 0.5},
            "q2": {"rollout_exact_match": True},
        }
    }

    active, metadata = _MODULE._filter_parent_advantage_reward_data(
        reward_data,
        parent_baseline,
        solved_exact_rate=1.0,
    )

    assert active == [{"project": "q1"}]
    assert metadata["parent_advantage_active_projects"] == 1
    assert metadata["parent_advantage_skipped_projects"] == 2
    assert metadata["parent_advantage_active_project_names"] == "q1"
    assert metadata["parent_advantage_skipped_project_names"] == "q0,q2"


def test_parent_advantage_filter_falls_back_when_all_projects_solved():
    reward_data = [{"project": "q0"}]

    active, metadata = _MODULE._filter_parent_advantage_reward_data(
        reward_data,
        {"project_metrics": {"q0": {"rollout_exact_rate": 1.0}}},
        solved_exact_rate=1.0,
    )

    assert active == reward_data
    assert metadata["parent_advantage_all_projects_solved"] is True


def test_flush_inference_cache_retries_transient_busy_response(monkeypatch):
    calls = []

    class _Response:
        def __init__(self, status_code, text):
            self.status_code = status_code
            self.text = text

    def fake_post(url, *, params, headers, timeout):
        calls.append((url, params, headers, timeout))
        if len(calls) == 1:
            return _Response(400, "Cache not flushed because there are pending requests.")
        return _Response(200, "Cache flushed.")

    monkeypatch.setattr(_MODULE.requests, "post", fake_post)
    monkeypatch.setattr(_MODULE.time, "sleep", lambda _seconds: None)

    _MODULE.flush_inference_cache("http://sglang:30060", timeout_s=12.0, attempts=2)

    assert len(calls) == 2
    assert calls[0][1] == {"timeout": 12.0}
    assert calls[0][3] == pytest.approx(30.0)


def test_flush_inference_cache_raises_last_error_after_retries(monkeypatch):
    class _Response:
        status_code = 400
        text = "Timed out waiting for idle state."

    monkeypatch.setattr(_MODULE.requests, "post", lambda *_args, **_kwargs: _Response())
    monkeypatch.setattr(_MODULE.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="Timed out waiting"):
        _MODULE.flush_inference_cache("http://sglang:30060", timeout_s=1.0, attempts=2)


def test_parent_probe_key_uses_prefix_as_tiebreaker():
    old_best = {
        "exact_count": 4,
        "reward_mean": 0.5,
        "reward_stats": {"rollout_prefix_mean": 0.25, "rollout_char_match_mean": 0.4},
    }
    improved_partial = {
        "exact_count": 4,
        "reward_mean": 0.5,
        "reward_stats": {"rollout_prefix_mean": 0.35, "rollout_char_match_mean": 0.4},
    }

    assert _MODULE._parent_probe_key(improved_partial) > _MODULE._parent_probe_key(old_best)
