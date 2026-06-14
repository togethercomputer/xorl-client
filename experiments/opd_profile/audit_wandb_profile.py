#!/usr/bin/env python3
"""Check that W&B history contains the critical local OPD profile rows."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any


RESULT_ROOT = Path(
    "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/"
    "er-opd-q36-35b-slots"
)
DEFAULT_RUN_PREFIX = "together-research/xorl-prefill-time-compute"
DEFAULT_KEYS = [
    "loss",
    "opd_kl",
    "opd_hidden_match_loss",
    "opd_teacher_entropy",
    "opd_top1_agreement",
    "opd_hidden_match_raw_loss",
    "opd_hidden_match_weight_mean",
    "opd_hidden_match_pos_loss",
    "opd_hidden_match_neg_loss",
    "opd_hidden_match_pos_raw_loss",
    "opd_hidden_match_neg_raw_loss",
    "opd_hidden_match_neg_minus_pos_raw",
    "opd_hidden_match_pos_weight_mean",
    "opd_hidden_match_neg_weight_mean",
    "num_opd_datums",
    "opd_contrastive_corrupt_buffer_weight",
    "opd_contrastive_corrupt_examples",
    "opd_contrastive_corrupt_answer_weight",
    "opd_contrastive_corrupt_answer_examples",
    "opd_positive_answer_weight",
    "opd_positive_answer_examples",
    "opd_ptc_positive_buffer_kl_weight",
    "opd_ptc_positive_answer_kl_weight",
    "opd_ptc_positive_hidden_weight",
    "opd_ptc_positive_examples",
    "opd_mask_zero_weight_positions",
    "opd_teacher_answer_source_gold",
    "opd_gold_answer_replacements",
    "opd_gold_answer_skipped",
    "opd_teacher_weight_prompt_mean",
    "opd_teacher_weight_buffer_mean",
    "opd_teacher_weight_answer_mean",
    "opd_hidden_weight_prompt_mean",
    "opd_hidden_weight_buffer_mean",
    "opd_hidden_weight_answer_mean",
    "opd_contrastive_data_multiplier",
    "opd_contrastive_corrupt_change_frac",
    "opd_contrastive_corrupt_noop_frac",
    "opd_cache_mismatch_memory_weight",
    "opd_cache_mismatch_balance_positive_hidden",
    "opd_cache_mismatch_positive_hidden_boost",
    "opd_memory_hidden_weight_balance_per_token",
    "opd_cache_mismatch_examples",
    "opd_cache_mismatch_skipped_examples",
    "opd_cache_mismatch_span_token_total",
    "opd_cache_mismatch_changed_cache_rows",
    "opd_cache_mismatch_change_frac",
    "opd_cache_mismatch_same_visible_input",
    "opd_cache_mismatch_negative_answer_kl_weight",
    "opd_teacher_memory_pair_diag_requested",
    "opd_teacher_memory_pair_diag_active",
    "opd_teacher_memory_pair_diag_failure",
    "opd_teacher_memory_pair_cross_cosine_distance_mean",
    "opd_teacher_memory_pair_within_adjacent_distance_mean",
    "opd_teacher_memory_pair_cross_minus_within_distance",
    "sync_inference_weights_s",
    "sync_transfer_time_s",
    "sync_endpoint_count",
    "sync_endpoint_success_count",
    "sync_endpoint_failure_count",
    "sync_serial_endpoint_sync",
    "sampler_metrics_available",
    "sampler_router_requests_delta",
    "sampler_worker_count",
    "sampler_worker_active_count",
    "sampler_worker_success_delta_total",
    "sampler_worker_success_delta_min",
    "sampler_worker_success_delta_max",
    "sampler_worker_success_balance_ratio",
    "sampler_policy_round_robin_active",
    "eval/accuracy",
    "eval/acc_pause",
    "eval/acc_nopause",
    "eval/acc_corrupt_pause",
    "eval/buffer_delta",
    "eval/buffer_delta_z",
    "eval/buffer_causal_margin",
    "eval/buffer_causal_z_min",
    "eval/buffer_lead_delta",
    "eval/buffer_vs_corrupt_delta",
    "eval/buffer_vs_corrupt_delta_z",
    "eval/buffer_vs_corrupt_lead_delta",
    "eval/cap_hit_frac",
    "eval/stop_sequence_seen_frac",
    "eval/pause_cap_hit_frac",
    "eval/nopause_cap_hit_frac",
    "eval/corrupt_pause_cap_hit_frac",
    "eval/pause_stop_sequence_seen_frac",
    "eval/nopause_stop_sequence_seen_frac",
    "eval/corrupt_pause_stop_sequence_seen_frac",
    "eval/control_cap_hit_frac_max",
    "eval/control_cap_hit_frac_mean",
    "eval/control_stop_sequence_seen_frac_max",
    "eval/control_stop_sequence_seen_frac_mean",
    "eval/corrupt_pause_request_failure_frac",
    "eval/control_request_failure_frac_max",
    "eval/control_max_completion_tokens",
    "eval/control_start_step",
    "eval/control_allowed_by_start_step",
    "eval/control_arms_concurrent",
    "eval/control_max_concurrency",
    "eval/control_configured_max_concurrency",
    "eval/control_bounded_concurrency_active",
    "eval/control_corrupt_pause_active",
    "eval/control_corrupt_pause_mode_preserve_boundary_ws",
    "eval/control_corrupt_pause_change_frac",
    "eval/control_corrupt_pause_len_delta_chars",
    "eval/control_corrupt_pause_leading_ws_match",
    "eval/control_corrupt_pause_trailing_ws_match",
    "eval/control_request_latency_mean_s",
    "eval/control_request_latency_p95_s",
    "eval/control_request_latency_max_s",
    "eval/control_client_queue_latency_mean_s",
    "eval/control_client_queue_latency_p95_s",
    "eval/control_client_queue_latency_max_s",
    "eval/control_service_latency_mean_s",
    "eval/control_service_latency_p95_s",
    "eval/control_service_latency_max_s",
    "eval/answer_logprob_group_latency_mean_s",
    "eval/answer_logprob_group_latency_max_s",
    "eval/answer_logprob_configured_batch_size",
    "eval/answer_logprob_batch_size",
    "eval/answer_logprob_chunk_count",
    "eval/answer_logprob_configured_max_concurrency",
    "eval/answer_logprob_max_concurrency",
    "eval/answer_logprob_bounded_concurrency_active",
    "eval/answer_logprob_chunk_latency_mean_s",
    "eval/answer_logprob_chunk_latency_max_s",
    "eval/answer_logprob_client_queue_latency_mean_s",
    "eval/answer_logprob_client_queue_latency_max_s",
    "eval/answer_logprob_service_latency_mean_s",
    "eval/answer_logprob_service_latency_max_s",
    "eval/answer_logprob_distractor_control_active",
    "eval/answer_logprob_distractor_pairs",
    "eval/answer_logprob_select_margin_pause",
    "eval/answer_logprob_select_margin_nopause",
    "eval/answer_logprob_select_margin_corrupt_pause",
    "eval/answer_logprob_select_delta",
    "eval/answer_logprob_select_delta_z",
    "eval/answer_logprob_select_vs_corrupt_delta",
    "eval/answer_logprob_select_vs_corrupt_delta_z",
    "eval/answer_logprob_select_causal_delta",
    "eval/answer_logprob_select_causal_z_min",
]


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _resolve_profile(arg: str) -> Path:
    if arg == "latest":
        profiles = sorted(glob.glob(str(RESULT_ROOT / "*" / "opd_profile.jsonl")))
        if not profiles:
            raise SystemExit(f"no profiles under {RESULT_ROOT}")
        return Path(profiles[-1])
    path = Path(arg)
    if path.is_dir():
        path = path / "opd_profile.jsonl"
    if not path.exists():
        raise SystemExit(f"profile not found: {path}")
    return path


def _resolve_run_path(run: str) -> str:
    if run.count("/") == 2:
        return run
    if "/" in run:
        raise SystemExit(
            "W&B run must be either a bare run id or entity/project/run_id, got "
            f"{run!r}"
        )
    return f"{DEFAULT_RUN_PREFIX}/{run}"


def _step(row: dict[str, Any], key: str) -> int | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _close(a: Any, b: Any, *, atol: float) -> bool:
    try:
        af = float(a)
        bf = float(b)
    except (TypeError, ValueError):
        return a == b
    if math.isnan(af) and math.isnan(bf):
        return True
    return abs(af - bf) <= atol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", help="profile JSONL, run dir, or 'latest'")
    parser.add_argument("--wandb-run", required=True, help="bare run id or entity/project/run_id")
    parser.add_argument("--key", action="append", dest="keys", help="metric key to audit; repeatable")
    parser.add_argument("--atol", type=float, default=1e-8)
    args = parser.parse_args()

    try:
        import wandb  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - depends on local env
        raise SystemExit(f"failed to import wandb: {exc}") from exc

    profile = _resolve_profile(args.profile)
    local_rows = _load_rows(profile)
    local_by_step = {int(row["step"]): row for row in local_rows if "step" in row}
    local_steps = sorted(local_by_step)
    if not local_steps:
        raise SystemExit(f"no step rows in {profile}")

    run_path = _resolve_run_path(args.wandb_run)
    api = wandb.Api(timeout=30)
    run = api.run(run_path)
    keys = args.keys or DEFAULT_KEYS

    print(f"profile={profile}")
    print(f"wandb_run={run_path} state={run.state} name={run.name}")
    print(f"local_steps={local_steps}")

    failed = False
    for key in keys:
        local_values = {step: row[key] for step, row in local_by_step.items() if key in row}
        if not local_values:
            continue
        wandb_values: dict[int, Any] = {}
        for row in run.scan_history(keys=["_step", key], page_size=200):
            step = _step(row, "_step")
            if step is not None and key in row:
                wandb_values[step] = row[key]
        local_key_steps = sorted(local_values)
        wandb_key_steps = sorted(wandb_values)
        missing = [step for step in local_key_steps if step not in wandb_values]
        mismatched = [
            step
            for step in local_key_steps
            if step in wandb_values and not _close(local_values[step], wandb_values[step], atol=args.atol)
        ]
        status = "ok"
        if missing or mismatched:
            status = "stale_or_mismatch"
            failed = True
        print(
            f"{key}\t{status}\tlocal_steps={local_key_steps}\t"
            f"wandb_steps={wandb_key_steps}\tmissing={missing}\tmismatched={mismatched}"
        )

    if failed:
        print("VERDICT: wandb_stale_or_mismatched")
        raise SystemExit(1)
    print("VERDICT: wandb_matches_profile")


if __name__ == "__main__":
    main()
