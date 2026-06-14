#!/usr/bin/env python3
"""Summarize OPD slot profile rows and apply strict promotion gates."""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path
from typing import Any


RESULT_ROOT = Path(
    "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/"
    "er-opd-q36-35b-slots"
)
SYNC_ENDPOINT_RE = re.compile(r"\bto\s+(\d+)\s+endpoint")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a JSON object")
    return data


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


def _same_run_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).expanduser().resolve(strict=False) == Path(right).expanduser().resolve(strict=False)


def _scalar_overlay_items(data: dict[str, Any]) -> dict[str, Any]:
    overlay: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict | list):
            continue
        if key.startswith("eval/") or key.startswith("opd_") or key.startswith("sampler_"):
            overlay[key] = value
        elif key.startswith("posthoc_"):
            overlay[key] = value
    return overlay


def _posthoc_overlay(
    path: Path,
    *,
    profile_run_dir: Path,
    result_key: str,
    allow_source_mismatch: bool,
) -> dict[str, Any]:
    data = _load_json(path)
    source_run = data.get("posthoc_source_run") or data.get("source_run")
    if source_run and not allow_source_mismatch and not _same_run_path(str(source_run), profile_run_dir):
        raise SystemExit(
            f"{path}: source run {source_run!r} does not match profile run {str(profile_run_dir)!r}; "
            "use --allow-posthoc-source-mismatch only for manual forensics"
        )

    selected_key = ""
    selected: dict[str, Any] = data
    results = data.get("results")
    if isinstance(results, dict):
        if result_key:
            if result_key not in results or not isinstance(results[result_key], dict):
                raise SystemExit(f"{path}: result key {result_key!r} not found")
            selected_key = result_key
            selected = results[result_key]
        elif len(results) == 1:
            selected_key, only_result = next(iter(results.items()))
            if not isinstance(only_result, dict):
                raise SystemExit(f"{path}: only result {selected_key!r} is not an object")
            selected = only_result
        else:
            raise SystemExit(
                f"{path}: contains multiple posthoc result keys; pass --posthoc-result-key "
                f"with one of {', '.join(sorted(results))}"
            )

    overlay = _scalar_overlay_items(selected)
    if source_run is not None:
        overlay["posthoc_source_run"] = source_run
    overlay["posthoc_overlay_path"] = str(path)
    if selected_key:
        overlay["posthoc_result_key"] = selected_key
    return overlay


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _delta_z(row: dict[str, Any]) -> float:
    if "eval/buffer_delta_z" in row:
        return _float(row, "eval/buffer_delta_z")
    pause = _float(row, "eval/acc_pause")
    nopause = _float(row, "eval/acc_nopause")
    pause_n = max(1.0, _float(row, "eval/control_scored_pause", _float(row, "eval/control_n")))
    nopause_n = max(1.0, _float(row, "eval/control_scored_nopause", _float(row, "eval/control_n")))
    se = math.sqrt(
        max(pause * (1.0 - pause), 0.0) / pause_n
        + max(nopause * (1.0 - nopause), 0.0) / nopause_n
    )
    return (pause - nopause) / se if se > 0 else 0.0


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _sync_endpoint_count(row: dict[str, Any]) -> int | None:
    if "sync_endpoint_count" in row:
        try:
            return int(row["sync_endpoint_count"])
        except (TypeError, ValueError):
            return None
    message = row.get("sync_message")
    if not isinstance(message, str):
        return None
    match = SYNC_ENDPOINT_RE.search(message)
    if not match:
        return None
    return int(match.group(1))


def _falseish(value: Any) -> bool:
    if value is False:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"0", "false", "no", "failed"}
    return False


def _require_present(failures: list[str], row: dict[str, Any], key: str) -> bool:
    if key not in row or row[key] is None:
        failures.append(f"{key} is missing")
        return False
    return True


def _fail_if_present(
    failures: list[str],
    row: dict[str, Any],
    key: str,
    *,
    maximum: float,
) -> None:
    if key in row and _float(row, key) > maximum:
        failures.append(f"{key}={_float(row, key):.4f} > {maximum:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", help="profile JSONL, run dir, or 'latest'")
    parser.add_argument("--min-delta", type=float, default=0.03)
    parser.add_argument("--min-z", type=float, default=2.0)
    parser.add_argument("--min-corrupt-delta", type=float, default=0.03)
    parser.add_argument("--min-corrupt-z", type=float, default=2.0)
    parser.add_argument("--min-answer-logprob-margin", type=float, default=0.0)
    parser.add_argument("--min-answer-logprob-z", type=float, default=0.0)
    parser.add_argument(
        "--require-answer-select-control",
        action="store_true",
        help="reject unless answer-logprob distractor controls show prompt-specific answer selection",
    )
    parser.add_argument("--min-answer-select-delta", type=float, default=0.0)
    parser.add_argument("--min-answer-select-z", type=float, default=0.0)
    parser.add_argument(
        "--min-answer-select-paired-n",
        type=float,
        default=0.0,
        help="minimum paired answer-selection examples; 0 uses --min-control-n",
    )
    parser.add_argument("--min-control-n", type=float, default=192.0)
    parser.add_argument("--min-consecutive", type=int, default=2)
    parser.add_argument("--include-warmup", action="store_true")
    parser.add_argument("--max-repeated-numeric", type=float, default=0.10)
    parser.add_argument("--max-leak", type=float, default=0.10)
    parser.add_argument("--max-cap-hit", type=float, default=0.50)
    parser.add_argument("--max-request-failure", type=float, default=0.0)
    parser.add_argument("--max-sampler-quiesce-outstanding", type=float, default=0.0)
    parser.add_argument("--max-sampler-quiesce-active-connections", type=float, default=0.0)
    parser.add_argument("--max-sampler-quiesce-inflight", type=float, default=0.0)
    parser.add_argument(
        "--expected-sync-endpoints",
        type=int,
        default=1,
        help="expected endpoint count in sync_message; set 0 to disable",
    )
    parser.add_argument(
        "--require-serial-sync",
        action="store_true",
        help="reject unless profile rows report serial endpoint sync was active",
    )
    parser.add_argument(
        "--require-sampler-routing",
        action="store_true",
        help="reject unless sampler-routing metrics prove traffic reached enough workers",
    )
    parser.add_argument("--min-sampler-active-workers", type=float, default=2.0)
    parser.add_argument("--min-sampler-balance-ratio", type=float, default=0.75)
    parser.add_argument("--min-control-max-tokens", type=float, default=32.0)
    parser.add_argument(
        "--no-require-control-artifacts",
        dest="require_control_artifacts",
        action="store_false",
        help="allow legacy profiles missing artifact/control-health metrics",
    )
    parser.add_argument(
        "--no-require-corrupt-control",
        dest="require_corrupt_control",
        action="store_false",
        help="allow profiles without the corrupted-pause causal control arm",
    )
    parser.add_argument(
        "--require-full-vocab-diagnostics",
        action="store_true",
        help="reject unless full-vocab entropy/top1 diagnostics are expected active",
    )
    parser.add_argument(
        "--posthoc-overlay",
        action="append",
        default=[],
        metavar="JSON",
        help="merge scalar metrics from a posthoc diagnostic JSON into the latest control row",
    )
    parser.add_argument(
        "--posthoc-result-key",
        default="",
        help="result key to select from nested posthoc JSON files with a top-level 'results' object",
    )
    parser.add_argument(
        "--allow-posthoc-source-mismatch",
        action="store_true",
        help="allow posthoc overlay files whose source_run does not match the profile directory",
    )
    args = parser.parse_args()

    profile = _resolve_profile(args.profile)
    rows = _load_rows(profile)
    control_rows = [r for r in rows if "eval/buffer_delta" in r]
    if control_rows and args.posthoc_overlay:
        latest_control = control_rows[-1]
        overlay_paths: list[str] = []
        for overlay_arg in args.posthoc_overlay:
            overlay_path = Path(overlay_arg)
            overlay = _posthoc_overlay(
                overlay_path,
                profile_run_dir=profile.parent,
                result_key=args.posthoc_result_key,
                allow_source_mismatch=args.allow_posthoc_source_mismatch,
            )
            latest_control.update(overlay)
            overlay_paths.append(str(overlay_path))
        latest_control["posthoc_overlay_active"] = 1.0
        latest_control["posthoc_overlay_count"] = float(len(overlay_paths))
        latest_control["posthoc_overlay_paths"] = ",".join(overlay_paths)
    print(f"profile={profile}")
    print(f"rows={len(rows)} control_rows={len(control_rows)}")
    if not control_rows:
        raise SystemExit("VERDICT: incomplete (no control rows)")

    header = [
        "step",
        "delta",
        "z",
        "causal_margin",
        "causal_z_min",
        "hm_raw",
        "hm_wmean",
        "hm_pos_raw",
        "hm_neg_raw",
        "hm_neg_minus_pos",
        "contrast_mult",
        "pos_ans_w",
        "pos_ans_n",
        "ptc_buf_w",
        "ptc_ans_w",
        "ptc_hid_w",
        "ptc_n",
        "mask0",
        "gold_src",
        "gold_n",
        "tw_prompt",
        "tw_buffer",
        "tw_answer",
        "hw_prompt",
        "hw_buffer",
        "hw_answer",
        "lead_delta",
        "acc_pause",
        "acc_nopause",
        "acc_corrupt",
        "health_acc",
        "corrupt_delta",
        "corrupt_z",
        "corrupt_lead_delta",
        "anslp_margin",
        "anslp_z",
        "anslp_corrupt_margin",
        "anslp_corrupt_z",
        "anslp_fail",
        "anslp_chunks",
        "anslp_batch",
        "anslp_maxconc",
        "sel_delta",
        "sel_z",
        "sel_corrupt",
        "sel_corrupt_z",
        "sel_causal",
        "pause_repnum",
        "nopause_repnum",
        "corrupt_repnum",
        "pause_leak",
        "pause_cue_leak",
        "pause_cap",
        "corrupt_cap",
        "pause_stop",
        "nopause_stop",
        "corrupt_stop",
        "pause_reqfail",
        "nopause_reqfail",
        "corrupt_reqfail",
        "ctrl_max_tok",
        "ctrl_start",
        "ctrl_allowed",
        "ctrl_conc",
        "ctrl_maxconc",
        "ctrl_cfgconc",
        "ctrl_q_p95",
        "ctrl_svc_p95",
        "corrupt_active",
        "corrupt_ws",
        "corrupt_chg",
        "corrupt_lend",
        "corrupt_lws",
        "corrupt_tws",
        "sync_ok",
        "sync_eps",
        "sync_ep_ok",
        "sync_ep_fail",
        "sync_serial",
        "sampler_req",
        "sampler_workers",
        "sampler_active",
        "sampler_balance",
        "quiesce_ok",
        "quiesce_out",
        "quiesce_conn",
        "quiesce_inflight",
        "sync_s",
        "cachemis",
        "cachemis_bal",
        "hm_wbal",
        "cachemis_skip",
        "cachemis_frac",
        "mempair_active",
        "mempair_delta",
        "diag_active",
        "diag_unavail",
        "posthoc",
        "posthoc_n",
        "posthoc_result",
    ]
    print("\t".join(header))
    for row in control_rows:
        values = [
            row.get("step"),
            _float(row, "eval/buffer_delta"),
            _delta_z(row),
            row.get("eval/buffer_causal_margin"),
            row.get("eval/buffer_causal_z_min"),
            row.get("opd_hidden_match_raw_loss"),
            row.get("opd_hidden_match_weight_mean"),
            row.get("opd_hidden_match_pos_raw_loss"),
            row.get("opd_hidden_match_neg_raw_loss"),
            row.get("opd_hidden_match_neg_minus_pos_raw"),
            row.get("opd_contrastive_data_multiplier"),
            row.get("opd_positive_answer_weight"),
            row.get("opd_positive_answer_examples"),
            row.get("opd_ptc_positive_buffer_kl_weight"),
            row.get("opd_ptc_positive_answer_kl_weight"),
            row.get("opd_ptc_positive_hidden_weight"),
            row.get("opd_ptc_positive_examples"),
            row.get("opd_mask_zero_weight_positions"),
            row.get("opd_teacher_answer_source_gold"),
            row.get("opd_gold_answer_replacements"),
            row.get("opd_teacher_weight_prompt_mean"),
            row.get("opd_teacher_weight_buffer_mean"),
            row.get("opd_teacher_weight_answer_mean"),
            row.get("opd_hidden_weight_prompt_mean"),
            row.get("opd_hidden_weight_buffer_mean"),
            row.get("opd_hidden_weight_answer_mean"),
            _float(row, "eval/buffer_lead_delta"),
            _float(row, "eval/acc_pause"),
            _float(row, "eval/acc_nopause"),
            row.get("eval/acc_corrupt_pause"),
            row.get("eval/accuracy"),
            row.get("eval/buffer_vs_corrupt_delta"),
            row.get("eval/buffer_vs_corrupt_delta_z"),
            row.get("eval/buffer_vs_corrupt_lead_delta"),
            row.get("eval/answer_logprob_margin"),
            row.get("eval/answer_logprob_margin_z"),
            row.get("eval/answer_logprob_vs_corrupt_margin"),
            row.get("eval/answer_logprob_vs_corrupt_margin_z"),
            row.get("eval/answer_logprob_request_failure_frac"),
            row.get("eval/answer_logprob_chunk_count"),
            row.get("eval/answer_logprob_batch_size"),
            row.get("eval/answer_logprob_max_concurrency"),
            row.get("eval/answer_logprob_select_delta"),
            row.get("eval/answer_logprob_select_delta_z"),
            row.get("eval/answer_logprob_select_vs_corrupt_delta"),
            row.get("eval/answer_logprob_select_vs_corrupt_delta_z"),
            row.get("eval/answer_logprob_select_causal_delta"),
            row.get("eval/pause_repeated_numeric_frac"),
            row.get("eval/nopause_repeated_numeric_frac"),
            row.get("eval/corrupt_pause_repeated_numeric_frac"),
            row.get("eval/pause_filler_leak_frac"),
            row.get("eval/pause_answer_cue_leak_frac"),
            row.get("eval/pause_cap_hit_frac"),
            row.get("eval/corrupt_pause_cap_hit_frac"),
            row.get("eval/pause_stop_sequence_seen_frac"),
            row.get("eval/nopause_stop_sequence_seen_frac"),
            row.get("eval/corrupt_pause_stop_sequence_seen_frac"),
            row.get("eval/pause_request_failure_frac"),
            row.get("eval/nopause_request_failure_frac"),
            row.get("eval/corrupt_pause_request_failure_frac"),
            row.get("eval/control_max_completion_tokens"),
            row.get("eval/control_start_step"),
            row.get("eval/control_allowed_by_start_step"),
            row.get("eval/control_arms_concurrent"),
            row.get("eval/control_max_concurrency"),
            row.get("eval/control_configured_max_concurrency"),
            row.get("eval/control_client_queue_latency_p95_s"),
            row.get("eval/control_service_latency_p95_s"),
            row.get("eval/control_corrupt_pause_active"),
            row.get("eval/control_corrupt_pause_mode_preserve_boundary_ws"),
            row.get("eval/control_corrupt_pause_change_frac"),
            row.get("eval/control_corrupt_pause_len_delta_chars"),
            row.get("eval/control_corrupt_pause_leading_ws_match"),
            row.get("eval/control_corrupt_pause_trailing_ws_match"),
            row.get("sync_success"),
            _sync_endpoint_count(row),
            row.get("sync_endpoint_success_count"),
            row.get("sync_endpoint_failure_count"),
            row.get("sync_serial_endpoint_sync"),
            row.get("sampler_router_requests_delta"),
            row.get("sampler_worker_count"),
            row.get("sampler_worker_active_count"),
            row.get("sampler_worker_success_balance_ratio"),
            row.get("sampler_quiesce_success"),
            row.get("sampler_quiesce_new_outstanding"),
            row.get("sampler_quiesce_connections_active"),
            row.get("sampler_quiesce_inflight_request_age_count"),
            row.get("sync_inference_weights_s"),
            row.get("opd_cache_mismatch_examples"),
            row.get("opd_cache_mismatch_balance_positive_hidden"),
            row.get("opd_memory_hidden_weight_balance_per_token"),
            row.get("opd_cache_mismatch_skipped_examples"),
            row.get("opd_cache_mismatch_change_frac"),
            row.get("opd_teacher_memory_pair_diag_active"),
            row.get("opd_teacher_memory_pair_cross_minus_within_distance"),
            row.get("opd_full_vocab_diag_active_expected"),
            row.get("opd_full_vocab_diag_unavailable_expected"),
            row.get("posthoc_overlay_active"),
            row.get("posthoc_prompt_count"),
            row.get("posthoc_result_key", row.get("posthoc_corrupt_mode")),
        ]
        print("\t".join(_fmt(v) for v in values))

    gate_rows = control_rows if args.include_warmup else [r for r in control_rows if not r.get("profile_warmup")]
    if not gate_rows:
        print("VERDICT: incomplete (no non-warmup control rows)")
        raise SystemExit(1)

    def row_failures(row: dict[str, Any]) -> list[str]:
        failures: list[str] = []
        delta = _float(row, "eval/buffer_delta")
        z = _delta_z(row)
        control_n = _float(row, "eval/control_n")
        lead_delta = _float(row, "eval/buffer_lead_delta")
        corrupt_delta = _float(row, "eval/buffer_vs_corrupt_delta")
        corrupt_z = _float(row, "eval/buffer_vs_corrupt_delta_z")
        corrupt_lead_delta = _float(row, "eval/buffer_vs_corrupt_lead_delta")
        sync_endpoints = _sync_endpoint_count(row)
        answer_logprob_active = _float(row, "eval/answer_logprob_control_active")
        if delta < args.min_delta:
            failures.append(f"eval/buffer_delta={delta:.4f} < {args.min_delta:.4f}")
        if z < args.min_z:
            failures.append(f"eval/buffer_delta_z={z:.4f} < {args.min_z:.4f}")
        if control_n < args.min_control_n:
            failures.append(f"eval/control_n={control_n:.0f} < {args.min_control_n:.0f}")
        if lead_delta <= 0.0:
            failures.append(f"eval/buffer_lead_delta={lead_delta:.4f} <= 0")
        if args.require_corrupt_control:
            required_corrupt = [
                "eval/control_corrupt_pause_active",
                "eval/acc_corrupt_pause",
                "eval/buffer_vs_corrupt_delta",
                "eval/buffer_vs_corrupt_delta_z",
                "eval/buffer_vs_corrupt_lead_delta",
                "eval/corrupt_pause_repeated_numeric_frac",
                "eval/corrupt_pause_cap_hit_frac",
                "eval/corrupt_pause_request_failure_frac",
            ]
            have_corrupt_metrics = True
            for key in required_corrupt:
                have_corrupt_metrics = _require_present(failures, row, key) and have_corrupt_metrics
            if have_corrupt_metrics:
                if _float(row, "eval/control_corrupt_pause_active") < 1.0:
                    failures.append(
                        "eval/control_corrupt_pause_active="
                        f"{_float(row, 'eval/control_corrupt_pause_active'):.4f} < 1.0000"
                    )
                if corrupt_delta < args.min_corrupt_delta:
                    failures.append(
                        f"eval/buffer_vs_corrupt_delta={corrupt_delta:.4f} < "
                        f"{args.min_corrupt_delta:.4f}"
                    )
                if corrupt_z < args.min_corrupt_z:
                    failures.append(
                        f"eval/buffer_vs_corrupt_delta_z={corrupt_z:.4f} < "
                        f"{args.min_corrupt_z:.4f}"
                    )
                if corrupt_lead_delta <= 0.0:
                    failures.append(
                        f"eval/buffer_vs_corrupt_lead_delta={corrupt_lead_delta:.4f} <= 0"
                    )
        if answer_logprob_active >= 1.0:
            required_answer_logprob = [
                "eval/answer_logprob_control_available",
                "eval/answer_logprob_margin",
                "eval/answer_logprob_margin_z",
                "eval/answer_logprob_margin_paired_n",
                "eval/answer_logprob_request_failure_frac",
            ]
            have_answer_logprob = True
            for key in required_answer_logprob:
                have_answer_logprob = _require_present(failures, row, key) and have_answer_logprob
            if have_answer_logprob:
                if _float(row, "eval/answer_logprob_control_available") < 1.0:
                    failures.append(
                        "eval/answer_logprob_control_available="
                        f"{_float(row, 'eval/answer_logprob_control_available'):.4f} < 1.0000"
                    )
                answer_margin = _float(row, "eval/answer_logprob_margin")
                answer_z = _float(row, "eval/answer_logprob_margin_z")
                if answer_margin < args.min_answer_logprob_margin:
                    failures.append(
                        "eval/answer_logprob_margin="
                        f"{answer_margin:.4f} < {args.min_answer_logprob_margin:.4f}"
                    )
                if answer_z < args.min_answer_logprob_z:
                    failures.append(
                        "eval/answer_logprob_margin_z="
                        f"{answer_z:.4f} < {args.min_answer_logprob_z:.4f}"
                    )
                if (
                    "eval/answer_logprob_vs_corrupt_margin" in row
                    and _float(row, "eval/answer_logprob_vs_corrupt_margin") < args.min_answer_logprob_margin
                ):
                    failures.append(
                        "eval/answer_logprob_vs_corrupt_margin="
                        f"{_float(row, 'eval/answer_logprob_vs_corrupt_margin'):.4f} < "
                        f"{args.min_answer_logprob_margin:.4f}"
                    )
                if (
                    "eval/answer_logprob_vs_corrupt_margin_z" in row
                    and _float(row, "eval/answer_logprob_vs_corrupt_margin_z") < args.min_answer_logprob_z
                ):
                    failures.append(
                        "eval/answer_logprob_vs_corrupt_margin_z="
                        f"{_float(row, 'eval/answer_logprob_vs_corrupt_margin_z'):.4f} < "
                        f"{args.min_answer_logprob_z:.4f}"
                    )
                _fail_if_present(
                    failures,
                    row,
                    "eval/answer_logprob_request_failure_frac",
                    maximum=args.max_request_failure,
                )
        if args.require_answer_select_control:
            required_answer_select = [
                "eval/answer_logprob_distractor_control_active",
                "eval/answer_logprob_distractor_pairs",
                "eval/answer_logprob_select_delta",
                "eval/answer_logprob_select_delta_z",
                "eval/answer_logprob_select_delta_paired_n",
                "eval/answer_logprob_select_vs_corrupt_delta",
                "eval/answer_logprob_select_vs_corrupt_delta_z",
                "eval/answer_logprob_select_vs_corrupt_delta_paired_n",
                "eval/answer_logprob_select_causal_delta",
                "eval/answer_logprob_select_causal_z_min",
            ]
            have_answer_select = True
            for key in required_answer_select:
                have_answer_select = _require_present(failures, row, key) and have_answer_select
            if have_answer_select:
                if _float(row, "eval/answer_logprob_distractor_control_active") < 1.0:
                    failures.append(
                        "eval/answer_logprob_distractor_control_active="
                        f"{_float(row, 'eval/answer_logprob_distractor_control_active'):.4f} < 1.0000"
                    )
                min_select_n = (
                    args.min_answer_select_paired_n
                    if args.min_answer_select_paired_n > 0.0
                    else args.min_control_n
                )
                for key in [
                    "eval/answer_logprob_distractor_pairs",
                    "eval/answer_logprob_select_delta_paired_n",
                    "eval/answer_logprob_select_vs_corrupt_delta_paired_n",
                ]:
                    if _float(row, key) < min_select_n:
                        failures.append(f"{key}={_float(row, key):.0f} < {min_select_n:.0f}")
                select_delta = _float(row, "eval/answer_logprob_select_delta")
                select_z = _float(row, "eval/answer_logprob_select_delta_z")
                select_corrupt_delta = _float(row, "eval/answer_logprob_select_vs_corrupt_delta")
                select_corrupt_z = _float(row, "eval/answer_logprob_select_vs_corrupt_delta_z")
                select_causal_delta = _float(row, "eval/answer_logprob_select_causal_delta")
                select_causal_z = _float(row, "eval/answer_logprob_select_causal_z_min")
                if select_delta < args.min_answer_select_delta:
                    failures.append(
                        "eval/answer_logprob_select_delta="
                        f"{select_delta:.4f} < {args.min_answer_select_delta:.4f}"
                    )
                if select_z < args.min_answer_select_z:
                    failures.append(
                        "eval/answer_logprob_select_delta_z="
                        f"{select_z:.4f} < {args.min_answer_select_z:.4f}"
                    )
                if select_corrupt_delta < args.min_answer_select_delta:
                    failures.append(
                        "eval/answer_logprob_select_vs_corrupt_delta="
                        f"{select_corrupt_delta:.4f} < {args.min_answer_select_delta:.4f}"
                    )
                if select_corrupt_z < args.min_answer_select_z:
                    failures.append(
                        "eval/answer_logprob_select_vs_corrupt_delta_z="
                        f"{select_corrupt_z:.4f} < {args.min_answer_select_z:.4f}"
                    )
                if select_causal_delta < args.min_answer_select_delta:
                    failures.append(
                        "eval/answer_logprob_select_causal_delta="
                        f"{select_causal_delta:.4f} < {args.min_answer_select_delta:.4f}"
                    )
                if select_causal_z < args.min_answer_select_z:
                    failures.append(
                        "eval/answer_logprob_select_causal_z_min="
                        f"{select_causal_z:.4f} < {args.min_answer_select_z:.4f}"
                    )
        if _float(row, "sampler_quiesce_enabled") >= 1.0:
            required_quiesce = [
                "sampler_quiesce_success",
                "sampler_quiesce_new_outstanding",
                "sampler_quiesce_connections_active",
                "sampler_quiesce_inflight_request_age_count",
            ]
            have_quiesce = True
            for key in required_quiesce:
                have_quiesce = _require_present(failures, row, key) and have_quiesce
            if have_quiesce:
                if _float(row, "sampler_quiesce_success") < 1.0:
                    failures.append(
                        f"sampler_quiesce_success={_float(row, 'sampler_quiesce_success'):.4f} < 1.0000"
                    )
                if _float(row, "sampler_quiesce_new_outstanding") > args.max_sampler_quiesce_outstanding:
                    failures.append(
                        "sampler_quiesce_new_outstanding="
                        f"{_float(row, 'sampler_quiesce_new_outstanding'):.0f} > "
                        f"{args.max_sampler_quiesce_outstanding:.0f}"
                    )
                if _float(row, "sampler_quiesce_connections_active") > args.max_sampler_quiesce_active_connections:
                    failures.append(
                        "sampler_quiesce_connections_active="
                        f"{_float(row, 'sampler_quiesce_connections_active'):.0f} > "
                        f"{args.max_sampler_quiesce_active_connections:.0f}"
                    )
                if _float(row, "sampler_quiesce_inflight_request_age_count") > args.max_sampler_quiesce_inflight:
                    failures.append(
                        "sampler_quiesce_inflight_request_age_count="
                        f"{_float(row, 'sampler_quiesce_inflight_request_age_count'):.0f} > "
                        f"{args.max_sampler_quiesce_inflight:.0f}"
                    )
        if _falseish(row.get("sync_success")):
            failures.append(f"sync_success={row.get('sync_success')!r}")
        if _float(row, "sync_endpoint_failure_count") > 0:
            failures.append(
                f"sync_endpoint_failure_count={_float(row, 'sync_endpoint_failure_count'):.0f} > 0"
            )
        if args.expected_sync_endpoints > 0:
            if sync_endpoints is None:
                failures.append("sync endpoint count is missing from sync_message")
            elif sync_endpoints != args.expected_sync_endpoints:
                failures.append(
                    f"sync endpoints={sync_endpoints} != expected {args.expected_sync_endpoints}"
                )
            if "sync_endpoint_success_count" in row and (
                _float(row, "sync_endpoint_success_count") < args.expected_sync_endpoints
            ):
                failures.append(
                    "sync_endpoint_success_count="
                    f"{_float(row, 'sync_endpoint_success_count'):.0f} < {args.expected_sync_endpoints}"
                )
        if args.require_serial_sync and _float(row, "sync_serial_endpoint_sync") < 1.0:
            failures.append(
                f"sync_serial_endpoint_sync={_float(row, 'sync_serial_endpoint_sync'):.4f} < 1.0000"
            )
        if args.require_sampler_routing:
            required_sampler = [
                "sampler_metrics_available",
                "sampler_worker_active_count",
                "sampler_worker_success_balance_ratio",
                "sampler_router_requests_delta",
            ]
            have_sampler_metrics = True
            for key in required_sampler:
                have_sampler_metrics = _require_present(failures, row, key) and have_sampler_metrics
            if have_sampler_metrics:
                if _float(row, "sampler_metrics_available") < 1.0:
                    failures.append(
                        f"sampler_metrics_available={_float(row, 'sampler_metrics_available'):.4f} < 1.0000"
                    )
                if _float(row, "sampler_worker_active_count") < args.min_sampler_active_workers:
                    failures.append(
                        "sampler_worker_active_count="
                        f"{_float(row, 'sampler_worker_active_count'):.0f} < "
                        f"{args.min_sampler_active_workers:.0f}"
                    )
                if _float(row, "sampler_worker_success_balance_ratio") < args.min_sampler_balance_ratio:
                    failures.append(
                        "sampler_worker_success_balance_ratio="
                        f"{_float(row, 'sampler_worker_success_balance_ratio'):.4f} < "
                        f"{args.min_sampler_balance_ratio:.4f}"
                    )
                if _float(row, "sampler_router_requests_delta") <= 0.0:
                    failures.append(
                        f"sampler_router_requests_delta={_float(row, 'sampler_router_requests_delta'):.0f} <= 0"
                    )
        if args.require_control_artifacts:
            required = [
                "eval/control_max_completion_tokens",
                "eval/control_arms_concurrent",
                "eval/pause_repeated_numeric_frac",
                "eval/nopause_repeated_numeric_frac",
                "eval/pause_cap_hit_frac",
                "eval/nopause_cap_hit_frac",
                "eval/pause_request_failure_frac",
                "eval/nopause_request_failure_frac",
            ]
            for key in required:
                _require_present(failures, row, key)
            if "eval/control_arms_concurrent" in row and _float(row, "eval/control_arms_concurrent") < 1.0:
                failures.append(
                    f"eval/control_arms_concurrent={_float(row, 'eval/control_arms_concurrent'):.4f} < 1.0000"
                )
        if (
            "eval/control_max_completion_tokens" in row
            and _float(row, "eval/control_max_completion_tokens") < args.min_control_max_tokens
        ):
            failures.append(
                "eval/control_max_completion_tokens="
                f"{_float(row, 'eval/control_max_completion_tokens'):.0f} < {args.min_control_max_tokens:.0f}"
            )
        if args.require_full_vocab_diagnostics and _float(row, "opd_full_vocab_diag_active_expected") < 1.0:
            failures.append("full-vocab diagnostics are not expected active")
        for arm in ("pause", "nopause"):
            _fail_if_present(
                failures,
                row,
                f"eval/{arm}_repeated_numeric_frac",
                maximum=args.max_repeated_numeric,
            )
            _fail_if_present(
                failures,
                row,
                f"eval/{arm}_cap_hit_frac",
                maximum=args.max_cap_hit,
            )
            _fail_if_present(
                failures,
                row,
                f"eval/{arm}_request_failure_frac",
                maximum=args.max_request_failure,
            )
        if args.require_corrupt_control:
            _fail_if_present(
                failures,
                row,
                "eval/corrupt_pause_repeated_numeric_frac",
                maximum=args.max_repeated_numeric,
            )
            _fail_if_present(
                failures,
                row,
                "eval/corrupt_pause_cap_hit_frac",
                maximum=args.max_cap_hit,
            )
            _fail_if_present(
                failures,
                row,
                "eval/corrupt_pause_request_failure_frac",
                maximum=args.max_request_failure,
            )
        _fail_if_present(failures, row, "eval/pause_filler_leak_frac", maximum=args.max_leak)
        _fail_if_present(failures, row, "eval/pause_answer_cue_leak_frac", maximum=args.max_leak)
        return failures

    latest = gate_rows[-1]
    failures = row_failures(latest)
    warnings: list[str] = []
    if _float(latest, "opd_full_vocab_diag_requested") >= 1.0 and _float(
        latest, "opd_full_vocab_diag_unavailable_expected"
    ) >= 1.0:
        warnings.append(
            "full-vocab entropy/top1 diagnostics were requested but are unavailable for this KL backend"
        )
    if "eval/control_arms_concurrent" not in latest:
        warnings.append("control arms concurrency metric is missing; profile likely predates the concurrent control patch")
    if "eval/control_corrupt_pause_active" not in latest:
        warnings.append("corrupted-pause control metric is missing; profile predates the causal control patch")

    consecutive = 0
    for row in reversed(gate_rows):
        if row_failures(row):
            break
        consecutive += 1

    if warnings:
        print("WARNINGS:")
        for warning in warnings:
            print(f"- {warning}")
    if failures:
        print("VERDICT: reject")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    if consecutive < args.min_consecutive:
        print("VERDICT: watch")
        print(f"- consecutive passing non-warmup controls={consecutive} < {args.min_consecutive}")
        raise SystemExit(1)
    print("VERDICT: promote")


if __name__ == "__main__":
    main()
