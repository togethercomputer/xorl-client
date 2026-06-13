#!/usr/bin/env python3
"""Autoresearch controller for ZORL experiment candidates."""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
ZORL_DIR = ROOT.parent
REPO_ROOT = ZORL_DIR.parents[1]

IDEAS_PATH = ROOT / "ideas.yaml"
RUN_LOG = ROOT / "runs.jsonl"
SCORECARD_DIR = ROOT / "scorecards"
RENDER_DIR = ROOT / "renders"
DEFAULT_RESULT_ROOT = ZORL_DIR / "results"

LOG_NAMES = (
    "job.log",
    "zorl_client.log",
    "letter_count_test.log",
    "password_test.log",
    "countdown_e2e.log",
)


class EnvString(str):
    """String scalar that must remain quoted for Kubernetes EnvVar.value."""


def _represent_env_string(dumper: yaml.SafeDumper, data: EnvString) -> yaml.nodes.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data), style='"')


yaml.SafeDumper.add_representer(EnvString, _represent_env_string)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ideas": []}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a YAML mapping")
    data.setdefault("ideas", [])
    if not isinstance(data["ideas"], list):
        raise SystemExit(f"{path}: ideas must be a list")
    return data


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def append_event(event: dict[str, Any]) -> None:
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    payload = {"time_utc": utc_now(), **event}
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def idea_by_id(data: dict[str, Any], idea_id: str) -> dict[str, Any]:
    for idea in data["ideas"]:
        if str(idea.get("id")) == idea_id:
            return idea
    raise SystemExit(f"unknown idea id {idea_id!r}")


def dependencies_satisfied(data: dict[str, Any], idea: dict[str, Any]) -> bool:
    complete_states = {
        "weak_signal",
        "promote_retest",
        "strong_signal",
        "complete",
        "rejected",
    }
    by_id = {str(item.get("id")): item for item in data["ideas"]}
    for dep in idea.get("requires", []) or []:
        dep_idea = by_id.get(str(dep))
        if dep_idea is None or str(dep_idea.get("status")) not in complete_states:
            return False
    return True


def next_idea(data: dict[str, Any]) -> dict[str, Any] | None:
    runnable = ready_ideas(data)
    if not runnable:
        return None
    return runnable[0]


def ready_ideas(data: dict[str, Any]) -> list[dict[str, Any]]:
    runnable = [
        idea
        for idea in data["ideas"]
        if str(idea.get("status", "queued")) in {"queued", "retest"}
        and dependencies_satisfied(data, idea)
    ]
    return sorted(runnable, key=lambda item: (-f(item.get("priority"), default=0.0), str(item.get("id"))))


def _resolve_path(raw: str | Path, *, bases: tuple[Path, ...]) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    for base in bases:
        candidate = base / path
        if candidate.exists():
            return candidate
    return bases[0] / path


def candidate_path_for(idea: dict[str, Any]) -> Path:
    raw = idea.get("candidate")
    if not raw:
        raise SystemExit(f"idea {idea.get('id')} has no candidate")
    return _resolve_path(str(raw), bases=(ROOT, ZORL_DIR, REPO_ROOT))


def load_candidate(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: candidate must be a YAML mapping")
    return data


def candidate_for_idea(data: dict[str, Any], idea: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = candidate_path_for(idea)
    return path, load_candidate(path)


def parse_set_env(items: list[str] | None) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set-env must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--set-env has an empty key: {item!r}")
        env[key] = value
    return env


def apply_env_override(candidate: dict[str, Any], items: list[str] | None) -> dict[str, Any]:
    overrides = parse_set_env(items)
    if not overrides:
        return candidate
    patched = copy.deepcopy(candidate)
    patched.setdefault("env", {})
    patched["env"].update(overrides)
    return patched


def candidate_from_args(
    data: dict[str, Any],
    *,
    idea_id: str | None,
    candidate_arg: str | None,
    set_env: list[str] | None = None,
) -> tuple[dict[str, Any] | None, Path, dict[str, Any]]:
    if idea_id and candidate_arg:
        raise SystemExit("use either --id or --candidate, not both")
    if candidate_arg:
        path = _resolve_path(candidate_arg, bases=(ROOT, ZORL_DIR, REPO_ROOT))
        return None, path, apply_env_override(load_candidate(path), set_env)
    if not idea_id:
        idea = next_idea(data)
        if idea is None:
            raise SystemExit("no queued ideas")
    else:
        idea = idea_by_id(data, idea_id)
    path, candidate = candidate_for_idea(data, idea)
    return idea, path, apply_env_override(candidate, set_env)


def f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(out) else out


def as_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def slugify(value: Any) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", str(value).lower()).strip("-")
    return slug or "unknown"


def merge_env(container: dict[str, Any], env: dict[str, Any]) -> None:
    entries = container.setdefault("env", [])
    by_name = {
        str(entry.get("name")): entry
        for entry in entries
        if isinstance(entry, dict) and "name" in entry
    }
    for key, value in env.items():
        entry = {"name": str(key), "value": str(value)}
        if str(key) in by_name:
            by_name[str(key)].clear()
            by_name[str(key)].update(entry)
        else:
            entries.append(entry)


def quote_env_values(container: dict[str, Any]) -> None:
    entries = container.get("env")
    if not isinstance(entries, list):
        return
    for entry in entries:
        if not isinstance(entry, dict) or "value" not in entry:
            continue
        entry["value"] = EnvString("" if entry["value"] is None else str(entry["value"]))


def cli_arg_lines(cli_args: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for raw_key, raw_value in cli_args.items():
        flag = "--" + str(raw_key).replace("_", "-").lstrip("-")
        if raw_value is None or raw_value is False:
            continue
        if raw_value is True:
            lines.append(f"                {shlex.quote(flag)} \\")
            continue
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        for value in values:
            lines.append(f"                {shlex.quote(flag)} {shlex.quote(str(value))} \\")
    return lines


def inject_cli_args(container: dict[str, Any], cli_args: dict[str, Any]) -> None:
    if not cli_args:
        return
    command = container.get("command")
    if not isinstance(command, list) or not command or not isinstance(command[-1], str):
        raise SystemExit("candidate cli_args require a container command script")
    marker = '                "${EXTRA_ARGS[@]}" \\'
    script = command[-1]
    if marker not in script:
        raise SystemExit("candidate cli_args require an ${EXTRA_ARGS[@]} marker in the manifest command")
    injected = "\n".join(cli_arg_lines(cli_args))
    command[-1] = script.replace(marker, f"{injected}\n{marker}", 1)


def pod_template_for(doc: dict[str, Any]) -> dict[str, Any] | None:
    kind = str(doc.get("kind", ""))
    if kind in {"Job", "Deployment", "StatefulSet", "DaemonSet"}:
        template = doc.get("spec", {}).get("template")
        return template if isinstance(template, dict) else None
    if kind == "Pod":
        return doc
    return None


def pod_template_requests_gpu(template: dict[str, Any]) -> bool:
    containers = template.get("spec", {}).get("containers")
    if not isinstance(containers, list):
        return False
    for container in containers:
        if not isinstance(container, dict):
            continue
        resources = container.get("resources") or {}
        for bucket in ("limits", "requests"):
            values = resources.get(bucket) or {}
            if isinstance(values, dict) and "nvidia.com/gpu" in values:
                return True
    return False


def apply_k8s_overrides(doc: dict[str, Any], candidate: dict[str, Any], *, slug: str, candidate_id: str) -> None:
    k8s = candidate.get("k8s") or {}
    app_label = str(k8s.get("app_label") or f"zorl-ar-{slug}")
    kind = str(doc.get("kind", ""))
    metadata = doc.setdefault("metadata", {})
    metadata.setdefault("labels", {})

    if kind == "Service":
        if k8s.get("service_name"):
            metadata["name"] = str(k8s["service_name"])
        metadata["labels"]["app"] = app_label
        metadata["labels"]["experiment-group"] = "zorl"
        metadata["labels"]["autoresearch-id"] = slug
        selector = doc.setdefault("spec", {}).setdefault("selector", {})
        selector["app"] = app_label
        return

    if kind in {"Deployment", "StatefulSet"}:
        if k8s.get("workload_name"):
            metadata["name"] = str(k8s["workload_name"])
        if k8s.get("replicas") is not None:
            doc.setdefault("spec", {})["replicas"] = int(k8s["replicas"])
        selector = doc.setdefault("spec", {}).setdefault("selector", {}).setdefault("matchLabels", {})
        selector["app"] = app_label
        if kind == "StatefulSet" and k8s.get("service_name"):
            doc.setdefault("spec", {})["serviceName"] = str(k8s["service_name"])

    if kind in {"Job", "Pod"}:
        metadata.pop("name", None)
        metadata["generateName"] = str(k8s.get("job_generate_name") or f"zorl-ar-{slug}-")

    metadata["labels"].update(
        {
            "experiment-group": "zorl",
            "autoresearch-id": slug,
            "app": app_label,
        }
    )
    metadata.setdefault("annotations", {})
    metadata["annotations"].update(
        {
            "autoresearch.together.ai/id": candidate_id,
            "autoresearch.together.ai/rendered_utc": utc_now(),
        }
    )

    template = pod_template_for(doc)
    if template is None:
        return
    pod_meta = template.setdefault("metadata", {})
    pod_meta.setdefault("labels", {})
    pod_meta["labels"].update(
        {
            "experiment-group": "zorl",
            "autoresearch-id": slug,
            "app": app_label,
        }
    )
    if pod_template_requests_gpu(template):
        pod_meta["labels"]["team"] = "turbo"
    pod_spec = template.setdefault("spec", {})
    if k8s.get("node_name"):
        pod_spec["nodeName"] = str(k8s["node_name"])
        if not k8s.get("node_hostname") and not k8s.get("node_selector"):
            pod_spec.pop("nodeSelector", None)
    if k8s.get("node_hostname"):
        if not k8s.get("node_selector"):
            pod_spec["nodeSelector"] = {}
        pod_spec.setdefault("nodeSelector", {})["kubernetes.io/hostname"] = str(k8s["node_hostname"])
    if k8s.get("node_selector"):
        pod_spec.setdefault("nodeSelector", {}).update(
            {str(key): str(value) for key, value in dict(k8s["node_selector"]).items()}
        )


def container_for_doc(doc: dict[str, Any], container_name: str | None) -> dict[str, Any] | None:
    template = pod_template_for(doc)
    if template is None:
        return None
    containers = template.setdefault("spec", {}).get("containers")
    if not isinstance(containers, list) or not containers:
        return None
    if container_name:
        for item in containers:
            if item.get("name") == container_name:
                return item
        raise SystemExit(f"container {container_name!r} not found in {doc.get('kind')} manifest")
    return containers[0]


def render_manifest(candidate_path: Path, candidate: dict[str, Any], *, output: Path | None = None) -> Path:
    raw_manifest = candidate.get("base_manifest")
    if not raw_manifest:
        raise SystemExit(f"{candidate_path}: candidate has no base_manifest")
    manifest_path = _resolve_path(str(raw_manifest), bases=(candidate_path.parent, ZORL_DIR, REPO_ROOT))
    docs = [
        doc
        for doc in yaml.safe_load_all(manifest_path.read_text(encoding="utf-8"))
        if doc is not None
    ]
    if not docs or not all(isinstance(doc, dict) for doc in docs):
        raise SystemExit(f"{manifest_path}: manifest must contain YAML mapping documents")

    candidate_id = str(candidate.get("id") or candidate_path.stem)
    slug = slugify(candidate_id)
    for doc in docs:
        apply_k8s_overrides(doc, candidate, slug=slug, candidate_id=candidate_id)
        doc.setdefault("metadata", {}).setdefault("annotations", {})[
            "autoresearch.together.ai/candidate"
        ] = str(candidate_path)

        container = container_for_doc(doc, candidate.get("container"))
        if container is None:
            continue
        env = {
            "AUTORESEARCH_ID": candidate_id,
            "AUTORESEARCH_CANDIDATE": str(candidate_path),
        }
        env.update(candidate.get("env") or {})
        merge_env(container, env)
        inject_cli_args(container, candidate.get("cli_args") or {})
        quote_env_values(container)

    if output is None:
        RENDER_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        output = RENDER_DIR / f"{stamp}-{slug}.yaml"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump_all(docs, sort_keys=False), encoding="utf-8")
    return output


def result_root_for_candidate(candidate: dict[str, Any]) -> Path:
    raw = candidate.get("local_result_root") or candidate.get("result_root")
    if not raw:
        return DEFAULT_RESULT_ROOT
    return _resolve_path(str(raw), bases=(REPO_ROOT, ZORL_DIR, ROOT))


def iter_result_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    files: list[Path] = []
    files.extend(Path(p) for p in glob.glob(str(path / "**" / "summary.json"), recursive=True))
    for name in LOG_NAMES:
        files.extend(Path(p) for p in glob.glob(str(path / "**" / name), recursive=True))
    return [item for item in files if item.is_file()]


def choose_latest(files: list[Path]) -> Path | None:
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def resolve_result(arg: str, *, data: dict[str, Any] | None = None, idea_id: str | None = None) -> Path:
    if arg != "latest":
        path = Path(arg)
        if path.is_dir():
            latest = choose_latest(iter_result_files(path))
            if latest is None:
                raise SystemExit(f"no result files under {path}")
            return latest
        if not path.exists():
            raise SystemExit(f"result not found: {path}")
        return path

    roots: list[Path] = []
    if data is not None and idea_id:
        candidate_path, candidate = candidate_for_idea(data, idea_by_id(data, idea_id))
        _ = candidate_path
        roots.append(result_root_for_candidate(candidate))
    elif data is not None:
        for idea in data["ideas"]:
            try:
                _candidate_path, candidate = candidate_for_idea(data, idea)
            except Exception:
                continue
            roots.append(result_root_for_candidate(candidate))
    roots.append(DEFAULT_RESULT_ROOT)

    files: list[Path] = []
    for root in dict.fromkeys(roots):
        files.extend(iter_result_files(root))
    latest = choose_latest(files)
    if latest is None:
        roots_text = ", ".join(str(root) for root in roots)
        raise SystemExit(f"no result files under {roots_text}")
    return latest


KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^,\s]+)")
COUNT_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([+-]?(?:\d+(?:\.\d*)?|\.\d+))/(\d+)")


def compact_number(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def parse_kv_fragment(fragment: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in KV_RE.findall(fragment):
        cleaned = value.strip().strip(",")
        if cleaned in {"True", "False"}:
            out[key] = cleaned == "True"
        else:
            number = f(cleaned, default=float("nan"))
            out[key] = cleaned if math.isnan(number) else number
    for key, num, den in COUNT_RE.findall(fragment):
        out[f"{key}_num"] = compact_number(float(num))
        out[f"{key}_den"] = int(den)
    return out


def parse_countdown_log(path: Path, candidate: dict[str, Any] | None = None) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    generation_rows: list[dict[str, Any]] = []
    parent_rows: list[dict[str, Any]] = []
    final_exact: int | None = None
    final_total: int | None = None
    sampler_exact: int | None = None
    sampler_total: int | None = None
    best_parent: dict[str, Any] | None = None

    for line in text.splitlines():
        generation = re.search(r"Generation\s+(\d+)/(\d+):\s+(.*)", line)
        if generation:
            row = {
                "generation": int(generation.group(1)),
                "planned_generations": int(generation.group(2)),
                **parse_kv_fragment(generation.group(3)),
            }
            generation_rows.append(row)
            continue

        parent = re.search(
            r"Parent probe\s+(\d+):\s+exact=(\d+)/(\d+),\s+reward_mean=([+-]?[0-9.eE-]+)",
            line,
        )
        if parent:
            parsed = parse_kv_fragment(line)
            parent_rows.append(
                {
                    **parsed,
                    "generation": int(parent.group(1)),
                    "exact": int(parent.group(2)),
                    "total": int(parent.group(3)),
                    "reward_mean": float(parent.group(4)),
                }
            )
            continue

        best = re.search(
            r"Best parent probe:\s+generation=(\d+),\s+exact=(\d+)/(\d+),\s+reward_mean=([+-]?[0-9.eE-]+)",
            line,
        )
        if best:
            best_parent = {
                "generation": int(best.group(1)),
                "exact": int(best.group(2)),
                "total": int(best.group(3)),
                "reward_mean": float(best.group(4)),
            }
            continue

        sampler = re.search(r"Sampler-LoRA score:\s+(\d+)/(\d+)", line)
        if sampler:
            sampler_exact = int(sampler.group(1))
            sampler_total = int(sampler.group(2))
            continue

        result = re.search(r"Result:\s+(\d+)/(\d+)\s+\[(PASS|FAIL)\]", line)
        if result:
            final_exact = int(result.group(1))
            final_total = int(result.group(2))

    gates = (candidate or {}).get("score_gates") or {}
    baseline_exact = int(gates.get("baseline_exact", 1))
    weak_exact = int(gates.get("weak_exact", baseline_exact + 1))
    promote_exact = int(gates.get("promote_exact", max(weak_exact + 1, 3)))
    strong_exact = int(gates.get("strong_exact", max(promote_exact + 2, 5)))
    min_generations = int(gates.get("min_generations", 1))

    parent_best_exact = None
    parent_best_total = None
    if best_parent is not None:
        parent_best_exact = int(best_parent["exact"])
        parent_best_total = int(best_parent["total"])
    elif parent_rows:
        best_row = max(parent_rows, key=lambda row: (f(row["exact"]), f(row.get("reward_mean"))))
        parent_best_exact = int(best_row["exact"])
        parent_best_total = int(best_row["total"])

    generation_best_exact = None
    generation_best_total = None
    for row in generation_rows:
        exact = f(row.get("best_candidate_exact_num"), default=float("nan"))
        total = as_int(row.get("best_candidate_exact_den"))
        if math.isnan(exact):
            continue
        if generation_best_exact is None or exact > generation_best_exact:
            generation_best_exact = compact_number(exact)
            generation_best_total = total

    scored_candidates = [
        (final_exact, final_total),
        (sampler_exact, sampler_total),
        (parent_best_exact, parent_best_total),
        (generation_best_exact, generation_best_total),
    ]
    scored_candidates = [(exact, total) for exact, total in scored_candidates if exact is not None]
    if scored_candidates:
        best_exact, best_total = max(scored_candidates, key=lambda item: f(item[0]))
    else:
        best_exact, best_total = None, gates.get("expected_total")

    completed_generations = max((int(row["generation"]) for row in generation_rows), default=0)
    planned_generations = max((int(row["planned_generations"]) for row in generation_rows), default=None)
    update_norms = [f(row.get("update_norm")) for row in generation_rows if "update_norm" in row]
    pair_deltas = [f(row.get("pair_delta_mean")) for row in generation_rows if "pair_delta_mean" in row]
    rollout_rates = [f(row.get("rollout_exact_rate")) for row in generation_rows if "rollout_exact_rate" in row]

    fatal_error = "Traceback (most recent call last)" in text or "  Result: ERROR" in text
    sync_failed = "SYNC FAILED" in text

    metrics = {
        "result_file": str(path),
        "completed_generations": completed_generations,
        "planned_generations": planned_generations,
        "final_exact": final_exact,
        "final_total": final_total,
        "sampler_exact": sampler_exact,
        "sampler_total": sampler_total,
        "best_parent_exact": parent_best_exact,
        "best_parent_total": parent_best_total,
        "best_candidate_exact": generation_best_exact,
        "best_candidate_total": generation_best_total,
        "best_exact": best_exact,
        "best_total": best_total,
        "baseline_exact": baseline_exact,
        "exact_gain": None if best_exact is None else f(best_exact) - baseline_exact,
        "generation_rows": len(generation_rows),
        "parent_probe_rows": len(parent_rows),
        "update_norm_positive_rows": sum(1 for value in update_norms if value > 0.0),
        "update_norm_last": update_norms[-1] if update_norms else None,
        "pair_delta_abs_mean": (
            sum(abs(value) for value in pair_deltas) / len(pair_deltas) if pair_deltas else None
        ),
        "rollout_exact_rate_max": max(rollout_rates) if rollout_rates else None,
        "fatal_error": fatal_error,
        "sync_failed": sync_failed,
    }

    if fatal_error and best_exact is None:
        verdict = "infra_invalid"
        reason = "harness ended with ERROR before any usable score"
    elif sync_failed and final_exact is None and parent_best_exact is None:
        verdict = "infra_invalid"
        reason = "final sync failed before sampler/parent score was available"
    elif best_exact is None:
        verdict = "incomplete"
        reason = "no final, sampler, parent, or candidate exact score found"
    elif completed_generations < min_generations and f(best_exact) < strong_exact:
        verdict = "incomplete"
        reason = f"completed_generations={completed_generations} < {min_generations}"
    elif f(best_exact) >= strong_exact:
        verdict = "strong_zorl_signal"
        reason = f"best exact {best_exact}/{best_total} passed strong gate {strong_exact}"
    elif f(best_exact) >= promote_exact:
        verdict = "promote_retest"
        reason = f"best exact {best_exact}/{best_total} passed promote gate {promote_exact}"
    elif f(best_exact) >= weak_exact or f(best_exact) > baseline_exact:
        verdict = "weak_zorl_signal"
        reason = f"best exact {best_exact}/{best_total} improved over baseline {baseline_exact}"
    elif final_exact is not None or completed_generations >= min_generations:
        verdict = "science_reject"
        reason = f"best exact {best_exact}/{best_total} did not beat baseline {baseline_exact}"
    else:
        verdict = "inconclusive"
        reason = "usable partial score but run did not meet completion gates"

    return {
        "result": str(path),
        "run_dir": str(path.parent),
        "score_source": "countdown_log",
        "verdict": verdict,
        "reason": reason,
        "metrics": metrics,
    }


def parse_standalone_zorl_log(path: Path, candidate: dict[str, Any] | None = None) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    step_rows: list[dict[str, Any]] = []
    cold_probe: dict[str, Any] | None = None
    best_probe: dict[str, Any] | None = None
    last_probe: dict[str, Any] | None = None
    run_metadata: dict[str, Any] = {}

    for line in text.splitlines():
        init = re.search(r"\[init\]\s+task=([^\s]+)\s+infer_url=([^\s]+)", line)
        if init:
            run_metadata["task"] = init.group(1)
            run_metadata["infer_url"] = init.group(2)
        for key in ("run_id", "run_dir", "task", "infer_url", "model_path", "adapter_dir", "seed"):
            prefix = f"{key}="
            if line.startswith(prefix):
                run_metadata[key] = line.split("=", 1)[1].strip()
        if line.startswith("recipe:"):
            run_metadata.update(parse_kv_fragment(line))

        cold = re.search(
            r"cold:\s+reward_mean=([+-]?[0-9.eE-]+)\s+exact_rate=([+-]?[0-9.eE-]+)\s+"
            r"exact_count=([+-]?[0-9.eE-]+)/(\d+)",
            line,
        )
        if cold:
            cold_probe = {
                "reward_mean": float(cold.group(1)),
                "exact_rate": float(cold.group(2)),
                "exact_count": float(cold.group(3)),
                "total": int(cold.group(4)),
            }
            continue

        step = re.search(r"step\s+(\d+)/(\d+):\s+(.*)", line)
        if step:
            row = {
                "step": int(step.group(1)),
                "planned_steps": int(step.group(2)),
                **parse_kv_fragment(step.group(3)),
            }
            step_rows.append(row)
            if "probe_reward" in row:
                last_probe = {
                    "step": int(row["step"]),
                    "reward_mean": f(row.get("probe_reward")),
                    "exact_rate": f(row.get("exact_rate")),
                    "exact_count": None,
                    "total": None,
                }
            continue

        done = re.search(
            r"\[done\]\s+best probe:\s+reward_mean=([+-]?[0-9.eE-]+)\s+"
            r"exact_count=([+-]?[0-9.eE-]+)/(\d+)",
            line,
        )
        if done:
            best_probe = {
                "reward_mean": float(done.group(1)),
                "exact_count": float(done.group(2)),
                "total": int(done.group(3)),
            }

    gates = (candidate or {}).get("score_gates") or {}
    min_steps = int(gates.get("min_steps", gates.get("min_generations", 1)))
    min_update_norm_positive_rows = int(gates.get("min_update_norm_positive_rows", 1))
    weak_exact_gain = float(gates.get("weak_exact_gain", 1.0))
    promote_exact_gain = float(gates.get("promote_exact_gain", max(weak_exact_gain + 1.0, 2.0)))
    strong_exact_gain = float(gates.get("strong_exact_gain", max(promote_exact_gain + 2.0, 4.0)))
    weak_reward_gain = float(gates.get("weak_reward_gain", 0.01))
    promote_reward_gain = float(gates.get("promote_reward_gain", max(weak_reward_gain, 0.05)))
    strong_reward_gain = float(gates.get("strong_reward_gain", max(promote_reward_gain, 0.10)))
    require_best_probe_after_positive_update = bool(gates.get("require_best_probe_after_positive_update", False))

    completed_steps = max((int(row["step"]) for row in step_rows), default=0)
    planned_steps = max((int(row["planned_steps"]) for row in step_rows), default=None)
    update_norms = [f(row.get("update_norm")) for row in step_rows if "update_norm" in row]
    pair_delta_means = [f(row.get("pair_delta_mean")) for row in step_rows if "pair_delta_mean" in row]
    pair_delta_stds = [f(row.get("pair_delta_std")) for row in step_rows if "pair_delta_std" in row]
    unclipped_update_norms = [f(row.get("unclipped_update_norm")) for row in step_rows if "unclipped_update_norm" in row]
    update_clip_scales = [f(row.get("update_clip_scale")) for row in step_rows if "update_clip_scale" in row]
    used_pairs = [f(row.get("used_pairs")) for row in step_rows if "used_pairs" in row]
    best_candidate_rewards = [f(row.get("best_cand")) for row in step_rows if "best_cand" in row]
    mean_candidate_rewards = [f(row.get("reward_mean")) for row in step_rows if "reward_mean" in row]
    probe_rewards = [f(row.get("probe_reward")) for row in step_rows if "probe_reward" in row]

    step_probes = [
        {
            "step": int(row["step"]),
            "reward_mean": f(row.get("probe_reward")),
            "exact_rate": f(row.get("exact_rate")),
            "exact_count": None,
            "total": gates.get("expected_total"),
        }
        for row in step_rows
        if "probe_reward" in row
    ]
    for probe in step_probes:
        if probe["exact_rate"] is not None and probe["total"] is not None:
            probe["exact_count"] = f(probe["exact_rate"]) * f(probe["total"])
    if last_probe is not None and not any(probe.get("step") == last_probe.get("step") for probe in step_probes):
        step_probes.append(last_probe)
    candidates_for_best_probe = [probe for probe in [*step_probes, best_probe] if probe is not None]
    if candidates_for_best_probe:
        best_probe = max(candidates_for_best_probe, key=lambda probe: f(probe.get("reward_mean")))
    if best_probe is not None and best_probe.get("step") is None:
        matching_step_probe = next(
            (
                probe
                for probe in step_probes
                if probe.get("reward_mean") is not None
                and abs(f(probe.get("reward_mean")) - f(best_probe.get("reward_mean"))) < 1e-9
            ),
            None,
        )
        if matching_step_probe is not None:
            best_probe = {**best_probe, "step": matching_step_probe.get("step")}

    cold_reward = f(cold_probe.get("reward_mean") if cold_probe else None)
    cold_exact = f(cold_probe.get("exact_count") if cold_probe else None)
    best_reward = f(best_probe.get("reward_mean") if best_probe else None)
    best_exact = best_probe.get("exact_count") if best_probe else None
    best_total = best_probe.get("total") if best_probe else gates.get("expected_total")
    exact_gain = None if best_exact is None or cold_probe is None else float(best_exact) - cold_exact
    reward_gain = None if best_probe is None or cold_probe is None else best_reward - cold_reward

    fatal_error = "Traceback (most recent call last)" in text or "RuntimeError:" in text
    update_norm_positive_rows = sum(1 for value in update_norms if value > 0.0)
    run_complete = planned_steps is not None and completed_steps >= planned_steps
    candidate_env = (candidate or {}).get("env") or {}
    update_norm_by_step = {
        int(row["step"]): f(row.get("update_norm"))
        for row in step_rows
        if row.get("step") is not None and row.get("update_norm") is not None
    }
    best_probe_step = best_probe.get("step") if best_probe is not None else None
    best_probe_update_norm = update_norm_by_step.get(int(best_probe_step)) if best_probe_step is not None else None
    best_probe_after_positive_update = best_probe_update_norm is not None and best_probe_update_norm > 0.0

    metrics = {
        "result_file": str(path),
        "task": run_metadata.get("task") or candidate_env.get("TASK"),
        "infer_url": run_metadata.get("infer_url") or candidate_env.get("INFER_URL"),
        "seed": candidate_env.get("SEED") or run_metadata.get("seed"),
        "lora_rank": candidate_env.get("LORA_RANK") or run_metadata.get("rank"),
        "num_pairs": candidate_env.get("NUM_PAIRS") or run_metadata.get("pairs"),
        "b_sigma": candidate_env.get("B_SIGMA") or run_metadata.get("sigma"),
        "learning_rate": candidate_env.get("LEARNING_RATE") or run_metadata.get("lr"),
        "max_update_norm": candidate_env.get("MAX_UPDATE_NORM") or run_metadata.get("max_update_norm"),
        "perturbation_mode": candidate_env.get("PERTURBATION_MODE") or run_metadata.get("perturbation_mode"),
        "score_mode": candidate_env.get("SCORE_MODE") or run_metadata.get("score_mode"),
        "completed_steps": completed_steps,
        "planned_steps": planned_steps,
        "cold_reward_mean": cold_reward if cold_probe is not None else None,
        "cold_exact": cold_probe.get("exact_count") if cold_probe else None,
        "cold_total": cold_probe.get("total") if cold_probe else None,
        "best_reward_mean": best_reward if best_probe is not None else None,
        "best_exact": best_exact,
        "best_total": best_total,
        "best_probe_step": best_probe_step,
        "best_probe_update_norm": best_probe_update_norm,
        "best_probe_after_positive_update": best_probe_after_positive_update,
        "exact_gain": exact_gain,
        "reward_gain": reward_gain,
        "step_rows": len(step_rows),
        "probe_rows": len(probe_rewards),
        "update_norm_positive_rows": update_norm_positive_rows,
        "pair_delta_abs_mean": (
            sum(abs(value) for value in pair_delta_means) / len(pair_delta_means) if pair_delta_means else None
        ),
        "pair_delta_std_last": pair_delta_stds[-1] if pair_delta_stds else None,
        "update_norm_last": update_norms[-1] if update_norms else None,
        "unclipped_update_norm_last": unclipped_update_norms[-1] if unclipped_update_norms else None,
        "update_clip_scale_last": update_clip_scales[-1] if update_clip_scales else None,
        "used_pairs_last": used_pairs[-1] if used_pairs else None,
        "best_candidate_reward_max": max(best_candidate_rewards) if best_candidate_rewards else None,
        "candidate_reward_mean_last": mean_candidate_rewards[-1] if mean_candidate_rewards else None,
        "fatal_error": fatal_error,
        "run_complete": run_complete,
    }

    if fatal_error and best_probe is None:
        verdict = "infra_invalid"
        reason = "standalone client ended with an error before any usable probe"
    elif best_probe is None:
        verdict = "incomplete"
        reason = "no standalone probe score found"
    elif completed_steps < min_steps:
        verdict = "incomplete"
        reason = f"completed_steps={completed_steps} < {min_steps}"
    elif update_norm_positive_rows < min_update_norm_positive_rows:
        verdict = "infra_invalid" if planned_steps is None or run_complete else "incomplete"
        reason = (
            f"update_norm_positive_rows={update_norm_positive_rows} "
            f"< {min_update_norm_positive_rows}"
        )
    elif require_best_probe_after_positive_update and not best_probe_after_positive_update:
        verdict = "science_reject" if planned_steps is None or run_complete else "incomplete"
        reason = "best_probe_after_positive_update=false"
    elif (exact_gain is not None and exact_gain >= strong_exact_gain) or (
        reward_gain is not None and reward_gain >= strong_reward_gain
    ):
        verdict = "strong_zorl_signal"
        reason = f"standalone reward_gain={reward_gain} exact_gain={exact_gain} passed strong gate"
    elif (exact_gain is not None and exact_gain >= promote_exact_gain) or (
        reward_gain is not None and reward_gain >= promote_reward_gain
    ):
        verdict = "promote_retest"
        reason = f"standalone reward_gain={reward_gain} exact_gain={exact_gain} passed promote gate"
    elif (exact_gain is not None and exact_gain >= weak_exact_gain) or (
        reward_gain is not None and reward_gain >= weak_reward_gain
    ):
        verdict = "weak_zorl_signal"
        reason = f"standalone reward_gain={reward_gain} exact_gain={exact_gain} passed weak gate"
    elif planned_steps is None or run_complete:
        verdict = "science_reject"
        reason = f"standalone reward_gain={reward_gain} exact_gain={exact_gain} did not pass weak gate"
    else:
        verdict = "incomplete"
        reason = f"completed_steps={completed_steps} < planned_steps={planned_steps}"

    return {
        "result": str(path),
        "run_dir": str(path.parent),
        "score_source": "standalone_zorl_log",
        "verdict": verdict,
        "reason": reason,
        "metrics": metrics,
    }


def score_smoke_summary(path: Path, candidate: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records") or []
    exports = data.get("exports") or []
    num_sessions = int(data.get("num_sessions") or 0)
    steps = int(data.get("steps") or 0)
    expected_records = num_sessions * steps if num_sessions and steps else None
    gates = (candidate or {}).get("score_gates") or {}
    min_records = int(gates.get("min_records", expected_records or 1))
    update_norms = [f(record.get("update_norm")) for record in records]
    pair_deltas = [f(record.get("pair_delta_mean")) for record in records]

    zorl_exports_ok = True
    for export in exports:
        weights_info = export.get("weights_info") or {}
        zorl_config = weights_info.get("zorl_config") or {}
        if not zorl_config.get("enabled"):
            zorl_exports_ok = False

    metrics = {
        "result_file": str(path),
        "run_id": data.get("run_id"),
        "num_sessions": num_sessions,
        "steps": steps,
        "records": len(records),
        "expected_records": expected_records,
        "exports": len(exports),
        "zorl_exports_ok": zorl_exports_ok,
        "update_norm_positive_rows": sum(1 for value in update_norms if value > 0.0),
        "update_norm_min": min(update_norms) if update_norms else None,
        "update_norm_max": max(update_norms) if update_norms else None,
        "pair_delta_abs_mean": (
            sum(abs(value) for value in pair_deltas) / len(pair_deltas) if pair_deltas else None
        ),
    }

    required_records = max(min_records, expected_records or 0)
    if len(records) < required_records:
        verdict = "incomplete"
        reason = f"records={len(records)} < required_records={required_records}"
    elif not zorl_exports_ok or not exports:
        verdict = "infra_invalid"
        reason = "missing exported ZORL checkpoint metadata"
    elif metrics["update_norm_positive_rows"] < len(records):
        verdict = "infra_invalid"
        reason = "at least one smoke generation produced a zero update norm"
    else:
        verdict = "smoke_pass"
        reason = "all smoke records, updates, and exported ZORL metadata are present"

    return {
        "result": str(path),
        "run_dir": str(path.parent.parent if path.name == "summary.json" else path.parent),
        "score_source": "smoke_summary",
        "verdict": verdict,
        "reason": reason,
        "metrics": metrics,
    }


def score_result(path: Path, candidate: dict[str, Any] | None = None) -> dict[str, Any]:
    score_source = (candidate or {}).get("score_source")
    if score_source == "smoke_summary" or path.name == "summary.json":
        return score_smoke_summary(path, candidate)
    if score_source == "standalone_zorl_log":
        return parse_standalone_zorl_log(path, candidate)
    return parse_countdown_log(path, candidate)


def progress_markers(score: dict[str, Any]) -> dict[str, float]:
    metrics = score.get("metrics") or {}
    markers: dict[str, float] = {}
    for key in (
        "completed_generations",
        "completed_steps",
        "records",
        "parent_probe_rows",
        "probe_rows",
    ):
        value = metrics.get(key)
        if value is None:
            continue
        markers[key] = f(value)
    return markers


def score_is_terminal_or_failed(score: dict[str, Any]) -> bool:
    metrics = score.get("metrics") or {}
    if metrics.get("fatal_error") or metrics.get("sync_failed"):
        return True
    for completed_key, planned_key in (
        ("completed_generations", "planned_generations"),
        ("completed_steps", "planned_steps"),
        ("records", "expected_records"),
    ):
        completed = metrics.get(completed_key)
        planned = metrics.get(planned_key)
        if completed is not None and planned is not None and f(planned) > 0 and f(completed) >= f(planned):
            return True
    if score.get("verdict") == "infra_invalid":
        return True
    return any(metrics.get(key) is not None for key in ("final_exact", "sampler_exact"))


STATUS_BY_VERDICT = {
    "smoke_pass": "complete",
    "strong_zorl_signal": "strong_signal",
    "promote_retest": "promote_retest",
    "weak_zorl_signal": "weak_signal",
    "science_reject": "rejected",
    "infra_invalid": "infra_invalid",
    "incomplete": "incomplete",
    "inconclusive": "inconclusive",
}


def score_needs_more_time(score: dict[str, Any]) -> bool:
    if score_is_terminal_or_failed(score):
        return False
    if str(score.get("verdict")) == "infra_invalid":
        return True
    return str(score.get("verdict")) in {"incomplete", "inconclusive"}


def apply_score_to_idea(
    data: dict[str, Any],
    *,
    idea_id: str,
    result: Path,
    score: dict[str, Any],
    scorecard: Path,
) -> dict[str, Any]:
    idea = idea_by_id(data, idea_id)
    idea["status"] = STATUS_BY_VERDICT.get(str(score["verdict"]), "inconclusive")
    idea["last_verdict"] = score["verdict"]
    idea["last_reason"] = score["reason"]
    idea["last_result"] = str(result)
    idea["last_scorecard"] = str(scorecard)
    idea["last_scored_utc"] = utc_now()
    return idea


def find_new_result(candidate: dict[str, Any], *, since_epoch: float) -> Path | None:
    root = result_root_for_candidate(candidate)
    files = []
    for path in iter_result_files(root):
        try:
            if path.stat().st_mtime >= since_epoch - 5.0:
                files.append(path)
        except FileNotFoundError:
            continue
    return choose_latest(files)


def refresh_result_in_run_dir(result: Path) -> Path:
    if not result.exists() or not result.parent.exists():
        return result
    latest = choose_latest(iter_result_files(result.parent))
    return latest or result


def wait_for_new_result(
    candidate: dict[str, Any],
    *,
    since_epoch: float,
    poll_seconds: float,
    timeout_seconds: float | None,
) -> Path:
    deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
    while True:
        result = find_new_result(candidate, since_epoch=since_epoch)
        if result is not None:
            return result
        if deadline is not None and time.monotonic() >= deadline:
            root = result_root_for_candidate(candidate)
            raise SystemExit(f"timed out waiting for a new result under {root}")
        time.sleep(max(1.0, poll_seconds))


def wait_for_score_ready(
    result: Path,
    candidate: dict[str, Any],
    *,
    poll_seconds: float,
    timeout_seconds: float | None,
) -> tuple[Path, dict[str, Any]]:
    deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
    current_result = result
    while True:
        current_result = refresh_result_in_run_dir(current_result)
        score = score_result(current_result, candidate)
        if not score_needs_more_time(score):
            return current_result, score
        if deadline is not None and time.monotonic() >= deadline:
            return current_result, score
        print(
            "waiting: "
            f"result={current_result} verdict={score['verdict']} "
            f"markers={progress_markers(score)}"
        )
        time.sleep(max(1.0, poll_seconds))


def wait_eval_reasons(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    min_generation_delta: int,
    min_step_delta: int,
    min_record_delta: int,
    wake_on_probe: bool,
) -> list[str]:
    reasons: list[str] = []
    previous_markers = progress_markers(previous)
    current_markers = progress_markers(current)

    deltas = {
        "completed_generations": min_generation_delta,
        "completed_steps": min_step_delta,
        "records": min_record_delta,
    }
    for key, minimum in deltas.items():
        if key not in previous_markers or key not in current_markers:
            continue
        if current_markers[key] >= previous_markers[key] + minimum:
            reasons.append(f"{key} advanced {previous_markers[key]:g}->{current_markers[key]:g}")

    if wake_on_probe:
        for key in ("parent_probe_rows", "probe_rows"):
            if key not in previous_markers or key not in current_markers:
                continue
            if current_markers[key] > previous_markers[key]:
                reasons.append(f"{key} advanced {previous_markers[key]:g}->{current_markers[key]:g}")

    if score_is_terminal_or_failed(current) and not score_is_terminal_or_failed(previous):
        reasons.append(f"terminal_or_failed verdict={current.get('verdict')}")

    return reasons


def write_scorecard(score: dict[str, Any], *, idea_id: str | None) -> tuple[Path, Path]:
    SCORECARD_DIR.mkdir(parents=True, exist_ok=True)
    stem_id = slugify(idea_id or "unknown")
    verdict = slugify(score["verdict"])
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    stem = f"{stamp}-{stem_id}-{verdict}"
    json_path = SCORECARD_DIR / f"{stem}.json"
    md_path = SCORECARD_DIR / f"{stem}.md"
    json_path.write_text(json.dumps(score, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metrics = score.get("metrics", {})
    metric_lines = []
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, (dict, list)):
            continue
        metric_lines.append(f"- {key}: `{value}`")

    lines = [
        f"# {idea_id or 'unknown'} scorecard",
        "",
        f"- verdict: `{score['verdict']}`",
        f"- reason: {score['reason']}",
        f"- result: `{score['result']}`",
        f"- source: `{score['score_source']}`",
        "",
        "## Metrics",
        "",
        *metric_lines,
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def launch_candidate(
    data: dict[str, Any],
    *,
    idea: dict[str, Any] | None,
    candidate_path: Path,
    candidate: dict[str, Any],
    dry_run: bool,
) -> tuple[Path, float]:
    rendered = render_manifest(candidate_path, candidate)
    cmd = ["kubectl", "create", "-f", str(rendered)]
    launch_epoch = time.time()
    if dry_run:
        print(f"rendered={rendered}")
        print(" ".join(shlex.quote(item) for item in cmd))
        return rendered, launch_epoch

    subprocess.run(cmd, check=True)
    if idea is not None:
        idea["status"] = "launched"
        idea["last_launched_utc"] = utc_now()
        idea["last_manifest"] = str(rendered)
        save_yaml(IDEAS_PATH, data)
    append_event(
        {
            "event": "launch",
            "idea_id": idea.get("id") if idea is not None else None,
            "candidate": str(candidate_path),
            "manifest": str(rendered),
        }
    )
    return rendered, launch_epoch


def command_next(args: argparse.Namespace) -> None:
    data = load_yaml(IDEAS_PATH)
    idea = next_idea(data)
    if idea is None:
        print("no queued ideas")
        return
    print(yaml.safe_dump(idea, sort_keys=False).strip())


def command_render(args: argparse.Namespace) -> None:
    data = load_yaml(IDEAS_PATH)
    _idea, candidate_path, candidate = candidate_from_args(
        data,
        idea_id=args.id,
        candidate_arg=args.candidate,
        set_env=args.set_env,
    )
    output = Path(args.output) if args.output else None
    rendered = render_manifest(candidate_path, candidate, output=output)
    print(rendered)


def command_launch(args: argparse.Namespace) -> None:
    data = load_yaml(IDEAS_PATH)
    idea, candidate_path, candidate = candidate_from_args(
        data,
        idea_id=args.id,
        candidate_arg=args.candidate,
        set_env=args.set_env,
    )
    launch_candidate(data, idea=idea, candidate_path=candidate_path, candidate=candidate, dry_run=args.dry_run)


def command_monitor(args: argparse.Namespace) -> None:
    data = load_yaml(IDEAS_PATH)
    result = resolve_result(args.result, data=data, idea_id=args.idea_id)
    if args.json:
        idea = idea_by_id(data, args.idea_id) if args.idea_id else None
        candidate = candidate_for_idea(data, idea)[1] if idea else None
        print(json.dumps(score_result(result, candidate), indent=2, sort_keys=True))
        return
    if result.name == "summary.json":
        payload = json.loads(result.read_text(encoding="utf-8"))
        print(
            f"{result}: records={len(payload.get('records') or [])} "
            f"exports={len(payload.get('exports') or [])} run_id={payload.get('run_id')}"
        )
        return
    lines = result.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[-args.tail :]:
        print(line)


def command_score(args: argparse.Namespace) -> None:
    data = load_yaml(IDEAS_PATH)
    idea = idea_by_id(data, args.idea_id) if args.idea_id else None
    candidate = candidate_for_idea(data, idea)[1] if idea else None
    result = resolve_result(args.result, data=data, idea_id=args.idea_id)
    score = score_result(result, candidate)
    json_path, md_path = write_scorecard(score, idea_id=args.idea_id)
    append_event({"event": "score", "idea_id": args.idea_id, "score": score, "scorecard": str(json_path)})
    if args.json:
        print(json.dumps(score, indent=2, sort_keys=True))
        return
    print(f"verdict={score['verdict']} reason={score['reason']}")
    print(f"scorecard_json={json_path}")
    print(f"scorecard_md={md_path}")


def command_wait_eval(args: argparse.Namespace) -> None:
    data = load_yaml(IDEAS_PATH)
    idea = idea_by_id(data, args.idea_id) if args.idea_id else None
    candidate = candidate_for_idea(data, idea)[1] if idea else None

    result = resolve_result(args.result, data=data, idea_id=args.idea_id)
    baseline = score_result(result, candidate)
    deadline = None if args.timeout_seconds is None else time.monotonic() + args.timeout_seconds
    if score_is_terminal_or_failed(baseline):
        print(
            json.dumps(
                {
                    "event": "eval_ready",
                    "idea_id": args.idea_id,
                    "reasons": [f"already_terminal_or_failed verdict={baseline.get('verdict')}"],
                    "previous_markers": progress_markers(baseline),
                    "current_markers": progress_markers(baseline),
                    "result": str(result),
                    "score": baseline,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    while True:
        if deadline is not None and time.monotonic() >= deadline:
            print(
                json.dumps(
                    {
                        "event": "timeout",
                        "idea_id": args.idea_id,
                        "result": str(result),
                        "score": baseline,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            raise SystemExit(124)

        time.sleep(max(1.0, args.poll_seconds))
        current_result = resolve_result(args.result, data=data, idea_id=args.idea_id)
        current = score_result(current_result, candidate)
        reasons = wait_eval_reasons(
            baseline,
            current,
            min_generation_delta=args.min_generation_delta,
            min_step_delta=args.min_step_delta,
            min_record_delta=args.min_record_delta,
            wake_on_probe=args.wake_on_probe,
        )
        if reasons:
            payload = {
                "event": "eval_ready",
                "idea_id": args.idea_id,
                "reasons": reasons,
                "previous_markers": progress_markers(baseline),
                "current_markers": progress_markers(current),
                "result": str(current_result),
                "score": current,
            }
            append_event(
                {
                    "event": "eval_ready",
                    "idea_id": args.idea_id,
                    "reasons": reasons,
                    "result": str(current_result),
                    "verdict": current.get("verdict"),
                    "metrics": current.get("metrics"),
                }
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return


def command_run_loop(args: argparse.Namespace) -> None:
    launched_total = 0
    while True:
        data = load_yaml(IDEAS_PATH)
        remaining = None if args.max_runs == 0 else max(args.max_runs - launched_total, 0)
        if remaining == 0:
            return

        if args.id:
            if launched_total > 0:
                return
            batch = [idea_by_id(data, args.id)]
        else:
            limit = max(1, args.parallel)
            if remaining is not None:
                limit = min(limit, remaining)
            batch = ready_ideas(data)[:limit]

        if not batch:
            print("no queued ideas")
            return

        active: list[tuple[str, dict[str, Any], Path, float]] = []
        for idea in batch:
            candidate_path, candidate = candidate_for_idea(data, idea)
            candidate = apply_env_override(candidate, args.set_env)
            idea_id = str(idea.get("id"))
            print(f"launching {idea_id}: {candidate_path}")
            _rendered, launch_epoch = launch_candidate(
                data,
                idea=idea,
                candidate_path=candidate_path,
                candidate=candidate,
                dry_run=args.dry_run,
            )
            launched_total += 1
            if not args.dry_run:
                active.append((idea_id, candidate, candidate_path, launch_epoch))

        if args.dry_run:
            return

        for idea_id, candidate, _candidate_path, launch_epoch in active:
            print(f"waiting for result for {idea_id}")
            result = wait_for_new_result(
                candidate,
                since_epoch=launch_epoch,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.result_timeout_seconds,
            )
            print(f"monitoring {idea_id}: {result}")
            result, score = wait_for_score_ready(
                result,
                candidate,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.eval_timeout_seconds,
            )
            json_path, md_path = write_scorecard(score, idea_id=idea_id)
            data = load_yaml(IDEAS_PATH)
            idea = apply_score_to_idea(data, idea_id=idea_id, result=result, score=score, scorecard=json_path)
            save_yaml(IDEAS_PATH, data)
            append_event(
                {
                    "event": "run_loop_advance",
                    "idea_id": idea_id,
                    "score": score,
                    "status": idea["status"],
                    "scorecard": str(json_path),
                }
            )
            print(f"{idea_id}: {idea['status']} ({score['verdict']})")
            print(f"scorecard_json={json_path}")
            print(f"scorecard_md={md_path}")

        if args.id:
            return
        if args.max_runs != 0 and launched_total >= args.max_runs:
            return


def command_advance(args: argparse.Namespace) -> None:
    data = load_yaml(IDEAS_PATH)
    idea = idea_by_id(data, args.id)
    candidate = candidate_for_idea(data, idea)[1]
    result = resolve_result(args.result, data=data, idea_id=args.id)
    score = score_result(result, candidate)
    json_path, _ = write_scorecard(score, idea_id=args.id)
    idea = apply_score_to_idea(data, idea_id=args.id, result=result, score=score, scorecard=json_path)
    save_yaml(IDEAS_PATH, data)
    append_event({"event": "advance", "idea_id": args.id, "score": score, "status": idea["status"]})
    print(f"{args.id}: {idea['status']} ({score['verdict']})")


def command_append_idea(args: argparse.Namespace) -> None:
    data = load_yaml(IDEAS_PATH)
    if any(str(idea.get("id")) == args.id for idea in data["ideas"]):
        raise SystemExit(f"idea {args.id!r} already exists")
    idea = {
        "id": args.id,
        "status": "queued",
        "priority": args.priority,
        "candidate": args.candidate,
        "hypothesis": args.hypothesis,
    }
    if args.rationale:
        idea["rationale"] = args.rationale
    data["ideas"].append(idea)
    save_yaml(IDEAS_PATH, data)
    append_event({"event": "append_idea", "idea": idea})
    print(f"appended {args.id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    next_cmd = sub.add_parser("next")
    next_cmd.set_defaults(func=command_next)

    render = sub.add_parser("render")
    render.add_argument("--id")
    render.add_argument("--candidate")
    render.add_argument("--set-env", action="append", default=[])
    render.add_argument("--output")
    render.set_defaults(func=command_render)

    launch = sub.add_parser("launch")
    launch.add_argument("--id")
    launch.add_argument("--candidate")
    launch.add_argument("--set-env", action="append", default=[])
    launch.add_argument("--dry-run", action="store_true")
    launch.set_defaults(func=command_launch)

    monitor = sub.add_parser("monitor")
    monitor.add_argument("--result", default="latest")
    monitor.add_argument("--idea-id")
    monitor.add_argument("--tail", type=int, default=80)
    monitor.add_argument("--json", action="store_true")
    monitor.set_defaults(func=command_monitor)

    score = sub.add_parser("score")
    score.add_argument("--result", default="latest")
    score.add_argument("--idea-id")
    score.add_argument("--json", action="store_true")
    score.set_defaults(func=command_score)

    wait_eval = sub.add_parser("wait-eval")
    wait_eval.add_argument("--result", default="latest")
    wait_eval.add_argument("--idea-id")
    wait_eval.add_argument("--poll-seconds", type=float, default=60.0)
    wait_eval.add_argument("--timeout-seconds", type=float)
    wait_eval.add_argument("--min-generation-delta", type=int, default=5)
    wait_eval.add_argument("--min-step-delta", type=int, default=5)
    wait_eval.add_argument("--min-record-delta", type=int, default=1)
    wait_eval.add_argument("--wake-on-probe", action=argparse.BooleanOptionalAction, default=True)
    wait_eval.set_defaults(func=command_wait_eval)

    run_loop = sub.add_parser("run-loop")
    run_loop.add_argument("--id")
    run_loop.add_argument("--set-env", action="append", default=[])
    run_loop.add_argument("--parallel", type=int, default=1)
    run_loop.add_argument("--max-runs", type=int, default=1, help="0 means keep launching until the queue is empty")
    run_loop.add_argument("--poll-seconds", type=float, default=60.0)
    run_loop.add_argument("--result-timeout-seconds", type=float, default=1800.0)
    run_loop.add_argument("--eval-timeout-seconds", type=float)
    run_loop.add_argument("--dry-run", action="store_true")
    run_loop.set_defaults(func=command_run_loop)

    advance = sub.add_parser("advance")
    advance.add_argument("--id", required=True)
    advance.add_argument("--result", default="latest")
    advance.set_defaults(func=command_advance)

    append_idea = sub.add_parser("append-idea")
    append_idea.add_argument("--id", required=True)
    append_idea.add_argument("--candidate", required=True)
    append_idea.add_argument("--hypothesis", required=True)
    append_idea.add_argument("--rationale")
    append_idea.add_argument("--priority", type=int, default=50)
    append_idea.set_defaults(func=command_append_idea)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
