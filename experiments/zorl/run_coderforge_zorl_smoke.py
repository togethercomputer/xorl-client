"""Run a dataset-backed synthetic-reward ZORL full-loop smoke test against a live training server."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
import torch
from datasets import load_from_disk
from safetensors.torch import load_file as safetensors_load_file


@dataclass(frozen=True)
class GenerationRecord:
    model_id: str
    session_index: int
    generation: int
    family_id: str
    family_refreshed: bool
    reward_mean: float
    pair_delta_mean: float
    update_norm: float
    learning_rate: float


class TrainingServerClient:
    """Minimal client for the server's async REST API."""

    def __init__(
        self,
        base_url: str,
        *,
        submit_timeout: float = 30.0,
        future_timeout: float = 1800.0,
        future_poll_interval: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.submit_timeout = submit_timeout
        self.future_timeout = future_timeout
        self.future_poll_interval = future_poll_interval
        self.session = requests.Session()

    def close(self) -> None:
        self.session.close()

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        response = self.session.request(
            method,
            f"{self.base_url}{path}",
            json=payload,
            timeout=timeout or self.submit_timeout,
        )
        response.raise_for_status()
        return response.json()

    def _post_json(self, path: str, payload: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        return self._request_json("POST", path, payload=payload, timeout=timeout)

    def _get_json(self, path: str, *, timeout: float | None = None) -> dict[str, Any]:
        return self._request_json("GET", path, timeout=timeout)

    def wait_for_service(self, *, timeout: float = 1800.0, poll_interval: float = 3.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                payload = self._get_json("/health", timeout=5.0)
                if payload.get("engine_running"):
                    return
            except Exception:
                pass
            time.sleep(poll_interval)
        raise TimeoutError(f"Training server at {self.base_url} did not become healthy within {timeout:.0f}s")

    def wait_for_future(self, request_id: str, *, context: str) -> dict[str, Any]:
        deadline = time.time() + self.future_timeout
        while time.time() < deadline:
            result = self._post_json(
                "/api/v1/retrieve_future",
                {"request_id": request_id},
                timeout=min(120.0, self.future_timeout),
            )
            result_type = result.get("type")
            if result_type == "try_again":
                time.sleep(self.future_poll_interval)
                continue
            if result_type == "request_failed" or ("error" in result and "category" in result):
                raise RuntimeError(f"{context} failed: {result.get('error', result)}")
            return result
        raise TimeoutError(f"{context} future {request_id} timed out after {self.future_timeout:.0f}s")

    def _submit_async(self, path: str, payload: dict[str, Any], *, context: str) -> dict[str, Any]:
        future = self._post_json(path, payload)
        request_id = future.get("request_id")
        if not request_id:
            raise RuntimeError(f"{context} did not return a request_id: {future}")
        return self.wait_for_future(request_id, context=context)

    def session_info(self) -> dict[str, Any]:
        return self._get_json("/api/v1/session_info", timeout=self.submit_timeout)

    def create_model(
        self,
        *,
        model_id: str,
        base_model: str,
        lora_rank: int,
        lora_alpha: int,
        learning_rate: float,
        zorl_b_sigma: float,
        zorl_num_pairs: int,
        zorl_refresh_interval: int,
        zorl_seed: int,
    ) -> dict[str, Any]:
        return self._submit_async(
            "/api/v1/create_model",
            {
                "model_id": model_id,
                "base_model": base_model,
                "lora_config": {
                    "rank": lora_rank,
                    "lora_rank": lora_rank,
                    "alpha": lora_alpha,
                    "lora_alpha": lora_alpha,
                },
                "optimizer_config": {
                    "type": "sgd",
                    "learning_rate": learning_rate,
                    "weight_decay": 0.0,
                    "optimizer_dtype": "fp32",
                },
                "zorl_config": {
                    "enabled": True,
                    "b_sigma": zorl_b_sigma,
                    "num_perturbation_pairs": zorl_num_pairs,
                    "a_refresh_interval": zorl_refresh_interval,
                    "antithetic_sampling": True,
                    "a_init": "gaussian_jl",
                    "seed": zorl_seed,
                },
            },
            context=f"create_model({model_id})",
        )

    def start_zorl_generation(self, *, model_id: str) -> dict[str, Any]:
        return self._submit_async(
            "/api/v1/zorl/start_generation",
            {"model_id": model_id},
            context=f"start_zorl_generation({model_id})",
        )

    def apply_zorl_rewards(
        self,
        *,
        model_id: str,
        generation_id: str,
        candidate_rewards: list[dict[str, Any]],
        learning_rate: float,
    ) -> dict[str, Any]:
        return self._submit_async(
            "/api/v1/zorl/apply_rewards",
            {
                "model_id": model_id,
                "generation_id": generation_id,
                "candidate_rewards": candidate_rewards,
                "learning_rate": learning_rate,
            },
            context=f"apply_zorl_rewards({model_id}, {generation_id})",
        )

    def abort_zorl_generation(self, *, model_id: str, generation_id: str) -> dict[str, Any]:
        return self._submit_async(
            "/api/v1/zorl/abort_generation",
            {
                "model_id": model_id,
                "generation_id": generation_id,
            },
            context=f"abort_zorl_generation({model_id}, {generation_id})",
        )

    def save_weights(self, *, model_id: str, path: str) -> dict[str, Any]:
        return self._submit_async(
            "/api/v1/save_weights",
            {
                "model_id": model_id,
                "path": path,
            },
            context=f"save_weights({model_id}, {path})",
        )

    def save_weights_for_sampler(self, *, model_id: str, name: str) -> dict[str, Any]:
        return self._submit_async(
            "/api/v1/save_weights_for_sampler",
            {
                "model_id": model_id,
                "name": name,
            },
            context=f"save_weights_for_sampler({model_id}, {name})",
        )

    def weights_info(self, *, xorl_path: str) -> dict[str, Any]:
        return self._post_json(
            "/api/v1/weights_info",
            {
                "xorl_path": xorl_path,
            },
        )

    def unload_model(self, *, model_id: str) -> dict[str, Any]:
        return self._submit_async(
            "/api/v1/unload_model",
            {
                "model_id": model_id,
            },
            context=f"unload_model({model_id})",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-url", required=True, help="Training server base URL")
    parser.add_argument("--model", required=True, help="Base model path/name")
    parser.add_argument("--output-dir", required=True, help="Directory for smoke-test artifacts")
    parser.add_argument(
        "--server-output-dir", required=True, help="Training server output_dir used to resolve sampler URIs"
    )
    parser.add_argument(
        "--dataset-path",
        default="/home/apanda/xorl-internal/experiments/local_benchmark/dataset_cache/prepared_dataset_tail20480_filtered_reward1",
        help="Prepared filtered_reward1 dataset path or parent directory",
    )
    parser.add_argument("--run-id", default=None, help="Stable run identifier")
    parser.add_argument("--num-sessions", type=int, default=2, help="How many LoRA sessions to create")
    parser.add_argument("--steps", type=int, default=2, help="ZORL generations to run per session")
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Examples per generation for synthetic reward aggregation"
    )
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Parent LoRA SGD learning rate")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--zorl-b-sigma", type=float, default=0.01, help="ZORL LoRA-B perturbation scale")
    parser.add_argument("--zorl-num-pairs", type=int, default=4, help="ZORL perturbation pairs per generation")
    parser.add_argument(
        "--zorl-refresh-interval",
        type=int,
        default=8,
        help="Generations between LoRA-A family refreshes",
    )
    parser.add_argument("--zorl-seed", type=int, default=1234, help="Base ZORL RNG seed")
    parser.add_argument("--future-timeout", type=float, default=1800.0, help="Seconds to wait for async futures")
    parser.add_argument("--health-timeout", type=float, default=1800.0, help="Seconds to wait for service health")
    return parser.parse_args()


def _is_prepared_dataset_dir(path: Path) -> bool:
    return path.is_dir() and (path / "dataset_info.json").exists() and (path / "state.json").exists()


def resolve_prepared_dataset_path(path: Path) -> Path | None:
    """Resolve a prepared dataset root or a specific fingerprint directory."""
    if _is_prepared_dataset_dir(path):
        return path
    if not path.exists():
        return None

    candidates = [child for child in sorted(path.iterdir()) if _is_prepared_dataset_dir(child)]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    preferred: list[Path] = []
    for candidate in candidates:
        dataset_info_path = candidate / "dataset_info.json"
        data = json.loads(dataset_info_path.read_text(encoding="utf-8"))
        checksums = data.get("download_checksums", {})
        if any(str(key).startswith("/home/apanda/xorl-internal/") for key in checksums):
            preferred.append(candidate)

    if preferred:
        return sorted(preferred)[-1]
    return max(candidates, key=lambda item: item.stat().st_mtime)


def build_synthetic_dataset(*, num_rows: int, seq_len: int = 32) -> list[dict[str, list[int]]]:
    """Build a tiny deterministic token dataset for smoke runs without a cache."""
    rows: list[dict[str, list[int]]] = []
    for row_index in range(num_rows):
        base_token = 100 + row_index * seq_len
        input_ids = [10 + ((base_token + offset) % 10000) for offset in range(seq_len)]
        rows.append(
            {
                "input_ids": input_ids,
                "labels": input_ids[1:] + [input_ids[-1]],
                "position_ids": list(range(seq_len)),
            }
        )
    return rows


def _to_int_list(values: list[Any]) -> list[int]:
    return [int(value) for value in values]


def build_reward_batch(dataset: Any, *, session_index: int, generation: int, batch_size: int) -> list[dict[str, Any]]:
    """Build a deterministic batch used to synthesize candidate rewards."""
    dataset_size = len(dataset)
    if dataset_size == 0:
        raise ValueError("Prepared dataset is empty")

    batch: list[dict[str, Any]] = []
    base_index = (session_index * batch_size) + (generation * batch_size)
    for offset in range(batch_size):
        row = dataset[(base_index + offset) % dataset_size]
        model_input = {"input_ids": _to_int_list(row["input_ids"])}
        if "position_ids" in row:
            model_input["position_ids"] = _to_int_list(row["position_ids"])
        batch.append(
            {
                "model_input": model_input,
                "loss_fn_inputs": {
                    "labels": _to_int_list(row["labels"]),
                },
            }
        )
    return batch


def mix_seed(base_seed: int, *components: int) -> int:
    """Mix a stable integer seed from generation metadata."""
    acc = int(base_seed) & 0x7FFFFFFFFFFFFFFF
    for index, component in enumerate(components, start=1):
        acc = (acc * 6364136223846793005 + int(component) * 1442695040888963407 + index) & 0x7FFFFFFFFFFFFFFF
    return acc


def stable_example_seed(example: dict[str, Any]) -> int:
    """Build a stable 64-bit seed from one reward example."""
    payload = json.dumps(example, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def candidate_uri_to_path(model_path: str, server_output_dir: Path) -> Path:
    """Resolve a candidate adapter URI or relative path to a local directory."""
    if model_path.startswith("xorl://"):
        parts = model_path[7:].split("/", 2)
        if len(parts) != 3:
            raise ValueError(f"Unsupported candidate URI: {model_path}")
        _model_id, checkpoint_type, checkpoint_name = parts
        return server_output_dir / checkpoint_type / checkpoint_name
    if model_path.startswith("sampler_weights/"):
        return server_output_dir / model_path
    return Path(model_path)


def load_lora_b_vectors(candidate_dir: Path) -> list[torch.Tensor]:
    """Load flattened LoRA-B tensors from an exported adapter checkpoint."""
    weights_path = candidate_dir / "adapter_model.safetensors"
    tensors = safetensors_load_file(str(weights_path))
    lora_bs = [
        tensor.float().reshape(-1).cpu()
        for name, tensor in sorted(tensors.items())
        if name.endswith("lora_B") or name.endswith("lora_B.weight") or name.endswith("lora_embedding_B")
    ]
    if not lora_bs:
        sample_keys = ", ".join(sorted(tensors.keys())[:8])
        raise RuntimeError(f"No LoRA-B tensors found in {weights_path}; sample keys: {sample_keys}")
    return lora_bs


def synthetic_reward_mean(
    *,
    candidate_dir: Path,
    reward_batch: list[dict[str, Any]],
    reward_seed: int,
    probes_per_tensor: int = 4,
) -> float:
    """Compute a deterministic synthetic reward from sampled LoRA-B coordinates."""
    lora_bs = load_lora_b_vectors(candidate_dir)
    reward_values: list[float] = []

    for example_index, example in enumerate(reward_batch):
        example_seed = mix_seed(reward_seed, stable_example_seed(example), example_index)
        total = 0.0
        total_probes = 0

        for tensor_index, flat_values in enumerate(lora_bs):
            tensor_size = int(flat_values.numel())
            if tensor_size == 0:
                continue
            for probe_index in range(probes_per_tensor):
                probe_seed = mix_seed(example_seed, tensor_index, probe_index)
                idx = probe_seed % tensor_size
                sign = 1.0 if ((probe_seed >> 8) & 1) else -1.0
                total += sign * float(flat_values[idx].item())
                total_probes += 1

        if total_probes == 0:
            raise RuntimeError(f"No synthetic probes were produced for {candidate_dir}")
        reward_values.append(total / float(total_probes))

    return sum(reward_values) / float(len(reward_values))


def expected_family_refresh(generation: int, refresh_interval: int) -> bool:
    """Return whether generation planning should refresh LoRA-A for this generation."""
    return generation == 0 or (refresh_interval > 0 and generation % refresh_interval == 0)


def best_effort_unload(client: TrainingServerClient, model_id: str) -> None:
    try:
        client.unload_model(model_id=model_id)
    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response is not None else None
        if status_code != 404:
            print(f"[cleanup] failed to unload {model_id}: {error}")
    except Exception as error:
        print(f"[cleanup] failed to unload {model_id}: {error}")


def build_markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# ZORL Smoke Run",
        "",
        "## Configuration",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- train_url: `{summary['train_url']}`",
        f"- dataset_path: `{summary['dataset_path']}`",
        f"- dataset_rows: `{summary['dataset_rows']}`",
        f"- num_sessions: `{summary['num_sessions']}`",
        f"- generations: `{summary['steps']}`",
        f"- batch_size: `{summary['batch_size']}`",
        "",
        "## Session Exports",
        "",
        "| model_id | checkpoint_path | sampler_path | b_sigma | num_pairs | seed |",
        "|---|---|---|---:|---:|---:|",
    ]

    for export in summary["exports"]:
        zorl_config = export["weights_info"]["zorl_config"]
        lines.append(
            "| "
            f"{export['model_id']} | "
            f"`{export['checkpoint_path']}` | "
            f"`{export['sampler_path']}` | "
            f"{zorl_config['b_sigma']} | "
            f"{zorl_config['num_perturbation_pairs']} | "
            f"{zorl_config['seed']} |"
        )

    lines.extend(
        [
            "",
            "## Generation Metrics",
            "",
            "| model_id | generation | family_id | refreshed | reward_mean | pair_delta_mean | update_norm | learning_rate |",
            "|---|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for record in summary["records"]:
        lines.append(
            "| "
            f"{record['model_id']} | "
            f"{record['generation']} | "
            f"`{record['family_id']}` | "
            f"{record['family_refreshed']} | "
            f"{record['reward_mean']:.6f} | "
            f"{record['pair_delta_mean']:.6f} | "
            f"{record['update_norm']:.6f} | "
            f"{record['learning_rate']:.6f} |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.run_id = args.run_id or time.strftime("zorl-smoke-%Y%m%dT%H%M%SZ", time.gmtime())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    server_output_dir = Path(args.server_output_dir)

    dataset_path = resolve_prepared_dataset_path(Path(args.dataset_path))
    if dataset_path is None:
        dataset_rows = max(args.num_sessions * args.steps * args.batch_size, 4)
        dataset = build_synthetic_dataset(num_rows=dataset_rows)
        dataset_label = f"synthetic://missing/{args.dataset_path}"
        print(f"[dataset] cache missing at {args.dataset_path}; using deterministic synthetic rows={len(dataset)}")
    else:
        dataset = load_from_disk(str(dataset_path))
        dataset_label = str(dataset_path)
        print(f"[dataset] using {dataset_path} rows={len(dataset)}")

    client = TrainingServerClient(args.train_url, future_timeout=args.future_timeout)
    records: list[GenerationRecord] = []
    exports: list[dict[str, Any]] = []
    model_ids = [f"{args.run_id}-session-{session_index:02d}" for session_index in range(args.num_sessions)]
    active_generations: dict[str, str] = {}
    session_info_before: dict[str, Any] = {}
    session_info_after: dict[str, Any] = {}

    try:
        client.wait_for_service(timeout=args.health_timeout)
        session_info_before = client.session_info()

        for session_index, model_id in enumerate(model_ids):
            zorl_seed = args.zorl_seed + session_index
            print(f"[create] model_id={model_id} zorl_seed={zorl_seed}")
            client.create_model(
                model_id=model_id,
                base_model=args.model,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                learning_rate=args.learning_rate,
                zorl_b_sigma=args.zorl_b_sigma,
                zorl_num_pairs=args.zorl_num_pairs,
                zorl_refresh_interval=args.zorl_refresh_interval,
                zorl_seed=zorl_seed,
            )

        for generation in range(args.steps):
            for session_index, model_id in enumerate(model_ids):
                reward_batch = build_reward_batch(
                    dataset,
                    session_index=session_index,
                    generation=generation,
                    batch_size=args.batch_size,
                )
                generation_result = client.start_zorl_generation(model_id=model_id)
                generation_id = generation_result["generation_id"]
                active_generations[model_id] = generation_id

                refreshed_expected = expected_family_refresh(generation, args.zorl_refresh_interval)
                if bool(generation_result["family_refreshed"]) != refreshed_expected:
                    raise RuntimeError(
                        f"Unexpected family refresh state for {model_id} generation {generation}: "
                        f"got {generation_result['family_refreshed']} expected {refreshed_expected}"
                    )

                reward_seed = mix_seed(args.zorl_seed + session_index, generation, 7)
                candidate_rewards: list[dict[str, Any]] = []
                for candidate in generation_result["candidates"]:
                    candidate_dir = candidate_uri_to_path(candidate["model_path"], server_output_dir)
                    reward_mean = synthetic_reward_mean(
                        candidate_dir=candidate_dir,
                        reward_batch=reward_batch,
                        reward_seed=reward_seed,
                    )
                    candidate_rewards.append(
                        {
                            "candidate_id": candidate["candidate_id"],
                            "reward_mean": reward_mean,
                            "num_rollouts": len(reward_batch),
                        }
                    )

                apply_result = client.apply_zorl_rewards(
                    model_id=model_id,
                    generation_id=generation_id,
                    candidate_rewards=candidate_rewards,
                    learning_rate=args.learning_rate,
                )
                active_generations.pop(model_id, None)

                record = GenerationRecord(
                    model_id=model_id,
                    session_index=session_index,
                    generation=generation,
                    family_id=str(apply_result["family_id"]),
                    family_refreshed=bool(generation_result["family_refreshed"]),
                    reward_mean=float(apply_result["metrics"]["reward_mean"]),
                    pair_delta_mean=float(apply_result["metrics"]["pair_delta_mean"]),
                    update_norm=float(apply_result["metrics"]["update_norm"]),
                    learning_rate=float(apply_result["metrics"]["learning_rate"]),
                )
                records.append(record)
                print(
                    "[zorl] "
                    f"model_id={model_id} generation={generation:03d} "
                    f"family={record.family_id} refreshed={record.family_refreshed} "
                    f"reward_mean={record.reward_mean:.6f} "
                    f"pair_delta_mean={record.pair_delta_mean:.6f} "
                    f"update_norm={record.update_norm:.6f}"
                )

        for session_index, model_id in enumerate(model_ids):
            checkpoint_result = client.save_weights(model_id=model_id, path="smoke-ckpt")
            weights_info = client.weights_info(xorl_path=checkpoint_result["path"])
            zorl_config = weights_info.get("zorl_config")
            if not zorl_config or not zorl_config.get("enabled", False):
                raise RuntimeError(f"Checkpoint metadata for {model_id} is missing ZORL config: {weights_info}")
            if int(zorl_config["num_perturbation_pairs"]) != int(args.zorl_num_pairs):
                raise RuntimeError(
                    f"Checkpoint metadata for {model_id} has wrong num_pairs: "
                    f"{zorl_config['num_perturbation_pairs']} != {args.zorl_num_pairs}"
                )
            if int(zorl_config["seed"]) != int(args.zorl_seed + session_index):
                raise RuntimeError(
                    f"Checkpoint metadata for {model_id} has wrong seed: "
                    f"{zorl_config['seed']} != {args.zorl_seed + session_index}"
                )

            sampler_name = f"zorl-smoke/{args.run_id}/{model_id}"
            sampler_result = client.save_weights_for_sampler(model_id=model_id, name=sampler_name)
            exports.append(
                {
                    "model_id": model_id,
                    "checkpoint_path": checkpoint_result["path"],
                    "sampler_path": sampler_result["path"],
                    "weights_info": weights_info,
                }
            )
            print(
                f"[export] model_id={model_id} checkpoint={checkpoint_result['path']} sampler={sampler_result['path']}"
            )

        session_info_after = client.session_info()

    finally:
        for model_id, generation_id in list(active_generations.items()):
            try:
                client.abort_zorl_generation(model_id=model_id, generation_id=generation_id)
            except Exception as error:
                print(f"[cleanup] failed to abort active generation {generation_id}: {error}")
        for model_id in model_ids:
            best_effort_unload(client, model_id)
        client.close()

    summary = {
        "run_id": args.run_id,
        "train_url": args.train_url,
        "dataset_path": dataset_label,
        "dataset_rows": len(dataset),
        "num_sessions": args.num_sessions,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "session_info_before": session_info_before,
        "session_info_after": session_info_after,
        "records": [asdict(record) for record in records],
        "exports": exports,
    }

    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"
    summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary_md_path.write_text(build_markdown_summary(summary), encoding="utf-8")

    print(f"[done] wrote {summary_json_path}")
    print(f"[done] wrote {summary_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
