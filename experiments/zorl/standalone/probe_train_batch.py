"""Greedy exact-match probe of the CURRENT parent on the STAT64 fixed train batch.

Disambiguates the stationary-64 run's loss descent: train accuracy rising while
the held-out probe stays flat = memorization (the stationary-8 failure mode);
both rising = ES found a generalizing direction.

Read-only against the serving pool (same machinery as the client's held-out
probe: probe_parent, temp=0, n=1). Usage:
  python3 probe_train_batch.py --parent-lora-name <session>/parent [--n 64]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("HF_HOME", "/shared/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import zorl_client  # noqa: E402


POOL_C_URLS = [
    f"http://zorl-ar-sglang-c-{i}.zorl-ar-sglang-c-headless.apanda.svc.cluster.local:30000"
    for i in range(8)
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-lora-name", default=None, help="omit to probe the served BASE model")
    ap.add_argument("--urls", nargs="*", default=None, help="override pool URLs (default: pool C)")
    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--task", default="mult")
    ap.add_argument("--seed", type=int, default=9234)
    ap.add_argument("--train-pool-size", type=int, default=512)
    ap.add_argument("--eval-size", type=int, default=128)
    ap.add_argument("--n", type=int, default=64, help="fixed train batch size (train_pool[:n])")
    args = ap.parse_args()

    task = zorl_client.load_task(args.task)
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_pool, _ = task.build_examples(
        tokenizer, train_size=args.train_pool_size, eval_size=args.eval_size, seed=args.seed
    )
    batch = list(train_pool[: args.n])

    probe_args = argparse.Namespace(
        task=args.task,
        score_mode="sft",
        probe_temperature=0.0,
        probe_n=1,
        rollout_temperature=0.0,
        rollout_max_new_tokens=64,
        rollouts_per_puzzle=1,
        score_max_workers=32,
        score_max_workers_per_owner=0,
        opsd_multi_turn=False,
        opsd_student_ignore_eos=True,
        invalid_retries=0,
    )
    result = zorl_client.probe_parent(
        args.urls if args.urls else POOL_C_URLS,
        parent_lora_name=args.parent_lora_name,
        examples=batch,
        task=task,
        args=probe_args,
        tokenizer=tokenizer,
    )
    print(
        f"TRAIN-BATCH PROBE n={len(batch)}: exact_rate={result.get('exact_match_rate')} "
        f"exact_count={result.get('exact_count')} reward_mean={result.get('reward_mean')}"
    )


if __name__ == "__main__":
    main()
