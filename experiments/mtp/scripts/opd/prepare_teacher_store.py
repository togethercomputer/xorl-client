#!/usr/bin/env python3
from __future__ import annotations

import argparse

from xorl.distillation import prepare_lm_head_teacher_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a row-sharded OPD teacher LM-head store.")
    parser.add_argument(
        "--model-path", required=True, help="HF model directory or safetensors file containing lm_head.weight"
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write manifest.json and shard files")
    parser.add_argument("--teacher-id", default="0", help="Teacher id recorded in the manifest")
    parser.add_argument("--tensor-key", default="lm_head.weight", help="Tensor key to extract from safetensors")
    parser.add_argument("--shard-rows", type=int, default=32768, help="Rows per lm_head shard")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing manifest")
    args = parser.parse_args()

    manifest = prepare_lm_head_teacher_store(
        args.model_path,
        args.output_dir,
        teacher_id=args.teacher_id,
        tensor_key=args.tensor_key,
        shard_rows=args.shard_rows,
        force=args.force,
    )
    print(manifest)


if __name__ == "__main__":
    main()
