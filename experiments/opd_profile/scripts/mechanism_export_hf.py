#!/usr/bin/env python3
"""MECHANISM-SANDBOX -> production-servable HF export (2026-07-06).

Converts a sandbox checkpoint (mechanism_sandbox_train.py save_ckpt output)
into an sglang-servable HF directory. AFTER this step the eval path is 100%
production: sglang serve -> comp_bench_eval.py [--pause-slots] (35B) /
eval_math_suite.py --prompt-style pause-slots (marin). No sandbox code at
eval time.

--model q35b:
    input : sharded safetensors dir (keys model.* / lm_head.weight, bf16)
    output: base-layout dir — model.* -> model.language_model.*, lm_head kept;
            model.visual.* and mtp.* copied VERBATIM from the base snapshot
            (untrained; required by the HF/sglang load); aux files copied
            incl. preprocessor_config.json + video_preprocessor_config.json
            (the 2026-07-06 converter patch — two tracks crashed on absence:
            "Can't load image processor").
    Mirrors convert_sampler_export_to_hf.py's output contract exactly; that
    script remains the converter for XORL-engine exports; this one is for
    sandbox-trained (already-HF-layout) weights.

--model marin:
    input : single safetensors (MechModel keys: embed_tokens.* /
            layers.N.inner.* / norm.* / lm_head.*; recur_stage0 ckpts are
            also accepted — adapter./state_norm. keys are DROPPED with a
            printed note, [I|0]-identity means base semantics at r=1)
    output: HF-layout dir (model.embed_tokens.* / model.layers.N.* /
            model.norm.* / lm_head.*) + config/tokenizer files copied from
            the RL checkpoint dir.

STRICT audit both ways: output key set must EQUAL the reference checkpoint's
key set (the "Unexpected key" doctrine) or the script raises.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

Q35_BD = ("/shared/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/"
          "995ad96eacd98c81ed38be0c5b274b04031597b0")
MARIN_CKPT = "/shared/xorl-marin-rl-6279/checkpoints/delphi-1e22-p33m67-rl-rlvr7500_w1-think-140-10B"
Q35_AUX = ("config.json", "generation_config.json", "tokenizer.json",
           "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
           "preprocessor_config.json", "video_preprocessor_config.json")
MARIN_AUX = ("config.json", "generation_config.json", "tokenizer.json",
             "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")


class Reader:
    def __init__(self, path):
        p = Path(path)
        self._open = {}
        if p.is_dir():
            self.dir = p
            self.index = json.load(open(p / "model.safetensors.index.json"))["weight_map"]
        else:
            self.dir = p.parent
            f = safe_open(p, framework="pt", device="cpu")
            self.index = {k: p.name for k in f.keys()}
            self._open[p.name] = f

    def keys(self):
        return list(self.index.keys())

    def get(self, key):
        shard = self.index[key]
        if shard not in self._open:
            self._open[shard] = safe_open(self.dir / shard, framework="pt", device="cpu")
        return self._open[shard].get_tensor(key)


def write_sharded(tensors_iter, out: Path, shard_gb: float = 8.0):
    out.mkdir(parents=True, exist_ok=True)
    shard, cur, sid, wmap = {}, 0, 1, {}

    def flush():
        nonlocal shard, cur, sid
        if not shard:
            return
        name = f"model-{sid:05d}.safetensors"
        save_file(shard, out / name)
        for k in shard:
            wmap[k] = name
        print(f"  wrote {name} ({len(shard)} tensors, {cur/1e9:.1f} GB)")
        shard, cur, sid = {}, 0, sid + 1

    for k, t in tensors_iter:
        shard[k] = t
        cur += t.numel() * t.element_size()
        if cur >= shard_gb * 1e9:
            flush()
    flush()
    json.dump({"metadata": {}, "weight_map": wmap},
              open(out / "model.safetensors.index.json", "w"))
    return wmap


def export_q35b(ckpt: str, out_dir: str):
    base = Reader(Q35_BD)
    trained = Reader(ckpt)
    tkeys = set(trained.keys())

    def gen():
        for key in base.keys():
            if key.startswith("model.visual.") or key.startswith("mtp."):
                yield key, base.get(key)
                continue
            src = "lm_head.weight" if key == "lm_head.weight" else key.replace(
                "model.language_model.", "model.", 1)
            if src not in tkeys:
                raise RuntimeError(f"STRICT EXPORT AUDIT: trained ckpt missing {src} "
                                   f"(for base key {key})")
            tkeys.discard(src)
            yield key, trained.get(src).to(torch.bfloat16).contiguous()

    wmap = write_sharded(gen(), Path(out_dir))
    if tkeys:
        raise RuntimeError(f"STRICT EXPORT AUDIT: {len(tkeys)} trained keys unused, "
                           f"e.g. {sorted(tkeys)[:8]}")
    assert set(wmap.keys()) == set(base.keys()), "output key set != base key set"
    for f in Q35_AUX:
        if (Path(Q35_BD) / f).exists():
            shutil.copy(Path(Q35_BD) / f, Path(out_dir) / f)
    assert (Path(out_dir) / "preprocessor_config.json").exists(), \
        "preprocessor_config.json missing — sglang AutoProcessor will crash"
    print(f"DONE q35b: {len(wmap)} tensors -> {out_dir}")


def export_marin(ckpt: str, out_dir: str):
    ref = Reader(MARIN_CKPT)
    trained = Reader(ckpt)
    dropped = []
    mapped = {}
    for k in trained.keys():
        if k.startswith("adapter.") or k.startswith("state_norm."):
            dropped.append(k)
            continue
        if k.startswith("layers.") and ".inner." in k:
            nk = "model." + k.replace(".inner.", ".", 1)
        elif k.startswith("embed_tokens.") or k.startswith("norm."):
            nk = "model." + k
        elif k.startswith("lm_head."):
            nk = k
        else:
            raise RuntimeError(f"unrecognized sandbox key {k}")
        mapped[nk] = k
    if dropped:
        print(f"NOTE: dropped {len(dropped)} recurrence-adapter keys (identity at r=1): "
              f"{dropped[:4]}...")
    refkeys = set(ref.keys())
    if set(mapped.keys()) != refkeys:
        miss = refkeys - set(mapped.keys())
        extra = set(mapped.keys()) - refkeys
        raise RuntimeError(f"STRICT EXPORT AUDIT: missing={sorted(miss)[:8]}"
                           f"({len(miss)}) extra={sorted(extra)[:8]}({len(extra)})")

    def gen():
        for nk in ref.keys():
            yield nk, trained.get(mapped[nk]).to(torch.bfloat16).contiguous()

    write_sharded(gen(), Path(out_dir))
    for f in MARIN_AUX:
        if (Path(MARIN_CKPT) / f).exists():
            shutil.copy(Path(MARIN_CKPT) / f, Path(out_dir) / f)
    print(f"DONE marin: {len(mapped)} tensors -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["marin", "q35b"], required=True)
    ap.add_argument("--ckpt", required=True,
                    help="sandbox safetensors file (marin) or sharded dir (q35b)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if args.model == "q35b":
        export_q35b(args.ckpt, args.out_dir)
    else:
        export_marin(args.ckpt, args.out_dir)


if __name__ == "__main__":
    main()
