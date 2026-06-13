"""Sample the current parent LoRA (the trained student) on public-reasoning Wordle
turn prompts and print the generated <reasoning>...</reasoning><guess>...</guess>.
Run ON a replica (has the tokenizer cache + localhost SGLang):

  kubectl exec zorl-ar-sglang-0 -n apanda -- \
    /workspace/home/xorl-sglang-internal/.venv/bin/python \
    /workspace/home/xorl-apanda-dev-zorl-consolidated/experiments/zorl/standalone/inspect_cots.py \
    "<parent_lora_name>"
"""
import sys

import requests
from transformers import AutoTokenizer

sys.path.insert(0, "/workspace/home/xorl-apanda-dev-zorl-consolidated/experiments/zorl/standalone")
import tasks.wordle as w  # noqa: E402

PARENT = sys.argv[1] if len(sys.argv) > 1 else None
URL = "http://localhost:30000"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")


def gen(input_ids, lora, temp=0.7, n=2):
    body = {
        "input_ids": list(input_ids),
        "sampling_params": {"temperature": temp, "max_new_tokens": 96, "n": n},
    }
    if lora is not None:
        body["lora_path"] = lora
    try:
        r = requests.post(f"{URL}/generate", json=body, timeout=180).json()
    except Exception as e:
        return [f"<error: {e}>"]
    if isinstance(r, dict):
        r = [r]
    return [x.get("text", "") for x in r]


scenarios = [
    ("lemon", [], "TURN 1 (no feedback)"),
    ("lemon", [("stare", "BBBBB"), ("cloud", "BBBYB")], "MID-GAME (2 prior guesses + feedback)"),
]
for target, hist, label in scenarios:
    ids = w._build_turn_input_ids(tok, target=target, history=hist, prompt_style="public_reasoning")
    cands = w.remaining_candidates(hist)
    print(f"\n========== {label}  target={target}  remaining_candidates={len(cands)} ==========")
    print("--- PARENT (trained student) ---")
    for i, t in enumerate(gen(ids, PARENT)):
        print(f"  [{i}] {t!r}")
    print("--- BASE (no LoRA) ---")
    for i, t in enumerate(gen(ids, None)):
        print(f"  [{i}] {t!r}")
