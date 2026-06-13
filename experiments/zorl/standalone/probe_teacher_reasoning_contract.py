"""Does the target-conditioned policy_hint TEACHER produce off-contract (target-revealing)
reasoning? OPSD KL over the student's <reasoning> tokens is only legitimate if the teacher
distribution there agrees with the public-reasoning contract (no private-target mentions).

Two measurements on the BASE 30B (no LoRA) over mid-game states:
  (1) GENERATION: sample the teacher prompt; does its <reasoning> mention the target / rely
      on oracle knowledge? (qualitative; the prompt explicitly forbids it, so this tests
      whether the instruction actually holds.)
  (2) LOGPROB CONTRAST (the precise test you suggested): teacher logprob of
        <reasoning>The private target is TARGET.</reasoning>
      vs
        <reasoning>Two candidates remain, so choose TARGET.</reasoning>
      Higher mass on the first => the reasoning-token KL is pulling the student off-contract
      => mask reasoning KL / train the action (guess) first.

Run ON a replica:
  kubectl exec zorl-ar-sglang-15 -n apanda -- \
    /workspace/home/xorl-sglang-internal/.venv/bin/python \
    /workspace/home/xorl-apanda-dev-zorl-consolidated/experiments/zorl/standalone/probe_teacher_reasoning_contract.py
"""
import sys

import requests
from transformers import AutoTokenizer


sys.path.insert(0, "/workspace/home/xorl-apanda-dev-zorl-consolidated/experiments/zorl/standalone")
import tasks.wordle as w  # noqa: E402


URL = "http://localhost:30000"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")


def teacher_prompt_ids(target, history):
    msgs = [
        {"role": "system", "content": w.PUBLIC_REASONING_SYSTEM_PROMPT},
        {"role": "user", "content": w._policy_hint_teacher_content(
            target=target, history=history, response_style="public_reasoning")},
    ]
    return tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                   enable_thinking=False, return_dict=False)


def gen(input_ids, temp=0.7, n=3, max_new=96):
    body = {"input_ids": list(input_ids), "sampling_params": {"temperature": temp, "max_new_tokens": max_new, "n": n}}
    r = requests.post(f"{URL}/generate", json=body, timeout=180).json()
    if isinstance(r, dict):
        r = [r]
    return [x.get("text", "") or "" for x in r]


def continuation_logprob(prompt_ids, cont_text):
    """Sum logprob the teacher assigns to `cont_text` appended after prompt_ids (prefill)."""
    cont_ids = tok.encode(cont_text, add_special_tokens=False)
    full = list(prompt_ids) + cont_ids
    body = {
        "input_ids": full,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
        "return_logprob": True,
        "logprob_start_len": len(prompt_ids),
    }
    r = requests.post(f"{URL}/generate", json=body, timeout=180).json()
    if isinstance(r, list):
        r = r[0]
    lps = (r.get("meta_info", {}) or {}).get("input_token_logprobs") or []
    vals = [t[0] for t in lps if t and t[0] is not None]
    total = sum(vals)
    return total, (total / max(len(vals), 1)), len(vals)


# A couple of mid-game states (target + public history).
states = [
    ("canon", [("stare", "BBBBB"), ("could", "BYBBB")]),
    ("brave", [("stare", "BYBYB"), ("crane", "BYYYB")]),
    ("plumb", [("stare", "BBBBB"), ("could", "BBYBB")]),
]

for target, hist in states:
    cands = w.remaining_candidates(hist)
    pid = teacher_prompt_ids(target, hist)
    print(f"\n========== target={target.upper()}  history={hist}  remaining={len(cands)} ==========")
    print("--- TEACHER generated <reasoning> (n=3, temp=0.7) ---")
    for i, t in enumerate(gen(pid)):
        leak = "  <<< MENTIONS TARGET" if target.upper() in t.upper() else ""
        print(f"  [{i}] {t!r}{leak}")
    leak_text = f"<reasoning>The private target is {target.upper()}.</reasoning>"
    pub_text = f"<reasoning>Two candidates remain, so choose {target.upper()}.</reasoning>"
    lt, la, ln = continuation_logprob(pid, leak_text)
    pt, pa, pn = continuation_logprob(pid, pub_text)
    print(f"--- LOGPROB CONTRAST (teacher prefers higher) ---")
    print(f"  leak   {leak_text!r}\n         sum={lt:.2f} avg/tok={la:.3f} ntok={ln}")
    print(f"  public {pub_text!r}\n         sum={pt:.2f} avg/tok={pa:.3f} ntok={pn}")
    print(f"  => teacher prefers {'LEAK (off-contract!)' if la > pa else 'PUBLIC (on-contract)'} by avg/tok {abs(la - pa):.3f}")
