#!/usr/bin/env python3
"""Build index-aligned, nonempty 5x5 prompt+CoT files for Q3.5-35B-A3B self-distill,
and measure the teacher's CoT correctness (= the achievable OPD ceiling).

Source: /shared/opd-coord/randnum_5digit_81920_q35_35b_cot_mt16384.json
  (81920 entries; ~32% are finish_reason='error:' with empty cot -> would hard-abort
   the trainer, so we keep only nonempty finish_reason='stop').
Writes:
  /shared/opd-coord/randnum_5digit_q35_nonempty_prompts.json  (list of message-lists)
  /shared/opd-coord/randnum_5digit_q35_nonempty_cot.json      (list of {prompt,cot,...})
"""
import json
import re

SRC = "/shared/opd-coord/randnum_5digit_81920_q35_35b_cot_mt16384.json"
OUT_PROMPTS = "/shared/opd-coord/randnum_5digit_q35_nonempty_prompts.json"
OUT_COT = "/shared/opd-coord/randnum_5digit_q35_nonempty_cot.json"

INT_RE = re.compile(r"[-+]?[0-9][0-9,]*")
MULT_RE = re.compile(r"(\d+)\s*\*\s*(\d+)")


def prompt_text(entry):
    p = entry["prompt"]
    if isinstance(p, list):
        return " ".join(m.get("content", "") for m in p)
    return str(p)


def ground_truth(entry):
    m = MULT_RE.search(prompt_text(entry))
    if not m:
        return None
    return int(m.group(1)) * int(m.group(2))


def teacher_answer(cot):
    # Prefer the number after the last 'Answer:'; else the last integer in the text.
    idx = cot.rfind("Answer:")
    region = cot[idx:] if idx != -1 else cot
    nums = INT_RE.findall(region)
    if not nums:
        nums = INT_RE.findall(cot)
    if not nums:
        return None
    try:
        return int(nums[-1].replace(",", ""))
    except ValueError:
        return None


def main():
    data = json.load(open(SRC))
    kept_prompts, kept_cot = [], []
    correct = 0
    lens = []
    for e in data:
        cot = (e.get("cot") or "").strip()
        if e.get("finish_reason") != "stop" or not cot:
            continue
        kept_prompts.append(e["prompt"])
        kept_cot.append(e)
        lens.append(int(e.get("tokens") or 0))
        gt = ground_truth(e)
        ta = teacher_answer(cot)
        if gt is not None and ta is not None and gt == ta:
            correct += 1
    n = len(kept_cot)
    json.dump(kept_prompts, open(OUT_PROMPTS, "w"))
    json.dump(kept_cot, open(OUT_COT, "w"))
    lens.sort()
    print(f"source entries: {len(data)}")
    print(f"kept (nonempty stop): {n}")
    print(f"teacher CoT correctness (the OPD ceiling): {correct}/{n} = {correct / n:.3f}")
    if lens:
        print(f"CoT token len: min={lens[0]} median={lens[n // 2]} p95={lens[int(n * 0.95)]} max={lens[-1]}")
        print(f"  -> match_cot K cap (<= min CoT len): {lens[0]}")
    print(f"wrote {OUT_PROMPTS}")
    print(f"wrote {OUT_COT}")


if __name__ == "__main__":
    main()
