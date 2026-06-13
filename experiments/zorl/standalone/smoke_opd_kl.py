"""Smoke-test the full-vocab OPSD-KL engine path on one restarted SGLang replica.

Validates end-to-end: capture_hidden_mode=FULL capture -> teacher cache ->
streaming full-vocab KL -> meta_info["opd_kl_reward"], including the TP=2 lm_head
gather. Uses raw token ids (no tokenizer needed) under the BASE model; the teacher
and student share the same fake CoT suffix but different prefixes, so a finite
non-zero forward KL is expected.

Run from a pod that can reach the replica, e.g. after:
    kubectl port-forward pod/zorl-ar-sglang-0 30000:30000 -n apanda
    python experiments/zorl/standalone/smoke_opd_kl.py http://localhost:30000
"""

import math
import sys

import requests


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:30000"
    cot = list(range(1000, 1040))  # 40 fake CoT tokens (valid ids < vocab)
    n = len(cot)
    teacher_in = list(range(10, 30)) + cot   # "hinted" prefix + CoT
    student_in = list(range(200, 210)) + cot  # "hint-free" prefix + same CoT

    def gen(input_ids, spec):
        body = {
            "input_ids": input_ids,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
            "opd_kl": spec,
        }
        r = requests.post(f"{url}/generate", json=body, timeout=120)
        r.raise_for_status()
        return r.json()

    t = gen(teacher_in, {"role": "teacher_set", "key": "smoke", "cot_len": n, "mode": "forward_kl_full"})
    print("teacher_set ->", t.get("meta_info", {}).get("opd_kl_reward"),
          t.get("meta_info", {}).get("opd_kl_role"))

    s = gen(student_in, {"role": "student", "key": "smoke", "cot_len": n, "mode": "forward_kl_full"})
    rew = s.get("meta_info", {}).get("opd_kl_reward")
    print("student   ->", rew)

    ok = isinstance(rew, list) and len(rew) == 1 and math.isfinite(rew[0])
    print("SMOKE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
