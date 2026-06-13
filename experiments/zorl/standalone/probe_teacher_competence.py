"""Teacher-competence / inferability probe for OPSD Wordle.

Plays full greedy (temp=0) games on the BASE 30B (no LoRA) under three info
conditions to locate the solving bottleneck:

  A  student_public      current public_reasoning student (PUBLIC transcript only)
  B  student_with_cands  student + the public consistent-candidate list each turn
  C  teacher_policyhint   the policy_hint teacher prompt (candidate list + tie-break answer)

Reasoning: the policy_hint TEACHER is handed remaining_candidates() in its prompt;
the STUDENT only sees the transcript and cannot enumerate the ~500-word list from
its weights. So "which words are still consistent" — the core inferable skill — is
not reachable from the student's context.

  * B >> A  -> bottleneck is candidate-enumeration; the inferable lever is to
              scaffold the public consistent set into the student (learnable).
  * B ~= A  -> the model can't use the list either (deeper reasoning failure).
  * C       -> privileged upper bound (what current OPSD distills toward).

Run ON a replica (has the tokenizer cache + localhost SGLang):

  kubectl exec zorl-ar-sglang-15 -n apanda -- \
    /workspace/home/xorl-sglang-internal/.venv/bin/python \
    /workspace/home/xorl-apanda-dev-zorl-consolidated/experiments/zorl/standalone/probe_teacher_competence.py 32
"""
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import requests
from transformers import AutoTokenizer


sys.path.insert(0, "/workspace/home/xorl-apanda-dev-zorl-consolidated/experiments/zorl/standalone")
import tasks.wordle as w  # noqa: E402


N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
URL = "http://localhost:30000"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")


def gen(input_ids) -> str:
    body = {"input_ids": list(input_ids), "sampling_params": {"temperature": 0.0, "max_new_tokens": 128}}
    try:
        r = requests.post(f"{URL}/generate", json=body, timeout=240).json()
    except Exception as e:  # noqa: BLE001
        return f"<error: {e}>"
    if isinstance(r, list):
        r = r[0]
    return r.get("text", "") or ""


def student_public_msgs(target, history):
    return [{"role": "system", "content": w.PUBLIC_REASONING_SYSTEM_PROMPT}] + w._turn_user_messages(
        target=target, history=history, prompt_style="public_reasoning"
    )


def student_with_cands_msgs(target, history):
    msgs = [{"role": "system", "content": w.PUBLIC_REASONING_SYSTEM_PROMPT}]
    base = w._turn_user_messages(target=target, history=history, prompt_style="public_reasoning")
    if history:
        cands = w.remaining_candidates(history)
        shown = cands[:100]
        more = f", showing 100" if len(cands) > 100 else ""
        cand_line = (
            f"\nReference: words still consistent with ALL feedback so far ({len(cands)} total{more}): "
            f"{w._format_candidates(shown)}. Your guess MUST be one of these."
        )
        base[-1] = {**base[-1], "content": base[-1]["content"] + cand_line}
    msgs += base
    return msgs


def teacher_policyhint_msgs(target, history):
    return [
        {"role": "system", "content": w.PUBLIC_REASONING_SYSTEM_PROMPT},
        {"role": "user", "content": w._policy_hint_teacher_content(
            target=target, history=history, response_style="public_reasoning")},
    ]


CONDITIONS = {
    "A student_public": student_public_msgs,
    "B student_with_cands": student_with_cands_msgs,
    "C teacher_policyhint": teacher_policyhint_msgs,
}


def play_game(build_msgs, target) -> dict:
    history: list[tuple[str, str]] = []
    info_scores: list[float] = []
    format_hits = 0
    turns = 0
    solved = False
    for _ in range(w.MAX_TURNS):
        turns += 1
        ids = tok.apply_chat_template(
            build_msgs(target, history), tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=False,
        )
        guess = w.extract_guess(gen(ids))
        if guess is None or len(guess) != 5 or not guess.isalpha():
            break
        format_hits += 1
        fb = w.compute_feedback(guess, target)
        info_scores.append(w._info_score(guess, target, fb))
        history.append((guess, fb))
        if guess == target:
            solved = True
            break
    return {
        "solved": float(solved),
        "turns": turns,
        "format_rate": format_hits / max(turns, 1),
        "info_gain": sum(info_scores) / max(len(info_scores), 1),
    }


rng = random.Random(9234)
pool = list(dict.fromkeys(w.WORD_LIST))  # dedup (WORD_LIST has repeats like "amber")
rng.shuffle(pool)
words = pool[20:20 + N]  # disjoint from a nominal train head

print(f"probe: {N} puzzles, base 30B (no LoRA), greedy temp=0\n")
for name, build in CONDITIONS.items():
    with ThreadPoolExecutor(max_workers=16) as pool_ex:
        results = list(pool_ex.map(lambda tgt: play_game(build, tgt), words))
    n = len(results)
    solve = sum(r["solved"] for r in results) / n
    fmt = sum(r["format_rate"] for r in results) / n
    info = sum(r["info_gain"] for r in results) / n
    turns = sum(r["turns"] for r in results) / n
    print(f"{name:24s}  solve_rate={solve:.3f} ({int(solve * n)}/{n})  "
          f"format={fmt:.3f}  info_gain={info:.3f}  avg_turns={turns:.2f}")
