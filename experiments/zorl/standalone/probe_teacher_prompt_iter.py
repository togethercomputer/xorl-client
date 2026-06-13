"""Iterate on the OPSD Wordle TEACHER prompt: keep the privileged target (paper-faithful)
and keep supervising the full CoT, but make the teacher's REASONING on-contract (public-facing,
no privileged-knowledge phrases) so distilling the reasoning tokens is legitimate.

For each candidate teacher prompt, over valid mid-game states, measure on the BASE 30B:
  * leak_rate  — fraction of generated <reasoning> that asserts privileged knowledge
                 ("private", "target is", "the answer is", "given", "hint", "i know", "secret")
  * solve_rate — fraction whose <guess> == target (teacher still uses the target to pick right)
A good teacher prompt => LOW leak_rate + HIGH solve_rate.

Run ON a replica (sequential, low load — does NOT hammer the pool):
  kubectl exec zorl-ar-sglang-15 -n apanda -- \
    /workspace/home/xorl-sglang-internal/.venv/bin/python \
    /workspace/home/xorl-apanda-dev-zorl-consolidated/experiments/zorl/standalone/probe_teacher_prompt_iter.py
"""
import sys

import requests
from transformers import AutoTokenizer


sys.path.insert(0, "/workspace/home/xorl-apanda-dev-zorl-consolidated/experiments/zorl/standalone")
import tasks.wordle as w  # noqa: E402


URL = "http://localhost:30000"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")
LEAK_PHRASES = ["private", "target is", "the target", "the answer is", "answer is", "given ",
                "i know", "i am told", "secret", "hint", "provided"]


def gen(input_ids, temp=0.7, n=3, max_new=80):
    body = {"input_ids": list(input_ids), "sampling_params": {"temperature": temp, "max_new_tokens": max_new, "n": n}}
    try:
        r = requests.post(f"{URL}/generate", json=body, timeout=180).json()
    except Exception as e:  # noqa: BLE001
        return [f"<error: {e}>"]
    if isinstance(r, dict):
        r = [r]
    return [x.get("text", "") or "" for x in r]


def _reasoning_of(text):
    import re
    m = re.search(r"<reasoning>(.*?)</reasoning>", text, re.IGNORECASE | re.DOTALL)
    return (m.group(1) if m else text).lower()


# ---- Teacher-prompt variants (all keep the privileged target) ----

def v_base(target, history):  # current policy_hint
    return w._policy_hint_teacher_content(target=target, history=history, response_style="public_reasoning")


def _common_state(target, history):
    cands = w.remaining_candidates(history)
    return w._format_transcript(history), cands


FEWSHOT = (
    "Examples of the REQUIRED public reasoning style (derive from feedback, never name a private answer):\n"
    "  Feedback CRANE->BYBBB, remaining BRISK,BRING,DRINK. "
    "<reasoning>R is present but not position 2; only words with R later fit, BRISK keeps the most live letters.</reasoning><guess>BRISK</guess>\n"
    "  Feedback SLATE->BBGYB, remaining PLANT,GIANT. "
    "<reasoning>A is green at position 3 and T is present off-position; PLANT places T legally.</reasoning><guess>PLANT</guess>\n"
)


def v_fewshot(target, history):  # base + few-shot public-reasoning exemplars
    return v_base(target, history) + "\n\n" + FEWSHOT


def v_oblique(target, history):
    # Do NOT label the target "private target". Frame it as the guess to justify publicly.
    transcript, cands = _common_state(target, history)
    shown = w._format_candidates(cands) if history else "(turn 1: all answer-list words)"
    pick = "choose a broad information-gathering opener" if not history else f"select the word {target.upper()}"
    return (
        "You are the strong-solver teacher for on-policy self-distillation. The student sees only the "
        "public transcript and must read your response as PUBLIC reasoning derived from feedback.\n\n"
        f"Public transcript:\n{transcript}\n\n"
        f"Words still consistent with the feedback (count={len(cands)}): {shown}\n\n"
        f"For this turn, {pick}. Write the rationale as a public solver would — cite the feedback "
        "constraints and the consistent words ONLY. NEVER write 'private', 'target', 'the answer is', "
        "'given', or 'hint'; never claim to know the answer in advance.\n"
        "Format: <reasoning>one public sentence under 15 words</reasoning><guess>WORD</guess>"
    )


def v_oblique_fewshot(target, history):
    return v_oblique(target, history) + "\n\n" + FEWSHOT


def v_enumerate(target, history):
    # Teacher whose CoT explicitly ENUMERATES the consistent words then narrows to the pick.
    # Teaches the inferable narrowing skill (distill into the student's own CoT, no scaffold).
    transcript, cands = _common_state(target, history)
    shown = w._format_candidates(cands[:40]) if history else "(turn 1: any answer-list word)"
    pick = "choose a broad information-gathering opener" if not history else f"narrow to {target.upper()}"
    return (
        "You are the strong-solver teacher for on-policy self-distillation. The student sees only the "
        "public transcript; write reasoning it could REPRODUCE from feedback alone.\n\n"
        f"Public transcript:\n{transcript}\n\n"
        f"Words still consistent with all feedback (count={len(cands)}): {shown}\n\n"
        f"For this turn, {pick}. Structure the reasoning as PUBLIC narrowing: (1) list the consistent "
        "words, (2) state the single feedback constraint that best discriminates among them, (3) name "
        "the chosen word. Cite ONLY the transcript feedback and the consistent-word list. Never write "
        "'private', 'target', 'the answer is', 'given', or 'hint'.\n"
        "Format: <reasoning>consistent words: ...; <constraint> ⇒ WORD</reasoning><guess>WORD</guess>"
    )


VARIANTS = {"oblique": v_oblique, "enumerate": v_enumerate}
MAXNEW = {"enumerate": 220}


def teacher_ids(content):
    msgs = [{"role": "system", "content": w.PUBLIC_REASONING_SYSTEM_PROMPT}, {"role": "user", "content": content}]
    return tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                   enable_thinking=False, return_dict=False)


# Build valid mid-game states: play 2 fixed openers, compute real feedback.
def make_state(target, openers):
    hist = []
    for g in openers:
        if g == target:
            break
        hist.append((g, w.compute_feedback(g, target)))
    return hist


targets = ["brave", "plumb", "shine", "crowd", "feast", "globe"]
states = [(t, make_state(t, ["stare", "cloud"])) for t in targets]

import re


def _enum_count(reasoning_lower, cands):
    """How many distinct consistent words the reasoning actually names (proxy for enumeration)."""
    words = set(re.findall(r"\b[a-z]{5}\b", reasoning_lower))
    return len(words & set(cands))


print("teacher-prompt iteration r2: base 30B, 6 mid-game states, n=3 gen/temp0.7")
print("(enum_rate = frac of reasonings naming >=2 consistent words = actually narrowing)\n")
print(f"{'variant':14s} {'leak_rate':>10s} {'solve_rate':>11s} {'enum_rate':>10s}")
for name, fn in VARIANTS.items():
    leaks = solves = enums = total = 0
    samples = []
    for target, hist in states:
        ids = teacher_ids(fn(target, hist))
        cands = w.remaining_candidates(hist)
        for text in gen(ids, max_new=MAXNEW.get(name, 80)):
            total += 1
            reasoning = _reasoning_of(text)
            if any(p in reasoning for p in LEAK_PHRASES):
                leaks += 1
            if w.extract_guess(text) == target:
                solves += 1
            if _enum_count(reasoning, cands) >= 2:
                enums += 1
            if len(samples) < 2:
                samples.append((target, text))
    print(f"{name:14s} {leaks / max(total,1):>10.3f} {solves / max(total,1):>11.3f} {enums / max(total,1):>10.3f}")
    for target, text in samples:
        print(f"     [{target}] {text!r}")
