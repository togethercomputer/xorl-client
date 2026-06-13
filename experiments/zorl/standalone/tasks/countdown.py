"""Countdown 24-puzzle task: combine 4 distinct ints with +, -, *, /, parens to
reach a target (24). Binary reward: 1.0 iff the parsed expression uses each
input number exactly once AND evaluates (within 1e-6) to the target.

Puzzle pool: 8 hand-picked seed puzzles + 24 enumerator-generated puzzles
with deterministically-shuffled distinct-integer 4-tuples drawn from [1, 12].
Seed shuffle is fixed (random.Random(20260520)) so the pool is reproducible
across runs.

The reward extractor strips a trailing ``= 24`` if present, then picks the
longest arithmetic-shaped contiguous substring containing at least one digit.
"""

from __future__ import annotations

import itertools
import random
import re

from .base import Example


is_multi_turn = False

# ---------------------------------------------------------------------------
# Puzzle pool
# ---------------------------------------------------------------------------

_SEED_COUNTDOWN_PUZZLES = [
    ([3, 4, 6, 9], 24, "(9 - 6 + 3) * 4"),
    ([2, 4, 6, 8], 24, "4 * 8 - 6 - 2"),
    ([1, 6, 7, 9], 24, "(1 + 7) * (9 - 6)"),
    ([3, 5, 7, 11], 24, "(11 - 5) * (7 - 3)"),
    ([2, 3, 8, 10], 24, "2 * 3 + 8 + 10"),
    ([1, 3, 7, 12], 24, "12 * (7 - 1) / 3"),
    ([2, 6, 8, 11], 24, "(11 - 8) * (2 + 6)"),
    ([1, 2, 7, 8], 24, "(7 - 1) * 8 / 2"),
]
_DEFAULT_POOL_SIZE = 32


SYSTEM_PROMPT = (
    "You solve arithmetic puzzles. Combine all of the given numbers using +, -, *, /, "
    "and parentheses so the expression equals the target. Use each number exactly once. "
    "Reply with only the expression — no words, no thinking, no equality sign."
)


def _eval_tree(tree):
    if not isinstance(tree, tuple):
        return float(tree)
    op, l, r = tree
    lv, rv = _eval_tree(l), _eval_tree(r)
    if lv is None or rv is None:
        return None
    if op == "+":
        return lv + rv
    if op == "-":
        return lv - rv
    if op == "*":
        return lv * rv
    if op == "/":
        return None if abs(rv) < 1e-12 else lv / rv


def _render_tree(t):
    if not isinstance(t, tuple):
        return str(t)
    op, l, r = t
    return f"({_render_tree(l)} {op} {_render_tree(r)})"


def _first_solution(nums, target):
    """Enumerate over 4! * 4^3 * 5 = 7680 expression-shaped candidates and
    return the first one that evaluates to target (or None)."""
    ops = ("+", "-", "*", "/")
    seen = set()
    for perm in itertools.permutations(nums):
        a, b, c, d = perm
        for o1 in ops:
            for o2 in ops:
                for o3 in ops:
                    for tree in (
                        (o3, (o2, (o1, a, b), c), d),
                        (o3, (o1, a, (o2, b, c)), d),
                        (o2, (o1, a, b), (o3, c, d)),
                        (o1, a, (o3, (o2, b, c), d)),
                        (o1, a, (o2, b, (o3, c, d))),
                    ):
                        v = _eval_tree(tree)
                        if v is None or abs(v - target) > 1e-6:
                            continue
                        s = _render_tree(tree)
                        if s in seen:
                            continue
                        seen.add(s)
                        return s
    return None


def _strip_outer_parens(s):
    """Drop a single redundant outer (...) wrapping the whole expression."""
    if not (s.startswith("(") and s.endswith(")")):
        return s
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and i < len(s) - 1:
                return s
    return s[1:-1]


def _build_puzzle_pool(target_size: int):
    """Return ``target_size`` puzzles: 8 seeds + (target_size-8) auto-generated.

    The auto-generated half draws from a deterministically-shuffled candidate
    list of distinct-integer 4-tuples from [1, 12], keeping the first that
    has a valid target=24 solution. Stable across runs because the rng seed
    is hardcoded."""
    extras_needed = max(0, target_size - len(_SEED_COUNTDOWN_PUZZLES))
    if extras_needed == 0:
        return list(_SEED_COUNTDOWN_PUZZLES[:target_size])
    seed_keys = {tuple(sorted(n)) for n, _, _ in _SEED_COUNTDOWN_PUZZLES}
    candidates = [m for m in itertools.combinations(range(1, 13), 4) if m not in seed_keys]
    random.Random(20260520).shuffle(candidates)
    extras = []
    for multiset in candidates:
        soln = _first_solution(list(multiset), 24)
        if soln is None:
            continue
        extras.append((list(multiset), 24, _strip_outer_parens(soln)))
        if len(extras) >= extras_needed:
            break
    return list(_SEED_COUNTDOWN_PUZZLES) + extras


# ---------------------------------------------------------------------------
# Reward extraction
# ---------------------------------------------------------------------------

_EXPR_ALLOWED_RE = re.compile(r"^[\s0-9+\-*/().]+$")
_EXPR_GREEDY_RE = re.compile(r"[0-9+\-*/().\s]+")
_NUMBER_TOKEN_RE = re.compile(r"\d+")


def _extract_expression(text: str):
    """Pull the longest arithmetic-shaped substring out of the model's reply.
    Strips a trailing ``= 24`` because models often append it."""
    if not text:
        return None
    cleaned = text.strip()
    if "=" in cleaned:
        cleaned = cleaned.split("=", 1)[0].strip()
    cands = _EXPR_GREEDY_RE.findall(cleaned)
    if not cands:
        return None
    cands.sort(key=len, reverse=True)
    for c in cands:
        s = c.strip()
        if any(ch.isdigit() for ch in s):
            return s
    return None


def _safe_eval(expr):
    if not isinstance(expr, str) or not expr.strip():
        return None
    if not _EXPR_ALLOWED_RE.match(expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))
    except (SyntaxError, ZeroDivisionError, ValueError, TypeError, OverflowError):
        return None


def _uses_each_once(expr, expected):
    found = [int(t) for t in _NUMBER_TOKEN_RE.findall(expr or "")]
    return sorted(found) == sorted(expected)


# ---------------------------------------------------------------------------
# Task contract
# ---------------------------------------------------------------------------


def _question(meta_numbers, target):
    nums = ", ".join(str(n) for n in meta_numbers)
    return (
        f"Combine the numbers {nums} using +, -, *, /, and parentheses "
        f"so the expression equals {target}. Use each number exactly once."
    )


def build_examples(tokenizer, *, train_size: int = 32, eval_size: int | None = None, seed: int = 0):
    """Build the Countdown puzzle set. Train and eval are the SAME pool for
    Countdown — there's no held-out set since the pool itself is small and
    deterministic. ``eval_size`` is ignored; we return the full pool for both."""
    puzzles = _build_puzzle_pool(train_size)
    examples = []
    for i, (numbers, target, solution) in enumerate(puzzles):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _question(numbers, target)},
        ]
        prompt_ids = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False
        )
        examples.append(Example(
            project=f"q_{i:02d}",
            prompt_ids=prompt_ids,
            metadata={"numbers": list(numbers), "target": target, "solution": solution},
        ))
    return examples, examples  # train and eval are the same set for Countdown


def score_completion(example: Example, generated_text: str) -> dict[str, float]:
    """Binary reward for Countdown: 1.0 iff the parsed expression uses each
    input number exactly once and evaluates to the target within 1e-6."""
    numbers = example.metadata["numbers"]
    target = example.metadata["target"]
    expr = _extract_expression(generated_text)
    if expr is None:
        return {"reward": 0.0, "exact_match": 0.0}
    if not _uses_each_once(expr, numbers):
        return {"reward": 0.0, "exact_match": 0.0}
    val = _safe_eval(expr)
    if val is None:
        return {"reward": 0.0, "exact_match": 0.0}
    if abs(val - float(target)) < 1e-6:
        return {"reward": 1.0, "exact_match": 1.0}
    return {"reward": 0.0, "exact_match": 0.0}


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Verify each seed puzzle's canonical solution scores 1.0.
    print("Running countdown sanity checks...")
    for i, (numbers, target, solution) in enumerate(_SEED_COUNTDOWN_PUZZLES):
        ex = Example(project=f"q_{i:02d}", prompt_ids=[], metadata={"numbers": list(numbers), "target": target, "solution": solution})
        r = score_completion(ex, solution)
        assert r["reward"] == 1.0, f"seed puzzle {i} ({solution!r}) should score 1.0, got {r}"
    # Junk text scores 0.
    ex0 = Example(project="q_00", prompt_ids=[], metadata={"numbers": [3, 4, 6, 9], "target": 24, "solution": ""})
    assert score_completion(ex0, "I don't know")["reward"] == 0.0
    # Wrong-arithmetic expression scores 0.
    assert score_completion(ex0, "3 + 4 + 6 + 9")["reward"] == 0.0
    # Wrong-number-set scores 0.
    assert score_completion(ex0, "5 + 5 + 5 + 9")["reward"] == 0.0
    # Trailing "= 24" gets stripped.
    assert score_completion(ex0, "(9 - 6 + 3) * 4 = 24")["reward"] == 1.0
    # Pool builder is reproducible + correct size.
    pool = _build_puzzle_pool(32)
    assert len(pool) == 32
    for nums, _t, soln in pool:
        assert _eval_tree(("+",) + (0, 0)) == 0  # eval helper baseline
        # Verify the canonical solution actually evaluates
        v = _safe_eval(soln)
        assert v is not None and abs(v - 24) < 1e-6, f"pool solution {soln!r} for {nums} doesn't eval to 24"
    print(f"OK — countdown sanity passed ({len(pool)} puzzles)")
