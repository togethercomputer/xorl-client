"""Alphabet Sort task: given a list of names, sort them alphabetically.

Adapted from Prime-RL's alphabet sort environment as described in the
May-2026 ZORL-on-Prime-RL blog. Single-shot (not multi-turn) variant: we
ask the model to sort one list per example and score similarity to the
ground-truth sorted list.

Reward: character-level "similarity" score raised to a power. For a candidate
output, we extract a list of names and compute mean(name_overlap_ratio) ** p,
where the per-pair overlap_ratio penalizes wrong positions. ``similarity_power``
defaults to 8 (Prime-RL's value); higher p makes the reward sharper around
exact matches and is what the researcher's blog used.

Examples are generated procedurally from a fixed name pool with a seeded
RNG, so train/eval are deterministic across runs but disjoint subsets.
"""

from __future__ import annotations

import random
import re

from .base import Example


is_multi_turn = False


SYSTEM_PROMPT = (
    "You are given a list of names. Sort them alphabetically (case-insensitive), "
    "one per line, with no commentary. Output the sorted list inside a single "
    "<sorted>...</sorted> block — one name per line, nothing else."
)


# A small pool of common English first names. Sized to give enough variety
# (~150 names) without overlap concerns for the small train/eval slices we
# use. Source: arbitrary common-name list; not loaded from a dataset to avoid
# external download.
_NAME_POOL = [
    "Aaron", "Abigail", "Adam", "Aiden", "Alex", "Alice", "Amelia", "Andrew", "Angela", "Anna",
    "Anthony", "Aria", "Ariel", "Asher", "Aubrey", "Audrey", "Aurora", "Austin", "Ava", "Avery",
    "Benjamin", "Bella", "Blake", "Brandon", "Brody", "Brooklyn", "Caleb", "Camila", "Carter", "Charles",
    "Charlotte", "Chase", "Chloe", "Christian", "Christopher", "Claire", "Cole", "Colton", "Connor", "Cooper",
    "Cora", "Daniel", "David", "Diana", "Dylan", "Eleanor", "Elena", "Eli", "Eliana", "Elijah",
    "Elizabeth", "Ella", "Ellie", "Emily", "Emma", "Eric", "Ethan", "Evan", "Evelyn", "Faith",
    "Gabriel", "Gabriella", "Genesis", "Gianna", "Grace", "Grayson", "Hailey", "Hannah", "Harper", "Hazel",
    "Henry", "Hudson", "Hunter", "Iris", "Isaac", "Isabella", "Isaiah", "Jack", "Jackson", "Jacob",
    "James", "Jameson", "Jasmine", "Jason", "Jaxon", "Jayden", "Jeremiah", "Jocelyn", "John", "Jonathan",
    "Jordan", "Joseph", "Joshua", "Josie", "Julia", "Julian", "Justin", "Kai", "Kayden", "Kaylee",
    "Kennedy", "Kevin", "Kinsley", "Landon", "Layla", "Leah", "Leo", "Levi", "Liam", "Lily",
    "Lincoln", "Logan", "Lucas", "Luke", "Luna", "Madeline", "Madison", "Mason", "Mateo", "Matthew",
    "Maverick", "Maya", "Mia", "Michael", "Mila", "Naomi", "Natalie", "Nathan", "Noah", "Nora",
    "Nova", "Oliver", "Olivia", "Owen", "Paisley", "Parker", "Penelope", "Quinn", "Riley", "Robert",
    "Roman", "Ruby", "Ryan", "Sadie", "Samantha", "Samuel", "Sarah", "Savannah", "Scarlett", "Sebastian",
]


def _generate_examples(num_examples: int, names_per_list: int, seed: int):
    """Pick disjoint random names per example. Per researcher, names_per_list
    is sampled uniformly from [1, 4] inclusive (original Prime-RL difficulty).
    We pass a fixed int so each example has a deterministic size."""
    rng = random.Random(seed)
    examples = []
    for i in range(num_examples):
        n = names_per_list if names_per_list > 0 else rng.randint(1, 4)
        names = rng.sample(_NAME_POOL, n)
        gold_sorted = sorted(names, key=lambda x: x.lower())
        examples.append({"names": names, "gold": gold_sorted})
    return examples


def build_examples(tokenizer, *, train_size: int = 16, eval_size: int = 32, seed: int = 0, names_per_list: int = 12):
    """Generate train + held-out eval. Disjoint seeds keep them from sampling
    identical lists by accident; we don't enforce strict disjointness of
    sampled names because the pool is large enough that collisions are rare.

    Difficulty: at names_per_list=4 the cold Qwen3-4B-Instruct nearly aces it
    (28/32). 12 names/list keeps the task meaningfully hard while still
    tractable in a 256-token completion. Increase further if you see the
    model saturating again."""
    train_data = _generate_examples(train_size, names_per_list=names_per_list, seed=seed)
    eval_data = _generate_examples(eval_size, names_per_list=names_per_list, seed=seed + 100000)

    def to_example(row, project_id: str) -> Example:
        names_str = ", ".join(row["names"])
        user_msg = f"Sort these names alphabetically:\n{names_str}"
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        prompt_ids = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False
        )
        return Example(
            project=project_id,
            prompt_ids=prompt_ids,
            metadata={"names": list(row["names"]), "gold": list(row["gold"])},
        )

    train = [to_example(r, f"alphasort_train_{i:04d}") for i, r in enumerate(train_data)]
    eval_ = [to_example(r, f"alphasort_eval_{i:04d}") for i, r in enumerate(eval_data)]
    return train, eval_


# ---------------------------------------------------------------------------
# Output parsing + reward
# ---------------------------------------------------------------------------

_BLOCK_RE = re.compile(r"<sorted>\s*(.*?)\s*</sorted>", re.DOTALL | re.IGNORECASE)


def _extract_predicted_names(text: str) -> list[str] | None:
    """Pull names out of a <sorted>...</sorted> block. Strip lines / commas /
    numbering. Falls back to the raw text if the block tag is missing —
    important because the cold model often forgets the tag."""
    if not text:
        return None
    m = _BLOCK_RE.search(text)
    body = m.group(1) if m else text
    # Allow either newline-separated or comma-separated names.
    raw = re.split(r"[\n,]", body)
    out = []
    for tok in raw:
        # Strip list bullets / numbering / quotes / surrounding punctuation.
        tok = re.sub(r"^\s*[\d.\-•*]+\s*", "", tok).strip().strip("\"'`.")
        if not tok:
            continue
        # Take just the first word — the model sometimes annotates lines.
        first = tok.split()[0].strip("\"'`.,;")
        if first:
            out.append(first)
    return out or None


def _similarity_score(pred: list[str], gold: list[str], power: int = 8) -> float:
    """Per-position match rate ** power. Returns 0.0 if length mismatch — a
    sharp penalty that pushes the model toward exactly the right list shape.
    Power=8 (researcher's setting) makes ~70% match score ~0.06, full match
    score 1.0, so the gradient strongly favors near-exact ordering."""
    if not pred or not gold:
        return 0.0
    if len(pred) != len(gold):
        return 0.0
    matches = sum(1 for p, g in zip(pred, gold) if p.lower() == g.lower())
    ratio = matches / len(gold)
    return ratio ** power


def score_completion(example: Example, generated_text: str) -> dict[str, float]:
    """Sharp similarity reward + diagnostic metrics. Returns:
      reward: similarity ** 8 (sharp around exact match)
      exact_match: 1.0 iff list matches exactly
      has_block: did the model emit the <sorted> wrapper
      length_ok: did the model produce the right number of names
    """
    gold = example.metadata["gold"]
    pred = _extract_predicted_names(generated_text)
    has_block = float(bool(_BLOCK_RE.search(generated_text or "")))
    length_ok = float(pred is not None and len(pred) == len(gold))
    sim = _similarity_score(pred or [], gold, power=8)
    exact = float(sim == 1.0)
    return {
        "reward": float(sim),
        "exact_match": exact,
        "has_block": has_block,
        "length_ok": length_ok,
    }


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("Running alphabet_sort sanity checks...")
    ex = Example(project="t", prompt_ids=[], metadata={"names": ["Charlie", "Alice", "Bob"], "gold": ["Alice", "Bob", "Charlie"]})
    # Exact match in block format
    r = score_completion(ex, "<sorted>\nAlice\nBob\nCharlie\n</sorted>")
    assert r["reward"] == 1.0 and r["exact_match"] == 1.0 and r["has_block"] == 1.0 and r["length_ok"] == 1.0
    # Exact match without block (fallback)
    r = score_completion(ex, "Alice\nBob\nCharlie")
    assert r["reward"] == 1.0 and r["has_block"] == 0.0
    # Comma-separated also works
    r = score_completion(ex, "<sorted>Alice, Bob, Charlie</sorted>")
    assert r["reward"] == 1.0
    # Wrong order: 0/3 match → 0 ** 8 = 0
    r = score_completion(ex, "<sorted>Bob\nCharlie\nAlice</sorted>")
    assert r["reward"] == 0.0 and r["exact_match"] == 0.0
    # Partial: 2/3 correct positions → (2/3) ** 8 ~ 0.039
    ex2 = Example(project="t", prompt_ids=[], metadata={"names": ["a", "b", "c"], "gold": ["Alice", "Bob", "Charlie"]})
    r = score_completion(ex2, "<sorted>Alice\nBob\nDavid</sorted>")
    assert 0.0 < r["reward"] < 0.1, f"expected small partial reward, got {r}"
    # Length mismatch → 0
    r = score_completion(ex, "<sorted>Alice\nBob</sorted>")
    assert r["reward"] == 0.0 and r["length_ok"] == 0.0
    # Junk → 0
    r = score_completion(ex, "I don't know how to sort")
    assert r["reward"] == 0.0
    # Build examples
    class FakeTok:
        def apply_chat_template(self, msgs, **kw): return [1, 2, 3]
    train, eval_ = build_examples(FakeTok(), train_size=4, eval_size=4, seed=42)
    assert len(train) == 4 and len(eval_) == 4
    assert all(len(e.metadata["names"]) == len(e.metadata["gold"]) for e in train + eval_)
    print("OK — alphabet_sort sanity passed")
