"""Wordle task: 5-letter word guessing with positional feedback.

The standard 6-turn Wordle game. Each turn the model emits a guess inside
``<guess>[WORD]</guess>`` (or bare ``<guess>WORD</guess>``) tags; the harness
reveals a feedback string using ``G`` for green (correct letter, correct
position), ``Y`` for yellow (correct letter, wrong position), and ``X`` for
grey (letter absent). Targets and legal guesses come from the vendored
legal-word file (else the pinned ``wordle-python`` dictionary, else the
built-in fallback list). Game ends on exact match or after 6 turns.

This is the multi-turn task implementation: ``is_multi_turn = True`` and
the client calls ``rollout_completion(...)`` instead of single-shot
``generate + score_completion``.

Reward components (each in [0, 1], task surfaces all of them so the run
logs can show the differential-signal pattern from the blog):
  * ``reward``: composite — 0.4*solved + 0.3*format_rate + 0.2*info_gain +
    0.1*turn_bonus. Solved is binary; format_rate is fraction of turns
    that produced a parseable guess; info_gain is normalized average
    letters-resolved per turn; turn_bonus rewards solving in fewer turns.
  * ``exact_match``: 1.0 iff target word was guessed within 6 turns.
  * ``format_rate``: fraction of turns producing a 5-letter alphabetic
    guess inside the ``<guess>`` tag.
  * ``info_gain``: average per-turn information score in [0, 1].
  * ``turns_used``: integer in [1, 6] — included for logging.
"""

from __future__ import annotations

import importlib.util
import os
import random
import re
from importlib import metadata as importlib_metadata


try:
    from .base import Example
except ImportError:  # pragma: no cover - direct script sanity path
    from base import Example


is_multi_turn = True


MAX_TURNS = 6


# ---------------------------------------------------------------------------
# Word list — common NYT-style 5-letter answers. Curated to ~500 entries
# from public Wordle answer-list dumps. Lower-cased; comparison is
# case-insensitive. This is the (c) fallback if neither the vendored
# legal-word file nor the wordle-python dictionary is available.
# ---------------------------------------------------------------------------

BUILTIN_FALLBACK_WORD_LIST = [
    "abide", "abled", "above", "abuse", "actor", "acute", "adept", "admit", "adobe", "adopt",
    "adore", "adorn", "adult", "after", "agent", "agile", "aging", "agony", "agree", "ahead",
    "aisle", "alarm", "album", "alert", "alien", "align", "alive", "alley", "allot", "allow",
    "alloy", "alone", "along", "aloof", "aloud", "alpha", "altar", "alter", "amber", "amend",
    "amber", "amply", "angel", "anger", "angle", "angry", "ankle", "annex", "annoy", "antic",
    "anvil", "apart", "apple", "apply", "apron", "arena", "argue", "arise", "armor", "aroma",
    "arose", "array", "arrow", "ascot", "ashen", "aside", "askew", "atoll", "atone", "audio",
    "audit", "avert", "avoid", "await", "awake", "award", "aware", "awful", "axiom", "azure",
    "bacon", "badge", "baker", "banal", "banjo", "barge", "basic", "basin", "basis", "batch",
    "bathe", "baton", "beach", "beady", "beard", "beast", "began", "begin", "begun", "being",
    "below", "bench", "bicep", "biome", "birch", "birth", "black", "blade", "blame", "bland",
    "blank", "blast", "blaze", "bleed", "blend", "bless", "blimp", "blind", "blink", "bliss",
    "block", "blond", "blood", "bloom", "blown", "bluff", "blunt", "blurt", "blush", "board",
    "boast", "bobby", "bonus", "booby", "boost", "booth", "booty", "booze", "borne", "bound",
    "brain", "braid", "brake", "brand", "brass", "brave", "bread", "break", "briar", "bribe",
    "brick", "bride", "brief", "brine", "bring", "brink", "broad", "broil", "broke", "brood",
    "brook", "brown", "brush", "buddy", "buggy", "bugle", "build", "built", "bunch", "bunny",
    "burly", "burnt", "burst", "buyer", "cable", "cacao", "cadet", "cagey", "candy", "canny",
    "canon", "carat", "cargo", "carry", "carve", "caste", "catch", "cause", "cease", "cedar",
    "chair", "chalk", "champ", "chant", "chaos", "chard", "charm", "chart", "chase", "chasm",
    "cheap", "cheat", "check", "cheek", "cheer", "chess", "chest", "chick", "chief", "child",
    "chili", "chill", "chime", "china", "chirp", "choir", "choke", "chord", "chose", "chunk",
    "civic", "civil", "claim", "clamp", "clang", "clank", "clash", "clasp", "class", "clean",
    "clear", "cleat", "clerk", "click", "cliff", "climb", "cling", "clink", "cloak", "clock",
    "clone", "close", "cloth", "cloud", "clout", "clown", "cluck", "clump", "clung", "coach",
    "coast", "cobra", "cocoa", "color", "comet", "comma", "comfy", "conic", "could", "count",
    "court", "couth", "cover", "covet", "covey", "crack", "craft", "cramp", "crane", "crank",
    "crash", "crass", "crate", "crave", "crawl", "craze", "crazy", "creak", "cream", "credo",
    "creed", "creek", "creep", "crepe", "crept", "cress", "crest", "crick", "cried", "crier",
    "crime", "crimp", "crisp", "croak", "crock", "crone", "crony", "crook", "cross", "crowd",
    "crown", "crude", "cruel", "crumb", "crush", "crust", "crypt", "cubic", "cumin", "curio",
    "curly", "curry", "curse", "curve", "curvy", "cynic", "daddy", "daily", "dairy", "dance",
    "dandy", "datum", "daunt", "dealt", "death", "debit", "debug", "debut", "decal", "decay",
    "decor", "decoy", "deify", "delta", "demon", "demur", "denim", "dense", "depot", "depth",
    "derby", "deter", "devil", "diary", "digit", "dingo", "dingy", "diode", "dirty", "disco",
    "ditch", "ditto", "ditty", "diver", "dizzy", "donor", "donut", "dough", "dowdy", "dowry",
    "dozen", "draft", "drain", "drake", "drama", "drank", "drape", "drawl", "drawn", "dread",
    "dream", "dress", "dried", "drier", "drift", "drill", "drink", "drive", "droll", "drone",
    "drool", "droop", "drove", "drown", "druid", "drunk", "dryer", "dryly", "duchy", "dully",
    "dummy", "dusky", "dusty", "duvet", "dwarf", "dwell", "dwelt", "dying", "eager", "eagle",
    "early", "earth", "easel", "eaten", "eater", "ebony", "eclat", "edict", "edify", "eerie",
    "egret", "eight", "eject", "eking", "elate", "elbow", "elder", "elect", "elegy", "elfin",
    "elite", "elope", "elude", "email", "embed", "ember", "emcee", "empty", "enact", "endow",
    "enema", "enemy", "enjoy", "ennui", "enrol", "ensue", "enter", "entry", "envoy", "epoch",
    "epoxy", "equal", "equip", "erase", "erect", "erode", "error", "erupt", "essay", "ester",
    "ether", "ethic", "ethos", "etude", "evade", "event", "every", "evict", "evoke", "exact",
    "exalt", "excel", "exert", "exile", "exist", "expel", "extol", "extra", "exult", "eying",
    "fable", "facet", "faint", "fairy", "faith", "false", "fancy", "fanny", "farce", "fatal",
    "fatty", "fault", "fauna", "favor", "feast", "fecal", "feign", "fella", "felon", "femme",
    "femur", "fence", "feral", "ferry", "fetch", "fetid", "fetus", "fever", "fewer", "fiber",
    "ficus", "field", "fiend", "fiery", "fifth", "fifty", "fight", "filer", "filet", "filly",
    "filmy", "filth", "final", "finch", "finer", "first", "fishy", "fixer", "fizzy", "fjord",
    "flack", "flair", "flame", "flank", "flare", "flash", "flask", "fleet", "flesh", "flick",
    "flier", "fling", "flint", "flirt", "float", "flock", "flood", "floor", "flora", "floss",
    "flour", "flout", "flown", "fluff", "fluid", "fluke", "flume", "flung", "flunk", "flush",
    "flute", "flyer", "foamy", "focal", "focus", "foggy", "foist", "folio", "folly", "foray",
    "force", "forge", "forgo", "forte", "forth", "forty", "forum", "found", "foyer", "frail",
    "frame", "frank", "fraud", "freak", "freed", "freer", "fresh", "friar", "fried", "frill",
    "frisk", "fritz", "frock", "frond", "front", "frost", "froth", "frown", "froze", "fruit",
    "fudge", "fugue", "fully", "fungi", "funky", "funny", "furor", "furry", "fussy", "fuzzy",
    "gaffe", "gaily", "gamer", "gamma", "gaudy", "gauge", "gaunt", "gauze", "gavel", "gawky",
    "gayly", "gecko", "geeky", "geese", "genie", "genre", "ghost", "ghoul", "giant", "giddy",
    "gipsy", "girly", "girth", "given", "giver", "glade", "gland", "glare", "glass", "glaze",
    "gleam", "glean", "glide", "glint", "gloat", "globe", "gloom", "glory", "gloss", "glove",
    "glyph", "gnash", "gnome", "godly", "going", "golem", "golly", "gonad", "goner", "goody",
    "gooey", "goofy", "goose", "gorge", "gouge", "gourd", "grace", "grade", "graft", "grail",
    "grain", "grand", "grant", "grape", "graph", "grasp", "grass", "grate", "grave", "gravy",
    "graze", "great", "greed", "green", "greet", "grief", "grill", "grime", "grimy", "grind",
    "gripe", "groan", "groin", "groom", "grope", "gross", "group", "grout", "grove", "growl",
]
# Dedupe in case of typos (e.g. "amber" appears twice above intentionally for
# the curated list to top out near 500; dedup keeps it clean):
_seen = set()
BUILTIN_FALLBACK_WORD_LIST = [w for w in BUILTIN_FALLBACK_WORD_LIST if not (w in _seen or _seen.add(w))]


def _load_vendored_legal_words() -> list[str]:
    """Load the vendored ``wordle_legal_words.txt`` (one lowercase word per line).

    The file lives next to THIS module; resolve relative to ``__file__`` so it
    works regardless of cwd. Returns [] if the file is missing/unreadable."""
    path = os.path.join(os.path.dirname(__file__), "wordle_legal_words.txt")
    try:
        with open(path, encoding="utf-8") as handle:
            words = [line.strip().lower() for line in handle]
    except OSError:
        return []
    return sorted({w for w in words if len(w) == 5 and w.isalpha()})


def _load_wordle_python_legal_words() -> list[str]:
    """Load the pinned ``wordle-python`` dictionary without importing ``wordle``.

    Normal ``import wordle`` is ambiguous here because this task file is also
    named ``wordle.py`` when run directly. The package publishes a static
    ``wordle/dictionary.py`` containing ``words``, so load that file through the
    installed distribution metadata."""
    try:
        dist = importlib_metadata.distribution("wordle-python")
    except importlib_metadata.PackageNotFoundError:
        return []
    dictionary_file = None
    for file in dist.files or ():
        if str(file) == "wordle/dictionary.py":
            dictionary_file = dist.locate_file(file)
            break
    if dictionary_file is None:
        return []
    spec = importlib.util.spec_from_file_location("_wordle_python_dictionary", dictionary_file)
    if spec is None or spec.loader is None:
        return []
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    words = getattr(module, "words", [])
    return sorted({str(word).lower() for word in words if isinstance(word, str) and len(word) == 5 and word.isalpha()})


def _load_legal_words() -> tuple[list[str], str]:
    """Resolve the legal-guess set: vendored file → wordle-python → builtin.

    Returns ``(words, source_tag)`` where source_tag is one of
    ``"vendored-file"`` / ``"wordle-python"`` / ``"builtin-fallback"``."""
    vendored = _load_vendored_legal_words()
    if vendored:
        return vendored, "vendored-file"
    wordle_python = _load_wordle_python_legal_words()
    if wordle_python:
        return wordle_python, "wordle-python"
    return list(BUILTIN_FALLBACK_WORD_LIST), "builtin-fallback"


_LEGAL_WORDS, LEGAL_GUESSES_SOURCE = _load_legal_words()
LEGAL_GUESSES = set(_LEGAL_WORDS)
# Targets/candidates are drawn from the legal set (matching SOURCE).
WORD_LIST = sorted(LEGAL_GUESSES)


def is_valid_guess(guess: str | None, history: list[tuple[str, str]] | None = None) -> bool:
    """Wordle validity gate using the configured legal-word source.

    Rejects None / non-5-letter / non-alpha / repeated guesses, and anything
    outside the legal-guess dictionary."""
    if guess is None:
        return False
    normalized = guess.lower()
    if len(normalized) != 5 or not normalized.isalpha():
        return False
    if history is not None and normalized in {prior for prior, _ in history}:
        return False
    return normalized in LEGAL_GUESSES


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are playing Wordle. The target is a 5-letter English word. You have "
    f"{MAX_TURNS} attempts. After each guess I will tell you which letters "
    "are correct.\n"
    "  G = correct letter and correct position\n"
    "  Y = letter is in the word but in the wrong position\n"
    "  X = letter is not in the word\n"
    "Reply on each turn with exactly one guess inside <guess>...</guess> "
    "tags. Use the square-bracket form <guess>[STARE]</guess>; bare "
    "<guess>STARE</guess> is also accepted. The guess must be exactly 5 "
    "alphabetic characters; do not put anything else on the line."
)


# Public-reasoning student prompt (ported from the OPSD-baseline worktree): the
# student emits an explicit, PUBLIC <reasoning> rationale then the guess, using
# only the transcript/feedback (never a private target). Paired with the
# policy_hint teacher below, the teacher's edge becomes feedback-reasoning
# (inferable from the student's state) rather than answer-knowledge (which isn't),
# which is the lever for SOLVING — not just play — to transfer. <reasoning> is
# used instead of <think> because Qwen wrapped <think> in </tool_call>.
PUBLIC_REASONING_SYSTEM_PROMPT = (
    "You are playing Wordle. The target is a 5-letter English word. You have "
    f"{MAX_TURNS} attempts. After each guess I will tell you which letters "
    "are correct.\n"
    "  G = correct letter and correct position\n"
    "  Y = letter is in the word but in the wrong position\n"
    "  X = letter is not in the word\n"
    "Use only the public transcript and feedback. Do not claim to know a private target. "
    "Your response must start with <reasoning>, then a short public rationale, then "
    "</reasoning><guess>[WORD]</guess>. "
    "Keep the rationale to one sentence under 15 words. "
    "Do not put the guess before the rationale. The guess must be exactly 5 alphabetic characters. "
    "Example reply: <reasoning>Choose a broad opener.</reasoning><guess>[STARE]</guess>"
)


# Strict public-reasoning system prompt (ported verbatim from SOURCE). Used by
# the constraints-summary student styles where no candidate list is shown.
PUBLIC_REASONING_STRICT_SYSTEM_PROMPT = (
    "You are playing Wordle. The hidden target is a 5-letter English word. You have "
    f"{MAX_TURNS} attempts. Feedback symbols mean:\n"
    "  G = correct letter and correct position\n"
    "  Y = letter is in the word but in the wrong position\n"
    "  X = letter is not in the word\n"
    "Use only the public transcript and feedback. Do not claim to know a private target.\n"
    "Critical validity rules:\n"
    "- Output exactly one line with exactly one <reasoning>...</reasoning> block and exactly one <guess>[WORD]</guess> tag.\n"
    "- WORD must be a real common Wordle word, exactly five alphabetic letters, not an abbreviation, not a name, and not an invented letter string.\n"
    "- Never repeat a previous guess.\n"
    "- After feedback is available, prefer a word consistent with every previous G/Y/X result.\n"
    "- If unsure, choose a conservative common answer word instead of inventing a probe word.\n"
    "Example reply: <reasoning>Choose a broad legal opener.</reasoning><guess>[STARE]</guess>"
)


# Enumerate-reasoning student prompt: the student must REPRODUCE the consistent-word
# narrowing from the public transcript alone (it is NOT given the candidate list — that is
# the inferable skill we distill from the enumerate teacher). Longer than public_reasoning.
ENUMERATE_REASONING_SYSTEM_PROMPT = (
    "You are playing Wordle. The target is a 5-letter English word. You have "
    f"{MAX_TURNS} attempts. After each guess I will tell you which letters are correct.\n"
    "  G = correct letter and correct position\n"
    "  Y = letter is in the word but in the wrong position\n"
    "  X = letter is not in the word\n"
    "Use only the public transcript and feedback. Reason by NARROWING: your response must start "
    "with <reasoning>, first list the 5-letter words still consistent with ALL feedback so far, "
    "then state the single feedback constraint that best discriminates among them, then choose one; "
    "close with </reasoning><guess>[WORD]</guess>. The guess must be exactly 5 alphabetic characters "
    "and consistent with every prior feedback line. Do not claim a private target. Example: "
    "<reasoning>consistent words: BRAKE, BRAVE, GRAVE; E is confirmed at position 5 ⇒ BRAVE</reasoning>"
    "<guess>[BRAVE]</guess>"
)


_GUESS_RE = re.compile(r"<guess>\s*\[?\s*([A-Za-z]{5})\s*\]?\s*</guess>", re.IGNORECASE)
_GUESS_TAG_RE = re.compile(r"<guess\b[^>]*>.*?</guess>", re.IGNORECASE | re.DOTALL)


def has_single_guess_tag(text: str) -> bool:
    """Format reward: exactly one well-formed 5-letter <guess> tag.

    Counts 5-letter guesses (`_GUESS_RE`), NOT bare ``<guess>...</guess>`` blocks
    (`_GUESS_TAG_RE`): thinking models echo the prompt template's 4-letter
    ``<guess>WORD</guess>`` example, which is not a real guess and previously
    inflated the `_GUESS_TAG_RE` count to >1 — killing ~91% of turns and making
    the run measure format compliance instead of Wordle solving (2026-06 fix)."""
    if not text:
        return False
    return len(_GUESS_RE.findall(text)) == 1


def extract_guess(text: str) -> str | None:
    """Pull a 5-letter alphabetic guess out of the model's output. Returns
    lower-case; or None if the model didn't emit a parseable tag."""
    if not text:
        return None
    m = _GUESS_RE.search(text)
    if m is None:
        return None
    return m.group(1).lower()


def extract_guesses(text: str) -> list[str]:
    """All 5-letter guesses emitted in <guess>...</guess> tags, lower-cased and
    in order. Used by the OPSD eval to check whether the hint-free student's CoT
    actually produced the target word."""
    if not text:
        return []
    return [m.group(1).lower() for m in _GUESS_RE.finditer(text)]


def strip_think_prefix(text: str) -> str:
    """Drop a private ``<think>...</think>`` prefix; the public response follows it.
    Think-contract models wrap reasoning (which itself routinely mentions the output
    tags) in <think>...</think>, so the guess must be parsed from the public suffix
    only — otherwise tags inside the think block pollute the count/extraction."""
    if text and "</think>" in text:
        return text.split("</think>")[-1].lstrip()
    return text


def compute_feedback(guess: str, target: str) -> str:
    """Standard Wordle feedback: G/Y/X per position with correct duplicate
    handling (green takes priority, then yellow only credits remaining
    target letters)."""
    g, t = guess.lower(), target.lower()
    if len(g) != 5 or len(t) != 5:
        raise ValueError(f"both words must be length 5: {g!r} vs {t!r}")
    feedback = ["X"] * 5
    target_remaining = list(t)
    # First pass: greens.
    for i in range(5):
        if g[i] == t[i]:
            feedback[i] = "G"
            target_remaining[t.index(g[i])] = "_"  # consume from remaining
    # Re-derive remaining since the above index-on-list trick is brittle.
    target_remaining = list(t)
    for i in range(5):
        if feedback[i] == "G":
            target_remaining[i] = "_"
    # Second pass: yellows.
    for i in range(5):
        if feedback[i] == "G":
            continue
        try:
            j = target_remaining.index(g[i])
        except ValueError:
            continue
        feedback[i] = "Y"
        target_remaining[j] = "_"
    return "".join(feedback)


def _info_score(guess: str, target: str, feedback: str) -> float:
    """Per-turn information: 0.5*greens + 0.25*yellows, scaled to [0, 1]
    so that all-green = 1.0, all-yellow = 0.5, all-grey = 0.0."""
    greens = sum(1 for c in feedback if c == "G")
    yellows = sum(1 for c in feedback if c == "Y")
    return min(1.0, 0.5 * (greens / 5.0) + 0.5 * ((greens + yellows) / 5.0))


# ---------------------------------------------------------------------------
# Examples + scoring
# ---------------------------------------------------------------------------


def build_examples(tokenizer, *, train_size: int = 20, eval_size: int = 64, seed: int = 0):
    """Pick disjoint random target words for train and eval. ``prompt_ids``
    is *only* the initial system+user message; multi-turn rollout in the
    client rebuilds the chat history each turn, so we just stash the static
    setup here."""
    rng = random.Random(seed)
    pool = list(WORD_LIST)
    # Parity with the GRPO floor eval (eval_wordle_sglang._pick_targets): reserve
    # random.Random(SEED).shuffle(WORD_LIST)[:COUNT] as the held-out set, keep it OUT
    # of TRAINING, and PROBE on exactly that set so ZORL's held-out == GRPO's held-out.
    # (WORD_LIST is byte-identical across the zorl/grpo repos — verified — so the
    # shuffle picks the same words.) Set WORDLE_TRAIN_EXCLUDE_SEED=777
    # WORDLE_TRAIN_EXCLUDE_COUNT=170 to match the canonical GRPO held-out.
    _excl_seed = int(os.environ.get("WORDLE_TRAIN_EXCLUDE_SEED", "0") or 0)
    _excl_count = int(os.environ.get("WORDLE_TRAIN_EXCLUDE_COUNT", "0") or 0)
    _reserved_list: list[str] = []
    if _excl_count > 0:
        _reserve = list(WORD_LIST)
        random.Random(_excl_seed).shuffle(_reserve)
        _reserved_list = _reserve[:_excl_count]
        _reserved = set(_reserved_list)
        pool = [w for w in pool if w not in _reserved]
        print(
            f"[wordle.build_examples] excluded {len(_reserved)} held-out targets "
            f"(seed={_excl_seed}, count={_excl_count}); train pool now {len(pool)} words",
            flush=True,
        )
    rng.shuffle(pool)
    # When exclusion is on, PROBE on the reserved floor-eval set (identical words +
    # order to GRPO's _pick_targets(seed=SEED, offset=0)); otherwise keep the legacy
    # behaviour: a disjoint slice of the (post-exclusion) train pool.
    if _reserved_list:
        if eval_size > len(_reserved_list):
            raise ValueError(
                f"eval_size={eval_size} exceeds reserved held-out count "
                f"{len(_reserved_list)} (raise WORDLE_TRAIN_EXCLUDE_COUNT)."
            )
        if len(pool) < train_size:
            raise ValueError(
                f"train pool has {len(pool)} entries after exclusion, "
                f"need ≥{train_size}."
            )
        train_words = pool[:train_size]
        eval_words = _reserved_list[:eval_size]
    else:
        if len(pool) < train_size + eval_size:
            raise ValueError(
                f"WORD_LIST has {len(pool)} entries, need ≥{train_size + eval_size} "
                f"for train+eval split."
            )
        train_words = pool[:train_size]
        eval_words = pool[train_size:train_size + eval_size]

    def to_example(word: str, pid: str) -> Example:
        # Render the *first-turn* prompt so the standalone client's
        # generate-with-lora helper has something to fall back on if a task
        # ever wants to short-circuit to single-shot. Multi-turn rollouts
        # ignore prompt_ids and rebuild per turn from metadata.
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Make your first guess."},
        ]
        prompt_ids = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False
        )
        return Example(
            project=pid,
            prompt_ids=list(prompt_ids),
            metadata={"target": word},
        )

    train = [to_example(w, f"wordle_train_{i:04d}") for i, w in enumerate(train_words)]
    eval_ = [to_example(w, f"wordle_eval_{i:04d}") for i, w in enumerate(eval_words)]
    return train, eval_


def _wordle_reward_components(
    *,
    solved: bool,
    turns_with_guess: int,
    latest_feedback: str,
    format_reward: float,
    valid_guess_rate: float = 1.0,
    terminal_valid: bool = True,
    invalid_action: bool = False,
) -> dict[str, float]:
    """Additive Wordle reward decomposition (reward-v2, ported to match the GRPO arm).

    Adds exact answer, latest partial answer, shorter-solution bonus, format,
    and validity terms. Invalid terminal actions get an explicit penalty; this
    keeps repeated guesses and invented words from receiving a good reward just
    because the earlier turns were well formatted."""
    correct = float(solved)
    if solved:
        partial = 0.0
    else:
        partial = 0.2 * latest_feedback.count("G") + 0.1 * latest_feedback.count("Y")
    length_bonus = correct / float(max(turns_with_guess, 1))
    invalid_penalty = 0.8 if invalid_action else 0.0
    terminal_valid_reward = 0.2 if terminal_valid else 0.0
    wordle_reward = (
        correct
        + 0.5 * length_bonus
        + 0.3 * partial
        + 0.2 * float(format_reward)
        + 0.2 * float(valid_guess_rate)
        + terminal_valid_reward
        - invalid_penalty
    )
    return {
        "wordle_reward": float(wordle_reward),
        "wordle_correct": correct,
        "wordle_partial": float(partial),
        "wordle_length_bonus": float(length_bonus),
        "wordle_format_reward": float(format_reward),
        "wordle_valid_guess_rate": float(valid_guess_rate),
        "wordle_terminal_valid": float(terminal_valid),
        "wordle_invalid_penalty": float(invalid_penalty),
    }


def _wordle_shaped_reward(
    *,
    solved: bool,
    format_rate: float,
    valid_guess_rate: float,
    info_gain: float,
    turns_used: int,
    max_turns: int,
    invalid_action: bool,
) -> float:
    """Compact scalar reward used by the generic ``reward`` metric (the ES fitness).

    Keeps the legacy ``reward`` from looking healthy for repeated or illegal
    terminal actions (the reward-v2 invalid-action penalty)."""
    turn_bonus = ((max_turns - turns_used + 1) / max_turns) if solved else 0.0
    invalid_penalty = 0.45 if invalid_action else 0.0
    return float(
        0.55 * float(solved)
        + 0.15 * float(format_rate)
        + 0.2 * float(valid_guess_rate)
        + 0.15 * float(info_gain)
        + 0.1 * float(turn_bonus)
        - invalid_penalty
    )


def wordle_retrieval_reward(
    turns: list[tuple[str, str]],
    *,
    solved: bool,
    invalid_action: bool = False,
    solve_weight: float = 2.0,
    consistent_bonus: float = 0.1,
    violate_penalty: float = 0.2,
    reduce_weight: float = 0.1,
    invalid_penalty: float = 0.8,
    valid_guess_rate: float | None = None,
    format_rate: float | None = None,
    format_weight: float = 0.5,
) -> dict[str, float]:
    """Retrieval-targeted Wordle reward — ported verbatim from the GRPO-CONVERGING
    science fork (OPSD_WORDLE_CANONICAL_RUNBOOK_2026_06_08, wordle.py:616).

    WHY (the env-audit verdict): the legacy ``_wordle_shaped_reward`` UNDER-PRICES
    solving (a clean non-solving game banks ~40% of a solve from format/info proxies)
    and contains NO signal for the actual bottleneck skill — constrained-vocabulary
    RETRIEVAL — so as an ES fitness it is nearly flat across the population (no
    gradient; explains the ES no-lift). This reward makes SOLVE dominate (solve_weight
    2.0 + fast-solve length bonus, ~3-5x a non-solve) AND adds a DENSE per-turn signal:
    +consistent_bonus if the guess is still a possible answer, -violate_penalty if it
    violates the public constraints (the exact failure mode), +reduce_weight*log(cand
    narrowing) — present even before the game is solved. Returns the scalar under
    ``wordle_retrieval_reward`` plus components."""
    import math

    history: list[tuple[str, str]] = []
    consistent_turns = 0
    violate_turns = 0
    reduction = 0.0
    for guess, feedback in turns:
        g = str(guess).lower()
        cand_before = remaining_candidates(history)
        n_before = max(1, len(cand_before))
        if history:  # constraints exist -> retrieval is being tested
            if g in {w.lower() for w in cand_before}:
                consistent_turns += 1
            else:
                violate_turns += 1
        history.append((guess, feedback))
        cand_after = remaining_candidates(history)
        n_after = max(1, len(cand_after))
        reduction += math.log(n_before / n_after)  # >= 0: a guess only narrows
    n_played = max(1, len(turns))
    length_bonus = 0.5 * solve_weight * (float(solved) / n_played)
    consistency_reward = consistent_bonus * consistent_turns - violate_penalty * violate_turns
    narrowing_reward = reduce_weight * reduction
    solve_reward = solve_weight * float(solved)
    if valid_guess_rate is None:
        invalid_rate = 1.0 if invalid_action else 0.0
    else:
        invalid_rate = max(0.0, 1.0 - float(valid_guess_rate))
    invalid_pen = invalid_penalty * invalid_rate
    format_reward = format_weight * float(format_rate) if format_rate is not None else 0.0
    total = solve_reward + length_bonus + consistency_reward + narrowing_reward + format_reward - invalid_pen
    return {
        "wordle_retrieval_reward": float(total),
        "wr_solve": float(solve_reward),
        "wr_length_bonus": float(length_bonus),
        "wr_consistency": float(consistency_reward),
        "wr_consistent_turns": float(consistent_turns),
        "wr_violate_turns": float(violate_turns),
        "wr_narrowing": float(narrowing_reward),
        "wr_format": float(format_reward),
        "wr_invalid_penalty": float(invalid_pen),
    }


def score_completion(example: Example, generated_text: str) -> dict[str, float]:
    """Single-shot scorer (reward-v2). Multi-turn rollout drives itself; this
    handles the degenerate single-completion case. Treats the completion as one guess."""
    target = example.metadata["target"]
    guess = extract_guess(generated_text)
    single_guess_tag = float(has_single_guess_tag(generated_text or ""))
    format_ok = float(single_guess_tag and guess is not None)
    if guess is None or not is_valid_guess(guess, []):
        valid_guess_rate = 0.0
        invalid_action = True
        reward = _wordle_shaped_reward(
            solved=False, format_rate=format_ok, valid_guess_rate=valid_guess_rate,
            info_gain=0.0, turns_used=1, max_turns=MAX_TURNS, invalid_action=invalid_action,
        )
        return {
            "reward": reward, "exact_match": 0.0, "format_rate": format_ok,
            "single_guess_tag_rate": single_guess_tag, "info_gain": 0.0, "turns_used": 1,
            "valid_guess_rate": valid_guess_rate, "invalid_action": float(invalid_action),
            **_wordle_reward_components(
                solved=False, turns_with_guess=int(single_guess_tag), latest_feedback="",
                format_reward=single_guess_tag, valid_guess_rate=valid_guess_rate,
                terminal_valid=False, invalid_action=invalid_action,
            ),
        }
    feedback = compute_feedback(guess, target)
    solved = float(guess == target)
    valid_guess_rate = 1.0
    info_gain = _info_score(guess, target, feedback)
    components = _wordle_reward_components(
        solved=bool(solved), turns_with_guess=1, latest_feedback=feedback,
        format_reward=single_guess_tag, valid_guess_rate=valid_guess_rate,
        terminal_valid=True, invalid_action=False,
    )
    reward = _wordle_shaped_reward(
        solved=bool(solved), format_rate=format_ok, valid_guess_rate=valid_guess_rate,
        info_gain=info_gain, turns_used=1, max_turns=MAX_TURNS, invalid_action=False,
    )
    return {
        "reward": reward, "exact_match": solved, "format_rate": format_ok,
        "single_guess_tag_rate": single_guess_tag, "info_gain": info_gain, "turns_used": 1,
        "valid_guess_rate": valid_guess_rate, "invalid_action": 0.0, **components,
    }


from functools import lru_cache as _lru_cache


@_lru_cache(maxsize=131072)
def _remaining_candidates_cached(history: tuple) -> tuple:
    """Cache-backed INCREMENTAL narrowing. remaining_candidates(history) is a pure
    function of history (WORD_LIST is constant), and each prefix is just its parent
    prefix filtered by ONE more (guess, feedback). So: filter the (already-tiny)
    parent result instead of re-scanning all ~4266 words every call, and memoize —
    each distinct prefix is computed exactly once; antithetic pairs and shared openers
    become cache hits. This turns the retrieval reward from ~640s/step of full rescans
    (~384M compute_feedback calls) into near-free. Returns a tuple (hashable/immutable)."""
    if not history:
        return tuple(WORD_LIST)
    prev = _remaining_candidates_cached(history[:-1])
    g, fb = history[-1]
    return tuple(w for w in prev if compute_feedback(g, w) == fb)


def remaining_candidates(history: list[tuple[str, str]]) -> list[str]:
    """Words in WORD_LIST consistent with all public (guess, feedback) so far.
    Cache-backed incremental narrowing (see ``_remaining_candidates_cached``) — same
    result as ``[w for w in WORD_LIST if all(...)]`` but ~instant on warm prefixes."""
    return list(_remaining_candidates_cached(tuple((str(g), str(fb)) for g, fb in history)))


def _format_transcript(history: list[tuple[str, str]]) -> str:
    return "\n".join(f"{g.upper()} -> {fb}" for g, fb in history) if history else "(none)"


def _format_candidates(cands: list[str]) -> str:
    return ", ".join(w.upper() for w in cands) if cands else "(none)"


def format_public_constraints(history: list[tuple[str, str]]) -> str:
    """Summarize public Wordle constraints without enumerating candidates."""
    if not history:
        return "No feedback yet."
    green_pattern = ["_"] * 5
    present_positions: dict[str, set[int]] = {}
    present_letters: set[str] = set()
    grey_letters: set[str] = set()
    for guess, feedback in history:
        guess = guess.lower()
        for idx, (letter, mark) in enumerate(zip(guess, feedback, strict=True)):
            pos = idx + 1
            if mark == "G":
                green_pattern[idx] = letter.upper()
                present_letters.add(letter)
            elif mark == "Y":
                present_letters.add(letter)
                present_positions.setdefault(letter, set()).add(pos)
            elif mark == "X":
                grey_letters.add(letter)
                if letter in present_letters:
                    present_positions.setdefault(letter, set()).add(pos)
    absent_letters = sorted(grey_letters - present_letters)
    excluded_parts = []
    for letter in sorted(present_positions):
        positions = ", ".join(str(pos) for pos in sorted(present_positions[letter]))
        excluded_parts.append(f"{letter.upper()} not in position(s) {positions}")
    lines = [
        f"Green pattern: {' '.join(green_pattern)}",
        f"Known present letters: {', '.join(letter.upper() for letter in sorted(present_letters)) or '(none)'}",
        f"Excluded positions for present letters: {'; '.join(excluded_parts) or '(none)'}",
        f"Letters likely absent: {', '.join(letter.upper() for letter in absent_letters) or '(none)'}",
    ]
    return "\n".join(lines)


def render_feedback_message(guess: str, feedback: str, remaining: int) -> str:
    """Render the latest feedback as a compact two-line Wordle board slice."""
    word_row = " ".join(guess.upper())
    feedback_row = " ".join(feedback)
    return f"{word_row}\n{feedback_row}\nYou have {remaining} guesses left."


def _policy_hint_teacher_content(*, target: str, history: list[tuple[str, str]], response_style: str) -> str:
    """Teacher 'policy_hint' context (ported from the OPSD baseline). The teacher is a
    strong PUBLIC solver over the candidate set derived from feedback, and may use the
    private target ONLY to tie-break among already-public-valid candidates (never on
    turn 1). Its edge is feedback-reasoning (in the student's state), not raw answer-
    knowledge (not in the student's state) — the lever for SOLVING to transfer."""
    target_u = target.upper()
    cands = remaining_candidates(history)
    transcript = _format_transcript(history)
    if not history:
        private_line = "Private target: omitted on the first turn."
        shown = "(omitted on turn 1 to avoid target leakage; all answer-list words apply)"
        target_rule = (
            "- On turn 1, ignore the private target completely (not provided). "
            "Choose a broad information-gathering opener."
        )
    else:
        private_line = f"Private target, for restricted tie-breaking only: {target_u}."
        shown = _format_candidates(cands)
        target_rule = (
            "- You may use the private target ONLY when all hold: it is in the public candidate set; "
            "the set came only from prior public feedback; it is among the best public-policy choices; "
            "the remaining choices are public-policy ties/near-ties; and this is not the first turn."
        )
    if response_style == "public_reasoning":
        contract = (
            "The student response being scored starts with <reasoning>, a short public rationale, "
            "</reasoning>, then exactly one <guess>[WORD]</guess>. Assign low probability to responses that "
            "put <guess> before <reasoning> or omit the rationale. The rationale is one sentence under 15 "
            "words, derivable from the public transcript/candidates, and must not reveal or rely on the "
            "private target."
        )
    else:
        contract = "Reply with exactly one guess tag and no other text:\n<guess>[WORD]</guess>"
    return (
        "You are scoring the student's next public Wordle action for on-policy self-distillation.\n"
        f"{private_line}\n\n"
        "The private target is not visible to the student. Act like a strong solver using only the public "
        "transcript, except for the restricted tie-break rule below.\n\n"
        "State before the next guess:\nPrevious guesses and feedback:\n"
        f"{transcript}\n\n"
        f"Remaining public candidate answers (from prior feedback only): count = {len(cands)}\n{shown}\n\n"
        "Public-policy rule:\n"
        f"{target_rule}\n"
        "- Choose only from the public candidate set after feedback.\n"
        "- Never use feedback for the student's current sampled guess; never repeat a previous guess.\n"
        "- Never choose a word inconsistent with the public transcript.\n"
        "- Penalize rationales that use oracle knowledge unavailable from the public transcript.\n\n"
        f"{contract}"
    )


def _enumerate_teacher_content(*, target: str, history: list[tuple[str, str]]) -> str:
    """Teacher 'enumerate' context (round-2 iteration winner). Keeps the privileged target but
    demonstrates PUBLIC candidate-narrowing the student can reproduce: list the consistent words,
    cite the discriminating feedback constraint, then land on the target. Probe: leak 0.0,
    solve 0.83, enum 0.67 on the base 30B. The student is NOT given the candidate set — distilling
    this teaches it to enumerate in its own CoT (the no-scaffold path past the enumeration ceiling)."""
    transcript = _format_transcript(history)
    cands = remaining_candidates(history)
    if not history:
        shown = "(turn 1: any answer-list word)"
        pick = "choose a broad information-gathering opener"
    else:
        shown = _format_candidates(cands[:40])
        pick = f"narrow to {target.upper()}"
    return (
        "You are the strong-solver teacher for on-policy self-distillation. The student sees only the "
        "public transcript; write reasoning it could REPRODUCE from feedback alone.\n\n"
        f"Public transcript:\n{transcript}\n\n"
        f"Words still consistent with all feedback (count={len(cands)}): {shown}\n\n"
        f"For this turn, {pick}. Structure the reasoning as PUBLIC narrowing: (1) list the consistent "
        "words, (2) state the single feedback constraint that best discriminates among them, (3) name "
        "the chosen word. Cite ONLY the transcript feedback and the consistent-word list. Never write "
        "'private', 'target', 'the answer is', 'given', or 'hint'.\n"
        "Format: <reasoning>consistent words: ...; <constraint> ⇒ WORD</reasoning><guess>[WORD]</guess>"
    )


def _turn_user_messages(*, target: str, history: list[tuple[str, str]], prompt_style: str):
    """STUDENT user-turn messages.

    Styles: 'default' (plain guess), 'public_reasoning' (explicit <reasoning> then guess),
    'enumerate' (CoT enumeration of consistent words), 'public_reasoning_constraints'
    (board summary only, no candidates/target), 'constraints_enumerate' (board summary +
    student enumerates the consistent words itself, then narrows — counterpart to the
    enumerate teacher). All assistant echoes use the bracketed <guess>[WORD]</guess> form."""
    msgs = []
    if prompt_style in {"public_reasoning_constraints", "constraints_enumerate"}:
        # Board-summary-only styles (SOURCE's public_reasoning_constraints layout).
        if history:
            transcript = "\n".join(
                f"{idx}. {guess.upper()} -> {feedback}" for idx, (guess, feedback) in enumerate(history, 1)
            )
            previous = ", ".join(guess.upper() for guess, _ in history)
            constraints = format_public_constraints(history)
            if prompt_style == "public_reasoning_constraints":
                tail = (
                    "Choose one real common five-letter Wordle answer word that fits the public constraints. "
                    "Do not invent words or use obscure letter strings. "
                    "Respond exactly as <reasoning>one short sentence under 15 words</reasoning><guess>[WORD]</guess>"
                )
            else:  # constraints_enumerate
                tail = (
                    "Choose one real common five-letter Wordle answer word that fits the public constraints. "
                    "Do not invent words or use obscure letter strings. "
                    "In <reasoning>, FIRST list the five-letter words still consistent with the constraints (from "
                    "your own knowledge), THEN state the single discriminating constraint, THEN name the choice. "
                    "Respond exactly as <reasoning>list the consistent words, then the discriminating constraint, "
                    "then the choice</reasoning><guess>[WORD]</guess>"
                )
            request = (
                "State before your next guess:\n"
                f"Previous guesses and feedback:\n{transcript}\n"
                f"Public constraint summary:\n{constraints}\n"
                f"Already guessed, do not repeat: {previous}\n"
                f"{tail}"
            )
        else:
            request = (
                "No previous guesses. Choose a strong common legal Wordle opener such as STARE, CRANE, SLATE, or AUDIO. "
                "Do not invent a word. "
                "Respond exactly as <reasoning>one short sentence under 15 words</reasoning><guess>[WORD]</guess>"
            )
        msgs.append({"role": "user", "content": request})
        return msgs
    if prompt_style == "public_reasoning":
        first = (
            "Use only public Wordle information. Start exactly with <reasoning>, briefly choose a broad "
            "information-gathering opener in one sentence under 15 words, then close </reasoning> and output "
            "exactly one guess tag: <guess>[WORD]</guess>. Do not mention, infer, or claim a private target."
        )
        msgs.append({"role": "user", "content": first})
        for i, (guess, feedback) in enumerate(history):
            msgs.append({"role": "assistant", "content": f"<guess>[{guess.upper()}]</guess>"})
            remaining = MAX_TURNS - (i + 1)
            if guess == target or remaining == 0:
                msgs.append({"role": "user", "content": f"Feedback: {feedback}. Game over."})
            else:
                msgs.append({"role": "user", "content": (
                    f"Feedback: {feedback} (for guess {guess.upper()}). You have {remaining} guess(es) left. "
                    "Start exactly with <reasoning>, update public constraints and choose a public-valid next "
                    "guess in one sentence under 15 words, then close </reasoning> and output exactly one guess "
                    "tag: <guess>[WORD]</guess>. Do not mention any private target.")})
    elif prompt_style == "enumerate":
        first = (
            "Use only public Wordle information. Start exactly with <reasoning>; on turn 1 there are no "
            "constraints yet, so briefly choose a broad information-gathering opener, then close "
            "</reasoning> and output exactly one guess tag: <guess>[WORD]</guess>."
        )
        msgs.append({"role": "user", "content": first})
        for i, (guess, feedback) in enumerate(history):
            msgs.append({"role": "assistant", "content": f"<guess>[{guess.upper()}]</guess>"})
            remaining = MAX_TURNS - (i + 1)
            if guess == target or remaining == 0:
                msgs.append({"role": "user", "content": f"Feedback: {feedback}. Game over."})
            else:
                msgs.append({"role": "user", "content": (
                    f"Feedback: {feedback} (for guess {guess.upper()}). You have {remaining} guess(es) left. "
                    "Start exactly with <reasoning>, list the 5-letter words still consistent with ALL feedback "
                    "so far, state the single constraint that best discriminates among them, and choose one; then "
                    "close </reasoning> and output exactly one guess tag: <guess>[WORD]</guess>.")})
    else:
        msgs.append({"role": "user", "content": "Make your first guess."})
        for i, (guess, feedback) in enumerate(history):
            msgs.append({"role": "assistant", "content": f"<guess>[{guess.upper()}]</guess>"})
            remaining = MAX_TURNS - (i + 1)
            if guess == target or remaining == 0:
                msgs.append({"role": "user", "content": f"Feedback: {feedback}. Game over."})
            else:
                msgs.append({"role": "user", "content": (
                    f"Feedback: {feedback} (for guess {guess.upper()}). "
                    f"You have {remaining} guess(es) left. Make your next guess.")})
    return msgs


# F_strat commit-instruction (prompt-search winner 2026-06-26): with native think
# DISABLED, this lifts cold solve ~0.05->~0.28 and terminate ~62%->~99% by forcing the
# model to commit instead of unboundedly enumerating candidate words inside <think>.
_FSTRAT_COMMIT = (
    "Pick the single most likely real five-letter answer that fits EVERY clue "
    "(prefer common words). Give ONE short sentence of reasoning, then output "
    "<guess>[WORD]</guess>. Do NOT enumerate or test lists of candidate words."
)


def _turn_system_prompt(prompt_style: str, enable_thinking: bool = False) -> str:
    """Map a STUDENT prompt_style to its system prompt.

    ``_FSTRAT_COMMIT`` (the no-think "commit to one word, don't enumerate" instruction) is
    appended ONLY for the no-think ``public_reasoning_constraints`` style — it contradicts a
    native <think> block, so the ``..._think`` style (which passes enable_thinking=True) gets
    the clean GRPO floor-protocol system prompt (strict + THINK_INSTR added by the caller)."""
    if prompt_style == "enumerate":
        return ENUMERATE_REASONING_SYSTEM_PROMPT
    if prompt_style == "public_reasoning":
        return PUBLIC_REASONING_SYSTEM_PROMPT
    if prompt_style == "public_reasoning_constraints":
        base = PUBLIC_REASONING_STRICT_SYSTEM_PROMPT
        return base if enable_thinking else base + "\n\n" + _FSTRAT_COMMIT
    if prompt_style == "constraints_enumerate":
        return PUBLIC_REASONING_STRICT_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def _build_turn_input_ids(tokenizer, *, target: str, history: list[tuple[str, str]], prompt_style: str = "default"):
    """STUDENT turn prompt. prompt_style: 'default' (plain guess), 'public_reasoning'
    (explicit <reasoning> then guess, public-only — paired with the policy_hint teacher),
    'enumerate' (CoT enumeration), 'public_reasoning_constraints' (board summary only),
    or 'constraints_enumerate' (board summary + student-side enumeration, counterpart to
    the enumerate teacher). A trailing '_think' (e.g. 'public_reasoning_constraints_think')
    opens the model's private <think> block (the GRPO floor-protocol that scores 0.469);
    the base layout is identical to the non-think style."""
    enable_thinking = prompt_style.endswith("_think")
    base_style = prompt_style[: -len("_think")] if enable_thinking else prompt_style
    system = _turn_system_prompt(base_style, enable_thinking=enable_thinking)
    if enable_thinking:
        # Opening the native <think> block is not enough: a base model rambles past
        # the token budget and never emits a guess (cold solve 0.0156, all invalid).
        # The GRPO floor-protocol (0.469) forces TERMINATION + a one-line format, so
        # the think must be told to close promptly and emit exactly one guess.
        system += (
            "\n\nThink briefly in your private reasoning, then STOP and reply with EXACTLY ONE "
            "line and nothing after it:\n"
            "<reasoning>RATIONALE</reasoning><guess>[WORD]</guess>\n"
            "- RATIONALE: one short sentence, under 15 words.\n"
            "- WORD: exactly 5 letters.\n"
            "Close your private reasoning promptly — do not keep thinking; the answer line is what counts."
        )
    msgs = [{"role": "system", "content": system}] + _turn_user_messages(
        target=target, history=history, prompt_style=base_style
    )
    return tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, enable_thinking=enable_thinking, return_dict=False
    )


def _build_teacher_turn_input_ids(
    tokenizer, *, target: str, history: list[tuple[str, str]],
    teacher_prompt_style: str = "policy_hint", response_style: str = "public_reasoning",
):
    """TEACHER turn prompt, teacher-forced on the student's tokens. teacher_prompt_style:
    'policy_hint' (public solver + restricted tie-break; new default), 'answer_hint'
    (legacy: raw private target — the control that taught play-not-solving), or 'enumerate'
    (privileged: lists the consistent candidates + narrows to the target — the privileged
    counterpart to the 'constraints_enumerate' student, which gets the board summary only).
    The 'enumerate' teacher ignores ``response_style`` and uses its own X/bracket format."""
    if teacher_prompt_style == "enumerate":
        msgs = [
            {"role": "system", "content": ENUMERATE_REASONING_SYSTEM_PROMPT},
            {"role": "user", "content": _enumerate_teacher_content(target=target, history=history)},
        ]
    elif teacher_prompt_style == "policy_hint":
        system = PUBLIC_REASONING_SYSTEM_PROMPT if response_style == "public_reasoning" else SYSTEM_PROMPT
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": _policy_hint_teacher_content(
                target=target, history=history, response_style=response_style)},
        ]
    else:  # answer_hint (legacy control)
        system = (
            f"{SYSTEM_PROMPT}\n\nPrivate hint: the target word is {target.upper()}. "
            "Use the hint and the feedback to guess optimally."
        )
        msgs = [{"role": "system", "content": system}] + _turn_user_messages(
            target=target, history=history,
            prompt_style=("public_reasoning" if response_style == "public_reasoning" else "default"),
        )
    return tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False
    )


def _default_teacher_opener(target: str) -> str:
    """Pick a stable broad opener that is not usually the answer itself."""
    for guess in ("stare", "crane", "slate", "audio"):
        if guess != target:
            return guess
    return "crane"


def _teacher_trace_text(*, target: str, opener: str, feedback: str, style: str) -> str:
    target_u = target.upper()
    opener_u = opener.upper()
    if style == "guess_only":
        return f"<guess>[{opener_u}]</guess>\n<guess>[{target_u}]</guess>"
    return (
        "<think>"
        f"The private hint says the target word is {target_u}. "
        f"I can make a legal opener {opener_u}; its feedback is {feedback}. "
        f"Given that hint, the next legal guess should be the target {target_u}."
        "</think>\n"
        f"<guess>[{opener_u}]</guess>\n"
        f"<guess>[{target_u}]</guess>"
    )


def build_teacher_forced_example(tokenizer, example: Example, *, args) -> Example:
    """Build a hinted Wordle teacher trace for OPSD-style ZORL scoring.

    The candidate model is scored on the logprob of a teacher trajectory whose
    context includes a private hint with the target word and a compact rationale.
    This gives dense per-candidate feedback without requiring sparse solved-rate
    reward during the ES step.
    """
    target = example.metadata["target"].lower()
    opener = _default_teacher_opener(target)
    feedback = compute_feedback(opener, target)
    style = getattr(args, "wordle_teacher_trace_style", "hinted_cot")
    teacher_text = _teacher_trace_text(target=target, opener=opener, feedback=feedback, style=style)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "You are a teacher solving one Wordle game for distillation. "
                f"Private hint: the target word is {target.upper()}. "
                "Write a compact hidden rationale, then emit a legal sequence of "
                "<guess>WORD</guess> guesses that solves the game."
            ),
        },
    ]
    prefix_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    target_ids = tokenizer.encode(teacher_text, add_special_tokens=False)
    return Example(
        project=example.project,
        prompt_ids=list(prefix_ids) + list(target_ids),
        metadata={
            **example.metadata,
            "teacher_target_token_count": len(target_ids),
            "teacher_target_text": teacher_text,
            "teacher_opener": opener,
            "teacher_opener_feedback": feedback,
            "teacher_trace_style": style,
        },
    )


def _opsd_user_content(target_upper: str, *, hinted: bool) -> str:
    """Hinted (teacher) vs hint-free (student) Wordle instruction for on-policy
    OPSD. Kept identical to train_opsd_baseline._asymmetric_prompt_ids so the
    ZORL (ES) and gradient/GRPO runs distill from the same teacher context."""
    if hinted:
        return (
            "You are a teacher solving one Wordle game for distillation. "
            f"Private hint: the target word is {target_upper}. "
            "Write a compact hidden rationale, then emit complete five-letter guesses in "
            "<guess>[WORD]</guess> tags that solve the game. Do not leave any guess tag empty."
        )
    return (
        "You are solving one Wordle game. The target word is hidden from you. "
        "Write a compact hidden rationale, then emit at least two complete five-letter guesses in "
        "<guess>[WORD]</guess> tags. If uncertain, use common Wordle openers; do not leave any guess tag empty."
    )


def build_opsd_prompts(tokenizer, example: Example, *, args) -> tuple[list[int], list[int]]:
    """Return ``(student_prompt_ids, teacher_prefix_ids)`` for on-policy OPSD ZORL
    scoring. The student prompt is hint-free; the teacher prefix carries the
    private hint. The candidate (student) samples a CoT from the student prompt;
    the frozen base teacher is then scored on that same CoT continuation, and the
    ZORL reward is ``mean(teacher_logprob - student_logprob)`` over those tokens
    (a single-sample estimate of -KL(student||teacher) on the student's own CoT).
    """
    target_u = str(example.metadata["target"]).upper()

    def prefix(hinted: bool) -> list[int]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _opsd_user_content(target_u, hinted=hinted)},
        ]
        return list(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
        )

    return prefix(hinted=False), prefix(hinted=True)


def _invalid_reason(text: str, guess: str | None, history: list[tuple[str, str]]) -> str:
    """Human-readable reason a turn was rejected (mirrors SOURCE eval)."""
    if not has_single_guess_tag(text or ""):
        return "not_exactly_one_guess_tag"
    if guess is None:
        return "no_parseable_five_letter_guess"
    normalized = guess.lower()
    if normalized in {prior for prior, _ in history}:
        return "repeated_guess"
    if len(normalized) != 5 or not normalized.isalpha():
        return "not_five_alpha_letters"
    if normalized not in LEGAL_GUESSES:
        return "not_in_legal_wordle_dictionary"
    return "unknown_invalid"


def _build_retry_input_ids(
    tokenizer,
    *,
    target: str,
    history: list[tuple[str, str]],
    prompt_style: str,
    rejected_attempts: list[dict],
) -> list[int]:
    """Re-prompt the model after a rejected turn (mirrors SOURCE ``_build_retry_input_ids``).

    Replays the turn messages, echoes the rejected raw text as an assistant turn, then
    appends a rejection note that does NOT advance the game / fabricate feedback."""
    messages = [{"role": "system", "content": _turn_system_prompt(prompt_style)}] + _turn_user_messages(
        target=target, history=history, prompt_style=prompt_style
    )
    rejected_words = [
        str(attempt.get("guess") or "").upper()
        for attempt in rejected_attempts
        if str(attempt.get("guess") or "")
    ]
    last = rejected_attempts[-1]
    messages.append({"role": "assistant", "content": str(last.get("raw_text") or "")})
    retry_note = (
        f"That response was rejected because: {last.get('reason')}. "
        "It did not count as a Wordle guess, and no Wordle feedback is available for it. "
    )
    if rejected_words:
        retry_note += f"Do not reuse these rejected guesses: {', '.join(rejected_words)}. "
    retry_note += (
        "Try again with exactly one real, common, legal five-letter Wordle word. "
        "Do not repeat any previous accepted guess. "
        "Respond exactly as <reasoning>one short sentence under 15 words</reasoning><guess>[WORD]</guess>"
    )
    messages.append({"role": "user", "content": retry_note})
    return list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
    )


def rollout_completion(example: Example, *, generate_turn, lora_path: str, tokenizer, args) -> dict[str, float]:
    """Play a 6-turn Wordle game against the target word in
    ``example.metadata['target']``. Returns the same score blob shape as
    ``score_completion`` so the client treats it uniformly.

    ``generate_turn`` is provided by the client; its contract is::

        generate_turn(input_ids, *, lora_path, temperature, max_new_tokens) -> str

    Returning just the text of the model's reply for that turn.

    reward-v2 (matches the gradient GRPO arm): an invalid action (bad format /
    not-legal / repeated) is a TERMINAL action — it ends the game and takes the
    invalid-action penalty (no retry here; retries are an eval-only mechanic and
    live in ``rollout_capture`` for OPSD). The ``reward`` field is the compact
    shaped reward (solve + format + valid_guess_rate + info + turn_bonus − penalty)."""
    target = example.metadata["target"].lower()
    prompt_style = getattr(args, "wordle_prompt_style", "default")
    history: list[tuple[str, str]] = []
    format_hits = 0
    single_guess_tag_hits = 0
    valid_hits = 0
    info_scores: list[float] = []
    latest_feedback = ""
    solved = False
    invalid_action = False

    for turn in range(MAX_TURNS):
        input_ids = _build_turn_input_ids(tokenizer, target=target, history=history, prompt_style=prompt_style)
        text = generate_turn(
            input_ids=list(input_ids),
            lora_path=lora_path,
            temperature=float(args.rollout_temperature),
            max_new_tokens=int(args.rollout_max_new_tokens),
        )
        # Lenient extraction: take the LAST 5-letter <guess> (thinking models
        # reason first, emit the real guess last). A legal guess lets the game
        # CONTINUE even when the format is messy — format is now a graded reward,
        # NOT a game-ending gate. The old `not format_ok` break + template-echo
        # tag over-count killed ~91% of turns, so the run measured format, not Wordle.
        action_text = strip_think_prefix(text or "")
        _guesses = extract_guesses(action_text)
        guess = _guesses[-1] if _guesses else None
        single_guess_tag_ok = has_single_guess_tag(action_text)
        if single_guess_tag_ok:
            single_guess_tag_hits += 1
        format_ok = single_guess_tag_ok and guess is not None
        if format_ok:
            format_hits += 1
        if guess is None or not is_valid_guess(guess, history):
            # No legal guess extractable: end this local rollout (don't invent
            # public feedback for a non-action). Messy-but-legal guesses continue.
            info_scores.append(0.0)
            invalid_action = True
            break
        valid_hits += 1
        feedback = compute_feedback(guess, target)
        latest_feedback = feedback
        info_scores.append(_info_score(guess, target, feedback))
        history.append((guess, feedback))
        if guess == target:
            solved = True
            break

    turns_used = len(info_scores)
    format_rate = format_hits / max(turns_used, 1)
    single_guess_tag_rate = single_guess_tag_hits / max(turns_used, 1)
    valid_guess_rate = valid_hits / max(turns_used, 1)
    info_gain = sum(info_scores) / max(len(info_scores), 1)
    # GRPO-converging RETRIEVAL reward (dense per-turn constraint-consistency + candidate-narrowing,
    # solve-dominant), ported from the science fork. Computed ALWAYS so its full breakdown (wr_*) is
    # LOGGED for GRPO parity — zorl_client.record_score surfaces every numeric score-blob key. The ES
    # fitness IS this reward by default; the legacy _wordle_shaped_reward gave ES ~no gradient (flat:
    # a valid non-solve banks ~40% of a solve from saturated proxies, NO retrieval signal). Env-gated
    # for A/B + revert: XORL_WORDLE_REWARD=shaped restores the legacy fitness.
    wr = wordle_retrieval_reward(
        history,  # the valid (guess, feedback) turns actually played
        solved=solved,
        invalid_action=invalid_action,
        valid_guess_rate=valid_guess_rate,
        format_rate=format_rate,
    )
    if os.environ.get("XORL_WORDLE_REWARD", "retrieval").lower() == "shaped":
        reward = _wordle_shaped_reward(
            solved=solved,
            format_rate=format_rate,
            valid_guess_rate=valid_guess_rate,
            info_gain=info_gain,
            turns_used=turns_used,
            max_turns=MAX_TURNS,
            invalid_action=invalid_action,
        )
    else:
        reward = wr["wordle_retrieval_reward"]
    wordle_components = _wordle_reward_components(
        solved=solved,
        turns_with_guess=valid_hits,
        latest_feedback=latest_feedback,
        format_reward=single_guess_tag_rate,
        valid_guess_rate=valid_guess_rate,
        terminal_valid=not invalid_action,
        invalid_action=invalid_action,
    )
    return {
        "reward": float(reward),
        "exact_match": float(solved),
        "format_rate": float(format_rate),
        "single_guess_tag_rate": float(single_guess_tag_rate),
        "valid_guess_rate": float(valid_guess_rate),
        "info_gain": float(info_gain),
        "turns_used": float(turns_used),
        "invalid_action": float(invalid_action),
        **wordle_components,
        **wr,  # GRPO-parity reward breakdown: wordle_retrieval_reward, wr_solve, wr_length_bonus,
               # wr_consistency, wr_consistent_turns, wr_violate_turns, wr_narrowing, wr_format, wr_invalid_penalty
    }


def rollout_capture(example: Example, *, generate_turn_ids, tokenizer, args) -> list[tuple]:
    """Play the hint-free multi-turn game and capture per-turn distillation segments
    for full-vocab OPSD (score_mode=opsd_kl_full + --opsd-multi-turn).

    ``generate_turn_ids(input_ids) -> (text, out_ids)`` runs one turn (typically the
    ZORL parent) and returns the reply text plus its generated token ids.

    Returns a list of ``(student_input_ids, teacher_input_ids, out_ids)`` — one per
    played turn. ``student_input_ids`` is the student turn prompt (args.wordle_prompt_style:
    default, public_reasoning, enumerate, public_reasoning_constraints, or
    constraints_enumerate); ``teacher_input_ids`` is the teacher prompt
    (args.wordle_teacher_prompt_style: policy_hint = public-solver + restricted tie-break,
    answer_hint = legacy raw target, or enumerate = privileged candidate-narrowing). Both
    share the SAME feedback history; ``out_ids`` are the parent's generated tokens for that
    turn (the shared trajectory all candidates are scored on). The per-turn full-vocab KL
    between teacher and candidate student over ``out_ids`` is the distillation reward.

    Invalid turns (unparseable / not-legal / repeat) are re-prompted up to
    ``args.invalid_retries`` times and are NOT captured: only the FIRST VALID turn's
    generation (the one that advances the game) becomes a distillation segment, so the
    captured ``out_ids`` always correspond to a guess/feedback the teacher/student contexts
    agree on (X + bracket format)."""
    target = example.metadata["target"].lower()
    prompt_style = str(getattr(args, "wordle_prompt_style", "default"))
    teacher_prompt_style = str(getattr(args, "wordle_teacher_prompt_style", "answer_hint"))
    invalid_retries = int(getattr(args, "invalid_retries", 2))
    history: list[tuple[str, str]] = []
    segments: list[tuple] = []
    for _turn in range(MAX_TURNS):
        student_ids = _build_turn_input_ids(tokenizer, target=target, history=history, prompt_style=prompt_style)
        text, out_ids = generate_turn_ids(list(student_ids))
        if not out_ids:
            break
        guess = extract_guess(text or "")
        rejected_attempts: list[dict] = []
        retry_index = 0
        # Invalid-retry loop: re-prompt on an invalid action; capture only a valid turn.
        while not is_valid_guess(guess, history) and retry_index < invalid_retries:
            rejected_attempts.append({
                "raw_text": text or "",
                "guess": guess,
                "reason": _invalid_reason(text or "", guess, history),
            })
            retry_ids = _build_retry_input_ids(
                tokenizer, target=target, history=history,
                prompt_style=prompt_style, rejected_attempts=rejected_attempts,
            )
            student_ids = retry_ids
            text, out_ids = generate_turn_ids(list(retry_ids))
            if not out_ids:
                break
            guess = extract_guess(text or "")
            retry_index += 1
        if not out_ids:
            break
        if not is_valid_guess(guess, history):
            # Exhausted retries without a valid guess: stop without capturing garbage.
            break
        teacher_ids = _build_teacher_turn_input_ids(
            tokenizer, target=target, history=history,
            teacher_prompt_style=teacher_prompt_style, response_style=prompt_style,
        )
        # student_ids is the prompt that produced this valid generation (initial or retry),
        # so student_in / teacher_in both prefix the SAME out_ids under matching format.
        segments.append((list(student_ids), list(teacher_ids), list(out_ids)))
        feedback = compute_feedback(guess, target)
        history.append((guess, feedback))
        if guess == target:
            break
    return segments


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("Running wordle sanity checks...")
    # Feedback edge cases (X = absent)
    assert compute_feedback("crate", "crate") == "GGGGG"
    assert compute_feedback("xxxxx", "crate") == "XXXXX"
    assert compute_feedback("trace", "crate") == "YGGYG", compute_feedback("trace", "crate")
    # Duplicate letters: 2 e's in guess vs 1 e in target — only the better-positioned
    # e should get a non-X (the green at pos 4 here); the other e is consumed already.
    # Also tests yellow-with-other-letters logic.
    assert compute_feedback("eerie", "crate") == "XXYXG", compute_feedback("eerie", "crate")
    # Greens consume target letters before yellows. "speed" vs "abide": e at pos 2
    # of guess matches e... wait, target "abide" = a,b,i,d,e. Guess "speed" = s,p,e,e,d.
    # Greens: none position-wise. Yellows: 'e' at guess[2] (in target at pos 4) → Y;
    # 'e' at guess[3] — target only has one 'e', already consumed → X.
    # 'd' at guess[4] — in target at pos 3 → Y. Final: XXYXY.
    assert compute_feedback("speed", "abide") == "XXYXY", compute_feedback("speed", "abide")

    # Guess extraction (bracketed + bare)
    assert extract_guess("I think the answer is <guess>CRATE</guess>") == "crate"
    assert extract_guess("<guess>crate</guess>") == "crate"
    assert extract_guess("<guess>[crate]</guess>") == "crate"
    assert extract_guess("<reasoning>x</reasoning><guess>[CRATE]</guess>") == "crate"
    assert extract_guess("<guess>cr at e</guess>") is None  # spaces not allowed
    assert extract_guess("no tag here") is None
    assert extract_guess("<guess>crater</guess>") is None  # too long
    assert has_single_guess_tag("<guess>[crane]</guess>")
    assert not has_single_guess_tag("<guess>[crane]</guess><guess>[apple]</guess>")

    # Validity gate
    assert is_valid_guess("stare", [])
    assert is_valid_guess("crate", [])
    assert not is_valid_guess("zzzzz", [])
    assert not is_valid_guess("crate", [("crate", "GGGGG")])  # repeat

    # score_completion (single-shot fallback)
    ex = Example(project="t", prompt_ids=[], metadata={"target": "crate"})
    r = score_completion(ex, "<guess>crate</guess>")
    assert r["exact_match"] == 1.0 and r["reward"] >= 0.7, r
    r = score_completion(ex, "<guess>trace</guess>")
    assert r["exact_match"] == 0.0 and 0 < r["reward"] < 0.5, r
    r = score_completion(ex, "lol no")
    assert r["reward"] < 0.0 and r["invalid_action"] == 1.0, r  # reward-v2 invalid penalty

    # Multi-turn rollout (mock generate_turn that always guesses target on
    # turn 2 — exercises the loop)
    from types import SimpleNamespace
    class FakeTok:
        def apply_chat_template(self, msgs, **kw): return list(range(len(msgs) * 5))
        def encode(self, text, add_special_tokens=False): return [ord(ch) % 251 for ch in text]
    calls = []
    def fake_generate(input_ids, *, lora_path, temperature, max_new_tokens):
        calls.append(len(input_ids))
        if len(calls) == 1:
            return "<guess>STARE</guess>"
        return "<guess>CRATE</guess>"
    args = SimpleNamespace(rollout_temperature=0.6, rollout_max_new_tokens=32, invalid_retries=2)
    result = rollout_completion(ex, generate_turn=fake_generate, lora_path="x", tokenizer=FakeTok(), args=args)
    assert result["exact_match"] == 1.0
    assert result["turns_used"] == 2.0
    assert result["format_rate"] == 1.0
    assert len(calls) == 2

    # reward-v2: a bad-format action is TERMINAL (no retry in rollout_completion) + penalized.
    def fake_bad(input_ids, **kw):
        return "i don't know"
    result = rollout_completion(ex, generate_turn=fake_bad, lora_path="x", tokenizer=FakeTok(), args=args)
    assert result["exact_match"] == 0.0
    assert result["format_rate"] == 0.0
    assert result["invalid_action"] == 1.0 and result["reward"] < 0.0, result

    # reward-v2: an illegal (not-in-dictionary) word is also a terminal invalid action, no retry.
    illegal_calls = {"n": 0}
    def fake_illegal(input_ids, *, lora_path, temperature, max_new_tokens):
        illegal_calls["n"] += 1
        return "<guess>ZZZZZ</guess>"  # parseable but not a legal word
    result = rollout_completion(ex, generate_turn=fake_illegal, lora_path="x", tokenizer=FakeTok(), args=args)
    assert result["exact_match"] == 0.0 and result["invalid_action"] == 1.0, result
    assert illegal_calls["n"] == 1, result  # rollout_completion does NOT retry

    # build_examples
    train, eval_ = build_examples(FakeTok(), train_size=4, eval_size=8, seed=42)
    assert len(train) == 4 and len(eval_) == 8
    targets = {e.metadata["target"] for e in train + eval_}
    assert len(targets) == 12, "train/eval should be disjoint"

    # Teacher-forced trace used by OPSD-style scoring.
    tf = build_teacher_forced_example(FakeTok(), ex, args=SimpleNamespace(wordle_teacher_trace_style="hinted_cot"))
    assert tf.metadata["teacher_target_token_count"] > 0
    assert "CRATE" in tf.metadata["teacher_target_text"]
    assert "<guess>[CRATE]</guess>" in tf.metadata["teacher_target_text"]

    print(
        "OK — wordle sanity passed "
        f"(answer list size: {len(WORD_LIST)}, legal source: {LEGAL_GUESSES_SOURCE}, "
        f"legal size: {len(LEGAL_GUESSES)})"
    )
