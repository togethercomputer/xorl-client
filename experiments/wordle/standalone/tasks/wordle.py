"""Wordle task: 5-letter word guessing with positional feedback.

The standard 6-turn Wordle game. Each turn the model emits a guess inside
``<guess>[WORD]</guess>`` or ``<guess>WORD</guess>`` tags; the harness reveals a feedback string using
``G`` for green (correct letter, correct position), ``Y`` for yellow
(correct letter, wrong position), and ``X`` for wrong/absent. Targets and
legal guesses come from the pinned ``wordle-python`` dictionary when installed.
Game ends on exact match or after 6 turns.

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
  * ``format_rate``: fraction of turns producing exactly one 5-letter
    alphabetic guess inside the ``<guess>`` tag.
  * ``info_gain``: average per-turn information score in [0, 1].
  * ``turns_used``: integer in [1, 6] — included for logging.
  * ``wordle_reward``: additive Wordle diagnostic reward:
    exact_match + latest partial answer + length bonus + 0.2*format_reward.
    The old ``textarena_*`` aliases are also emitted for W&B continuity.
"""

from __future__ import annotations

import importlib.util
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
# case-insensitive.
# ---------------------------------------------------------------------------

WORD_LIST = [
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
WORD_LIST = [w for w in WORD_LIST if not (w in _seen or _seen.add(w))]

BUILTIN_FALLBACK_WORD_LIST = list(WORD_LIST)


def _load_wordle_python_legal_words() -> list[str]:
    """Load the pinned `wordle-python` dictionary without importing `wordle`.

    Normal `import wordle` is ambiguous here because this task file is also
    named `wordle.py` when run directly. The package publishes a static
    `wordle/dictionary.py` containing `words`, so load that file through the
    installed distribution metadata.
    """
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


WORDLE_PYTHON_LEGAL_WORDS = _load_wordle_python_legal_words()
FALLBACK_COMMON_LEGAL_GUESSES = {"adieu", "audio", "crane", "slate", "stare", "trace"}
WORD_LIST_SOURCE = "wordle-python" if WORDLE_PYTHON_LEGAL_WORDS else "builtin-fallback"
WORD_LIST = list(WORDLE_PYTHON_LEGAL_WORDS) if WORDLE_PYTHON_LEGAL_WORDS else list(BUILTIN_FALLBACK_WORD_LIST)
LEGAL_GUESSES_SOURCE = WORD_LIST_SOURCE
LEGAL_GUESSES = set(WORD_LIST) if WORDLE_PYTHON_LEGAL_WORDS else set(WORD_LIST) | FALLBACK_COMMON_LEGAL_GUESSES


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
    "tags. Use the square-bracket form <guess>[STARE]</guess>; "
    "bare <guess>STARE</guess> is also accepted. The guess must be exactly "
    "5 alphabetic characters; do not put anything else on the line."
)

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

SHARED_PUBLIC_SYSTEM_PROMPT = (
    "You are playing Wordle. Follow the public Wordle prompt exactly."
)


def build_shared_public_prompt_content(
    history: list[tuple[str, str]],
    *,
    include_begin: bool = True,
    include_candidates: bool = False,
    max_candidate_display: int = 100,
) -> str:
    """Public Wordle prompt shared by student and OPSD teacher contexts.

    ``include_candidates`` adds the remaining public candidate answers (derived
    only from prior feedback) to the shared state. This closes the
    candidate-enumeration gap between teacher and student: the teacher's edge
    reduces to split quality instead of enumeration.
    """
    candidate_block = ""
    if history:
        transcript = "\n".join(f"{idx}. {guess.upper()} -> {feedback}" for idx, (guess, feedback) in enumerate(history, 1))
        constraints = format_public_constraints(history)
        previous = ", ".join(guess.upper() for guess, _ in history)
        if include_candidates:
            candidates = remaining_candidates(history)
            candidate_text = format_candidate_list(candidates, max_display=max_candidate_display)
            candidate_block = (
                "Remaining public candidate answers (computed only from the feedback above):\n"
                f"count = {len(candidates)}\n"
                f"{candidate_text}\n\n"
            )
    else:
        transcript = "(none)"
        constraints = "No constraints yet."
        previous = "(none)"
        if include_candidates:
            candidate_block = "Remaining public candidate answers: all answers are possible before the first guess.\n\n"
    prompt = (
        "You are playing Wordle. The hidden target is a 5-letter English word. "
        f"You have {MAX_TURNS} attempts.\n\n"
        "Feedback symbols:\n"
        "G = correct letter and correct position\n"
        "Y = letter is in the word but in the wrong position\n"
        "X = letter is not in the word\n\n"
        "Use only the public transcript and feedback in the public response.\n"
        "Do not claim to know a private target.\n\n"
        "State before your next guess:\n"
        "Previous guesses and feedback:\n"
        f"{transcript}\n\n"
        "Public constraint summary:\n"
        f"{constraints}\n\n"
        f"{candidate_block}"
        "Already guessed, do not repeat:\n"
        f"{previous}\n\n"
        "Think briefly in the required public style, then give one Wordle guess.\n\n"
        "Output exactly one line:\n"
        "<reasoning>PUBLIC_REASONING</reasoning><guess>WORD</guess>\n\n"
        "Rules:\n"
        "- PUBLIC_REASONING must be one short sentence under 15 words.\n"
        "- PUBLIC_REASONING must not mention private targets, hints, hidden information, or oracle knowledge.\n"
        "- WORD must be exactly five alphabetic letters.\n"
        "- WORD must be a common valid Wordle answer word.\n"
        "- WORD must not repeat a previous guess.\n"
        "- After feedback is available, WORD must fit all public constraints."
    )
    if include_begin:
        prompt += "\n\nBegin your response now."
    return prompt


_GUESS_RE = re.compile(r"<guess>\s*\[?\s*([A-Za-z]{5})\s*\]?\s*</guess>", re.IGNORECASE)
_GUESS_TAG_RE = re.compile(r"<guess\b[^>]*>.*?</guess>", re.IGNORECASE | re.DOTALL)
_REASONING_TAG_RE = re.compile(
    r"<(?P<tag>think|reasoning)>\s*(.*?)\s*</(?P=tag)>",
    re.IGNORECASE | re.DOTALL,
)
_STRICT_REASONING_GUESS_RE = re.compile(
    r"^\s*<(?P<tag>think|reasoning)>\s*(?P<reasoning>.*?)\s*</(?P=tag)>\s*"
    r"<guess>\s*\[?\s*(?P<guess>[A-Za-z]{5})\s*\]?\s*</guess>\s*$",
    re.IGNORECASE | re.DOTALL,
)
_PRIVATE_REASONING_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bprivate\b",
        r"\bhint\b",
        r"\boracle\b",
        r"\bhidden (?:target|answer|word)\b",
        r"\bi know\b",
        r"\btarget (?:is|must be|has)\b",
        r"\banswer (?:is|must be)\b",
    )
]
MAX_REASONING_WORDS = 18


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


def has_single_guess_tag(text: str) -> bool:
    """Format reward: every assistant turn should contain exactly one guess tag."""
    if not text:
        return False
    return len(_GUESS_TAG_RE.findall(text)) == 1


def public_constraints_satisfied(guess: str | None, history: list[tuple[str, str]] | None = None) -> bool:
    """Whether `guess` could still be the hidden answer under public feedback."""
    if guess is None:
        return False
    return all(compute_feedback(prior_guess, guess) == feedback for prior_guess, feedback in history or [])


def _reasoning_mentions_private_info(reasoning: str) -> bool:
    return any(pattern.search(reasoning or "") for pattern in _PRIVATE_REASONING_PATTERNS)


def parse_turn_response(text: str, history: list[tuple[str, str]] | None = None) -> dict[str, object]:
    """Strict one-line Wordle action parser used for rewards and rollout state.

    A valid public response is exactly:
    ``<reasoning>...</reasoning><guess>WORD</guess>``
    with no unrelated continuation after the guess.
    """
    raw = text or ""
    stripped = raw.strip()
    guess_tags = _GUESS_TAG_RE.findall(raw)
    reasoning_matches = list(_REASONING_TAG_RE.finditer(raw))
    match = _STRICT_REASONING_GUESS_RE.fullmatch(stripped)
    guess = match.group("guess").lower() if match is not None else extract_guess(raw)
    reasoning = match.group("reasoning").strip() if match is not None else ""
    reasoning_words = re.findall(r"[A-Za-z0-9']+", reasoning)
    errors: list[str] = []
    if len(guess_tags) != 1:
        errors.append("guess_tag_count")
    if len(reasoning_matches) != 1:
        errors.append("reasoning_tag_count")
    if match is None:
        errors.append("not_exact_one_line_response")
    if "\n" in stripped or "\r" in stripped:
        errors.append("extra_line_or_trailing_text")
    if len(reasoning_words) > MAX_REASONING_WORDS:
        errors.append("reasoning_too_long")
    target_leak = _reasoning_mentions_private_info(reasoning)
    if target_leak:
        errors.append("target_leaky_reasoning")
    valid_guess = is_valid_guess(guess, history)
    if not valid_guess:
        errors.append("invalid_or_repeated_guess")
    public_consistent = public_constraints_satisfied(guess, history)
    if not public_consistent:
        errors.append("public_constraint_violation")
    format_ok = (
        match is not None
        and len(guess_tags) == 1
        and len(reasoning_matches) == 1
        and "\n" not in stripped
        and "\r" not in stripped
        and len(reasoning_words) <= MAX_REASONING_WORDS
        and not target_leak
    )
    action_ok = bool(format_ok and valid_guess and public_consistent)
    return {
        "guess": guess or "",
        "reasoning": reasoning,
        "format_ok": format_ok,
        "single_guess_tag_ok": len(guess_tags) == 1,
        "single_reasoning_tag_ok": len(reasoning_matches) == 1,
        "valid_guess": valid_guess,
        "public_constraint_valid": public_consistent,
        "target_leak": target_leak,
        "extra_text": "extra_line_or_trailing_text" in errors,
        "action_ok": action_ok,
        "errors": errors,
    }


def is_valid_guess(guess: str | None, history: list[tuple[str, str]] | None = None) -> bool:
    """Wordle validity gate using the configured legal-word source.

    We use `wordle-python`'s static 4k-word dictionary when installed; the
    local fallback only keeps direct-script sanity checks usable.
    """
    if guess is None:
        return False
    normalized = guess.lower()
    if len(normalized) != 5 or not normalized.isalpha():
        return False
    if history is not None and normalized in {prior for prior, _ in history}:
        return False
    return normalized in LEGAL_GUESSES


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


def remaining_candidates(history: list[tuple[str, str]]) -> list[str]:
    """Return target candidates consistent with the public feedback history."""
    candidates: list[str] = []
    for word in WORD_LIST:
        if all(compute_feedback(guess, word) == feedback for guess, feedback in history):
            candidates.append(word)
    return candidates


def format_candidate_list(candidates: list[str], *, max_display: int = 200) -> str:
    """Render candidates without allowing huge dictionaries to dominate prompts."""
    if not candidates:
        return "(none)"
    displayed = candidates[:max_display]
    text = ", ".join(word.upper() for word in displayed)
    if len(candidates) > max_display:
        text += f"\n(displaying first {max_display} of {len(candidates)} candidates)"
    return text


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


def _info_score(guess: str, target: str, feedback: str) -> float:
    """Per-turn information: 0.5*greens + 0.25*yellows, scaled to [0, 1]
    so that all-green = 1.0, all-yellow = 0.5, all-grey = 0.0."""
    greens = sum(1 for c in feedback if c == "G")
    yellows = sum(1 for c in feedback if c == "Y")
    return min(1.0, 0.5 * (greens / 5.0) + 0.5 * ((greens + yellows) / 5.0))


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
    """Additive Wordle reward decomposition.

    Adds exact answer, latest partial answer, shorter-solution bonus, format,
    and validity terms. Invalid terminal actions get an explicit penalty; this
    keeps repeated guesses and invented words from receiving a good reward just
    because the earlier turns were well formatted.
    """
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
        # Legacy metric aliases kept so existing W&B panels do not disappear.
        "textarena_reward": float(wordle_reward),
        "textarena_correct": correct,
        "textarena_partial": float(partial),
        "textarena_length_bonus": float(length_bonus),
        "textarena_format_reward": float(format_reward),
        "textarena_valid_guess_rate": float(valid_guess_rate),
        "textarena_terminal_valid": float(terminal_valid),
        "textarena_invalid_penalty": float(invalid_penalty),
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
    """Compact scalar reward used by the generic `reward` metric.

    The training run uses `wordle_reward`, but this keeps the legacy `reward`
    metric from looking healthy for repeated or illegal terminal actions.
    """
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


def render_feedback_message(guess: str, feedback: str, remaining: int) -> str:
    """Render the latest feedback as a compact two-line Wordle board slice."""
    word_row = " ".join(guess.upper())
    feedback_row = " ".join(feedback)
    return f"{word_row}\n{feedback_row}\nYou have {remaining} guesses left."


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
    rng.shuffle(pool)
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
            {"role": "system", "content": SHARED_PUBLIC_SYSTEM_PROMPT},
            {"role": "user", "content": build_shared_public_prompt_content([])},
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


def score_completion(example: Example, generated_text: str) -> dict[str, float]:
    """Single-shot scorer required by the protocol. Multi-turn rollout
    drives the rollout itself; this fallback handles the degenerate case
    where the client sends a single completion (e.g. for one-turn debug).
    Treats the completion as a single guess attempt."""
    target = example.metadata["target"]
    parsed = parse_turn_response(generated_text, [])
    guess = str(parsed["guess"]) or None
    single_guess_tag = float(bool(parsed["single_guess_tag_ok"]))
    format_ok = float(bool(parsed["format_ok"]))
    valid_guess = bool(parsed["valid_guess"]) and bool(parsed["public_constraint_valid"])
    if guess is None or not valid_guess:
        valid_guess_rate = 0.0
        invalid_action = True
        reward = _wordle_shaped_reward(
            solved=False,
            format_rate=format_ok,
            valid_guess_rate=valid_guess_rate,
            info_gain=0.0,
            turns_used=1,
            max_turns=MAX_TURNS,
            invalid_action=invalid_action,
        )
        return {
            "reward": reward,
            "exact_match": 0.0,
            "format_rate": format_ok,
            "single_guess_tag_rate": single_guess_tag,
            "info_gain": 0.0,
            "turns_used": 1,
            "valid_guess_rate": valid_guess_rate,
            "invalid_action": float(invalid_action),
            "strict_format_rate": format_ok,
            "public_constraint_rate": float(bool(parsed["public_constraint_valid"])),
            "target_leak_rate": float(bool(parsed["target_leak"])),
            "extra_text_rate": float(bool(parsed["extra_text"])),
            **_wordle_reward_components(
                solved=False,
                turns_with_guess=int(single_guess_tag),
                latest_feedback="",
                format_reward=format_ok,
                valid_guess_rate=valid_guess_rate,
                terminal_valid=False,
                invalid_action=invalid_action,
            ),
        }
    feedback = compute_feedback(guess, target)
    solved = float(guess == target)
    valid_guess_rate = 1.0
    info_gain = _info_score(guess, target, feedback)
    components = _wordle_reward_components(
        solved=bool(solved),
        turns_with_guess=1,
        latest_feedback=feedback,
        format_reward=format_ok,
        valid_guess_rate=valid_guess_rate,
        terminal_valid=True,
        invalid_action=False,
    )
    reward = _wordle_shaped_reward(
        solved=bool(solved),
        format_rate=format_ok,
        valid_guess_rate=valid_guess_rate,
        info_gain=info_gain,
        turns_used=1,
        max_turns=MAX_TURNS,
        invalid_action=False,
    )
    return {
        "reward": reward,
        "exact_match": solved,
        "format_rate": format_ok,
        "single_guess_tag_rate": single_guess_tag,
        "strict_format_rate": format_ok,
        "info_gain": info_gain,
        "turns_used": 1,
        "valid_guess_rate": valid_guess_rate,
        "public_constraint_rate": float(bool(parsed["public_constraint_valid"])),
        "target_leak_rate": float(bool(parsed["target_leak"])),
        "extra_text_rate": float(bool(parsed["extra_text"])),
        "invalid_action": 0.0,
        **components,
    }


def _build_turn_messages(*, target: str, history: list[tuple[str, str]], prompt_style: str = "default"):
    """Build chat messages reflecting the running game state.
    ``history`` is a list of (guess, feedback) pairs from prior turns."""
    if prompt_style == "public_reasoning":
        system_prompt = PUBLIC_REASONING_SYSTEM_PROMPT
    elif prompt_style in {
        "public_reasoning_constraints",
        "public_reasoning_constraints_candidates",
        "public_reasoning_constraints_think",
        "public_reasoning_constraints_candidates_think",
    }:
        return [
            {"role": "system", "content": SHARED_PUBLIC_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_shared_public_prompt_content(
                    history,
                    include_candidates="candidates" in prompt_style,
                ),
            },
        ]
    elif prompt_style in {
        "public_reasoning_strict",
        "public_reasoning_strict_nocandidates",
    }:
        system_prompt = PUBLIC_REASONING_STRICT_SYSTEM_PROMPT
    else:
        system_prompt = SYSTEM_PROMPT
    msgs = [{"role": "system", "content": system_prompt}]
    if prompt_style == "candidate_list":
        if history:
            transcript = "\n".join(f"{guess.upper()} -> {feedback}" for guess, feedback in history)
        else:
            transcript = "(none)"
        candidates = remaining_candidates(history)
        candidate_text = format_candidate_list(candidates)
        msgs.append({
            "role": "user",
            "content": (
                "Use only the public Wordle feedback. Do not assume the target word.\n"
                f"Previous guesses:\n{transcript}\n"
                f"Remaining candidate answer list ({len(candidates)}): {candidate_text}\n"
                "Pick a five-letter candidate that is consistent with all feedback. "
                "If more than one candidate remains, choose the most useful or likely answer. "
                "Reply with exactly one guess tag and no other text: <guess>[WORD]</guess>"
            ),
        })
    elif prompt_style == "public_reasoning":
        first_turn_request = (
            "Use only public Wordle information. Start exactly with <reasoning>, briefly choose a broad "
            "information-gathering opener in one sentence under 15 words, then close </reasoning> and output "
            "exactly one guess tag: <guess>[WORD]</guess>. Do not mention, infer, or claim a private target."
        )
        if not history:
            msgs.append({"role": "user", "content": first_turn_request})
        else:
            msgs.append({"role": "user", "content": first_turn_request})
            for i, (guess, feedback) in enumerate(history):
                msgs.append({"role": "assistant", "content": f"<guess>[{guess.upper()}]</guess>"})
                remaining = MAX_TURNS - (i + 1)
                if guess == target or remaining == 0:
                    msgs.append({"role": "user", "content": f"{render_feedback_message(guess, feedback, remaining)}\nGame over."})
                else:
                    msgs.append({
                        "role": "user",
                        "content": (
                            f"{render_feedback_message(guess, feedback, remaining)}\n"
                            "Start exactly with <reasoning>, update public "
                            "constraints and choose a public-valid next guess in one sentence under 15 words, "
                            "then close </reasoning> and output exactly one guess tag: <guess>[WORD]</guess>. "
                            "Do not mention any private target or hidden answer."
                        ),
                    })
    elif prompt_style in {
        "public_reasoning_strict",
        "public_reasoning_strict_nocandidates",
        "public_reasoning_constraints",
    }:
        if history:
            transcript = "\n".join(f"{idx}. {guess.upper()} -> {feedback}" for idx, (guess, feedback) in enumerate(history, 1))
            previous = ", ".join(guess.upper() for guess, _ in history)
            if prompt_style == "public_reasoning_strict":
                candidates = remaining_candidates(history)
                candidate_hint = format_candidate_list(candidates, max_display=40)
                request = (
                    "State before your next guess:\n"
                    f"Previous guesses and feedback:\n{transcript}\n"
                    f"Already guessed, do not repeat: {previous}\n"
                    f"Remaining public candidate answers count: {len(candidates)}\n"
                    f"Examples of public-valid candidate answers: {candidate_hint}\n"
                    "Choose one real five-letter Wordle word. Prefer a listed candidate if possible; otherwise choose a common legal word "
                    "that is consistent with all feedback. Do not invent words. "
                    "Respond exactly as <reasoning>one short sentence under 15 words</reasoning><guess>[WORD]</guess>"
                )
            elif prompt_style == "public_reasoning_strict_nocandidates":
                request = (
                    "State before your next guess:\n"
                    f"Previous guesses and feedback:\n{transcript}\n"
                    f"Already guessed, do not repeat: {previous}\n"
                    "Choose one real common five-letter Wordle answer word that is consistent with all feedback. "
                    "Do not repeat a previous guess. Do not invent words or use obscure letter strings. "
                    "Respond exactly as <reasoning>one short sentence under 15 words</reasoning><guess>[WORD]</guess>"
                )
            else:
                constraints = format_public_constraints(history)
                request = (
                    "State before your next guess:\n"
                    f"Previous guesses and feedback:\n{transcript}\n"
                    f"Public constraint summary:\n{constraints}\n"
                    f"Already guessed, do not repeat: {previous}\n"
                    "Choose one real common five-letter Wordle answer word that fits the public constraints. "
                    "Do not invent words or use obscure letter strings. "
                    "Respond exactly as <reasoning>one short sentence under 15 words</reasoning><guess>[WORD]</guess>"
                )
        else:
            request = (
                "No previous guesses. Choose a strong common legal Wordle opener such as STARE, CRANE, SLATE, or AUDIO. "
                "Do not invent a word. "
                "Respond exactly as <reasoning>one short sentence under 15 words</reasoning><guess>[WORD]</guess>"
            )
        msgs.append({"role": "user", "content": request})
    elif not history:
        msgs.append({"role": "user", "content": "Make your first guess."})
    else:
        # Re-render the entire chat history. Each prior turn the model
        # produced an assistant message containing the guess tag, and the
        # user (us) replied with the feedback line. Re-playing the full
        # transcript keeps the model's view consistent each turn.
        msgs.append({"role": "user", "content": "Make your first guess."})
        for i, (guess, feedback) in enumerate(history):
            msgs.append({"role": "assistant", "content": f"<guess>[{guess.upper()}]</guess>"})
            remaining = MAX_TURNS - (i + 1)
            if guess == target or remaining == 0:
                # Game over; we won't actually call again, but for completeness
                # include the user message form.
                msgs.append({"role": "user", "content": f"{render_feedback_message(guess, feedback, remaining)}\nGame over."})
            else:
                msgs.append({
                    "role": "user",
                    "content": (
                        f"{render_feedback_message(guess, feedback, remaining)}\n"
                        "Make your next guess."
                    ),
                })
    return msgs


def strip_think_prefix(text: str) -> str:
    """Drop a private <think>...</think> prefix; the public response follows it."""
    if "</think>" in text:
        return text.split("</think>")[-1].lstrip()
    return text


def extract_action_text(text: str) -> str:
    """Return the canonical ``<reasoning>...</reasoning><guess>WORD</guess>`` action
    line that turn validity should be judged on.

    Think-contract models wrap a private ``<think>...</think>`` block (which itself
    routinely mentions the output tags) and frequently emit trailing chatter after
    the guess. Judging the strict one-line parser on that raw text spuriously fails
    (``guess_tag_count``/``not_exact_one_line_response``/``extra_line_or_trailing_text``)
    and ends the game on a valid action. We therefore: strip the think prefix,
    truncate after the first complete guess tag, and drop any preamble before the
    tag block. Both the held-out rollout and the on-policy training rollout use this
    so they agree on whether a turn is a legal action."""
    action_text = strip_think_prefix(text or "")
    tag_match = _GUESS_TAG_RE.search(action_text)
    if tag_match:
        action_text = action_text[: tag_match.end()]
    block_start = action_text.find("<reasoning>")
    if block_start < 0:
        block_start = action_text.find("<guess>")
    if block_start > 0:
        action_text = action_text[block_start:]
    return action_text


def _build_turn_input_ids(
    tokenizer,
    *,
    target: str,
    history: list[tuple[str, str]],
    prompt_style: str = "default",
):
    """Build chat-template input_ids reflecting the running game state."""
    msgs = _build_turn_messages(target=target, history=history, prompt_style=prompt_style)
    # *_think styles open a private thinking block (the enumeration budget);
    # everything else pins the closed-think rendering (PTC-118 hazard).
    enable_thinking = prompt_style.endswith("_think")
    return tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, enable_thinking=enable_thinking, return_dict=False
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
                "<guess>[WORD]</guess> guesses that solves the game."
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


def rollout_completion(example: Example, *, generate_turn, lora_path: str, tokenizer, args) -> dict[str, float]:
    """Play a 6-turn Wordle game against the target word in
    ``example.metadata['target']``. Returns the same score blob shape as
    ``score_completion`` so the client treats it uniformly.

    ``generate_turn`` is provided by the client; its contract is::

        generate_turn(input_ids, *, lora_path, temperature, max_new_tokens) -> str

    Returning just the text of the model's reply for that turn."""
    target = example.metadata["target"].lower()
    prompt_style = getattr(args, "wordle_prompt_style", "default")
    history: list[tuple[str, str]] = []
    format_hits = 0
    single_guess_tag_hits = 0
    valid_hits = 0
    public_constraint_hits = 0
    strict_format_hits = 0
    target_leak_hits = 0
    extra_text_hits = 0
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
        raw_text = strip_think_prefix(text or "")
        # Score the ACTION on the response truncated after the first complete
        # guess tag (think prefix stripped, preamble dropped) — see
        # extract_action_text. Serving-side newline stops are unreliable, and
        # un-stopped continuation chatter would otherwise kill the game as an
        # invalid action (measures serving, not play). Style counters below
        # still use the raw (think-stripped) text.
        action_text = extract_action_text(text or "")
        parsed_raw = parse_turn_response(raw_text, history)
        parsed = parse_turn_response(action_text, history)
        guess = str(parsed["guess"]) or None
        single_guess_tag_ok = bool(parsed["single_guess_tag_ok"])
        if single_guess_tag_ok:
            single_guess_tag_hits += 1
        format_ok = bool(parsed["format_ok"])
        if format_ok:
            format_hits += 1
        if bool(parsed_raw["format_ok"]):
            strict_format_hits += 1
        if bool(parsed["public_constraint_valid"]):
            public_constraint_hits += 1
        if bool(parsed["target_leak"]):
            target_leak_hits += 1
        if bool(parsed_raw["extra_text"]):
            extra_text_hits += 1
        if not bool(parsed["action_ok"]):
            # Bad format or invalid Wordle action: end this local rollout and
            # avoid inventing public feedback for an invalid game action.
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
    public_constraint_rate = public_constraint_hits / max(turns_used, 1)
    strict_format_rate = strict_format_hits / max(turns_used, 1)
    target_leak_rate = target_leak_hits / max(turns_used, 1)
    extra_text_rate = extra_text_hits / max(turns_used, 1)
    info_gain = sum(info_scores) / max(len(info_scores), 1)
    reward = _wordle_shaped_reward(
        solved=solved,
        format_rate=format_rate,
        valid_guess_rate=valid_guess_rate,
        info_gain=info_gain,
        turns_used=turns_used,
        max_turns=MAX_TURNS,
        invalid_action=invalid_action,
    )
    wordle_components = _wordle_reward_components(
        solved=solved,
        turns_with_guess=single_guess_tag_hits,
        latest_feedback=latest_feedback,
        format_reward=1.0 if turns_used > 0 and strict_format_hits == turns_used else 0.0,
        valid_guess_rate=valid_guess_rate,
        terminal_valid=not invalid_action,
        invalid_action=invalid_action,
    )
    return {
        "reward": float(reward),
        "exact_match": float(solved),
        "format_rate": float(format_rate),
        "single_guess_tag_rate": float(single_guess_tag_rate),
        "strict_format_rate": float(strict_format_rate),
        "valid_guess_rate": float(valid_guess_rate),
        "public_constraint_rate": float(public_constraint_rate),
        "target_leak_rate": float(target_leak_rate),
        "extra_text_rate": float(extra_text_rate),
        "info_gain": float(info_gain),
        "turns_used": float(turns_used),
        "invalid_action": float(invalid_action),
        **wordle_components,
    }


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("Running wordle sanity checks...")
    # Feedback edge cases
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

    # Guess extraction
    assert extract_guess("I think the answer is <guess>CRATE</guess>") == "crate"
    assert extract_guess("<guess>crate</guess>") == "crate"
    assert extract_guess("<guess>[crate]</guess>") == "crate"
    assert extract_guess("<guess>cr at e</guess>") is None  # spaces not allowed
    assert extract_guess("no tag here") is None
    assert extract_guess("<guess>crater</guess>") is None  # too long
    assert has_single_guess_tag("<guess>[crane]</guess>")
    assert not has_single_guess_tag("<guess>[crane]</guess><guess>[apple]</guess>")
    assert is_valid_guess("apple", [])
    assert is_valid_guess("stare", [])
    assert not is_valid_guess("zzzzz", [])
    assert not is_valid_guess("apple", [("apple", "GGGGG")])
    strict = parse_turn_response("<reasoning>Choose a broad opener.</reasoning><guess>CRANE</guess>", [])
    assert strict["action_ok"], strict
    strict = parse_turn_response("<reasoning>Choose a broad opener.</reasoning><guess>CRANE</guess>\nWrite a blog post.", [])
    assert not strict["format_ok"] and strict["extra_text"], strict
    strict = parse_turn_response("<reasoning>Target has C in position 1.</reasoning><guess>CRANE</guess>", [])
    assert not strict["format_ok"] and strict["target_leak"], strict
    strict = parse_turn_response(
        "<reasoning>Try a consistent answer.</reasoning><guess>APPLE</guess>",
        [("crane", "XXXXX")],
    )
    assert not strict["action_ok"] and not strict["public_constraint_valid"], strict

    # score_completion (single-shot fallback)
    ex = Example(project="t", prompt_ids=[], metadata={"target": "crate"})
    r = score_completion(ex, "<reasoning>Use the consistent answer.</reasoning><guess>CRATE</guess>")
    assert r["exact_match"] == 1.0 and r["reward"] >= 0.7, r
    r = score_completion(ex, "<reasoning>Try a valid anagram.</reasoning><guess>TRACE</guess>")
    assert r["exact_match"] == 0.0 and 0 < r["reward"] < 0.5, r
    r = score_completion(ex, "lol no")
    assert r["reward"] < 0.0 and r["wordle_reward"] < 0.0, r

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
            return "<reasoning>Choose a broad opener.</reasoning><guess>STARE</guess>"
        return "<reasoning>Use the remaining answer.</reasoning><guess>CRATE</guess>"
    args = SimpleNamespace(rollout_temperature=0.6, rollout_max_new_tokens=32)
    result = rollout_completion(ex, generate_turn=fake_generate, lora_path="x", tokenizer=FakeTok(), args=args)
    assert result["exact_match"] == 1.0
    assert result["turns_used"] == 2.0
    assert result["format_rate"] == 1.0
    assert len(calls) == 2

    # Failed rollout (bad format on turn 1)
    def fake_bad(input_ids, **kw):
        return "i don't know"
    result = rollout_completion(ex, generate_turn=fake_bad, lora_path="x", tokenizer=FakeTok(), args=args)
    assert result["exact_match"] == 0.0
    assert result["format_rate"] == 0.0
    assert result["turns_used"] == 1.0
    assert result["invalid_action"] == 1.0
    assert result["wordle_reward"] < 0.0

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
