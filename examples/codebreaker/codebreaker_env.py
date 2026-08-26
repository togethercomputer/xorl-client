"""Procedural Codebreaker environment: generalized Mastermind over per-episode symbol alphabets.

Successor env to wordle-multiturn, designed to fix the separation weaknesses measured in the
wordle program (RESULTS.md): a fixed 4,266-word target list invites instance memorization and a
single near-optimal opener tree; positional G/Y/X feedback plus a re-rendered board makes each
turn an almost-fresh CSP with shallow cross-turn credit; and the deterministic game leaves a
learned value head nothing to model beyond obo bias. Codebreaker counters each point:

  * PROCEDURAL INSTANCES: every episode samples a fresh alphabet of ``k`` symbols from a
    ~1.5k-symbol CVC universe and a fresh secret code of ``code_length`` symbols (repeats
    allowed by default). Train/eval is a split over generator draws, not a word list — there
    is no instance to memorize and no universal opener.
  * COUNT-ONLY FEEDBACK: each guess returns ``exact`` (right symbol, right position) and
    ``close`` (right symbol, wrong position) COUNTS with standard Mastermind duplicate
    handling — no per-position marks, so disambiguation requires intersecting every prior
    (guess, feedback) pair across turns.
  * DIFFICULTY DIAL: the hypothesis space is k^L (repeats) — (6,3)=216 below Wordle,
    (10,4)=10^4 ~ Wordle-hard, (16,6)=1.7e7 far above — tuned by two integers instead of
    empirical word mining.
  * NEUTRAL VERIFIER (user rule: no human bias in the verifier): any well-formed L-tuple over
    the instance alphabet is a legal guess (repeat guesses excepted, as in wordle); solved is
    exact string match. No strategy judgment anywhere in the reward path.
  * SEPARATION TOGGLES: ``apply_feedback_noise`` (stochastic feedback -> value estimation
    becomes load-bearing) and the ``multiturn_last_only`` prompt style (constraints survive
    only in the model's own generated text -> cross-turn credit actually binds).

Mirrors ``wordle_env.py``'s role and interface shape so the rollout/eval/recipe layers port
with minimal diffs: stdlib only (``re``, ``random``, ``math``, ``itertools``, ``dataclasses``),
``Example`` with the same fields, ``build_examples`` returning (train, eval, val), the
``build_mt_*`` conversational prompt builders, and the same validity/consistency predicate
split. Differences the porting layer must absorb (both flagged in the relevant docstrings):
the turn-1 prompt is INSTANCE-DEPENDENT (per-episode alphabet — no global initial-prompt
cache), and candidate/info-bits tracking is exact only while k^L <= CANDIDATE_ENUM_CAP.

The multiturn recipe's loose per-turn format gate (single guess tag + parseable guess, as
computed in the rollout) is kept; the wordle-baseline strict one-line ``parse_turn_response``
grader is deliberately not ported.
"""

from __future__ import annotations

import itertools
import math
import random
import re
from dataclasses import dataclass, field


is_multi_turn = True

MAX_TURNS = 10

# Exact candidate tracking (info-bits telemetry / td step rewards) is enabled only while the
# instance hypothesis space k^L fits under this cap; larger instances play identically but
# report info_bits untracked. (10,4)=1e4 and (12,5)=2.5e5 track; (16,6)=1.7e7 does not.
CANDIDATE_ENUM_CAP = 300_000


@dataclass
class Example:
    """One Codebreaker game seed. ``prompt_ids`` is intentionally left EMPTY: unlike wordle's
    target-independent turn-1 prompt, the codebreaker initial prompt embeds the per-episode
    alphabet, so the rollout renders it at play time from ``metadata['target']`` (the encoded
    instance) — pre-tokenizing tens of thousands of distinct prompts at startup buys nothing."""

    project: str
    prompt_ids: list[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Symbol universe — CVC syllables over restricted consonant/vowel sets.
# Uppercase, whitespace-separated in prompts and guesses, so parsing is unambiguous and
# tokenizer-friendly. Common English 3-letter words are blocklisted so alphabet symbols do
# not import lexical priors (the secret is uniform over the alphabet either way; this keeps
# guessing behavior from skewing toward "wordy" symbols).
# ---------------------------------------------------------------------------

SYMBOL_CONSONANTS = "BDFGHJKLMNPRSTVWXZ"  # no C/Q/Y: fewer real words, no soft-C ambiguity
SYMBOL_VOWELS = "AEIOU"

COMMON_WORD_BLOCKLIST = frozenset(
    """
    BAD BAG BAN BAR BAT BED BEG BET BID BIG BIN BIT BOB BOG BOP BOT BUD BUG BUM BUN BUS BUT
    DAB DAD DAM DEN DID DIG DIM DIN DIP DOG DOT DUB DUD DUG FAD FAN FAR FAT FED FIB FIG FIN
    FIT FIX FOG FOX FUN FUR GAG GAL GAP GAS GEL GEM GET GIG GIN GOB GOD GOT GUM GUN GUT HAD
    HAG HAM HAS HAT HEM HEN HID HIM HIP HIS HIT HOG HOP HOT HUB HUG HUM HUT JAB JAM JAR JET
    JIG JOB JOG JOT JUG KEG KID KIN KIT LAB LAD LAG LAP LEG LET LID LIP LIT LOG LOT MAD MAN
    MAP MAT MEN MET MID MIX MOB MOM MOP MUD MUG NAB NAG NAP NET NIL NIP NOD NOR NOT NUN NUT
    PAD PAL PAN PAR PAT PEG PEN PET PIG PIN PIT POD POP POT PUB PUG PUN PUP PUT RAG RAM RAN
    RAP RAT RED RIB RID RIG RIM RIP ROB ROD ROT RUB RUG RUM RUN RUT SAD SAG SAP SAT SET SIN
    SIP SIR SIT SIX SOB SON SUB SUM SUN TAB TAG TAN TAP TAR TAX TEN TIN TIP TON TOP TOT TUB
    TUG VAN VAT VET VIM WAG WAR WAS WAX WEB WED WET WIG WIN WIT WON ZAP ZIP
    """.split()
)

SYMBOL_UNIVERSE = [
    c1 + v + c2
    for c1 in SYMBOL_CONSONANTS
    for v in SYMBOL_VOWELS
    for c2 in SYMBOL_CONSONANTS
    if c1 + v + c2 not in COMMON_WORD_BLOCKLIST
]


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Instance:
    """One episode's ground truth: the sampled alphabet (k distinct symbols, uppercase) and the
    secret code (``code_length`` symbols drawn from it). ``allow_repeats`` records the GENERATION
    rule (whether the secret may repeat symbols) — it changes the prompt's stated rules and the
    hypothesis count, but never guess validity (a repeated-symbol guess is informative and legal
    regardless)."""

    alphabet: tuple[str, ...]
    secret: tuple[str, ...]
    allow_repeats: bool = True

    @property
    def k(self) -> int:
        return len(self.alphabet)

    @property
    def code_length(self) -> int:
        return len(self.secret)


def generate_instance(
    rng: random.Random, *, k: int, code_length: int, allow_repeats: bool = True
) -> Instance:
    """Sample one instance: alphabet = k distinct symbols from the universe, secret = code of
    ``code_length`` symbols over it (with replacement iff ``allow_repeats``)."""
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if k > len(SYMBOL_UNIVERSE):
        raise ValueError(f"k={k} exceeds the symbol universe ({len(SYMBOL_UNIVERSE)})")
    if code_length < 1:
        raise ValueError(f"code_length must be >= 1, got {code_length}")
    if not allow_repeats and code_length > k:
        raise ValueError(f"no-repeat code needs code_length <= k, got L={code_length} k={k}")
    alphabet = tuple(rng.sample(SYMBOL_UNIVERSE, k))
    if allow_repeats:
        secret = tuple(rng.choice(alphabet) for _ in range(code_length))
    else:
        secret = tuple(rng.sample(list(alphabet), code_length))
    return Instance(alphabet=alphabet, secret=secret, allow_repeats=allow_repeats)


def encode_instance(instance: Instance) -> str:
    """Canonical single-string form (rides through ``Example.metadata['target']`` and
    ``Trajectory.target`` without widening the harness plumbing): ``R|ALPHABET|SECRET`` with
    space-separated symbols; leading flag R/N = repeats allowed / not."""
    flag = "R" if instance.allow_repeats else "N"
    return f"{flag}|{' '.join(instance.alphabet)}|{' '.join(instance.secret)}"


def decode_instance(encoded: str) -> Instance:
    parts = encoded.strip().upper().split("|")
    if len(parts) != 3 or parts[0] not in ("R", "N"):
        raise ValueError(f"malformed instance encoding: {encoded!r}")
    alphabet = tuple(parts[1].split())
    secret = tuple(parts[2].split())
    if len(set(alphabet)) != len(alphabet):
        raise ValueError(f"alphabet has duplicate symbols: {encoded!r}")
    if not secret or any(s not in alphabet for s in secret):
        raise ValueError(f"secret not drawn from alphabet: {encoded!r}")
    return Instance(alphabet=alphabet, secret=secret, allow_repeats=parts[0] == "R")


def hypothesis_count(k: int, code_length: int, *, allow_repeats: bool = True) -> int:
    """Size of the code space the model must search: k^L, or falling factorial without repeats."""
    if allow_repeats:
        return k**code_length
    n = 1
    for i in range(code_length):
        n *= k - i
    return n


# ---------------------------------------------------------------------------
# Feedback — standard Mastermind counting with correct duplicate handling.
# ---------------------------------------------------------------------------


def compute_feedback(guess: tuple[str, ...], secret: tuple[str, ...]) -> tuple[int, int]:
    """(exact, close): exact = positions matching; close = remaining multiset overlap. Duplicate
    handling is the classic rule — each secret symbol credits at most one guess symbol, exacts
    first. Hard-raises on length mismatch (validity gating upstream must prevent it)."""
    if len(guess) != len(secret):
        raise ValueError(f"guess/secret length mismatch: {len(guess)} vs {len(secret)}")
    exact = sum(1 for g, s in zip(guess, secret, strict=True) if g == s)
    overlap = sum(min(guess.count(sym), secret.count(sym)) for sym in set(guess))
    return exact, overlap - exact


def render_feedback(exact: int, close: int) -> str:
    return f"exact={exact} close={close}"


def apply_feedback_noise(
    exact: int, close: int, code_length: int, rng: random.Random, eps: float
) -> tuple[int, int]:
    """Stochastic-feedback toggle (default OFF): each count independently gets +/-1 with prob
    ``eps``, clamped to the basic invariants (0 <= exact <= L-1 for an unsolved guess — a false
    "exact=L" would announce a solve the game does not grant; 0 <= close <= L - exact). Solve
    detection stays ground truth upstream (guess == secret ends the game before feedback), so
    noise only perturbs the message. CALLER CONTRACT: derive ``rng`` deterministically from the
    trajectory's stable identifiers + turn (seed discipline — coroutines resume in completion
    order, so drawing from a shared RNG here would break reproducibility). Candidate tracking
    is only meaningful at eps=0 (noise can filter out the true secret); disable it when on."""
    if eps <= 0.0:
        return exact, close
    L = code_length
    if exact >= L:  # true solve: never corrupted (games end before feedback anyway)
        return exact, close

    def jitter(v: int) -> int:
        return v + rng.choice((-1, 1)) if rng.random() < eps else v

    e = min(max(jitter(exact), 0), L - 1)
    c = min(max(jitter(close), 0), L - e)
    return e, c


# ---------------------------------------------------------------------------
# Parsing & validity
# ---------------------------------------------------------------------------

_GUESS_TAG_RE = re.compile(r"<guess\b[^>]*>.*?</guess>", re.IGNORECASE | re.DOTALL)
_GUESS_CONTENT_RE = re.compile(r"<guess>\s*(.*?)\s*</guess>", re.IGNORECASE | re.DOTALL)
# ASCII-only, no IGNORECASE on the symbol class: under re.IGNORECASE [A-Za-z] admits unicode
# look-alikes (U+0130 crashed wordle jobs 34680/34685 via a length-changing .lower()).
_SYMBOL_RUN_RE = re.compile(r"[A-Za-z]+")


def extract_guess(text: str) -> tuple[str, ...] | None:
    """Pull the guessed symbol tuple out of the FIRST guess tag (uppercased), or None. Accepts
    bracketed ``<guess>[VEX KOB]</guess>`` and bare ``<guess>VEX KOB</guess>``; symbols may be
    separated by any non-letter runs (spaces, commas). Arity/membership are validity's job."""
    if not text:
        return None
    m = _GUESS_CONTENT_RE.search(text)
    if m is None:
        return None
    symbols = _SYMBOL_RUN_RE.findall(m.group(1))
    if not symbols:
        return None
    return tuple(s.upper() for s in symbols)


def has_single_guess_tag(text: str) -> bool:
    """Format gate: every assistant turn should contain exactly one guess tag."""
    if not text:
        return False
    return len(_GUESS_TAG_RE.findall(text)) == 1


def is_well_formed_guess(guess: tuple[str, ...] | None, code_length: int) -> bool:
    """Right arity, every symbol a pure-ASCII alphabetic token. The single well-formedness
    predicate for every consumer (validity, consistency telemetry) — same convention as
    wordle_env.is_well_formed_guess."""
    return (
        guess is not None
        and len(guess) == code_length
        and all(s.isascii() and s.isalpha() for s in guess)
    )


def is_valid_guess(
    guess: tuple[str, ...] | None,
    instance: Instance,
    history: list[tuple[tuple[str, ...], tuple[int, int]]] | None = None,
) -> bool:
    """Neutral validity gate: well-formed, every symbol in the instance alphabet, not a repeat
    of a prior guess. Nothing else — no strategy judgment, no dictionary beyond the alphabet."""
    if not is_well_formed_guess(guess, instance.code_length):
        return False
    assert guess is not None
    if any(s not in instance.alphabet for s in guess):
        return False
    if history is not None and guess in {prior for prior, _ in history}:
        return False
    return True


def invalid_reason(
    guess: tuple[str, ...] | None,
    text: str,
    instance: Instance,
    history: list[tuple[tuple[str, ...], tuple[int, int]]],
) -> str:
    """Reason code for an invalid turn (feeds INVALID_REASON_MESSAGES under burn_turn)."""
    if not has_single_guess_tag(text or ""):
        return "not_exactly_one_guess_tag"
    if guess is None:
        return "no_parseable_guess"
    if len(guess) != instance.code_length:
        return "wrong_code_length"
    if any(not (s.isascii() and s.isalpha()) or s not in instance.alphabet for s in guess):
        return "symbol_not_in_alphabet"
    if guess in {prior for prior, _ in history}:
        return "repeated_guess"
    return "unknown_invalid"


def constraints_satisfied(
    guess: tuple[str, ...] | None,
    history: list[tuple[tuple[str, ...], tuple[int, int]]] | None = None,
    *,
    code_length: int,
) -> bool:
    """Whether ``guess`` could still be the secret under every delivered feedback — the direct
    cross-turn information-use gauge (graded only for well-formed guesses; malformed is False,
    never an exception). Exact and cheap: no candidate enumeration involved."""
    if not is_well_formed_guess(guess, code_length):
        return False
    assert guess is not None
    return all(compute_feedback(prior, guess) == fb for prior, fb in history or [])


# ---------------------------------------------------------------------------
# Candidate tracking (info-bits ground truth) — exact only under CANDIDATE_ENUM_CAP.
# ---------------------------------------------------------------------------


def enumerate_candidates(instance: Instance) -> list[tuple[str, ...]] | None:
    """All codes consistent with zero feedback = the full hypothesis space, or None when it
    exceeds CANDIDATE_ENUM_CAP (instance plays identically; info_bits reported untracked)."""
    if hypothesis_count(instance.k, instance.code_length, allow_repeats=instance.allow_repeats) > CANDIDATE_ENUM_CAP:
        return None
    if instance.allow_repeats:
        return list(itertools.product(instance.alphabet, repeat=instance.code_length))
    return [c for c in itertools.permutations(instance.alphabet, instance.code_length)]


def filter_candidates(
    candidates: list[tuple[str, ...]], guess: tuple[str, ...], feedback: tuple[int, int]
) -> list[tuple[str, ...]]:
    """One incremental consistency step (same invariant as wordle's: folding filter_candidates
    over a history equals filtering the full enumeration by every pair at once). Only sound at
    noise eps=0 — noisy feedback can filter out the true secret."""
    return [c for c in candidates if compute_feedback(guess, c) == feedback]


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

SHARED_PUBLIC_SYSTEM_PROMPT = "You are playing Codebreaker. Follow the game prompt exactly."

# multiturn_transcript: feedback messages re-show the full numbered (guess -> counts) history —
# information parity with a re-rendered board (the baseline arm). multiturn_plain: latest
# feedback + already-guessed list only; earlier COUNTS are not re-shown, so the model must
# retain them. multiturn_last_only: the leanest style — latest feedback and the format
# reminder, nothing else; ALL constraint state must be carried in the model's own generated
# text, so cross-turn credit assignment actually binds (the arm where game-level methods
# should separate from turn-level ones).
MT_PROMPT_STYLES = ("multiturn_transcript", "multiturn_plain", "multiturn_last_only")

INVALID_REASON_MESSAGES = {
    "not_exactly_one_guess_tag": "your reply must contain exactly one <guess>[...]</guess> tag",
    "no_parseable_guess": "no symbols could be read from your <guess> tag",
    "wrong_code_length": "your guess must contain exactly the code's number of symbols",
    "symbol_not_in_alphabet": "every symbol in your guess must come from the game alphabet",
    "repeated_guess": "you already tried that exact guess",
    "unknown_invalid": "the guess was not accepted",
}


def render_guess(guess: tuple[str, ...]) -> str:
    return " ".join(guess)


def format_example_guess(instance: Instance) -> str:
    """Deterministic legal example for the format spec: the first code_length alphabet symbols
    (same bias class as wordle's STARE example — a legal, concrete action)."""
    return render_guess(tuple(instance.alphabet[: instance.code_length]))


def mt_format_reminder(instance: Instance) -> str:
    return (
        "Reply in this format, replacing the example with your own guess "
        "consistent with all feedback: <reasoning>your deduction</reasoning>"
        f"<guess>[{format_example_guess(instance)}]</guess>"
    )


def build_shared_public_prompt_content(instance: Instance, *, max_turns: int = MAX_TURNS) -> str:
    """Turn-1 user content: full rules + the per-episode alphabet + the output format. NOTE:
    instance-dependent (unlike wordle's shared prompt) — the rollout must render per game, not
    once per tokenizer."""
    L = instance.code_length
    repeat_rule = (
        "Symbols MAY repeat within the code." if instance.allow_repeats else "Each symbol appears at most once in the code."
    )
    return (
        f"You are playing Codebreaker. A secret code of {L} symbols has been chosen. "
        f"{repeat_rule} The symbols come from this alphabet:\n"
        f"{' '.join(instance.alphabet)}\n\n"
        f"You have {max_turns} attempts. After each guess you receive two counts:\n"
        "exact = symbols that are correct AND in the correct position\n"
        "close = additional symbols that are in the code but in the wrong position\n\n"
        "Use only the feedback you receive; there is no other information about the code.\n\n"
        "Reply in this format, replacing the example with your own guess "
        f"of {L} symbols from the alphabet:\n"
        f"<reasoning>your deduction</reasoning><guess>[{format_example_guess(instance)}]</guess>\n\n"
        "Rules:\n"
        "- In <reasoning>, work through what the feedback implies before choosing; a few sentences are fine.\n"
        f"- The guess must be exactly {L} symbols separated by single spaces, each from the alphabet above.\n"
        "- Symbols may repeat within a guess.\n"
        "- Do not repeat a previous guess.\n\n"
        "Begin your response now."
    )


def build_mt_initial_messages(instance: Instance, *, max_turns: int = MAX_TURNS) -> list[dict]:
    """Turn-1 chat messages (system + user). Instance-dependent; see build_shared_public_prompt_content."""
    return [
        {"role": "system", "content": SHARED_PUBLIC_SYSTEM_PROMPT},
        {"role": "user", "content": build_shared_public_prompt_content(instance, max_turns=max_turns)},
    ]


def format_public_transcript(history: list[tuple[tuple[str, ...], tuple[int, int]]]) -> str:
    if not history:
        return "(none)"
    return "\n".join(
        f"{idx}. {render_guess(guess)} -> {render_feedback(*fb)}" for idx, (guess, fb) in enumerate(history, 1)
    )


def _mt_style_state_blocks(
    history: list[tuple[tuple[str, ...], tuple[int, int]]], prompt_style: str
) -> list[str]:
    """Style-dependent state blocks shared by the feedback and invalid-turn messages. Validates
    the style — both callers rely on this raise (same contract as wordle_env)."""
    if prompt_style not in MT_PROMPT_STYLES:
        raise ValueError(f"unknown multiturn prompt_style {prompt_style!r}; expected one of {MT_PROMPT_STYLES}")
    parts: list[str] = []
    if not history:
        return parts
    if prompt_style == "multiturn_transcript":
        parts.append(f"All guesses and feedback so far:\n{format_public_transcript(history)}")
    elif prompt_style == "multiturn_plain":
        previous = "; ".join(render_guess(g) for g, _ in history)
        parts.append(f"Already guessed, do not repeat: {previous}")
    return parts


def render_feedback_message(guess: tuple[str, ...], exact: int, close: int, remaining: int) -> str:
    return f"Your guess: {render_guess(guess)}\n{render_feedback(exact, close)}\nYou have {remaining} guesses left."


def build_mt_feedback_content(
    history: list[tuple[tuple[str, ...], tuple[int, int]]],
    *,
    instance: Instance,
    prompt_style: str = "multiturn_transcript",
    max_turns: int = MAX_TURNS,
    remaining: int | None = None,
) -> str:
    """User-message content delivering the latest feedback mid-game. ``history`` must be
    non-empty (its last entry is the guess being answered) and non-terminal. ``remaining``
    overrides the guesses-left count: under burn_turn, burned turns consume budget WITHOUT
    entering history, so the len(history) default would overstate what the loop grants."""
    state_blocks = _mt_style_state_blocks(history, prompt_style)
    if not history:
        raise ValueError("build_mt_feedback_content needs at least one (guess, feedback) pair")
    guess, (exact, close) = history[-1]
    if remaining is None:
        remaining = max_turns - len(history)
    parts = [render_feedback_message(guess, exact, close, remaining), *state_blocks, mt_format_reminder(instance)]
    return "\n\n".join(parts)


def build_mt_invalid_feedback_content(
    reason: str,
    history: list[tuple[tuple[str, ...], tuple[int, int]]],
    *,
    instance: Instance,
    prompt_style: str = "multiturn_transcript",
    remaining: int = 1,
) -> str:
    """User-message content after an INVALID turn under burn_turn: rejection reason, spent-turn
    notice, then the same style-dependent state blocks the normal feedback message carries."""
    state_blocks = _mt_style_state_blocks(history, prompt_style)
    message = INVALID_REASON_MESSAGES.get(reason, INVALID_REASON_MESSAGES["unknown_invalid"])
    parts = [
        f"Invalid guess: {message}. That attempt is spent — no feedback.\nYou have {remaining} guesses left.",
        *state_blocks,
        mt_format_reminder(instance),
    ]
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Reward primitives (binary reward is assembled in codebreaker_scoring; these are the
# env-owned pieces + diagnostics)
# ---------------------------------------------------------------------------


def _info_score(exact: int, close: int, code_length: int) -> float:
    """Per-turn feedback richness scaled like wordle's: all-exact=1.0, all-close=0.5, none=0.0.
    Diagnostic only — never a training reward in the binary phase."""
    return min(1.0, (exact + 0.5 * close) / max(1, code_length))


def _sparse_reward(*, solved: bool, turns_used: int, max_turns: int) -> float:
    """Solve (+ faster-solve bonus) or nothing; leaves ALL intra-game credit to the estimator."""
    if not solved:
        return 0.0
    return 1.0 + 0.5 * (max_turns - turns_used) / max(1, max_turns)


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


def build_examples(
    tokenizer,
    *,
    train_size: int = 1024,
    eval_size: int = 256,
    val_size: int = 0,
    seed: int = 0,
    k: int = 10,
    code_length: int = 4,
    allow_repeats: bool = True,
    k_range: tuple[int, int] | None = None,
    code_length_range: tuple[int, int] | None = None,
):
    """Generate disjoint procedural instances for train / eval(TEST) / validation splits.

    One seeded stream draws train+eval+val instances in order with duplicate rejection (a seen
    set over encodings), then slices — so the splits are disjoint by construction and each
    slice's identity is stable under the same (seed, sizes, difficulty) regardless of how the
    others are consumed. ``k_range`` / ``code_length_range`` (inclusive) sample per-instance
    difficulty for mixed-difficulty pools; omitted means fixed ``k`` / ``code_length``.

    ``tokenizer`` is accepted for interface parity but unused: prompt_ids stays empty (see
    Example — the instance-dependent turn-1 prompt is rendered at play time)."""
    del tokenizer
    rng = random.Random(seed)
    total = train_size + eval_size + val_size
    seen: set[str] = set()
    instances: list[Instance] = []
    while len(instances) < total:
        ik = rng.randint(*k_range) if k_range is not None else k
        il = rng.randint(*code_length_range) if code_length_range is not None else code_length
        inst = generate_instance(rng, k=ik, code_length=il, allow_repeats=allow_repeats)
        enc = encode_instance(inst)
        if enc in seen:
            continue
        seen.add(enc)
        instances.append(inst)

    def to_example(inst: Instance, pid: str) -> Example:
        return Example(project=pid, prompt_ids=[], metadata={"target": encode_instance(inst)})

    train = [to_example(inst, f"cb_train_{i:05d}") for i, inst in enumerate(instances[:train_size])]
    eval_ = [
        to_example(inst, f"cb_eval_{i:05d}")
        for i, inst in enumerate(instances[train_size : train_size + eval_size])
    ]
    val = [
        to_example(inst, f"cb_val_{i:05d}")
        for i, inst in enumerate(instances[train_size + eval_size : total])
    ]
    return train, eval_, val


def describe_difficulty(k: int, code_length: int, *, allow_repeats: bool = True) -> str:
    """Human-readable difficulty line for run logs: hypothesis count and bits."""
    n = hypothesis_count(k, code_length, allow_repeats=allow_repeats)
    return f"k={k} L={code_length} repeats={'yes' if allow_repeats else 'no'}: {n:,} codes ({math.log2(n):.1f} bits)"
