"""CPU test suite for the codebreaker-multiturn slime port.

Three layers:
  1. env + scoring tests carried over VERBATIM from xorl's test_codebreaker_smoke.py (the two
     modules are byte-identical copies; these tests passing proves the copy is undamaged);
  2. chat-glue derivation tests against the FakeChatTok ChatML tokenizer (also from xorl);
  3. slime-side tests: generate_with_codebreaker against a scripted fake engine (token
     exactness, loss-mask alignment, truncation, terminate/burn_turn, seed determinism) and
     reward_post_process (xorl-reference equality + the obo +-1/sqrt(n) property).

Layers 1-2 need only stdlib+pytest. Layer 3 additionally needs torch + httpx (slime.utils
imports); it is skipped automatically where those are missing.

Run:  python -m pytest examples/codebreaker_multiturn/test_codebreaker_port.py -q
"""

import asyncio
import math
import os
import random
import re

import pytest

from examples.codebreaker import codebreaker_env as cb


HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Symbol universe
# ---------------------------------------------------------------------------


def test_symbol_universe_shape():
    assert len(cb.SYMBOL_UNIVERSE) > 1000
    assert len(set(cb.SYMBOL_UNIVERSE)) == len(cb.SYMBOL_UNIVERSE)
    for sym in cb.SYMBOL_UNIVERSE:
        assert len(sym) == 3
        assert sym[0] in cb.SYMBOL_CONSONANTS
        assert sym[1] in cb.SYMBOL_VOWELS
        assert sym[2] in cb.SYMBOL_CONSONANTS
        assert sym == sym.upper()


def test_symbol_universe_blocklists_common_words():
    for word in ("DOG", "BAT", "SUN", "MAP", "RUN", "WEB"):
        assert word not in cb.SYMBOL_UNIVERSE


# ---------------------------------------------------------------------------
# Instance generation / encoding
# ---------------------------------------------------------------------------


def test_generate_instance_basic():
    inst = cb.generate_instance(random.Random(7), k=10, code_length=4)
    assert inst.k == 10
    assert inst.code_length == 4
    assert len(set(inst.alphabet)) == 10
    assert all(s in cb.SYMBOL_UNIVERSE for s in inst.alphabet)
    assert all(s in inst.alphabet for s in inst.secret)


def test_generate_instance_deterministic():
    a = cb.generate_instance(random.Random(123), k=12, code_length=5)
    b = cb.generate_instance(random.Random(123), k=12, code_length=5)
    assert a == b
    c = cb.generate_instance(random.Random(124), k=12, code_length=5)
    assert a != c


def test_generate_instance_no_repeats_mode():
    inst = cb.generate_instance(random.Random(3), k=8, code_length=6, allow_repeats=False)
    assert len(set(inst.secret)) == 6


def test_generate_instance_validation():
    rng = random.Random(0)
    with pytest.raises(ValueError):
        cb.generate_instance(rng, k=1, code_length=3)
    with pytest.raises(ValueError):
        cb.generate_instance(rng, k=10, code_length=0)
    with pytest.raises(ValueError):
        cb.generate_instance(rng, k=4, code_length=5, allow_repeats=False)
    with pytest.raises(ValueError):
        cb.generate_instance(rng, k=len(cb.SYMBOL_UNIVERSE) + 1, code_length=3)


def test_encode_decode_roundtrip():
    for seed in range(20):
        inst = cb.generate_instance(random.Random(seed), k=10, code_length=4)
        assert cb.decode_instance(cb.encode_instance(inst)) == inst
    inst = cb.generate_instance(random.Random(5), k=8, code_length=5, allow_repeats=False)
    assert cb.decode_instance(cb.encode_instance(inst)) == inst


def test_decode_instance_rejects_malformed():
    with pytest.raises(ValueError):
        cb.decode_instance("VEX KOB|ZAL")  # missing flag
    with pytest.raises(ValueError):
        cb.decode_instance("R|VEX VEX KOB|VEX")  # duplicate alphabet symbol
    with pytest.raises(ValueError):
        cb.decode_instance("R|VEX KOB|MIR")  # secret outside alphabet
    with pytest.raises(ValueError):
        cb.decode_instance("R|VEX KOB|")  # empty secret


def test_hypothesis_count():
    assert cb.hypothesis_count(10, 4) == 10_000
    assert cb.hypothesis_count(6, 3) == 216
    assert cb.hypothesis_count(16, 6) == 16_777_216
    assert cb.hypothesis_count(5, 3, allow_repeats=False) == 60


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

SECRET = ("ZAL", "VEX", "ZAL", "DAK")


def test_feedback_transcript_example():
    # The worked example from the design discussion.
    assert cb.compute_feedback(("VEX", "KOB", "MIR", "ZAL"), SECRET) == (0, 2)
    assert cb.compute_feedback(("ZAL", "VEX", "TUF", "PYN"), SECRET) == (2, 0)
    assert cb.compute_feedback(("ZAL", "VEX", "GOR", "WEB"), SECRET) == (2, 0)
    assert cb.compute_feedback(SECRET, SECRET) == (4, 0)


def test_feedback_duplicate_handling():
    # Each secret symbol credits at most one guess symbol; exacts take priority.
    assert cb.compute_feedback(("VEX", "KOB", "VEX"), ("VEX", "VEX", "KOB")) == (1, 2)
    assert cb.compute_feedback(("VEX", "VEX", "VEX"), ("VEX", "KOB", "MIR")) == (1, 0)
    assert cb.compute_feedback(("KOB", "VEX"), ("VEX", "KOB")) == (0, 2)
    assert cb.compute_feedback(("VEX", "VEX"), ("VEX", "VEX")) == (2, 0)


def test_feedback_length_mismatch_raises():
    with pytest.raises(ValueError):
        cb.compute_feedback(("VEX",), ("VEX", "KOB"))


def test_feedback_invariants_random():
    rng = random.Random(99)
    for _ in range(500):
        inst = cb.generate_instance(rng, k=8, code_length=5)
        guess = tuple(rng.choice(inst.alphabet) for _ in range(5))
        exact, close = cb.compute_feedback(guess, inst.secret)
        assert 0 <= exact <= 5
        assert 0 <= close <= 5 - exact
        assert (exact == 5) == (guess == inst.secret)


def test_render_feedback():
    assert cb.render_feedback(2, 1) == "exact=2 close=1"


# ---------------------------------------------------------------------------
# Feedback noise
# ---------------------------------------------------------------------------


def test_noise_eps_zero_is_identity():
    rng = random.Random(1)
    for _ in range(50):
        e, c = rng.randint(0, 4), rng.randint(0, 2)
        assert cb.apply_feedback_noise(e, c, 4, rng, 0.0) == (e, c)


def test_noise_deterministic_given_rng():
    a = cb.apply_feedback_noise(1, 2, 4, random.Random(42), 0.5)
    b = cb.apply_feedback_noise(1, 2, 4, random.Random(42), 0.5)
    assert a == b


def test_noise_invariants():
    rng = random.Random(7)
    L = 4
    changed = 0
    for _ in range(2000):
        e0, c0 = rng.randint(0, L - 1), rng.randint(0, 2)
        c0 = min(c0, L - e0)
        e, c = cb.apply_feedback_noise(e0, c0, L, rng, 0.5)
        assert 0 <= e <= L - 1  # unsolved guess can never be reported as a full solve
        assert 0 <= c <= L - e
        changed += (e, c) != (e0, c0)
    assert changed > 200  # noise actually fires


def test_noise_never_corrupts_true_solve():
    assert cb.apply_feedback_noise(4, 0, 4, random.Random(0), 1.0) == (4, 0)


# ---------------------------------------------------------------------------
# Parsing & validity
# ---------------------------------------------------------------------------


def _instance() -> cb.Instance:
    return cb.Instance(
        alphabet=("VEX", "KOB", "MIR", "ZAL", "TUF", "PYN", "GOR", "WEB", "DAK", "SUL"),
        secret=("ZAL", "VEX", "ZAL", "DAK"),
    )


def test_extract_guess_variants():
    want = ("VEX", "KOB", "MIR", "ZAL")
    assert cb.extract_guess("<reasoning>x</reasoning><guess>[VEX KOB MIR ZAL]</guess>") == want
    assert cb.extract_guess("<guess>VEX KOB MIR ZAL</guess>") == want
    assert cb.extract_guess("<guess>[vex, kob, mir, zal]</guess>") == want
    assert cb.extract_guess("<GUESS>[VEX KOB MIR ZAL]</GUESS>") == want
    assert cb.extract_guess("no tag here") is None
    assert cb.extract_guess("<guess>[]</guess>") is None
    assert cb.extract_guess("") is None


def test_extract_guess_uses_first_tag():
    text = "<guess>[VEX VEX VEX VEX]</guess> then <guess>[KOB KOB KOB KOB]</guess>"
    assert cb.extract_guess(text) == ("VEX", "VEX", "VEX", "VEX")
    assert not cb.has_single_guess_tag(text)


def test_has_single_guess_tag():
    assert cb.has_single_guess_tag("<guess>[VEX KOB]</guess>")
    assert not cb.has_single_guess_tag("")
    assert not cb.has_single_guess_tag("nothing")


def test_is_valid_guess():
    inst = _instance()
    assert cb.is_valid_guess(("VEX", "KOB", "MIR", "ZAL"), inst)
    assert cb.is_valid_guess(("VEX", "VEX", "VEX", "VEX"), inst)  # repeats within a guess: legal
    assert not cb.is_valid_guess(("VEX", "KOB", "MIR"), inst)  # wrong arity
    assert not cb.is_valid_guess(("VEX", "KOB", "MIR", "QIZ"), inst)  # outside alphabet
    assert not cb.is_valid_guess(None, inst)
    history = [(("VEX", "KOB", "MIR", "ZAL"), (0, 2))]
    assert not cb.is_valid_guess(("VEX", "KOB", "MIR", "ZAL"), inst, history)  # repeated guess
    assert cb.is_valid_guess(("VEX", "KOB", "MIR", "TUF"), inst, history)


def test_invalid_reason_codes():
    inst = _instance()
    hist = [(("VEX", "KOB", "MIR", "ZAL"), (0, 2))]
    assert cb.invalid_reason(None, "two <guess>[A]</guess> <guess>[B]</guess>", inst, []) == "not_exactly_one_guess_tag"
    assert cb.invalid_reason(None, "<guess>[]</guess>", inst, []) == "no_parseable_guess"
    assert cb.invalid_reason(("VEX", "KOB"), "<guess>[VEX KOB]</guess>", inst, []) == "wrong_code_length"
    assert (
        cb.invalid_reason(("VEX", "KOB", "MIR", "QIZ"), "<guess>[VEX KOB MIR QIZ]</guess>", inst, [])
        == "symbol_not_in_alphabet"
    )
    assert (
        cb.invalid_reason(("VEX", "KOB", "MIR", "ZAL"), "<guess>[VEX KOB MIR ZAL]</guess>", inst, hist)
        == "repeated_guess"
    )
    # every reason code emitted has a burn_turn message
    for code in (
        "not_exactly_one_guess_tag",
        "no_parseable_guess",
        "wrong_code_length",
        "symbol_not_in_alphabet",
        "repeated_guess",
        "unknown_invalid",
    ):
        assert code in cb.INVALID_REASON_MESSAGES


def test_constraints_satisfied():
    # After (VEX KOB MIR ZAL) -> (0, 2): the true secret must satisfy; a contradicted code must not.
    hist = [(("VEX", "KOB", "MIR", "ZAL"), (0, 2))]
    assert cb.constraints_satisfied(SECRET, hist, code_length=4)
    assert not cb.constraints_satisfied(("VEX", "KOB", "MIR", "ZAL"), hist, code_length=4)  # would be (4,0)
    assert not cb.constraints_satisfied(("VEX", "KOB"), hist, code_length=4)  # malformed: False, no raise
    assert not cb.constraints_satisfied(None, hist, code_length=4)
    assert cb.constraints_satisfied(("TUF", "PYN", "GOR", "WEB"), [], code_length=4)  # vacuous


# ---------------------------------------------------------------------------
# Candidate tracking
# ---------------------------------------------------------------------------


def test_enumerate_candidates_small():
    inst = cb.Instance(alphabet=("VEX", "KOB", "MIR"), secret=("VEX", "KOB"))
    cands = cb.enumerate_candidates(inst)
    assert cands is not None
    assert len(cands) == 9
    assert inst.secret in cands


def test_enumerate_candidates_no_repeats():
    inst = cb.Instance(alphabet=("VEX", "KOB", "MIR"), secret=("VEX", "KOB"), allow_repeats=False)
    cands = cb.enumerate_candidates(inst)
    assert cands is not None
    assert len(cands) == 6
    assert all(len(set(c)) == len(c) for c in cands)


def test_enumerate_candidates_over_cap_returns_none():
    inst = cb.generate_instance(random.Random(1), k=16, code_length=6)
    assert cb.hypothesis_count(16, 6) > cb.CANDIDATE_ENUM_CAP
    assert cb.enumerate_candidates(inst) is None


def test_filter_candidates_fold_matches_bruteforce():
    rng = random.Random(11)
    inst = cb.generate_instance(rng, k=6, code_length=3)
    full = cb.enumerate_candidates(inst)
    assert full is not None
    history = []
    running = list(full)
    for _ in range(3):
        guess = tuple(rng.choice(inst.alphabet) for _ in range(3))
        fb = cb.compute_feedback(guess, inst.secret)
        history.append((guess, fb))
        running = cb.filter_candidates(running, guess, fb)
    brute = [c for c in full if all(cb.compute_feedback(g, c) == fb for g, fb in history)]
    assert running == brute
    assert inst.secret in running  # the truth is never filtered out at eps=0


def test_info_bits_are_positive_for_informative_guesses():
    inst = _instance()
    full = cb.enumerate_candidates(inst)
    assert full is not None and len(full) == 10_000
    guess = ("VEX", "KOB", "MIR", "ZAL")
    remaining = cb.filter_candidates(full, guess, cb.compute_feedback(guess, inst.secret))
    bits = math.log2(len(full)) - math.log2(len(remaining))
    assert bits > 0.5


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def test_initial_prompt_contains_rules_and_alphabet():
    inst = _instance()
    msgs = cb.build_mt_initial_messages(inst, max_turns=10)
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    content = msgs[1]["content"]
    for sym in inst.alphabet:
        assert sym in content
    assert "4 symbols" in content
    assert "10 attempts" in content
    assert "exact =" in content and "close =" in content
    assert "<guess>[" in content
    assert "MAY repeat" in content


def test_initial_prompt_no_repeats_wording():
    inst = cb.Instance(alphabet=("VEX", "KOB", "MIR", "ZAL"), secret=("VEX", "KOB"), allow_repeats=False)
    content = cb.build_shared_public_prompt_content(inst)
    assert "at most once" in content


def test_initial_prompt_never_leaks_secret():
    # The alphabet contains the secret's symbols by construction; the prompt must not single
    # them out — check the encoded secret sequence never appears verbatim.
    rng = random.Random(21)
    for _ in range(20):
        inst = cb.generate_instance(rng, k=10, code_length=4)
        content = cb.build_shared_public_prompt_content(inst)
        example = cb.format_example_guess(inst)
        secret_str = cb.render_guess(inst.secret)
        if secret_str != example:  # astronomically unlikely collision, but exact-check it
            assert secret_str not in content


def test_feedback_message_styles():
    inst = _instance()
    # turn-1 guess deliberately differs from the format reminder's example (the first
    # code_length alphabet symbols), so the last_only leakage check below is meaningful
    hist = [
        (("SUL", "DAK", "GOR", "WEB"), (0, 1)),
        (("ZAL", "VEX", "TUF", "PYN"), (2, 0)),
    ]
    transcript = cb.build_mt_feedback_content(hist, instance=inst, prompt_style="multiturn_transcript", max_turns=10)
    plain = cb.build_mt_feedback_content(hist, instance=inst, prompt_style="multiturn_plain", max_turns=10)
    last_only = cb.build_mt_feedback_content(hist, instance=inst, prompt_style="multiturn_last_only", max_turns=10)

    for content in (transcript, plain, last_only):
        assert "Your guess: ZAL VEX TUF PYN" in content
        assert "exact=2 close=0" in content
        assert "You have 8 guesses left." in content
        assert "<guess>[" in content  # format reminder present in every style

    assert "1. SUL DAK GOR WEB -> exact=0 close=1" in transcript
    assert "Already guessed" not in transcript
    assert "Already guessed, do not repeat: SUL DAK GOR WEB; ZAL VEX TUF PYN" in plain
    assert "exact=0 close=1" not in plain  # earlier feedback values NOT re-shown
    assert "SUL DAK GOR WEB" not in last_only  # nothing but the latest feedback survives


def test_feedback_message_remaining_override():
    inst = _instance()
    hist = [(("VEX", "KOB", "MIR", "ZAL"), (0, 2))]
    content = cb.build_mt_feedback_content(
        hist, instance=inst, prompt_style="multiturn_last_only", max_turns=10, remaining=3
    )
    assert "You have 3 guesses left." in content


def test_feedback_message_validates_style_and_history():
    inst = _instance()
    with pytest.raises(ValueError):
        cb.build_mt_feedback_content([], instance=inst, prompt_style="bogus_style")
    with pytest.raises(ValueError):
        cb.build_mt_feedback_content([], instance=inst, prompt_style="multiturn_plain")


def test_invalid_feedback_message():
    inst = _instance()
    hist = [(("VEX", "KOB", "MIR", "ZAL"), (0, 2))]
    content = cb.build_mt_invalid_feedback_content(
        "symbol_not_in_alphabet", hist, instance=inst, prompt_style="multiturn_plain", remaining=7
    )
    assert "every symbol in your guess must come from the game alphabet" in content
    assert "You have 7 guesses left." in content
    assert "Already guessed" in content
    unknown = cb.build_mt_invalid_feedback_content("never_seen_code", hist, instance=inst, remaining=1)
    assert "was not accepted" in unknown


# ---------------------------------------------------------------------------
# Reward primitives
# ---------------------------------------------------------------------------


def test_info_score_scaling():
    assert cb._info_score(4, 0, 4) == 1.0
    assert cb._info_score(0, 4, 4) == 0.5
    assert cb._info_score(0, 0, 4) == 0.0
    assert cb._info_score(2, 0, 4) == 0.5


def test_sparse_reward():
    assert cb._sparse_reward(solved=False, turns_used=10, max_turns=10) == 0.0
    assert cb._sparse_reward(solved=True, turns_used=10, max_turns=10) == 1.0
    assert cb._sparse_reward(solved=True, turns_used=1, max_turns=10) == pytest.approx(1.45)


# ---------------------------------------------------------------------------
# build_examples
# ---------------------------------------------------------------------------


def test_build_examples_splits_disjoint_and_deterministic():
    train, eval_, val = cb.build_examples(None, train_size=50, eval_size=20, val_size=10, seed=7)
    assert (len(train), len(eval_), len(val)) == (50, 20, 10)
    enc = lambda exs: {ex.metadata["target"] for ex in exs}
    assert enc(train) & enc(eval_) == set()
    assert enc(train) & enc(val) == set()
    assert enc(eval_) & enc(val) == set()
    train2, eval2, val2 = cb.build_examples(None, train_size=50, eval_size=20, val_size=10, seed=7)
    assert enc(train) == enc(train2) and enc(eval_) == enc(eval2) and enc(val) == enc(val2)
    train3, _, _ = cb.build_examples(None, train_size=50, eval_size=20, val_size=10, seed=8)
    assert enc(train) != enc(train3)


def test_build_examples_metadata_decodes():
    train, _, _ = cb.build_examples(None, train_size=5, eval_size=2, seed=1, k=10, code_length=4)
    for ex in train:
        inst = cb.decode_instance(ex.metadata["target"])
        assert inst.k == 10 and inst.code_length == 4
        assert ex.prompt_ids == []
        assert ex.project.startswith("cb_train_")


def test_build_examples_difficulty_ranges():
    train, _, _ = cb.build_examples(
        None, train_size=100, eval_size=0, seed=3, k_range=(8, 12), code_length_range=(3, 5)
    )
    ks = {cb.decode_instance(ex.metadata["target"]).k for ex in train}
    ls = {cb.decode_instance(ex.metadata["target"]).code_length for ex in train}
    assert ks <= set(range(8, 13)) and len(ks) > 1
    assert ls <= set(range(3, 6)) and len(ls) > 1


def test_describe_difficulty():
    line = cb.describe_difficulty(10, 4)
    assert "10,000 codes" in line and "13.3 bits" in line


# ---------------------------------------------------------------------------
# Scoring (codebreaker_scoring over duck-typed trajectories)
# ---------------------------------------------------------------------------

from examples.codebreaker import codebreaker_scoring


class _Sample:
    def __init__(self, *, format_ok=True, single_guess_tag=True, valid_guess=True, guess=None,
                 feedback=(0, 0), info_bits=0.0, constraint_consistent=None):
        self.format_ok = format_ok
        self.single_guess_tag = single_guess_tag
        self.valid_guess = valid_guess
        self.guess = guess
        self.feedback = feedback
        self.info_bits = info_bits
        self.constraint_consistent = constraint_consistent


class _Traj:
    def __init__(self, samples, solved, stopped_reason="max_turns"):
        self.samples = samples
        self.solved = solved
        self.stopped_reason = stopped_reason


def test_binary_reward_four_cases():
    # solved + clean format = 2; solved + any format miss = 1; unsolved + clean = 1;
    # unsolved + miss = 0 — the same contract as wordle's binary phase.
    clean = [_Sample(feedback=(1, 1)), _Sample(feedback=(4, 0))]
    dirty = [_Sample(feedback=(1, 1)), _Sample(format_ok=False, feedback=(4, 0))]
    L = 4
    assert codebreaker_scoring.score_trajectory(_Traj(clean, True), code_length=L)["binary_reward"] == 2.0
    assert codebreaker_scoring.score_trajectory(_Traj(dirty, True), code_length=L)["binary_reward"] == 1.0
    assert codebreaker_scoring.score_trajectory(_Traj(clean, False), code_length=L)["binary_reward"] == 1.0
    assert codebreaker_scoring.score_trajectory(_Traj(dirty, False), code_length=L)["binary_reward"] == 0.0


def test_score_blob_fields():
    samples = [
        _Sample(feedback=(0, 2), info_bits=3.5, constraint_consistent=None),
        _Sample(feedback=(2, 0), info_bits=4.0, constraint_consistent=True),
        _Sample(valid_guess=False, format_ok=False, feedback=None, info_bits=0.0, constraint_consistent=False),
    ]
    sc = codebreaker_scoring.score_trajectory(_Traj(samples, False), code_length=4, max_turns=10)
    assert sc["exact_match"] == 0.0
    assert sc["turns_used"] == 3.0
    assert sc["format_rate"] == pytest.approx(2 / 3)
    assert sc["valid_guess_rate"] == pytest.approx(2 / 3)
    assert sc["invalid_action"] == 1.0
    assert sc["info_bits_total"] == pytest.approx(7.5)
    assert sc["constraint_consistent_rate"] == pytest.approx(0.5)
    assert sc["info_gain"] == pytest.approx(((0 + 0.5 * 2) / 4 + (2 + 0.5 * 0) / 4) / 2)
    assert sc["reward"] == sc["binary_reward"]


def test_score_untracked_info_bits_excluded():
    # info_bits < 0 marks over-cap instances (untracked): must not poison the total
    samples = [_Sample(feedback=(1, 0), info_bits=-1.0)]
    sc = codebreaker_scoring.score_trajectory(_Traj(samples, False), code_length=4)
    assert sc["info_bits_total"] == 0.0


def test_score_empty_trajectory():
    sc = codebreaker_scoring.score_trajectory(_Traj([], False, stopped_reason="sampling_error"), code_length=4)
    assert sc["binary_reward"] == 0.0  # no turns played: format component must NOT award
    assert sc["turns_used"] == 0.0
    assert sc["constraint_consistent_rate"] == 1.0  # vacuous


def test_solve_le_turn_conditioned_scoring():
    """F(k) = 1{solved using <= k turns} over SOLVE_TURN_KS (xorl aa05361): cumulative in k,
    exact at the boundary, all-zero for unsolved games, and F(10) == exact_match (the existing
    solve metric, identical under the max_turns=10 protocol)."""
    ks = codebreaker_scoring.SOLVE_TURN_KS
    assert ks == (4, 5, 6, 7, 8, 9, 10)

    def fk(n_turns, solved):
        sc = codebreaker_scoring.score_trajectory(
            _Traj([_Sample() for _ in range(n_turns)], solved), code_length=4, max_turns=10
        )
        assert sc["solve_le_10"] == sc["exact_match"]  # F(10) == the existing solve metric
        vals = [sc[f"solve_le_{k}"] for k in ks]
        assert all(a <= b for a, b in zip(vals, vals[1:]))  # cumulative in k
        return vals

    assert fk(3, True) == [1.0] * 7  # fast solve counts for every budget
    assert fk(4, True) == [1.0] * 7  # boundary: <= is inclusive
    assert fk(7, True) == [0.0] * 3 + [1.0] * 4  # solve on turn 7: misses budgets 4..6
    assert fk(10, True) == [0.0] * 6 + [1.0]  # slowest solve: only the full budget
    assert fk(10, False) == [0.0] * 7  # unsolved: no budget is met
    assert fk(2, False) == [0.0] * 7  # early termination unsolved: still all-zero


# ---------------------------------------------------------------------------
# Fake ChatML tokenizer (from xorl test_codebreaker_smoke.py)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# xorl-client port layer: obo rescue spec (mirrors the slime rpp tests) + task
# ---------------------------------------------------------------------------

from examples.codebreaker.rollout import grpo_with_obo_rescue  # noqa: E402
from examples.codebreaker.task import CodebreakerTask  # noqa: E402


def test_obo_matches_grpo_on_variance_groups():
    import statistics

    rng = random.Random(11)
    rewards = [float(rng.choice([0.0, 1.0, 1.0, 2.0])) for _ in range(32)]
    group_ids = [i // 8 for i in range(32)]
    for g in range(4):
        rewards[g * 8] = 0.0
        rewards[g * 8 + 1] = 2.0
    exact = [1.0 if r >= 2.0 else 0.0 for r in rewards]
    out = grpo_with_obo_rescue(rewards, exact, group_ids)
    for g in range(4):
        rs = rewards[g * 8 : (g + 1) * 8]
        mean_r, std_r = sum(rs) / 8, statistics.pstdev(rs)
        for j in range(8):
            want = (rs[j] - mean_r) / (std_r + 1e-6)
            assert abs(out[g * 8 + j] - want) < 1e-12


def test_obo_all_failed_group():
    out = grpo_with_obo_rescue([0.0] * 8, [0.0] * 8, [0] * 8)
    assert all(abs(a + 1.0 / math.sqrt(8)) < 1e-9 for a in out)


def test_obo_all_solved_group():
    out = grpo_with_obo_rescue([2.0] * 8, [1.0] * 8, [0] * 8)
    assert all(abs(a - 1.0 / math.sqrt(8)) < 1e-9 for a in out)


def test_obo_mixed_zero_variance_group_zeroed():
    out = grpo_with_obo_rescue([1.0] * 4, [1.0, 0.0, 1.0, 0.0], [0] * 4)
    assert out == [0.0, 0.0, 0.0, 0.0]


def test_task_pools_disjoint_and_cyclic_selection():
    task = CodebreakerTask(
        train_size=32, eval_size=8, seed=3, k=6, code_length=3, max_turns=10
    )
    assert len(task.train_pool) == 32 and len(task.held_out) == 8
    assert not set(task.train_pool) & set(task.held_out)
    first = task.select_train_targets(step=1, count=8, seed=17)
    again = task.select_train_targets(step=1, count=8, seed=17)
    assert first == again
    epoch = [
        t for s in range(1, 5) for t in task.select_train_targets(step=s, count=8, seed=17)
    ]
    assert sorted(epoch) == sorted(task.train_pool)


def test_task_prompt_messages_roundtrip():
    task = CodebreakerTask(
        train_size=4, eval_size=2, seed=3, k=6, code_length=3, max_turns=10
    )
    inst = task.decode(task.train_pool[0])
    msgs = task.prompt_messages(inst, [])
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    guess = inst.secret
    history = [(guess, cb.compute_feedback(guess, inst.secret))]
    user = cb.build_mt_feedback_content(
        history, instance=inst, prompt_style=task.prompt_style, max_turns=10, remaining=9
    )
    msgs2 = task.prompt_messages(
        inst, [{"assistant": f"<guess>{cb.render_guess(guess)}</guess>", "user": user}]
    )
    assert len(msgs2) == 4 and msgs2[2]["role"] == "assistant"
