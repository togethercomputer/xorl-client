from __future__ import annotations

import json

from experiments.wordle.standalone import train_opsd_baseline as opsd
from experiments.wordle.standalone.tasks import wordle


def test_policy_hint_first_turn_is_target_invariant():
    prompt_a = opsd._wordle_policy_hint_user_content(wordle, target="apple", history=[])
    prompt_b = opsd._wordle_policy_hint_user_content(wordle, target="crane", history=[])

    assert prompt_a == prompt_b
    assert "Target word: omitted on the first turn." in prompt_a
    assert "APPLE" not in prompt_a
    assert "Reference next guess: STARE" in prompt_a


def test_public_policy_hint_first_turn_is_target_invariant():
    prompt_a = opsd._wordle_policy_hint_user_content(
        wordle, target="apple", history=[], teacher_prompt_style="public_policy_hint"
    )
    prompt_b = opsd._wordle_policy_hint_user_content(
        wordle, target="crane", history=[], teacher_prompt_style="public_policy_hint"
    )

    assert prompt_a == prompt_b
    assert "Target word:" not in prompt_a
    assert "APPLE" not in prompt_a
    assert "Strongest guess: STARE" in prompt_a


def test_public_policy_hint_never_shows_target_word():
    target = "apple"
    history = [("stare", wordle.compute_feedback("stare", target))]

    prompt = opsd._wordle_policy_hint_user_content(
        wordle,
        target=target,
        history=history,
        response_style="public_reasoning",
        teacher_prompt_style="public_policy_hint",
    )

    assert "Target word:" not in prompt
    assert "Public candidate analysis" in prompt
    assert "Remaining candidate answers consistent with the feedback so far:" in prompt
    assert "Strongest guess:" in prompt
    # The strongest guess must be a public candidate, not necessarily the target.
    candidates = opsd._wordle_public_candidates(wordle, target=target, history=history)
    reference_line = next(line for line in prompt.splitlines() if line.startswith("Strongest guess: "))
    reference_guess = reference_line.removeprefix("Strongest guess: ").strip().lower()
    assert reference_guess in candidates


def test_public_policy_action_prefers_information_over_target():
    # "abcde"/"abcdf"/"abcdg" each split the pool better than "zzzzz"; the
    # target must not override a strictly better public guess.
    candidates = ["abcde", "abcdf", "abcdg", "zzzzz"]
    history = [("stare", "XXXXX")]

    guess, reasoning = opsd._wordle_public_policy_action(
        wordle, candidates, target="zzzzz", history=history
    )

    assert guess != "zzzzz"
    assert guess in candidates
    assert "4 words fit the clues" in reasoning


def test_public_policy_action_uses_target_only_on_near_ties():
    # All four words differ only in the first letter, so every guess ties;
    # the private target may break the tie.
    candidates = ["batch", "match", "patch", "watch"]
    history = [("stare", "XYXXX")]

    guess, reasoning = opsd._wordle_public_policy_action(
        wordle, candidates, target="watch", history=history
    )

    assert guess == "watch"
    assert "WATCH" in reasoning


def test_public_policy_action_single_candidate():
    guess, reasoning = opsd._wordle_public_policy_action(
        wordle, ["apple"], target="apple", history=[("stare", "XXYXY")]
    )

    assert guess == "apple"
    assert reasoning == "Only one word fits every clue."


def test_public_reasoning_student_prompt_has_no_private_target():
    messages = wordle._build_turn_messages(target="apple", history=[], prompt_style="public_reasoning")
    content = "\n".join(message["content"] for message in messages)

    assert "<reasoning>" in content
    assert "<guess>[WORD]</guess>" in content
    assert "Start exactly with <reasoning>" in content
    assert "under 15 words" in content
    assert "private target" in content
    assert "APPLE" not in content


def test_policy_hint_public_reasoning_contract_scores_student_cot():
    prompt = opsd._wordle_policy_hint_user_content(
        wordle,
        target="apple",
        history=[],
        response_style="public_reasoning",
    )

    assert "Target word: omitted on the first turn." in prompt
    assert "<reasoning>PUBLIC_REASONING</reasoning><guess>WORD</guess>" in prompt
    assert "under 15 words" in prompt
    assert "Do not reveal the private target or private reference." in prompt
    assert "APPLE" not in prompt


def test_policy_hint_uses_pre_action_state_only():
    target = "apple"
    prior_guess = "stare"
    prior_feedback = wordle.compute_feedback(prior_guess, target)
    history = [(prior_guess, prior_feedback)]

    prompt = opsd._wordle_policy_hint_user_content(wordle, target=target, history=history)
    sampled_current_guess = "crane"
    current_feedback = wordle.compute_feedback(sampled_current_guess, target)

    assert "Target word: APPLE" in prompt
    assert "STARE ->" in prompt
    assert prior_feedback in prompt
    assert "CRANE ->" not in prompt
    assert f"{sampled_current_guess.upper()} -> {current_feedback}" not in prompt


def test_public_policy_hint_uses_pre_action_state_only():
    target = "apple"
    prior_guess = "stare"
    prior_feedback = wordle.compute_feedback(prior_guess, target)
    history = [(prior_guess, prior_feedback)]

    prompt = opsd._wordle_policy_hint_user_content(
        wordle, target=target, history=history, teacher_prompt_style="public_policy_hint"
    )
    sampled_current_guess = "crane"
    current_feedback = wordle.compute_feedback(sampled_current_guess, target)

    assert "Target word:" not in prompt
    assert "STARE ->" in prompt
    assert prior_feedback in prompt
    assert "CRANE ->" not in prompt
    assert f"{sampled_current_guess.upper()} -> {current_feedback}" not in prompt


def test_public_candidates_handle_duplicate_letter_targets():
    for target in ("apple", "eerie", "grass"):
        history = [("crane", wordle.compute_feedback("crane", target))]
        candidates = opsd._wordle_public_candidates(wordle, target=target, history=history)

        assert target in candidates
        assert "crane" not in candidates
        for candidate in candidates:
            assert wordle.compute_feedback("crane", candidate) == history[0][1]


def test_wordle_legal_guess_check_rejects_non_words_and_repeats():
    assert wordle.is_valid_guess("apple", [])
    assert wordle.is_valid_guess("stare", [])
    assert not wordle.is_valid_guess("zzzzz", [])
    assert not wordle.is_valid_guess("apple", [("apple", "GGGGG")])


class _ToyTokenizer:
    pieces = {
        1: "<",
        2: "guess",
        3: ">",
        4: "CR",
        5: "ANE",
        6: "</",
        7: "guess",
        8: ">",
    }

    def decode(self, token_ids, *, skip_special_tokens=False):
        return "".join(self.pieces[token_id] for token_id in token_ids)


def test_wordle_output_weights_prioritize_guess_content_tokens():
    weights, content_count, reasoning_count = opsd._wordle_output_weights(
        _ToyTokenizer(),
        [1, 2, 3, 4, 5, 6, 7, 8],
        turn_weight=1.0,
        tag_token_weight=0.05,
        reasoning_token_weight=1.0,
    )

    assert content_count == 2
    assert reasoning_count == 0
    assert weights == [0.05, 0.05, 0.05, 1.0, 1.0, 0.05, 0.05, 0.05]


class _ToyReasoningTokenizer:
    pieces = {
        1: "<",
        2: "reasoning",
        3: ">",
        4: "A",
        5: "B",
        6: "</",
        7: "reasoning",
        8: ">",
        9: "<",
        10: "guess",
        11: ">",
        12: "CR",
        13: "ANE",
        14: "</",
        15: "guess",
        16: ">",
    }

    def decode(self, token_ids, *, skip_special_tokens=False):
        return "".join(self.pieces[token_id] for token_id in token_ids)


def test_wordle_output_weights_include_reasoning_content_tokens():
    weights, guess_count, reasoning_count = opsd._wordle_output_weights(
        _ToyReasoningTokenizer(),
        list(range(1, 17)),
        turn_weight=1.0,
        tag_token_weight=0.05,
        reasoning_token_weight=0.5,
    )

    assert guess_count == 2
    assert reasoning_count == 2
    assert weights[3:5] == [0.5, 0.5]
    assert weights[11:13] == [1.0, 1.0]
    assert weights[:3] == [0.05, 0.05, 0.05]


class _ToyUnclosedReasoningTokenizer:
    pieces = {
        1: "<reasoning>",
        2: "A",
        3: "B",
    }

    def decode(self, token_ids, *, skip_special_tokens=False):
        return "".join(self.pieces[token_id] for token_id in token_ids)


def test_wordle_output_weights_include_unclosed_reasoning_content():
    weights, guess_count, reasoning_count = opsd._wordle_output_weights(
        _ToyUnclosedReasoningTokenizer(),
        [1, 2, 3],
        turn_weight=1.0,
        tag_token_weight=0.05,
        reasoning_token_weight=0.5,
    )

    assert guess_count == 0
    assert reasoning_count == 2
    assert weights == [0.05, 0.5, 0.5]


class _ToyContinuationTokenizer:
    pieces = {
        1: "<guess>",
        2: "CRANE",
        3: "</guess>",
        4: " extra",
        5: " text",
    }

    def decode(self, token_ids, *, skip_special_tokens=False):
        return "".join(self.pieces[token_id] for token_id in token_ids)


def test_truncate_output_ids_after_first_guess_removes_continuation():
    output_ids, text, truncated = opsd._truncate_output_ids_after_first_guess(
        _ToyContinuationTokenizer(),
        [1, 2, 3, 4, 5],
    )

    assert output_ids == [1, 2, 3]
    assert text == "<guess>CRANE</guess>"
    assert truncated is True


class _ChatTokenizer:
    def __init__(self):
        self.messages = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking, return_dict):
        assert tokenize is True
        assert add_generation_prompt is True
        assert enable_thinking is False
        assert return_dict is False
        self.messages.append(messages)
        return [len(self.messages)]

    def decode(self, token_ids, *, skip_special_tokens=False):
        return "<decoded>"

    def encode(self, text, *, add_special_tokens=False):
        return [1]


def test_teacher_wordle_diagnostics_logs_generated_reasoning(monkeypatch, tmp_path):
    tokenizer = _ChatTokenizer()
    generated = [
        "<reasoning>Choose broad opener.</reasoning><guess>CRANE</guess>",
        "<reasoning>Only APPLE remains.</reasoning><guess>APPLE</guess>",
    ]

    def fake_generate_base_with_sglang(teacher_url, *, input_ids, temperature, max_new_tokens, ignore_eos, stop):
        assert teacher_url == "http://teacher"
        text = generated.pop(0)
        return {"text": text, "output_ids": [1, 2, 3]}

    monkeypatch.setattr(opsd, "generate_base_with_sglang", fake_generate_base_with_sglang)
    metrics, rows = opsd.sample_teacher_wordle_diagnostics(
        tokenizer=tokenizer,
        task=wordle,
        examples=[opsd.Example(project="eval_0", prompt_ids=[], metadata={"target": "apple"})],
        teacher_url="http://teacher",
        temperature=0.2,
        max_new_tokens=96,
        ignore_eos=True,
        stop=[],
        workers=1,
        step=7,
        log_path=tmp_path / "teacher_samples.jsonl",
        log_examples=-1,
        wordle_prompt_style="public_reasoning",
        wordle_teacher_prompt_style="policy_hint",
    )

    assert metrics["exact_count"] == 1.0
    assert metrics["exact_match_rate"] == 1.0
    assert len(rows) == 2
    assert rows[0]["teacher_reasoning"] == "Choose broad opener."
    assert rows[0]["teacher_guess"] == "crane"
    assert rows[1]["teacher_guess"] == "apple"
    assert "APPLE" not in tokenizer.messages[0][1]["content"]
    assert "Target word: omitted on the first turn." in tokenizer.messages[0][1]["content"]
    assert "Target word: APPLE" in tokenizer.messages[1][1]["content"]
    assert (tmp_path / "teacher_samples.jsonl").read_text(encoding="utf-8").count("teacher_sample_turn") == 2


def test_teacher_reason_first_scoring_context_is_teacher_side_only():
    tokenizer = _ChatTokenizer()
    reason_prompt = opsd._wordle_teacher_reasoning_prompt_ids(
        tokenizer,
        wordle,
        target="apple",
        history=[],
        prompt_style="public_reasoning",
        teacher_prompt_style="policy_hint",
    )

    assert reason_prompt == [1]
    assert tokenizer.messages[0][0]["role"] == "system"
    assert tokenizer.messages[0][1]["role"] == "user"
    assert "<teacher_reasoning>PRIVATE_TEACHER_NOTE</teacher_reasoning>" in tokenizer.messages[0][1]["content"]
    assert "APPLE" not in tokenizer.messages[0][1]["content"]

    scoring_prompt = opsd._wordle_turn_prompt_ids(
        tokenizer,
        wordle,
        target="apple",
        history=[],
        hinted=True,
        prompt_style="public_reasoning",
        teacher_prompt_style="policy_hint",
        teacher_reasoning_context="<teacher_reasoning>Choose a broad opener.</teacher_reasoning>",
    )

    assert scoring_prompt == [2]
    assert [message["role"] for message in tokenizer.messages[1]] == ["system", "user", "assistant", "user"]
    assert tokenizer.messages[1][2]["content"] == "<teacher_reasoning>Choose a broad opener.</teacher_reasoning>"
    assert "must not reveal private information" in tokenizer.messages[1][3]["content"]
    assert "<reasoning>PUBLIC_REASONING</reasoning><guess>WORD</guess>" in tokenizer.messages[1][3]["content"]
    assert "APPLE" not in tokenizer.messages[1][1]["content"]


def test_teacher_reasoning_context_is_normalized_without_guess_tag():
    context = opsd._normalize_teacher_reasoning_context(
        "<reasoning>Public constraints leave two candidates.</reasoning><guess>APPLE</guess>"
    )

    assert context == "<teacher_reasoning>Public constraints leave two candidates.</teacher_reasoning>"
    assert "<guess>" not in context


def test_candidates_student_prompt_includes_consistent_candidate_block():
    target = "apple"
    prior_guess = "stare"
    prior_feedback = wordle.compute_feedback(prior_guess, target)
    history = [(prior_guess, prior_feedback)]

    messages = wordle._build_turn_messages(
        target=target, history=history, prompt_style="public_reasoning_constraints_candidates"
    )
    content = "\n".join(message["content"] for message in messages)
    candidates = wordle.remaining_candidates(history)

    assert "Remaining public candidate answers (computed only from the feedback above):" in content
    assert f"count = {len(candidates)}" in content
    assert target in candidates
    assert "APPLE" in content  # target appears only as one of the public-consistent candidates
    assert prior_guess.upper() not in [c.upper() for c in candidates]
    # candidate block must not exist in the no-candidates style
    base = "\n".join(
        m["content"]
        for m in wordle._build_turn_messages(target=target, history=history, prompt_style="public_reasoning_constraints")
    )
    assert "Remaining public candidate answers" not in base


def test_candidates_student_prompt_first_turn_is_target_invariant():
    contents = set()
    for target in ("apple", "crane", "shine"):
        messages = wordle._build_turn_messages(
            target=target, history=[], prompt_style="public_reasoning_constraints_candidates"
        )
        contents.add("\n".join(message["content"] for message in messages))
    assert len(contents) == 1
    only = next(iter(contents))
    assert "all answers are possible before the first guess" in only


def test_candidates_teacher_prefix_matches_student_shared_block():
    target = "apple"
    prior_guess = "stare"
    prior_feedback = wordle.compute_feedback(prior_guess, target)
    history = [(prior_guess, prior_feedback)]

    student_shared = wordle.build_shared_public_prompt_content(
        history, include_begin=False, include_candidates=True
    )
    teacher_prompt = opsd._wordle_policy_hint_user_content(
        wordle,
        target=target,
        history=history,
        response_style="public_reasoning",
        teacher_prompt_style="public_policy_hint",
        student_prompt_style="public_reasoning_constraints_candidates",
    )
    assert student_shared in teacher_prompt
    assert "Target word:" not in teacher_prompt

    # without the candidates style, the teacher shared block must omit candidates
    teacher_prompt_plain = opsd._wordle_policy_hint_user_content(
        wordle,
        target=target,
        history=history,
        response_style="public_reasoning",
        teacher_prompt_style="public_policy_hint",
        student_prompt_style="public_reasoning_constraints",
    )
    assert "Remaining public candidate answers (computed only from the feedback above):" not in teacher_prompt_plain.split(
        "<private_reference>"
    )[0]


class _ThinkChatTokenizer:
    def __init__(self):
        self.messages = []
        self.enable_thinking_calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking, return_dict):
        assert tokenize is True
        assert add_generation_prompt is True
        assert return_dict is False
        self.messages.append(messages)
        self.enable_thinking_calls.append(enable_thinking)
        return [len(self.messages)]

    def decode(self, token_ids, *, skip_special_tokens=False):
        return "<decoded>"

    def encode(self, text, *, add_special_tokens=False):
        return [1]


def test_public_policy_hint_think_teacher_opens_think_block():
    tokenizer = _ThinkChatTokenizer()
    opsd._wordle_turn_prompt_ids(
        tokenizer,
        wordle,
        target="apple",
        history=[("crane", "XXYXG")],
        hinted=True,
        prompt_style="public_reasoning_constraints_think",
        teacher_prompt_style="public_policy_hint",
    )
    assert tokenizer.enable_thinking_calls == [True]
    content = tokenizer.messages[0][1]["content"]
    assert "privately first" in content
    # The target may appear inside the public candidate list (that is the hint),
    # but must never be singled out.
    assert "Target word: APPLE" not in content
    assert "target word is APPLE" not in content


def test_teacher_score_transition_supports_think_style():
    text = opsd._wordle_teacher_score_transition_content(response_style="public_reasoning_think")
    assert "think privately first" in text
    assert "<reasoning>PUBLIC_REASONING</reasoning><guess>WORD</guess>" in text


class _ThinkOutputTokenizer:
    pieces = {
        1: "I could output <guess>",
        2: "SLATE</guess> now, but BLIMP",
        3: " splits better.",
        4: "</think>",
        5: "<reasoning>",
        6: "BLIMP tests new letters.",
        7: "</reasoning>",
        8: "<guess>",
        9: "BL",
        10: "IMP",
        11: "</guess>",
        12: " extra chatter",
    }

    def decode(self, token_ids, *, skip_special_tokens=False):
        return "".join(self.pieces[token_id] for token_id in token_ids)


def test_wordle_output_weights_think_region_supervised():
    weights, guess_count, reasoning_count = opsd._wordle_output_weights(
        _ThinkOutputTokenizer(),
        list(range(1, 12)),
        turn_weight=1.0,
        tag_token_weight=0.0,
        reasoning_token_weight=0.5,
        think_token_weight=0.25,
        think=True,
    )
    # think content supervised at think weight; in-think <guess> mention is NOT the action
    assert weights[0] == 0.25 and weights[1] == 0.25 and weights[2] == 0.25
    assert weights[3] == 0.0  # </think> tag
    assert weights[4] == 0.0  # <reasoning> tag
    assert weights[5] == 0.5  # public reasoning content
    assert weights[7] == 0.0  # <guess> tag
    assert weights[8] == 1.0 and weights[9] == 1.0  # guess content
    assert guess_count == 2
    assert reasoning_count == 1


def test_truncate_after_first_guess_think_ignores_in_think_mention():
    output_ids, text, truncated = opsd._truncate_output_ids_after_first_guess(
        _ThinkOutputTokenizer(),
        list(range(1, 13)),
        think=True,
    )
    assert output_ids == list(range(1, 12))
    assert text.endswith("</guess>")
    assert "BLIMP tests new letters." in text
    assert truncated is True


def test_truncate_after_first_guess_think_unclosed_returns_full():
    output_ids, text, truncated = opsd._truncate_output_ids_after_first_guess(
        _ThinkOutputTokenizer(),
        [1, 2, 3],
        think=True,
    )
    assert output_ids == [1, 2, 3]
    assert truncated is False


def test_teacher_reasoning_cache_roundtrip(tmp_path):
    key = opsd.teacher_cot_cache_key("Apple", [("CRANE", "XXYXG")])
    assert key == opsd.teacher_cot_cache_key("apple", [("crane", "XXYXG")])
    cache_path = tmp_path / "cot_cache.jsonl"
    cache_path.write_text(
        json.dumps(
            {
                "key": key,
                "target": "apple",
                "history": [["crane", "XXYXG"]],
                "teacher_reasoning_context": "<teacher_reasoning>Note.</teacher_reasoning>",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cache = opsd._load_teacher_reasoning_cache(str(cache_path))
    assert cache[key] == "<teacher_reasoning>Note.</teacher_reasoning>"
    # memoized: same object on re-load
    assert opsd._load_teacher_reasoning_cache(str(cache_path)) is cache


def test_extract_action_text_think_contract_is_legal_action():
    # Think-contract turn-1 output: the <think> block mentions the output tags
    # and the model emits trailing chatter after the guess. The strict one-line
    # parser fails on the raw text but must accept the extracted action line, so
    # the multi-turn rollout advances past the opener.
    raw = (
        "Thinking Process:\n"
        "Output format: <reasoning>PUBLIC_REASONING</reasoning><guess>WORD</guess>\n"
        "CRANE is a strong opener.\n"
        "</think>\n\n"
        "<reasoning>Starting with CRANE to test common letters.</reasoning><guess>CRANE</guess>\n"
        "The user wants me to continue playing Wordle."
    )
    assert wordle.parse_turn_response(raw, [])["action_ok"] is False
    action = wordle.extract_action_text(raw)
    assert action == "<reasoning>Starting with CRANE to test common letters.</reasoning><guess>CRANE</guess>"
    parsed = wordle.parse_turn_response(action, [])
    assert parsed["guess"] == "crane"
    assert parsed["format_ok"] is True
    assert parsed["action_ok"] is True


def test_extract_action_text_preserves_constraint_rejection():
    # A guess inconsistent with public feedback is still rejected after extraction.
    history = [("crane", "XXYXX")]  # 'a' present but not in position 3
    raw = (
        "<think>'A' is in the word, not position 3. Trying PLAID.</think>\n"
        "<reasoning>Place A elsewhere.</reasoning><guess>PLAID</guess>"
    )
    parsed = wordle.parse_turn_response(wordle.extract_action_text(raw), history)
    assert parsed["guess"] == "plaid"
    assert parsed["action_ok"] is False
    assert "public_constraint_violation" in parsed["errors"]


def test_extract_action_text_non_think_is_identity_for_clean_line():
    clean = "<reasoning>Open broad.</reasoning><guess>slate</guess>"
    assert wordle.extract_action_text(clean) == clean
    assert wordle.parse_turn_response(wordle.extract_action_text(clean), [])["action_ok"] is True


# --- algorithmic enumeration-think gold generator (generate_wordle_algo_think_gold) ---
from experiments.wordle.standalone import generate_wordle_algo_think_gold as algo  # noqa: E402


def _algo_turns(target: str, **kw):
    game = algo.play_algo_game(
        target=target, opener="slate", enum_cap=24, sample_cap=12,
        use_target_tiebreak=False, **kw,
    )
    return game["turns"]


def test_algo_think_completions_parse_and_are_feedback_consistent():
    # Every recorded turn must: close the think block, parse to the recorded
    # guess via the same extractor the trainer uses, and have correct feedback.
    for target in ("crane", "fused", "tardy", "igloo", "abbey"):
        for turn in _algo_turns(target):
            comp = turn["completion"]
            assert "</think>" in comp, comp
            parsed = wordle.parse_turn_response(wordle.strip_think_prefix(comp), turn["history_before"])
            assert (parsed.get("guess") or "").lower() == turn["guess"].lower(), comp
            assert wordle.compute_feedback(turn["guess"], target) == turn["feedback"]


def test_algo_think_enumerates_the_consistent_candidate_set():
    # On a small-n endgame turn the think must list exactly the env's consistent words.
    turns = _algo_turns("fused")
    enum_turns = [t for t in turns if t["history_before"]]
    assert enum_turns, "expected at least one post-opener turn"
    checked = 0
    for turn in enum_turns:
        cands = wordle.remaining_candidates(turn["history_before"])
        if 1 < len(cands) <= 24:
            think = turn["completion"].split("</think>")[0]
            for word in cands:
                assert word.upper() in think, f"{word} missing from enumerated think"
            checked += 1
    assert checked > 0, "no small-n enumeration turn was exercised"


def test_algo_think_guess_is_public_best_splitter_not_target_forced():
    # Selection is purely public: the guess is the lowest-expected-remaining
    # candidate, never forced to the (private) target when tiebreak is off.
    for target in ("tardy", "hosed", "vivid"):
        for turn in _algo_turns(target):
            if not turn["history_before"]:
                continue
            cands = wordle.remaining_candidates(turn["history_before"])
            if not cands:
                continue
            assert turn["guess"] in cands  # hard-mode: guess is itself a candidate
            best = algo._best_public_guess(cands)
            assert turn["guess"] == best
