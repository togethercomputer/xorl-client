from experiments.marin.standalone.prompts import (
    FORCED_THINK_PREFILL,
    QUESTION_SUFFIX,
    encode_forced_thinking_prefix,
    encode_prompt_prefix,
    ensure_boxed_instruction,
    render_chat_prompt,
    render_evalchemy_math_prompt,
    render_forced_thinking_chat_prompt,
    render_forced_thinking_prompt,
    render_gsm8k_cot_prompt,
    render_gsm8k_lm_eval_prompt,
    render_gsm8k_prompt,
)


class DummyTokenizer:
    chat_template = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) for char in text]


class ChatTemplateTokenizer(DummyTokenizer):
    chat_template = "dummy"

    def apply_chat_template(self, messages, tokenize: bool, add_generation_prompt: bool) -> str:
        assert not tokenize
        assert add_generation_prompt
        return f"<bos><user>{messages[0]['content']}<assistant>"


def test_ensure_boxed_instruction_checks_final_question_tail() -> None:
    prompt = r"Question: demo? Answer: \boxed{1}" "\n\nQuestion: new?"
    assert ensure_boxed_instruction(prompt).endswith(QUESTION_SUFFIX)


def test_ensure_boxed_instruction_is_idempotent_for_prompt_tail() -> None:
    prompt = r"Question: new? Write your answer in \boxed{} format."
    assert ensure_boxed_instruction(prompt) == prompt


def test_render_forced_thinking_prompt_appends_answer_prefix_and_prefill() -> None:
    rendered = render_forced_thinking_prompt("Question: What is 1+1?")
    assert rendered.endswith(f"Answer:{FORCED_THINK_PREFILL}")
    assert QUESTION_SUFFIX in rendered


def test_render_forced_thinking_chat_prompt_uses_template_when_available() -> None:
    rendered = render_forced_thinking_chat_prompt(ChatTemplateTokenizer(), "Question: What is 1+1?")
    assert rendered.startswith("<bos><user>")
    assert rendered.endswith(f"<assistant>{FORCED_THINK_PREFILL}")
    assert QUESTION_SUFFIX in rendered


def test_render_chat_prompt_uses_template_without_forced_thinking_prefill() -> None:
    rendered = render_chat_prompt(ChatTemplateTokenizer(), "Question: What is 1+1?")
    assert rendered.startswith("<bos><user>")
    assert rendered.endswith("<assistant>")
    assert FORCED_THINK_PREFILL not in rendered
    assert QUESTION_SUFFIX in rendered


def test_render_evalchemy_math_prompt_matches_evalchemy_template() -> None:
    assert render_evalchemy_math_prompt("What is 1+1?") == "Problem: What is 1+1?\nMark your solution with \\boxed\nAnswer:"


def test_encode_forced_thinking_prefix_falls_back_without_chat_template() -> None:
    encoded = encode_forced_thinking_prefix(DummyTokenizer(), "Question: What is 1+1?")
    assert encoded.text.endswith(f"Answer:{FORCED_THINK_PREFILL}")
    assert encoded.token_ids == [ord(char) for char in encoded.text]
    assert encoded.original_token_count == len(encoded.token_ids)
    assert not encoded.truncated_for_length


def test_encode_forced_thinking_prefix_can_tail_trim_to_max_prompt_tokens() -> None:
    encoded = encode_forced_thinking_prefix(
        DummyTokenizer(),
        "Question: " + ("x" * 50),
        max_prompt_tokens=10,
        truncate_to_max=True,
    )
    assert len(encoded.token_ids) == 10
    assert not encoded.skipped_for_length
    assert encoded.truncated_for_length
    assert encoded.token_ids == [ord(char) for char in encoded.text][-10:]


def test_encode_prompt_prefix_supports_chat_style() -> None:
    encoded = encode_prompt_prefix(ChatTemplateTokenizer(), "Question: What is 1+1?", prompt_style="chat")
    assert encoded.text.endswith("<assistant>")
    assert FORCED_THINK_PREFILL not in encoded.text


def test_encode_prompt_prefix_can_omit_boxed_instruction() -> None:
    encoded = encode_prompt_prefix(
        ChatTemplateTokenizer(),
        "Question: What is 1+1?",
        prompt_style="chat",
        add_boxed_instruction=False,
    )
    assert QUESTION_SUFFIX not in encoded.text


def test_render_gsm8k_cot_prompt_uses_eight_shot_query_format() -> None:
    rendered = render_gsm8k_cot_prompt("Janet has 2 eggs. How many?", num_fewshot=8)
    assert rendered.count("\n\nQ:") == 8
    assert rendered.startswith("Q: There are 15 trees in the grove.")
    assert rendered.endswith("\n\nQ: Janet has 2 eggs. How many?\nA:")
    assert "The answer is 8." in rendered


def test_render_gsm8k_cot_prompt_can_disable_fewshot() -> None:
    assert render_gsm8k_cot_prompt("How many?", num_fewshot=0) == "Q: How many?\nA:"


def test_render_gsm8k_prompt_matches_lm_eval_zero_shot_wrapper() -> None:
    assert render_gsm8k_prompt("How many?") == "Question: How many?\nAnswer:"


def test_render_gsm8k_lm_eval_prompt_includes_fewshot_supports() -> None:
    rendered = render_gsm8k_lm_eval_prompt(
        "How many?",
        [("One plus one?", "1 + 1 = <<1+1=2>>2. #### 2"), ("Two plus two?", "#### 4")],
    )
    assert rendered == (
        "Question: One plus one?\n"
        "Answer:1 + 1 = <<1+1=2>>2. #### 2\n\n"
        "Question: Two plus two?\n"
        "Answer:#### 4\n\n"
        "Question: How many?\n"
        "Answer:"
    )
