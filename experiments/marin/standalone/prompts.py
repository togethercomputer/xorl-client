from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol


START_THINK = "<|start_think|>"
END_THINK = "<|end_think|>"
FORCED_THINK_PREFILL = f"{START_THINK}\n"
QUESTION_SUFFIX = r" Write your answer in \boxed{} format."
GSM8K_COT_FEWSHOT_EXAMPLES: tuple[tuple[str, str], ...] = (
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, "
        "there will be 21 trees. How many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6. The answer is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. "
        "After eating 35, they had 74 - 35 = 39. The answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
        "How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
        "So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
        "How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. "
        "5 + 4 = 9. The answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers were installed each day, "
        "from monday to thursday. How many computers are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more computers were added. "
        "So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. "
        "After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    ),
)


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True)
class PromptEncoding:
    text: str
    token_ids: list[int]
    skipped_for_length: bool
    original_token_count: int | None = None
    truncated_for_length: bool = False


def ensure_boxed_instruction(prompt: str) -> str:
    stripped = prompt.rstrip()
    tail_start = max(stripped.rfind("Question:"), stripped.rfind("Problem:"))
    tail = stripped[tail_start:] if tail_start >= 0 else stripped
    if "boxed" in tail.lower():
        return stripped
    return f"{stripped}{QUESTION_SUFFIX}"


def render_forced_thinking_prompt(
    prompt: str,
    *,
    add_boxed_instruction: bool = True,
    append_answer_prefix: bool = True,
) -> str:
    rendered = ensure_boxed_instruction(prompt) if add_boxed_instruction else prompt.rstrip()
    if append_answer_prefix and not rendered.rstrip().lower().endswith("answer:"):
        rendered = f"{rendered}\nAnswer:"
    return f"{rendered}{FORCED_THINK_PREFILL}"


def render_forced_thinking_chat_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    add_boxed_instruction: bool = True,
) -> str:
    rendered = ensure_boxed_instruction(prompt) if add_boxed_instruction else prompt.rstrip()
    if getattr(tokenizer, "chat_template", None) and hasattr(tokenizer, "apply_chat_template"):
        chat_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": rendered}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return f"{chat_text}{FORCED_THINK_PREFILL}"
    return render_forced_thinking_prompt(
        prompt,
        add_boxed_instruction=add_boxed_instruction,
        append_answer_prefix=True,
    )


def render_chat_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    add_boxed_instruction: bool = True,
) -> str:
    rendered = ensure_boxed_instruction(prompt) if add_boxed_instruction else prompt.rstrip()
    if getattr(tokenizer, "chat_template", None) and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": rendered}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return render_forced_thinking_prompt(
        prompt,
        add_boxed_instruction=add_boxed_instruction,
        append_answer_prefix=True,
    )


def render_evalchemy_math_prompt(prompt: str) -> str:
    return f"Problem: {prompt.rstrip()}\nMark your solution with \\boxed\nAnswer:"


def render_gsm8k_prompt(prompt: str) -> str:
    return f"Question: {prompt.rstrip()}\nAnswer:"


def render_gsm8k_lm_eval_prompt(prompt: str, fewshot_examples: Sequence[tuple[str, str]] = ()) -> str:
    query = render_gsm8k_prompt(prompt)
    if not fewshot_examples:
        return query
    supports = [f"Question: {question.rstrip()}\nAnswer:{answer.rstrip()}" for question, answer in fewshot_examples]
    return "\n\n".join([*supports, query])


def render_gsm8k_cot_prompt(prompt: str, *, num_fewshot: int = 8) -> str:
    query = f"Q: {prompt.rstrip()}\nA:"
    if num_fewshot <= 0:
        return query
    supports = [
        f"Q: {question}\nA: {answer}"
        for question, answer in GSM8K_COT_FEWSHOT_EXAMPLES[: min(num_fewshot, len(GSM8K_COT_FEWSHOT_EXAMPLES))]
    ]
    return "\n\n".join([*supports, query])


def _length_fit_tokens(
    token_ids: list[int],
    *,
    max_prompt_tokens: int,
    truncate_to_max: bool,
) -> tuple[list[int], bool, bool, int]:
    original_count = len(token_ids)
    if original_count <= max_prompt_tokens:
        return token_ids, False, False, original_count
    if truncate_to_max:
        return token_ids[-max_prompt_tokens:], False, True, original_count
    return token_ids, True, False, original_count


def encode_forced_thinking_prefix(
    tokenizer: TokenizerLike,
    prompt: str,
    *,
    max_prompt_tokens: int = 512,
    add_special_tokens: bool = False,
    truncate_to_max: bool = False,
) -> PromptEncoding:
    text = render_forced_thinking_chat_prompt(tokenizer, prompt)
    token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    token_ids, skipped, truncated, original_count = _length_fit_tokens(
        token_ids,
        max_prompt_tokens=max_prompt_tokens,
        truncate_to_max=truncate_to_max,
    )
    return PromptEncoding(
        text=text,
        token_ids=token_ids,
        skipped_for_length=skipped,
        original_token_count=original_count,
        truncated_for_length=truncated,
    )


def encode_prompt_prefix(
    tokenizer: TokenizerLike,
    prompt: str,
    *,
    max_prompt_tokens: int = 512,
    prompt_style: str = "forced-thinking",
    add_boxed_instruction: bool = True,
    add_special_tokens: bool = False,
    truncate_to_max: bool = False,
) -> PromptEncoding:
    if prompt_style == "forced-thinking":
        text = render_forced_thinking_chat_prompt(tokenizer, prompt, add_boxed_instruction=add_boxed_instruction)
        token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        token_ids, skipped, truncated, original_count = _length_fit_tokens(
            token_ids,
            max_prompt_tokens=max_prompt_tokens,
            truncate_to_max=truncate_to_max,
        )
        return PromptEncoding(
            text=text,
            token_ids=token_ids,
            skipped_for_length=skipped,
            original_token_count=original_count,
            truncated_for_length=truncated,
        )
    if prompt_style == "chat":
        text = render_chat_prompt(tokenizer, prompt, add_boxed_instruction=add_boxed_instruction)
        token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        token_ids, skipped, truncated, original_count = _length_fit_tokens(
            token_ids,
            max_prompt_tokens=max_prompt_tokens,
            truncate_to_max=truncate_to_max,
        )
        return PromptEncoding(
            text=text,
            token_ids=token_ids,
            skipped_for_length=skipped,
            original_token_count=original_count,
            truncated_for_length=truncated,
        )
    raise ValueError(f"Unknown prompt_style={prompt_style!r}")


def thinking_prefill_token_ids(tokenizer: TokenizerLike, *, add_special_tokens: bool = False) -> list[int]:
    return tokenizer.encode(FORCED_THINK_PREFILL, add_special_tokens=add_special_tokens)
