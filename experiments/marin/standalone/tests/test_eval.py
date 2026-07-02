from pathlib import Path

from experiments.marin.standalone.eval_math_suite import (
    _apply_chat_template_override,
    _grade,
    _last_number,
    _sampling_seed_for_repeat,
)


def test_gsm8k_flex_grades_decimal_equivalent_last_number() -> None:
    assert _grade("gsm8k", "auto", "Therefore, the answer is $20.00.", "20")


def test_gsm8k_flex_uses_lm_eval_last_regex_match() -> None:
    assert _last_number("Therefore, Kylar needs to pay **$64** for 16 glasses.") == "16"


def test_apply_chat_template_override_sets_tokenizer_template(tmp_path: Path) -> None:
    template_path = tmp_path / "template.jinja2"
    template_path.write_text("{{ bos_token }} custom", encoding="utf-8")

    class Tokenizer:
        chat_template = "old"

    tokenizer = Tokenizer()
    _apply_chat_template_override(tokenizer, str(template_path))

    assert tokenizer.chat_template == "{{ bos_token }} custom"


def test_sampling_seed_for_repeat_defaults_to_outer_seed() -> None:
    assert _sampling_seed_for_repeat(seed=42, repeat_index=0, repeat_seed_base=None) == 42
    assert _sampling_seed_for_repeat(seed=42, repeat_index=3, repeat_seed_base=None) == 45


def test_sampling_seed_for_repeat_can_use_evalchemy_seed_base() -> None:
    assert _sampling_seed_for_repeat(seed=42, repeat_index=0, repeat_seed_base=0) == 0
    assert _sampling_seed_for_repeat(seed=42, repeat_index=9, repeat_seed_base=0) == 9
