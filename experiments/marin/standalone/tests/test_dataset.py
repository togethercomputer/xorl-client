from experiments.marin.standalone.dataset import iter_examples, load_examples, row_to_math_example


def test_row_to_math_example_extracts_rlvr_messages() -> None:
    row = {
        "messages": [{"role": "user", "content": "Question: What is 2+2?"}],
        "ground_truth": "4",
        "dataset": "MATH",
    }
    example = row_to_math_example(row, source="rlvr_math_7500", index=3)
    assert example.prompt == "Question: What is 2+2?"
    assert example.gold_answer == "4"
    assert example.example_id == "rlvr_math_7500_3"


def test_row_to_math_example_keeps_only_final_rlvr_question() -> None:
    row = {
        "messages": [
            {
                "role": "user",
                "content": "Question: solved one?\nAnswer: boxed{1}\n\nQuestion: target one?",
            }
        ],
        "ground_truth": "2",
    }
    example = row_to_math_example(row, source="rlvr_math_7500", index=0)
    assert example.prompt == "Question: target one?"
    assert example.metadata["raw_prompt"].startswith("Question: solved one?")


def test_rlvr_loader_skips_rows_without_ground_truth() -> None:
    rows = [
        {
            "messages": [{"role": "user", "content": "Question: unscored multiple choice"}],
            "ground_truth": "",
        },
        {
            "messages": [{"role": "user", "content": "Question: What is 3+5?"}],
            "ground_truth": "8",
        },
    ]
    examples = list(iter_examples(rows, source="rlvr_math_7500"))
    assert len(examples) == 1
    assert examples[0].prompt == "Question: What is 3+5?"
    assert examples[0].gold_answer == "8"


def test_row_to_math_example_extracts_boxed_solution_when_answer_missing() -> None:
    row = {"problem": "Compute.", "solution": r"Thus \boxed{\frac{2}{3}}."}
    example = row_to_math_example(row, source="math500", index=1)
    assert example.prompt == "Compute."
    assert example.gold_answer == r"\frac{2}{3}"


def test_row_to_math_example_extracts_evalchemy_style_aime_row() -> None:
    row = {"ID": "2024-II-4", "Problem": "Solve me.", "Answer": "033", "Solution": "Work."}
    example = row_to_math_example(row, source="aime24", index=4)
    assert example.prompt == "Solve me."
    assert example.gold_answer == "033"
    assert example.example_id == "aime24_4"


def test_aime24_loader_uses_evalchemy_jsonl() -> None:
    examples = list(load_examples("aime24", limit=1))
    assert len(examples) == 1
    example = examples[0]
    assert example.example_id == "aime24-0"
    assert example.gold_answer == "73"
    assert "Aimeville" in example.prompt
