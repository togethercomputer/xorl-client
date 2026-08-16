# Word data

`targets.txt` and `legal_guesses.txt` are a compact, repository-owned example
vocabulary extracted from the Apache-2.0 `xorl-client` standalone Wordle task.
They are provided under this repository's Apache-2.0 license. The runner hashes
both files into `source_info.json`; users can replace either path with a larger
licensed vocabulary. Targets must be a subset of legal guesses, and the runner
constructs disjoint deterministic training and held-out pools before training.
