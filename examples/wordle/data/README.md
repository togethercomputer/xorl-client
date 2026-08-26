# Word data

This directory contains the exact production data used by the shared Qwen3.6
configuration:

- `legal_guesses.txt`: the complete ordered 4,266-word dictionary.
- `train_targets.txt`: the exact ordered 4,000-target training pool.
- `eval_targets.txt`: the exact ordered 170-target held-out panel.

Every shipped preset loads these files directly without recreating or
reshuffling the train/eval split. Small smoke presets shorten the run geometry;
they do not define a different dataset split.

The runner hashes every input file into `source_info.json`. All training and
evaluation targets must be dictionary entries, and the two canonical splits
must remain disjoint.

The dictionary is the sorted, unique, lowercase five-letter word set from
`wordle-python==2.4.0`. The eval panel is the first 170 words after shuffling
that dictionary with `random.Random(777)`. The train pool removes those eval
words, shuffles the remainder with `random.Random(9234)`, and takes the first
4,000 words. The committed files themselves are the runtime source of truth.
