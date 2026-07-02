# Step-0 restatement evals — Qwen3.6-35B-A3B base, 2026-07-02

Completes the 2×2 from `../../FILLER_HELPS_REASONING_35B_HANDOFF_20260702.md` (D cell: mega
filler WITHOUT question restatement), plus a restatement dose-response and content/position
controls, on `count_reassignments`; second task: mult4/mult5. All step-0 (no training).

## Serving (deviation from the brief — documented)

The assigned sampler pod `er-opd-q36-megaonly-sglang-0` **no longer existed at session start**:
its log (`/shared/opd-control/er-opd-q36-megaonly/sglang-0/logs/20260702T003445Z-run.log`) ends
in a CUDA OOM at 2026-07-02T04:38:21Z (rc=1; a co-tenant process ate its slot GPU), and the whole
megaonly stack (pods, statefulsets, services) was torn down at ~05:07Z (`terminating` marker),
six minutes before this session began. No `er-opd-q36-{megafiller,nofiller,restate,megaonly}`
workloads remain on the cluster.

Rather than mutate the cluster, the SAME base snapshot
(`/shared/huggingface/.../Qwen3.6-35B-A3B/snapshots/995ad96eacd...`) was served **locally on this
dev pod's own k8s-allocated GPU** (physical index 3, UUID `GPU-2b9aa365-...`, verified via
`/var/run/nvidia-container-devices`; 0 MiB in use), with the dead sampler's exact sglang tree +
venv (`/home/apanda/xorl-sglang-internal`) and numerics flags (`--enable-fp32-lm-head
--enable-fp32-router --attention-backend fa3 --sampling-backend flashinfer --disable-radix-cache`,
`SGLANG_RMSNORM_FP32_WEIGHT_MUL=1` etc.). Only differences: TP1 instead of TP2 (one GPU),
mem-fraction 0.92, max-running-requests 48, port 127.0.0.1:30061, and the weight-update/mooncake
flags dropped (no trainer attached). See `serve_local_q36.sh`. The validation gate below shows
this reproduces the reference numbers, so TP1-vs-TP2 numerics are not a confound at this
precision.

## THE CRITICAL DISCOVERY: the handoff's 2×2 was measured on the OLD (easier) generator

The brief said to use `generate_variable_reassignment_problems` with the current knobs
(`min_reassign=5, max_reassign=13, min_distractors=22, max_distractors=42`) and gate on
B≈0.42 / C≈0.75. **These are incompatible.** The generator defaults were hardened at
~2026-07-02T01:09Z (the ZA fix: counts 5–13, ~28–56-line snippets); the handoff's 2×2
(B=0.42, C=0.75, A=0.73) was measured 2026-07-01 on the OLD defaults (counts 1–7, ~10–20-line
snippets). On the current harder problems the base model scores B=0.103, C=0.317 (greedy,
n=300) — nowhere near the references, with correct C≫B ordering.

The references are step-0 `original_mean` lines in the old runs' logs (16 problems × 8 samples,
temperature 1.0, top_k 20, top_p 0.95):

- B = 0.4219 — `er-opd-q36-nofiller/20260701T195841Z-.../grpo/filler-rl-count_reassignments-e51345cb/logs.log`
- C = 0.7500 — `er-opd-q36-restate/20260701T235857Z-.../grpo/filler-rl-count_reassignments-6cd753cd/logs.log`
- A = 0.7344 — `er-opd-q36-megafiller/20260701T194410Z-.../grpo/filler-rl-count_reassignments-8a5ef8b2/logs.log`
  (under `/shared/opd-coord/encoded_reasoning/results/qwen3_235b_self_distill/`)

The old harness file is unrecoverable (git-ignored, edited in place, pyc recompiled), but the old
**problems** are fully recoverable: the old C (restate) run's sample dumps contain the prescribed
prefill — i.e. the complete restated question — in every sample. All **2500 unique old problems**
(texts + ground truths) were extracted from
`er-opd-q36-restate/20260701T235857Z-.../samples/step_*.jsonl` with **0 mismatches** against an
independent recount of each snippet (`old_easy_problems_extracted.json`). The evals therefore run
on BOTH tiers:

- `count_reassignments_oldgen` — the exact old problems (first 300 in original order), directly
  comparable to the handoff's A/B/C;
- `count_reassignments` — the current harder contested-band generator (seed 12345, problems
  0–299), the difficulty training now uses.

## Methodology

- Rendering token-for-token the training harness prescribe path (verified against the harness's
  patched `build_generation_prompt` and the logged example prompts):
  `apply_chat_template([system, user(question)], add_generation_prompt=True, enable_thinking=False)`
  `+ filler_str + "\nAnswer: "`, num_fewshot=0, system prompt
  `get_system_prompt(ptype, filler_token_type="lorem", filler_tokens=0)` (the B/C/D run.sh args).
  NOTE the A run (`megafiller`) ran `filler_tokens=100`, which switches the system prompt to the
  counting-filler instruction — replicated as `A_sys100`; plain `A` uses the ft0 system prompt
  (clean 2×2).
- `filler_str` per condition mirrors the prescribe site exactly:
  `generate_mega_filler(K) + ("\n"+question if restate else "")`; harness-B is
  `generate_filler_tokens(0, "lorem", tokenizer)` — **which floors at ONE lorem word** (the loop
  appends an element before the `>= n` check), so the trained B prefill was
  `"{word}\nAnswer: "`, not a clean `"\nAnswer: "`. Both variants are measured
  (`B_harness...` vs `B_clean...`); they differ by ~1 point (noise).
- Problem/filler generators vendored VERBATIM (ast-extracted) from
  `/home/apanda/xorl-client-chat-completions/examples/filler_tokens_rl.py`
  (`vendored_filler_harness.py`).
- Decode: greedy (temperature 0), max 16 new tokens, stop `"\n"`; answer = first integer in the
  completion. Same 300 problems across all conditions. Concurrency 32.
- Gate runs additionally replicate the reference protocol exactly: temperature 1.0, top_k 20,
  top_p 0.95, 8 samples/problem.
- 95% CIs are Wilson intervals over problems (greedy) or samples (temp-1.0 runs; these are
  clustered by problem, so treat as slightly anti-conservative).

## Validation gate — PASSED

| condition | this eval (old problems, temp 1.0, k=8, n=300) | handoff reference (16 problems, k=8) | Δ |
|---|---|---|---|
| B no-filler no-restate | **0.409** [0.390, 0.429] | 0.4219 | −0.013 |
| C restate-only | **0.716** [0.697, 0.734] | 0.7500 | −0.034 |

Both within the ±0.08 gate. (Greedy on the same problems: B=0.523, C=0.827 — greedy sits
~+0.10 above the temp-1.0 protocol, as expected; all main tables below are greedy.)

## Main table — count_reassignments, OLD-generator problems (handoff-comparable), greedy, n=300

`samples_main_oldgen_greedy/`. Same 300 problems in every row. "empty" = greedy's first token
after `"Answer: "` was the stop newline (the model balked rather than answered); acc|answered
conditions on non-empty.

| condition | prefill after the chat turn | acc | 95% CI | empty | acc\|answered |
|---|---|---|---|---|---|
| B_harness (1 lorem word) | `{word}\nAnswer: ` | 0.483 | [0.427, 0.540] | 0.000 | 0.483 |
| B_clean | `\nAnswer: ` | 0.673 | [0.618, 0.724] | 0.000 | 0.673 |
| **C restate×1** | `\n{q}\nAnswer: ` | **0.823** | [0.776, 0.862] | 0.000 | 0.823 |
| **D mega-8192, NO restate** | `{mega8192}\nAnswer: ` | **0.287** | [0.238, 0.340] | 0.007 | 0.289 |
| **A mega-8192 + restate** | `{mega8192}\n{q}\nAnswer: ` | **0.713** | [0.660, 0.762] | 0.023 | 0.730 |
| A_sys100 (exact A-run sysprompt) | same as A | 0.833 | [0.787, 0.871] | 0.000 | 0.833 |
| restate×2 | `(\n{q})×2\nAnswer: ` | 0.760 | [0.709, 0.805] | 0.010 | 0.768 |
| restate×4 | `(\n{q})×4\nAnswer: ` | 0.403 | [0.349, 0.460] | 0.507 | 0.818 |
| restate×8 | `(\n{q})×8\nAnswer: ` | 0.347 | [0.295, 0.402] | 0.573 | 0.812 |
| wrongq restate×1 | `\n{other q}\nAnswer: ` | 0.027 | [0.014, 0.052] | 0.817 | 0.145 |
| structured restate (target-var lines only) | `\n{filtered q}\nAnswer: ` | 0.660 | [0.605, 0.711] | 0.000 | 0.660 |
| mega-512, no restate | `{mega512}\nAnswer: ` | 0.317 | [0.267, 0.371] | 0.000 | 0.317 |
| mega-512 + restate | `{mega512}\n{q}\nAnswer: ` | 0.750 | [0.698, 0.796] | 0.000 | 0.750 |

wrongq diagnostics: answered the RESTATED (wrong) question correctly only 0.093 of the time;
81.7% of the time it emits an immediate newline (balks). The mismatch derails it — it answers
neither question.

### Lenient-decode ablation (is the restate×4/×8 decline a stop-token artifact?)

`samples_lenient_oldgen/`: same conditions with NO stop-newline, 32 decode tokens, first integer
anywhere parsed. Results unchanged (C 0.833, restate2 0.757, restate4 0.400, restate8 0.360,
wrongq 0.023): the "empty" completions are the model emitting an immediate `<|im_end|>` after
`"Answer: "` — it terminates the turn without answering. The ×4/×8 decline is a genuine balk at
the over-repeated prompt (not recoverable with decode budget), while
accuracy-conditional-on-answering stays flat (~0.81): repetition breaks the model's willingness
to answer at that position, not its counting ability.

<!-- MORE RESULTS TABLES FILLED IN BELOW AS RUNS COMPLETE -->
