# OPD / Encoded-Reasoning — Complete Handoff & Runbook (2026-05-31 → 06-01)

Single self-contained doc to pick up this work exactly where it was left. Covers the
research question, every experiment + result, the full infra runbook (how to launch /
restart an OPD deployment), all file locations, current cluster state, and exact next steps.

---

## 0. TL;DR / current state

- **Research goal:** test whether "encoded reasoning" can be induced in *filler/buffer tokens*
  via on-policy distillation (OPD/OPSD) — i.e., can a model learn to do useful hidden
  computation in pause/filler tokens, transferred from a teacher.
- **Bottom-line result: NO, robustly.** Across every variant tried (buffer-only CoT
  distillation, ICL 10-shot→0-shot distillation, and direct reproduction of the published
  filler-token "uplift"), there is **no genuine, distillable filler capability**. The
  published +14.9 / +20.1pp filler lifts are **format-recovery of an artificially-suppressed
  baseline**, not a capability gain. Details in §3.
- **Infra win:** Qwen3-235B-A22B OPD now runs end-to-end at scale (8-node, EP=8, ~85s/step).
  Reusable manifests + a documented **clean-restart recipe** (§5) for the 3 bring-up hazards.
- **Cluster state (at handoff):** all my pods torn down. The parallel `er-q35-35b-genuine-*`
  investigation (another agent / canonical filler-probe harness) is still running — **leave it
  alone**; its conclusion ("no clean cross-model filler lift") matches mine.
- **One open item (needs a human go):** the only filler cell I couldn't *validly* test is the
  doc's largest (Qwen3.5-397B anti-EOS, +20.1) because the local **FP8 397B is broken**
  (multimodal, mis-served → emits `!`). Proper test needs **BF16 397B on TP=16 / 2 nodes**.
  It would only *confirm* the recovery mechanism, not change the conclusion. See §8.

---

## 1. Background & hypothesis

Prior project finding (pre-existing): the pause buffer is a **no-op** — `buffer_delta` =
`acc_with_buffer − acc_without_buffer` ≈ 0 on Qwen3.6; OPD tends to *internalize* any benefit
into weights so the buffer becomes unnecessary, and can reward-hack via early-EOS
(see memory `project-opd-reward-hack-eos-collapse`).

This session pushed three new angles to try to *make* the buffer load-bearing:
1. **Buffer-only distillation** — mask the answer, force the teacher's CoT-conditioned reasoning
   to be distilled *only* into the student's pause buffer (+ optional hidden-state matching).
2. **ICL-distillation** — teacher = 10-shot, student = 0-shot, both use the filler; distill the
   in-context-learning benefit into the 0-shot+filler student.
3. **Reproduce the published filler "uplift"** — the multimodel filler-sweep doc reports
   +14.9pp (Q3.5-35B-A3B) / +20.1pp (Q3.5-397B) at 10-shot 4-digit mult; find the setting.

Source docs (read these first):
- `/old-data/apanda/tomi/outputs/multimodel_unified_summary_20260601.md` (the reconciled, 11-model summary — **most current**)
- `/old-data/apanda/tomi/outputs/multimodel_filler_and_cot_summary_20260528.md` (earlier, mechanism detail)
- The harness that produced them: `/old-data/apanda/tomi/examples/filler_tokens_rl.py`
  (functions `get_system_prompt`, `create_fewshot_prompt`, `generate_filler_tokens`,
  `parse_filler_response_permissive`, `generate_multiplication_problems`). NOTE: importing it
  needs internal `chz` + `tinker_cookbook` libs that are NOT installed here → replicate by hand.

---

## 2. The OPD/OPSD system — how it's wired

OPD here = **self-distillation**: teacher and student are the *same base model*; the teacher sees
extra context (CoT, or 10-shot examples) the student doesn't, and we distill teacher hidden/logit
distributions into the student at kept positions.

**Components (one k8s "stack" per run):**
- **Trainer** (the student being trained): `xorl.server.launcher` torchrun across N nodes. A Job
  (`<run>-trainer-head`) + N−1 bare worker Pods (`<run>-trainer-worker-1..N-1`). Serves an HTTP
  API on :26050; runs the OPD *client* (`on_policy_distillation.py`) as the foreground process.
- **Teacher** (frozen): sglang server(s) `<run>-teacher-sglang-{0,1}` behind an SMG router
  `<run>-teacher-smg:8080`. Launched with `--enable-return-hidden-states --disable-radix-cache
  --chunked-prefill-size 16384`. Computes the teacher hidden-state cache.
- **Samplers** (student rollouts): sglang `<run>-sglang-{0,1}`, weights synced from the trainer
  via **Mooncake P2P** (RDMA). Behind an SMG **dispatch** router `<run>-dispatch:8080`.
- **Dispatch** (SMG): routes the client's sampling requests to the samplers (cache_aware policy,
  circuit breaker).

**The OPD client** (`/home/apanda/xorl-client-chat-completions/examples/on_policy_distillation.py`)
orchestrates each step: (a) control eval (every `eval_accuracy_every` steps, with/without buffer →
`buffer_delta`), (b) sample student rollouts via dispatch, (c) build teacher hidden cache via
teacher-smg, (d) `forward_backward` on the trainer, (e) P2P weight sync to samplers.

**Key client mechanics (read the docstrings around lines 169-460):**
- `teacher_cot_mode="insert"`: teacher_seq = prompt + CoT + pause + answer; student_seq = prompt +
  pause + answer; distill at kept positions.
- `teacher_system_prefix` / `teacher_system_prefix_path`: a **teacher-only system prompt** prepended
  before the user prompt (used for 10-shot ICL examples).
- **CRITICAL CONSTRAINT:** `K = int(student_filler_count)` (buffer width = the *count* param, line
  ~354). So a filler must be **1 token per unit** (" pause" = 1 Qwen token → count=100 → K=100).
  A **multi-token filler (e.g. NATO words) silently breaks K** → teacher-cache length mismatch →
  crash. Multi-token/random fillers need the per-sample `teacher_filler_tokens: list[list[int]]`
  path (unvalidated).
- `student_prefill_suffix` (e.g. `"</think>Answer: "`) is the answer-cue appended after the pause
  buffer; required so the student emits an answer instead of looping on pause/EOS. **But it also
  un-suppresses the baseline** (see §3 — this matters a lot).
- `opd_supervise_buffer_only=true` + `mask_answer`: distill only the buffer (answer masked).
- `opd_hidden_match_coef>0`: adds (1 − cosine(student_hidden, teacher_hidden)) at kept positions.
  NOTE the reported `opd_hidden_match_loss` metric shows 0 even when active (reporting bug) — verify
  via `loss ≠ opd_kl` gap.

**Proven trainer config (235B):** `experiments/opd_profile/configs/qwen3_235b_a22b_opd_8node.yaml`
- `expert_parallel_size=8` (intra-node, ep_intranode default True), `data_parallel_shard_size=64`,
  `ulysses=1`, `sample_packing_sequence_len=512`, `optimizer=adamw bf16`, `init_device=meta`,
  `load_weights_mode=all_ranks`, FA3, `enable_gradient_checkpointing=true`, `ce_mode=compiled`.
- **Parallelism lessons (hard-won, see memory `project-opd-235b-working-config`):**
  - **EP=64 is a TRAP** → 35 min/step (cross-node all-to-all). Use EP=8 intra-node → ~61s fwd_bwd.
  - **shard=1 DEADLOCKS** the OPD loss all_reduce (need dp_size>1). Use shard=64.
  - adamw optimizer state allocates *after* step-0 optim (+~18 GB) → step-1 OOM unless packing≤512.
  - `operation_timeout` bumped 1800→7200 in `src/xorl/server/launcher.py` (235B steps are slow cold).

---

## 3. Findings (the science) — all negative on encoded reasoning

### 3a. Buffer-only CoT distillation (Qwen3-235B-A22B)
- Runs: `er-opd-235b-053102` (hidden_match=1.0, head `9txvp`) and `er-opd-235b-nohm` (hidden_match=0.0).
- Result: `buffer_delta` **plateaus ~0.22** over 100+ steps, *regardless of hidden-match*. acc_pause
  flat ~0.40, acc_nopause flat ~0.18. No encoded-reasoning emergence. `empty_frac=0` (no EOS-collapse).
- An early apparent rise in the nohm run (0.25→0.28 over a few control evals) was **n=96 eval noise**.
- Checkpoints saved: `.../er-opd-235b-053102/server_output/weights/student/opd-235b-bufonly-step{50,100}`.

### 3b. ICL-distillation (teacher 10-shot, student 0-shot, both + pause)
- Implemented via `teacher_system_prefix_path` (the 10 ICL examples as the teacher's system prompt) +
  `teacher_cot_mode=insert` with a 1-token benign filler. Run `er-opd-235b-icl`.
- Measured directly first (smart — avoided a wasted run): with the **format instruction controlled**
  in the system prompt, the 10 examples add only **+0.025** (0.40→0.425). The "ICL benefit" on
  4-digit mult is **output-format, not reasoning**. Under `/no_think` the model is intuition-capped.

### 3c. Reproducing the published filler "uplift" (the main investigation)
The doc reports clean cells: **Q3.5-35B-A3B 4dmult-10 pause +14.9** (digit-refinement), **Q3.5-397B
4dmult-10 ellipsis +20.1** (anti-EOS). Exhaustive attempt to reproduce on locally-served models:

- **Methodology gotchas discovered (in order):**
  1. The doc's format = **system prompt with a filler *instruction*** + few-shot examples
     demonstrating `[filler]\nAnswer: [product]` + **free generation** (NOT a forced prefill).
  2. Qwen3.5 no-think toggle is **`chat_template_kwargs={"enable_thinking": false}`** — putting
     `/no_think` in the text alone leaves the model reasoning (emits "Thinking Process...").
  3. The doc used **T=1.0** sampling (not greedy). Greedy hides sampling-stabilization effects.
  4. Problem text is **`"What is {a} * {b}?"`** (`generate_multiplication_problems`, random pairs).
  5. The doc's grader (`parse_filler_response_permissive`) **requires the filler format** (filler
     before "Answer:", model must *stop* after) — it structurally rewards the filler convention.
- **Result on Qwen3.5-35B-A3B (BF16, validly served, TP=2), n=200, T=1.0, faithful format:**
  filler **HURTS or is flat at every difficulty** — 4-digit v3=0.80 (ceiling), 5×4 v3=0.705 (−0.075),
  5×5 v3=0.185 (−0.06). Even at the doc's baseline regime (~0.55-0.70), no lift. My checkpoint is
  simply more capable than the doc's (0.80 vs doc's 55.6% at 4-digit) → computes in one pass.
- **Qwen3.5-397B-A17B-FP8 is BROKEN:** arch `Qwen3_5MoeForConditionalGeneration` (multimodal/vision);
  this sglang build mis-serves it → outputs `!` for *everything* (incl 0-shot with thinking). The
  "40-50% empty completions" I briefly saw were the broken model, **not** real anti-EOS. Invalidated.

### 3d. The resolution (what "the setting" actually is)
**The published filler uplift is FORMAT-RECOVERY of a suppressed baseline, not capability.** The V3
"answer-only" prompt drives the baseline *below the model's true ability* (under-computing on weak
checkpoints; premature-EOS on the 397B mamba/hybrid), and the filler tokens recover it toward what the
model can already do. So "filler helps" iff `(model, prompt)` baseline is suppressed below true
ability. On validly-served models with no such suppression, there is **no genuine lift** — matching
the parallel `er-q35-35b-genuine` investigation and memory `project-filler-no-genuine-lift-cross-model`.
CoT itself reaches 93-99% on these tasks; fillers don't unlock a distillable fraction of it.

---

## 4. RUNBOOK — launch a 235B OPD run from scratch

1. **Pick / edit the manifest:** `experiments/opd_profile/k8s/generated/er-opd-235b-053102.yaml`
   (template). It defines: 2 teacher-sglang + teacher-smg, 2 samplers + dispatch, trainer Job + 7
   worker Pods. The trainer-head command (one big bash block) sources repos, waits for
   teacher/SMG/samplers health, launches `xorl.server.launcher` + the OPD client with all flags.
2. **Key client flags (in the trainer-head command):** `teacher_cot_json_path`,
   `teacher_filler_text/count`, `student_prefill_text/count`, `student_prefill_suffix`,
   `teacher_cot_mode`, `supervise_student_cot`, `opd_supervise_buffer_only`, `opd_hidden_match_coef`,
   `eval_accuracy_every`, `prompts_per_step`, `group_size`, `num_steps`, `learning_rate`, `sync_method=p2p`.
3. **Launch:** `kubectl apply -n apanda -f <manifest>`. Pods land on the `nccl` node-group.
4. **Bring-up takes ~10-15 min** (235B load on samplers + trainer). The trainer-head `sleep 5`-waits
   for samplers to be healthy, then loads its own 235B (~118 shards, ~7 min), then steps.
5. **Watch:** the run dir is
   `experiments/encoded_reasoning/results/qwen3_235b_self_distill/<run>/<timestamp>-<head-pod>/`;
   `trainer_job.log` (client) + `server.log` (trainer server). Steady state ~85s/step; weight sync
   470 GB in ~8-12 s warm.
6. **k8s gotchas (memories):** privileged pods need `CUDA_VISIBLE_DEVICES` pinned (the
   `q35-*-teach.yaml` manifests use an `nvidia-smi`-based free-GPU picker — copy that pattern);
   sglang exits 0 on CUDA-OOM (looks "Completed"); `backoffLimit:0` means a crash permanently fails
   the Job (resubmit manually).

### 4b. CLEAN-RESTART RECIPE (critical — 3 bring-up hazards)
When a trainer crashes/restarts, the **long-lived sglang pods accumulate stale state** that crashes
the *next* trainer at step 0. Three hazards (full detail: memory `project-opd-inference-stack-restart-recipe`):
1. Samplers stuck in `pause_generation` (resume: `curl -XPOST .../continue_generation -d '{}'`).
2. Samplers hold a stuck P2P `weight_sync_group` → `/prepare_weights_update` 400 → "Failed to
   initialize p2p backend".
3. Dispatch circuit breaker OPEN (samplers marked unhealthy during their reload) → HTTP 503
   `no_available_workers`.
**Recipe:** delete trainer Job + all worker pods **+ samplers + dispatch**; KEEP teachers warm
(stateless, expensive to reload); wait for terminating pods to clear; `kubectl apply -f <manifest>`.
A *brand-new* run name does NOT hit these (the hazards are restart-induced). Verify the fresh dispatch
routes before trusting it:
`kubectl exec <sampler> -- curl -s -XPOST http://<run>-dispatch:8080/v1/chat/completions -d '{"model":"...","messages":[{"role":"user","content":"hi"}],"max_tokens":3}'` → expect 200.

---

## 5. Filler-eval RUNBOOK (reproduce / extend the uplift hunt)

A standalone sglang server + a Python sweep is the fast iteration loop (no trainer needed).

1. **Serve a model** (template `experiments/opd_profile/k8s/generated/q35-35b-teach.yaml`): single-node
   sglang, TP=2, `--enable-return-hidden-states --disable-radix-cache --chunked-prefill-size 16384
   --mem-fraction-static 0.8`, and an **`nvidia-smi` free-GPU picker** for `CUDA_VISIBLE_DEVICES`
   (privileged pods see all 8 host GPUs). For TP=8 single-node use `--disable-custom-all-reduce`
   (avoids the SymmDeviceMemory TP-init hang). FP8 multimodal 397B does NOT work on this build.
2. **Sweep scripts** (on the shared home PVC `/home/apanda/`, runnable in-pod via the sglang venv
   `/home/apanda/xorl-sglang-internal/.venv/bin/python`; launch detached with
   `setsid ... > out 2>err </dev/null &`; output is readable locally since `/home/apanda` is a shared PVC):
   - `sweep_filler.py` — v3/counting/pause/ellipsis × dual grading (lenient + strict), T=1.0.
   - `sweep_diff.py` — difficulty sweep (4/5/6-digit) × fillers.
   - `sweep_sweet.py` — 4×3 / 5×4 / 5×5-digit × fillers, n=200 (the definitive null).
   - `sweep_397b*.py` — 397B variants (broken — kept for reference).
3. **Faithful format (must match all 5 gotchas from §3c):** system = `/no_think\n{problem_desc}\n\n{filler_instr}`;
   10 few-shot examples `[filler]\nAnswer: {product}`; eval question; **free generation**
   (`add_generation_prompt`, no continue); `chat_template_kwargs={"enable_thinking": false}`; T=1.0;
   problem text `"What is {a} * {b}?"`; grade product after "Answer:".
4. **Grading nuance:** the doc's `parse_filler_response_permissive` requires filler-before + stop-after
   — it rewards the filler convention. Use a capability-fair grader (product correct, location-agnostic)
   to measure *capability*; use the strict one to reproduce the doc's *numbers*.

---

## 6. File / artifact index

| What | Path |
|---|---|
| Findings narrative (chronological) | `experiments/opd_profile/OPD_ENCODED_REASONING_FINDINGS_2026_06_01.md` |
| **This handoff** | `experiments/opd_profile/HANDOFF_OPD_ENCODED_REASONING.md` |
| 235B trainer config | `experiments/opd_profile/configs/qwen3_235b_a22b_opd_8node.yaml` |
| 35B-A3B OPD configs | `experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_*.yaml` |
| 235B OPD manifests | `experiments/opd_profile/k8s/generated/er-opd-235b-{053102,icl,nohm}.yaml` |
| Single-server eval manifests | `experiments/opd_profile/k8s/generated/q35-{35b,397b}-teach.yaml` |
| OPD client (orchestrator) | `/home/apanda/xorl-client-chat-completions/examples/on_policy_distillation.py` |
| Filler-probe harness (reference) | `/old-data/apanda/tomi/examples/filler_tokens_rl.py` |
| Filler-sweep source docs | `/old-data/apanda/tomi/outputs/multimodel_{unified_summary_20260601,filler_and_cot_summary_20260528}.md` |
| Eval sweep scripts | `/home/apanda/sweep_*.py` |
| 235B CoT + prompts (15979, filtered) | `/shared/opd-coord/randnum_4digit_q3_235b_{cot,prompts}_filtered.json` |
| 10-shot ICL examples (correct products) | `/shared/opd-coord/icl_10shot_4digit.txt` |
| 235B run results | `experiments/encoded_reasoning/results/qwen3_235b_self_distill/er-opd-235b-053102/<ts>-<head>/` |
| Models | `/shared/huggingface/hub/models--Qwen--{Qwen3-235B-A22B, Qwen3.5-35B-A3B, Qwen3.5-397B-A17B[-FP8], Qwen3.6-35B-A3B}` |

**Relevant memories** (`/home/apanda/.claude/projects/-home-apanda-xorl-internal/memory/`):
`project-opd-filler-mechanism-mismatch`, `project-opd-inference-stack-restart-recipe`,
`project-opd-235b-working-config`, `project-filler-no-genuine-lift-cross-model`,
`project-opd-reward-hack-eos-collapse`, `project-opd-sglang-teacher-stability`,
`project-opd-sync-gpu-fp8-quant`. (Index: `MEMORY.md`.)

---

## 7. Cluster state at handoff
- All of my pods are **torn down** (235B stacks, q35-35b-teach, q35-397b-teach).
- The `er-q35-35b-genuine-*` pods are a **parallel investigation** (canonical filler-probe harness) —
  **do not touch**.
- Namespace: `apanda`. Node-group: `nccl`. PVCs: `shared-data` (→ /shared), `home-apanda` (→ /home/apanda).

---

## 8. Open items / exact next steps
1. **(Optional, confirmatory) BF16 397B anti-EOS.** The doc's largest cell, the only one not validly
   tested (FP8 broken). Needs the **BF16** `Qwen3.5-397B-A17B` (188 shards, multimodal) on **TP=16 /
   2 nodes** = a multi-node sglang deploy (`--nnodes 2 --node-rank {0,1} --dist-init-addr <node0>:port
   --tp-size 16`). Even if it shows +20.1, it's the *recovery* mechanism (premature-EOS suppression →
   filler un-suppresses), not capability — so it won't change the conclusion. Watch `empty_frac` of the
   v3 baseline; a clean reproduction = high empty_frac at v3, lower with filler + accuracy recovers.
2. **If pursuing capability transfer (a real pivot, needs sign-off):** the productive direction is
   *ordinary* CoT→direct-answer distillation on tasks the model **cannot already do directly**
   (where direct-answer ≈ 0 but CoT is high). That's standard distillation, not "encoded reasoning,"
   and is what OPD is actually good at — but it abandons the filler/buffer hypothesis.
3. **Data:** the 235B CoT set (15979 filtered) cycled ~1.6× over 400 steps — adequate; the negative
   result is a recipe property, not data-limited. More CoTs were precomputed (Run B mt=8192) but not
   needed for the encoded-reasoning question.

## 9. Pitfalls cheat-sheet (don't re-learn these the hard way)
- Buffer-only `K = int(student_filler_count)` ⇒ **single-token fillers only**; multi-token NATO crashes.
- Restarting a trainer without recreating samplers+dispatch ⇒ step-0 crash (3 hazards, §4b).
- Filler-eval must use **enable_thinking=false** (kwarg, not text), **T=1.0**, free generation,
  `"What is A*B?"` text — or you measure the wrong thing.
- The published filler "+Xpp" is **format-recovery**, not capability — don't chase it as if it were a
  distillable skill.
- FP8 Qwen3.5-397B (multimodal) is broken on this sglang build (emits `!`). Use BF16.
- n=80 eval is too noisy (±0.05); use n≥200 before believing a few-pp filler delta.
- sglang "Completed" pod usually = a crash (OOM/SIGQUIT), not success.

---
## 10. Verification: is the 35B's 0.80 no-CoT real? (added on follow-up)
Audited the no-CoT 0-shot baseline (someone reasonably doubted 0.80 exact 4-digit mult with no
scratch work). `/home/apanda/verify_0shot.py` on Qwen3.5-35B-A3B (0-shot, enable_thinking=False,
max_tokens=512 to capture any reasoning, n=100):
- **T=0 greedy: strict_acc=0.820, substring_acc=0.820, inline_reasoning_frac=0.000.**
- **T=1.0: strict_acc=0.750, substring_acc=0.750, inline_reasoning_frac=0.000.**
- strict==substring ⇒ no grading false-positives. inline_reasoning_frac=0 + median 0 chars before
  "Answer:" ⇒ genuinely no CoT (bare `Answer: <number>`). Wrong answers are close-but-middle-digits
  off (e.g. 2432*7649 → 18601168 vs 18602368) ⇒ real single-pass arithmetic, not memorization/noise.
- **Conclusion holds & strengthened:** my 35B is near its true single-pass ceiling (~0.80), so a
  filler has no suppressed baseline to recover → no lift. The doc's 0.556 baseline (weaker checkpoint
  and/or V3-suppressed) is where filler has headroom. Filler never exceeds single-pass ability.
