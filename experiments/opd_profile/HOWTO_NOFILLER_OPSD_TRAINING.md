# How to run no-filler (no-CoT) OPSD training

**Standalone recipe.** Train a model to answer a hard task **directly** (no chain-of-thought, no filler/pause buffer) by on-policy self-distillation against a **CoT-conditioned teacher**. The student emits only `prompt + "Answer:" + answer`; the teacher answers *after* a private CoT; reverse-KL distills the teacher's answer distribution onto the student's direct answer.

This is independently useful: it **amortizes** multi-step reasoning into a single forward pass. On 4-digit×4-digit multiplication it lifts direct accuracy **0.594 → ~0.77 (step-30) → ~0.88 (step-100)** with `acc_pause≈acc_nopause` (i.e. the model learns the algorithm in-weights; no test-time CoT needed). It is *not* prefill-time compute — there is no buffer — it's the fair direct-answer baseline.

> Scope: this doc is **only** the no-filler recipe. For the buffer/filler science + the all-layer-OPRD machinery, see `autoresearch/CANONICAL_RUNBOOK.md` (S1, S11, O12).

---

## 1. The objective (what makes it "no-filler")

- **Student input:** `prompt + student_prefill_suffix("Answer: ")` — no pause/filler tokens, no CoT.
- **Teacher:** same weights (self-distill) seeing `prompt + CoT + answer`; the CoT is the teacher's private scratch.
- **Loss:** reverse-KL on the student's answer positions only (prompt masked from the denominator). No hidden-match, no OPRD, no buffer.
- **Knobs that define it:** `student_prefill_count: 0`, `student_prefill_text: ''`, `opd_hidden_match_coef: 0.0`, **no** `opd_oprd_*`, **no** `opd_buffer_equals_cot`. `teacher_cot_mode` left at the default (`replace`).

---

## 2. The recipe (candidate YAML)

Drop this in `experiments/opd_profile/autoresearch/candidates/`. This is the **5×5 Qwen3.5** version (verified running 2026-06-09; eval/accuracy rising 0.09→0.31 over the first steps). For 4×4 Qwen3.6 swap `trainer_config` + data + `num_prompts` (see §5).

```yaml
id: NOFILLER                          # rename per run
base_config: AM
slug: nofiller-opsd-baseline
trainer_config: experiments/opd_profile/configs/qwen3_5_35b_a3b_opd_opdb_8node.yaml
gdn_backend: fla
score_mode: pause_vs_nopause          # control eval still runs; gate on eval/accuracy
score_min_control_n: 128
student_prefill_text: ''              # ← no filler
student_prefill_count: 0              # ← no filler
student_prefill_suffix: 'Answer: '
student_stop_sequences: '["\n"]'
teacher_cot_json_path: /shared/opd-coord/randnum_5digit_q35_le6000_cot.json
prompts_json_path: /shared/opd-coord/randnum_5digit_q35_le6000_prompts.json
num_prompts: 5179
default_num_steps: 101                # 4×4 reached ~0.88 here; raise to reach plateau
default_prompts_per_step: 128         # bigger batch = less overfitting, cleaner trend
supervise_student_cot: true
opd_supervise_buffer_only: false
opd_hidden_match_coef: 0.0            # ← no hidden-match / no OPRD
opd_contrastive_corrupt_buffer_weight: 0.0
opd_contrastive_corrupt_answer_weight: 0.0
learning_rate: 3e-6                   # the proven OPD lr (lr matters a lot; ~5e-6 also good early)
max_new_tokens: 64
eval_max_new_tokens: 64
eval_num_problems: 128                # control-eval n; raise for a cleaner accuracy read
eval_accuracy_every: 5
eval_control_start_step: 100          # MUST equal num_steps-1 AND be %eval_accuracy_every
eval_control_max_concurrency: 64
eval_answer_logprob_control: false
opd_pipeline_rl: true                 # ok for no-filler; see §6 caveat if on an OPRD tree
sampler_quiesce_before_sync: false
sync_method: p2p
serial_endpoint_sync: false
request_timeout: 1800
weight_sync_timeout: 900
opd_loss_max_clamp: 5.0
client_args:
  eval_corrupt_pause_control: false
```

(The original 4×4 baseline is `candidates/PTC-091.yaml`; the verified 5×5 one is `candidates/PTC-302N.yaml`.)

---

## 3. Launch

The stack is the reprogrammable-slots stack `er-opd-q36-35b-slots` (trainer head + 7 workers, ≥1 sglang sampler, ≥1 teacher-sglang, teacher-smg, dispatch). With the stack already up (pods `1/1 Running`):

```bash
cd /home/apanda/xorl-apanda-dev-opd-port
python3 experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py write-trainer-control \
  --candidate experiments/opd_profile/autoresearch/candidates/NOFILLER.yaml \
  --num-steps 101 --prompts-per-step 128 \
  --sampler-replicas 1 --sampler-layout dedicated
```

- `--sampler-layout dedicated` uses only `sglang-0` (keeps `teacher-sglang-1` a pure teacher). `spare-teacher1` overloads teacher-sglang-1 as a 2nd sampler — avoid unless you need the throughput.
- Trainer reads `xorl` from `${XORL_REPO}/src` via PYTHONPATH, so repo edits take effect on the next `write-trainer-control`.
- Bring-up ≈ 5–7 min, then steady-state. **Do not** hand-launch `controller.py autopilot` — drive single runs with `write-trainer-control`.

To stop: `python3 .../q36_35b_reprogrammable_slots.py stop-trainer-control --remove-run`.

---

## 4. What to watch (gate on accuracy, NOT the delta)

Profile rows land in `…/<run>/opd_profile.jsonl`; `eval/*` every `eval_accuracy_every` steps.

- **Headline = `eval/accuracy`** (greedy correctness, no filler). Healthy: rises with training, `eval/empty_frac≈0`, `eval/mean_completion_tokens` sane (~15–30), `eval/has_digit_frac≈1`. 4×4: 0.594→0.77→0.88. 5×5 Q3.5 (verified): 0.09→0.31 in the first ~6 steps, then watch the plateau.
- **`loss`/`opd_kl`** drop steadily (e.g. 0.67→0.26 early) — fine, but loss convergence ≠ success (see the EOS-collapse lesson below); always read `eval/accuracy`.
- **Ignore `buffer_delta` / the autopilot verdict.** For no-filler there is no buffer to be load-bearing; the pause-vs-nopause control delta is confounded (it's inflated by whichever arm is starved). The number that matters is the absolute `eval/accuracy` trajectory.

---

## 5. Variants / data

| task | model / trainer_config | prompts json | CoT json | num_prompts |
|---|---|---|---|---|
| 4×4 (the 0.88 result) | Qwen3.6-35B, `configs/qwen3_6_35b_a3b_opd_opdb_8node.yaml` | `randnum_4digit_8192_nonempty_cot.json` | `randnum_4digit_8192_cot_nonempty.json` | 8185 |
| 5×5 (direct-fail regime) | Qwen3.5-35B, `configs/qwen3_5_35b_a3b_opd_opdb_8node.yaml` | `randnum_5digit_q35_le6000_prompts.json` | `randnum_5digit_q35_le6000_cot.json` | 5179 |

All under `/shared/opd-coord/`. CoT files are `{prompt, cot, finish_reason, tokens}` lists; **prompts and CoT must be index-aligned and non-empty** (an empty CoT hard-aborts the trainer — filter both at the same indices). Use the long-`mt` CoT files (4×4 `mt8192`, 5×5 `mt16384`), never the old amputated `_cot.json`.

**Budget:** 4×4 reached the plateau at **128 prompts/step × 101 steps** (~13k exposures). Smaller budgets are under-trained and bottom out low — don't read a plateau from <few-thousand exposures. le6000 packs ~1 sample/micro-batch (~4 s/sample) → 128 prompts/step ≈ 8–9 min/step on 5×5; size `num_steps` to your wall-clock.

---

## 6. Gotchas (the ones that bite no-filler runs)

1. **`teacher_cache_indices` missing crash on an OPRD-modified tree.** If your `src/xorl/server/orchestrator/packing.py` has the all-layer-OPRD changes, confirm the fix that makes the `teacher_cache_indices` generic-loop exclusion **conditional** (it must be excluded *only* when the OPRD block handled it, i.e. `teacher_input_ids` present; non-OPRD runs concatenate it via the generic loop). Without that fix, every non-OPRD run dies with `opd_loss requires teacher_cache_indices when teacher_hidden_states are not provided`. (Regression introduced + fixed 2026-06-09; see CANONICAL_RUNBOOK O12.)
2. **EOS reward-hack (the loss≠success trap).** Keep `student_prefill_suffix: 'Answer: '` so there's an answer cue; without it the model learns to emit `<eos>` right after the prefix → 0% despite low loss. Watch `eval/empty_frac` (must stay ~0) and `eval/mean_completion_tokens` (must not collapse to the cap).
3. **P2P sync `ret=-1` after a trainer restart.** Each trainer restart can stale `sglang-0`'s Mooncake buffers. If the step-0 weight sync fails with `[P2P] batch_transfer_sync … ret=-1`, recreate **sglang-0 and dispatch** (`kubectl delete pod … --force --grace-period=0` then `kubectl apply -f <rendered manifest>`); keep the teacher pods warm. Proactively recreate before a long run.
4. **Init hang = dead bare pod.** If a deploy hangs in "Engine Core initialization timeout" / "Waiting for rank 0 ready" with GPU mem ≈ 627 MiB, run `kubectl get pods -n apanda -l stack=er-opd-q36-35b-slots` FIRST — a worker stuck in `Error` (bare Pods don't auto-restart) stalls the 64-rank rendezvous. Delete+apply that one pod (others no-op/stay warm).
5. **Loop config rule:** `eval_control_start_step` must equal `num_steps-1` AND be a multiple of `eval_accuracy_every` (so e.g. `num_steps=101, eval_accuracy_every=5, eval_control_start_step=100`), or the eval/control loop stalls.
6. **NCCL/IB + Mooncake:** trainer pods that init Mooncake must NOT set `NCCL_IB_GID_INDEX`/`NCCL_IB_HCA`; never set `CUDA_VISIBLE_DEVICES` manually; team label `team: turbo` on the pod template; no `privileged: true`. (Handled by the generator's manifest; noted for manual edits.)

---

## 7. One-paragraph summary

No-filler OPSD = distill a CoT-teacher's answer distribution into a direct-answer student. `student_prefill_count=0`, `opd_hidden_match_coef=0`, `supervise_student_cot=true`, lr `3e-6`, 128 prompts/step, ~101 steps; `write-trainer-control --candidate …`; gate on `eval/accuracy` (4×4: → ~0.88), not the buffer delta. It compresses the teacher's multi-step CoT into a single forward pass — real amortized latent reasoning, and the fair baseline any filler/buffer claim must beat.
