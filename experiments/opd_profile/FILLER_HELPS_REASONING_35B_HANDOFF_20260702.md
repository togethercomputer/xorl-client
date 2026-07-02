# Handoff: does prescribed filler help REASONING? — Qwen3.6-35B-A3B, 2026-07-02

Entry point for the next agent. This session moved the filler-token line from **235B/4-digit-mult**
(the "Value of Exploration" thread, see `FILLER_GRPO_VALUE_OF_EXPLORATION_HANDOFF_20260630.md`) to a
**35B, cheaper/faster GRPO** test of the core question: **does prescribing a big random-token
"scratchpad" actually help the model, or does it wash out?** It ends with a **clean disentangling
result + a validated ZA-free training recipe**, mid-relaunch.

---

## TL;DR / current verdict

1. **On arithmetic (4/5/6-digit mult), random filler cannot help** — random tokens don't manufacture
   computation. 4-digit is near-ceiling (model already ~0.8), 6-digit is floor (0.0), 5-digit works
   mechanically (~0.06, gradient flows) but a mega-filler-vs-none comparison there would wash out.
2. **Pivoted to a scan/count task** (`count_reassignments`: "how many times is variable X reassigned
   in this code snippet?") where the model is *partially* capable and extra forward-pass compute over
   filler *could* plausibly help. This is the right vehicle to test the hypothesis.
3. **KEY RESULT — the filler is NOT the lever; the question-RESTATEMENT is.** Step-0 (base model,
   same seed = same problems), on `count_reassignments`:

   | | no restatement | restatement |
   |---|---|---|
   | **no filler** | **B = 0.42** | **C = 0.75** |
   | **mega filler (8192)** | D (not measured; blocked on capacity) | **A = 0.73** |

   `C ≈ A ≫ B`: the +31-pt "mega-scratchpad benefit" is entirely from **restating the question right
   before the answer position** (short-range attention), not from the 8192 random filler tokens.
   Adding filler on top of the restatement (A) buys nothing over restatement alone (C). **This refutes
   the "filler compute helps" hypothesis for this task** — the apparent benefit was a prompt-structure
   artifact. (D — mega + no restatement — would confirm the bottom-left cell ≈ B; it was set up but its
   8-GPU head couldn't schedule on a full cluster.)
4. **ZA (zero-advantage) blew up during training and I mishandled it** (leaned on `max_za_replacements`
   resampling, which can't fix *structural* ZA). Root cause: binary exact-match reward + a task easy
   enough that GRPO climbs to near-ceiling → groups go all-correct → zero advantage → gradient stalls.
   **FIX (validated 2026-07-02, za/rate=0.000):** (a) **harder task** (counts 5–13, ~41-line snippets),
   (b) **graded reward** `reward_proximity=0.5 proximity_threshold=1.0` (partial credit by count
   distance → differing rollouts get differing rewards → groups keep variance → *no group can be
   zero-advantage*), (c) refill `max_za_replacements=256` (belt-and-suspenders, unused in practice).
   Calibration run C on this recipe: **za/rate=0.000, train_problems=16, za_replacements=0,
   original_mean≈0.49** (dead-center contested band).

**Next action:** relaunch the full A/B(/C/D) comparison on the validated harder+graded recipe and
compare held-out `val/correct` trajectories. But note the honest scientific conclusion is *already*
indicated by the step-0 2×2: **restatement, not filler, is the lever.**

---

## The task (`count_reassignments`)
Client generator `generate_variable_reassignment_problems` in the harness (below). A code snippet with
a target variable assigned K times (interleaved with distractor assignments to *other* vars, some of
which *read* the target on the RHS = a use, NOT counted). Answer = reassignments = K−1. Verified
0/200 mismatch between stated answer and true count. Integer answer → works with the existing
`Answer: N` reward + prescribe path. **Difficulty knobs** (defaults now, for the contested band):
`min_reassign=5, max_reassign=13, min_distractors=22, max_distractors=42` → counts 5–13 (mean 8.7),
~28–56 line snippets. Tune these if step-0 accuracy leaves the ~0.4–0.6 band.

## The 4 conditions / stacks (all Qwen3.6-35B-A3B, EP8, IS loss, corrected val)
| stack | condition | prescribe args | topology |
|---|---|---|---|
| `er-opd-q36-megafiller` | **A** mega-filler + restate | `prescribe_filler=true mega_filler=true mega_filler_tokens=8192` | 4-node |
| `er-opd-q36-nofiller`   | **B** no filler, no restate | `prescribe_filler=true mega_filler=false filler_tokens=0` (prefill `\nAnswer:`) | 1-node |
| `er-opd-q36-restate`    | **C** restate, no filler | `mega_filler=true mega_filler_tokens=0` (prefill `\n{question}\nAnswer:`) | 1-node |
| `er-opd-q36-megaonly`   | **D** mega-filler, no restate | `mega_filler=true mega_filler_tokens=8192 mega_restate_question=false` | 1-node (head was Pending on capacity) |

All stopped as of 2026-07-02 ~00:51Z (stop files) except C, which is the live **calibration** run
(relaunched 01:11Z on the harder+graded recipe, za=0). B/C/D are 1-node stacks I stood up this session
(teacher pods stripped from the generator render — GRPO needs no teacher).

## Code changes (all in the git-ignored client harness)
`XORL_CLIENT_REPO = /home/apanda/xorl-client-chat-completions/examples/filler_tokens_rl.py`:
- **`count_reassignments`** problem type: added `generate_variable_reassignment_problems` + entries in
  BOTH `load_problems` AND `PROBLEM_DESCRIPTIONS` (a second registry — omitting it crashed the head
  with `ValueError: Unknown problem_type`). Also added `multiplication_5digit`/`6digit`.
- **`mega_restate_question: bool = True`** config flag: when False (with `mega_filler`), prefill omits
  the question restatement (`filler_str = mega` only) → isolates filler-compute from question-position
  (condition D). Applied at BOTH prescribe sites (training loop + `build_eval_sample_prompt`).
- **`generate_mega_filler(<=0) → ""`** guard: `mega_filler_tokens=0` now yields a clean
  question-restated-no-filler prefill (condition C). Without it, `per=max(1,…)` floored at ~15 tokens.
- **Corrected held-out val** (was WRONG — evaluated a plain `filler_tokens=100` prompt, not the
  training condition): added `build_eval_sample_prompt` helper; both val loops now prescribe the same
  condition as training (mega/none) and reconstruct `filler_str + "\nAnswer:" + answer` before scoring,
  answer-only sampling. `val_every=50`, `val_size=100`, disjoint seeded slice (`val_start_idx =
  num_fewshot + N`), seed 12345 (A/B/C/D share problems → fair).
- **Graded reward**: `reward_proximity`/`proximity_threshold` were always available; this session turned
  them on (0.5 / 1.0) as the ZA fix.

## ZA — how to handle it PROPERLY (the lesson)
- GRPO gradient comes only from groups with reward **variance**. All-correct (`r1`) or all-wrong (`r0`)
  groups → advantage 0 → dropped. High `za/rate` = wasted rollouts + (near ceiling) gradient collapse.
- `max_za_replacements` (resampling) does **NOT** fix structural ZA — it redraws from the same
  distribution. Do not rely on it.
- The real fix is **(1) graded/continuous reward** so differing rollouts → differing rewards (variance
  survives even at high accuracy, as long as answers differ) **+ (2) difficulty in the contested band**
  so the model doesn't saturate. Validated: za/rate went 0.25–0.69 → **0.000**.
- Caveat: conditions have very different base accuracy (restatement ≫ none), so one difficulty may not
  hold all four perfectly in-band; the graded reward covers the low-accuracy end (variance among wrong
  counts).

## Config / paths
- Trainer configs: `/shared/apanda/filler_grpo/trainer_grpo_q36_{mega,nofiller,restate,megaonly}.yaml`
  (mega=4-node `dp_replicate=4`; the 1-node ones `dp_replicate=1`; EP8/dp_shard8, muon bf16, packing
  9216, `sync_inference_method: p2p`, R3 filesystem transport, fp32 lm_head/router).
- Control run.sh: `/shared/opd-control/er-opd-q36-{megafiller,nofiller,restate,megaonly}/{trainer-head,sglang-0,dispatch}/run.sh`
  (+ trainer-worker-1/2/3 for the 4-node A).
- Results: `/shared/opd-coord/encoded_reasoning/results/qwen3_235b_self_distill/er-opd-q36-*/…-trainer-head/`
  (`trainer_job.log` = client/steps, `server.log` = engine/weight-sync).
- wandb: project `xorl-prefill-time-compute`, runs `grpo-q36-35b-…-countreassign-…`.
- Engine (trainer) = `/home/apanda/xorl-qwen-k3-reconciliation` (src); venv `.venv-cu132-latest-probe`
  (cu13) with the libcudart-cu12 mooncake shim. Sampler venv = `/home/apanda/xorl-sglang-internal/.venv`.

## Infra gotchas hit this session (details in the infra runbook UPDATE 2026-07-02)
1. **`expandable_segments:True` breaks GPUDirect-RDMA MoE weight-sync** → `ibv_reg_mr` EFAULT "Bad
   address [14]" on the direct-EP expert transfer (dense layers fine). Set trainer
   `PYTORCH_ALLOC_CONF=expandable_segments:False` (+ `PYTORCH_CUDA_ALLOC_CONF`). Memory:
   `expandable-segments-breaks-gpudirect-rdma-weightsync`. **This was the multi-node sync blocker; NOT
   a topology/venv problem — don't disable direct-EP.**
2. **Editing a live slot's `run.sh` with an atomic-rename tool races the slot agent's `sha256sum` poll**
   → `No such file or directory` → slot agent exits → bare pod stuck `Error` (restartPolicy=Never). Use
   **`>>` append** for hash-busts; use in-place truncate-write (python `open('w')`) for content edits.
3. **Flaky free nodes** (057/058/080 were 7-GPU or dead-CUDA: `nvidia-smi` OK but `torch.cuda.init()`
   fails "CUDA unknown error"). Blacklist in pod `nodeAffinity` NotIn.
4. Recreating a dead bare slot pod: extract `metadata.annotations["kubectl.kubernetes.io/last-applied-configuration"]`
   (pristine, no nodeName pin) → delete + apply. (Needs explicit user OK for the delete.)
5. Generator `render-manifest` includes **teacher** pods (2×8 GPU) — strip them (drop docs whose name
   contains `teacher`) for GRPO. `--teacher-replicas` min is 1.
6. **Watchdog false-hang**: the run watchdog (`wd.sh`) execs `trainer-worker-1` (absent on 1-node) and
   latched an empty RUN_DIR when armed before the head wrote it. Fixed to recompute RUN_DIR each poll;
   copy at `experiments/opd_profile/scripts/opd_run_watchdog.sh`.

## Open items / next steps
1. **Relaunch the full comparison on the validated recipe** (harder task + `reward_proximity=0.5
   proximity_threshold=1.0` + `max_za_replacements=256`). Only C is currently running it; apply the same
   config edits to A/B (and D if capacity frees) and relaunch (bounce sampler via `>>`, in-place head
   edit). Compare held-out `val/correct` at step 50+.
2. **Capacity**: 4 stacks (A 4-node + B/C/D 1-node) overcommit a busy cluster; D's 8-GPU head went
   Pending. Either shrink A to 1-node (science only needs EP8; the 4-node was throughput) or run
   conditions sequentially.
3. The **step-0 2×2 already answers the hypothesis** (restatement, not filler). The trajectories mainly
   confirm it survives GRPO; consider whether the training comparison is even needed, or whether to
   report the base-model 2×2 as the finding.

## Pointers
- Prior 235B filler handoff: `FILLER_GRPO_VALUE_OF_EXPLORATION_HANDOFF_20260630.md`
- Science runbook (2026-07-02 UPDATE): `autoresearch/CANONICAL_SCIENCE_RUNBOOK.md`
- Infra runbook (2026-07-02 UPDATE): `autoresearch/CANONICAL_INFRA_RUNBOOK.md`
- Memory: `expandable-segments-breaks-gpudirect-rdma-weightsync`, `reprogrammable-slots-235b-launch`,
  `sampler-p2p-warmcache-relaunch-hang`
