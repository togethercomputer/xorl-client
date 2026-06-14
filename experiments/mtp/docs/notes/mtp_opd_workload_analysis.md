# SingleShot-MTP OPD: what the workload actually is, and where the MFU goes

**Date:** 2026-06-11. Written in response to "MFU is still really really low — are we only
training on the first turn of each sample? maybe the thing you are hillclimbing doesn't
make that much sense." Short answer: we train on *arbitrary mid-trajectory 512-token
windows*, not on assistant turns at all; MFU is 0.4% useful, and the biggest levers are
workload-level (supervision per context pass), not executor-level.

## 1. The dataset vs. what the pipeline does with it

**Dataset** (`/shared/opd-datasets/coderforge_assistant_turns_ctx8k_tgt256_qwen36_files16_stream`,
built from CoderForge-Preview trajectories):
- 2,368 trajectories -> **128,034 per-turn rows** (~55 assistant turns per trajectory,
  p90 81, max 100). NOT first-turn-only: every qualifying assistant turn is a row.
- Each row: `prompt_ids` = the trajectory tail-trimmed to the **last 8,192 tokens ending
  exactly at that turn's assistant-content start** (mean 8,063; 94% hit the 8k cap);
  `target_ids` = the turn's first 256 tokens (22% truncated at the cap).
- Because windows are tail-trimmed, consecutive turn-rows of the same trajectory have
  **zero materialized common prefix** (the window start slides) even though they overlap
  almost entirely in trajectory coordinates.

**Pipeline** (standing stack `er-opd-q36-mtp-ss-0605c`, `launch_args_...txt`):
- `--prompt-dataset-turn-strategy prefix --prompt-len 512`: takes `prompt_ids[:512]` —
  the **first** 512 tokens of the 8k window. The generation point therefore sits ~7.7k
  tokens **before** the assistant turn the row was built around. The turn alignment the
  dataset was constructed for is unused; we sample 256-token continuations from arbitrary
  mid-trajectory cut points. (Self-distillation stays well-defined — teacher and student
  see the same context — but the workload is "continue random mid-trajectory windows",
  not "imitate assistant turns".)
- Prompts advance **sequentially** through the row pool (`batch_for_step`: `start =
  step*batch_size`, offset 0, no shuffle) — each step's 32 prompts are consecutive rows,
  i.e. mostly successive turns of the *same* 1-2 trajectories. Batches are highly
  correlated.
- The `assistant` turn strategy (already implemented, `_assistant_prompt_from_row`) is
  tail-aligned: `prompt_end = assistant content start`, `prompt_start = prompt_end -
  prompt_len`. On these rows it positions at the last *closed* assistant span inside the
  window. A trivial `suffix` strategy (`prompt_ids[-prompt_len:]`) would align exactly to
  the row's own turn. Either fixes the positioning; today neither is used.

## 2. MFU accounting (v1 payload = the production 512+256 shape, 16 H100s)

From `replay_results.jsonl` FLOPs instrumentation (promised 989 TF/GPU):

| config | steady FB | MFU (useful) | MFU (executed) |
|---|---|---|---|
| CP1 promoted-shape control | 6.4-6.9s | **0.39%** | 1.58% |
| CP4 triton (rebalanced executor) | 15.4s | 0.17% | 0.71% |
| CP4 deepep (rebalanced executor) | 19.4s | 0.14% | 0.56% |

Decomposition of the gap (multiplicative):
1. **Executed/useful = 4.1x** — GDN replay re-executes context for branch suffixes
   (211k GDN-executed tokens for 27.4k layout tokens / 5,773 supervised tokens per
   32-prompt step) plus unsupervised context compute. GDN is 74% of executed FLOPs
   (1,280 TF of 1,724 TF; attention 86, MLP 340, lm_head 18).
2. **Executed-MFU is itself only 1.6%** — the work runs as many small packed varlen
   calls (~28 GDN calls/layer/mb at mb=1 prompt) over a 2k-hidden A3B model; kernels are
   launch/memory-bound, nothing reaches GEMM-shaped utilization.
3. CP adds its residual tax on top (now 1.8x/2.2x at CP2/CP4 after the 2026-06-11
   executor rebalance; was 2.7-3.7x).

## 3. Ranked levers

1. **More supervision per context pass (workload-level, biggest).** Today each context
   window pays full price for 256 supervised tokens. Two composable fixes:
   - **N continuations per prompt, prefix-shared**: sample N rollouts per prompt; the
     stateful replay plan already supports many branches re-using one context's boundary
     states — this is literally its design. N=8 multiplies supervised tokens/context by
     8 at ~zero extra context cost. (Memory note `project_opd_mtp_compute_bound_verdict`
     already named this "the real lever".)
   - **Multi-turn-per-sequence**: one 8k trajectory window contains ~55 assistant turns;
     supervising every turn boundary in one pass (branch per turn) yields ~55x256 ≈ 14k
     supervised tokens per window vs 256 today. Needs turn-aligned sampling (per-turn
     rollouts) wired into one replay row; the mask/plan machinery supports it.
2. **Packing (H5)** — multiple prompts per micro-batch: bigger packed GDN calls (attacks
   the 1.6% executed-MFU) and, under CP, parallel context chains across ranks (attacks
   the CP residual). Config-level (`--trainer-enable-packing`); first failures were
   per-rank OOM under the short-shape memory config presenting as collective hang/IMA
   (root-caused 2026-06-11; details in `gdn_cp_handoff.md` §6.2) — packed cells need
   `recompute_before_dispatch` + per-layer reshard.
3. **Turn-aligned positioning** (`assistant`/`suffix` strategy + longer prompt-len):
   doesn't change MFU by itself (longer context *lowers* supervision density) but makes
   the workload mean what the dataset intended; if adopted, lever #1 becomes mandatory
   to keep MFU acceptable at 4-8k contexts.
4. **CP executor rebalance** — landed (`b78ae079`): 1.35-1.64x on CP cells; CP is the
   memory lever for the long-context shapes lever #3 implies.
5. Longer `max-new-tokens` — supervision per context scales linearly; task-dependent.

## 4. Is the FB-time hillclimb the right target?

Partially. FB time on the v1 payload measures the *current* production shape (512+256,
mb=1 prompt) faithfully, and the CP/executor wins are real and portable. But the
end-to-end run is sampling-gated (async overlay; Amdahl analysis in the main handoff
§-1.11), and useful-MFU is dominated by supervision density, which no trainer-side
optimization touches. The highest-value next experiments are (1) N-continuations-per-
prompt capture payloads (same harness, new payload with 8 rollouts/prompt) and (2)
packed micro-batches — both measurable with the existing replay harness before touching
production.
