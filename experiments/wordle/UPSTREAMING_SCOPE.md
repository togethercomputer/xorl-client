# Upstreaming scope: Wordle RL session work → apanda-dev branches

> **🏁 2026-07-02 (later): MERGE QUEUE EXECUTED — consolidation CLOSED.** All engine/sglang/client
> PRs are MERGED into their target branches (squash, head branches deleted):
>
> - **Engine `apanda-dev` (17 merges)**: #433 (marin sglang-parity) → k3 atomic stack #435–#444
>   (rebased to deltas) → **#445** (wordle remainder: MoE EP/TP parity + token-diagnostics capture;
>   supersedes the auto-closed #430) → #431/#432 (marin drgrpo + KV-cache preserve) → #434 (OPD
>   Line A, 4-hunk diagnostics-capture reconciliation + width-clamp fix) → **#446** (zorl-ps
>   PS-as-trainer + fresh_ab). Plus 2 direct test-mock fixes (high_precision_bf16 post-#439).
>   Final-tip suite: 489 passing; only pre-existing base failures remain (module_utils_broadcast ×3,
>   qwen_hidden_states ×1 — fail identically at the June fork point 2093a6c82).
> - **SGLang `apanda-dev`**: #54 merged; #55 re-landed as **#57** (auto-closed on stacked-base
>   deletion) and merged. #56 stays draft (W12); #51 disposition-commented.
> - **Client**: #9 merged to `apanda-dev`; #7 disposition-commented (pre-consolidation, owner's call).
> - **Infra — RESOLVED via `apanda-dev` branch**: `main`'s review ruleset blocked PRs #1–#4, so
>   (per user) a new **`apanda-dev`** branch now carries everything (merge of local main +1,
>   populate-opd-mtp, opd-battery-consolidation, wordle-consolidation, zorl-consolidation; all 5
>   lines verified contained). `~/xorl-infra` is checked out on it; PRs #1–#4 closed as superseded.
>   Promotion `apanda-dev` → `main` = one reviewed PR later, at the user's leisure.
> - Deferred: k3pnr3-v2 science conclusion (run still live), zorl live-gate readout (running in the
>   zorl stream's relaunched stack), k3diag rebuild smoke, k3-recon-r4 rebuild + donor trees (k3 agent).
>
> Earlier same-day status (per-area landing map, pre-merge):
>
> | Area | Where it landed | Notes |
> |---|---|---|
> | **SGLang** | `xorl-sglang-internal` `apanda-dev` @ `e9593887f` "Return expert routing weights from SGLang" (+tests) — **pushed** | The "11 uncommitted files" claim below is stale. Labeling caveat: `8b177f6c8` (subject "lora: remove stale guard…") still bundles the selective batch-invariant ops + weight-version routing files. Left as-is deliberately: `zorl-ps-fp32` is pushed on top of this apanda-dev, so a relabeling force-push would orphan its base. Content map: expert-logits = `e9593887f`; BI + weight-version routing + lora guard = `8b177f6c8`; lora FP8 = `6a218506a`. |
> | **Client (core)** | Already upstream: `internal/apanda-dev` @ `bcd037d` "Add Wordle behavior-logprob replay controls" contains the identical `chunked_helpers.py` blob — **no cherry-pick needed** | |
> | **Client (science)** | Branch `science/wordle-retrieval-sft-20260614` **pushed to internal** — docs/handoffs, river port, and eval orchestration tooling (`experiments/wordle/orchestration/`, incl. the `engine_connect_host` fix) committed | |
> | **Engine** | **PR #430**, restacked onto **PR #433** (marin stream) — base `marin-sglang-parity-numerics`; merge #433 first. 5 themed commits, CPU suites green (159 pass incl. #433's tests) | **Scope corrections:** (1) upstream `2093a6c82` superseded the R3 externalize work (side_payloads Mooncake) — PR keeps upstream's R3; (2) #433 landed batch-invariant ops + rmsnorm-sglang first — #430 is now the delta over it (BI-ops superset w/ `XORL_BATCH_INVARIANT_OPS` selective mode, MoE/attention parity, k3 debug metrics, token diagnostics). Dense-K3 excluded. ⚠️ Five engine PRs are in flight (#430–#434); #430/#434 share collision files — second to merge rebases. |
> | **Infra** | `xorl-infra` branch `wordle-consolidation` **pushed** — `k8s/wordle/{builders,launch}/` + `configs/wordle/` + README | |
> | **SMG (gap CLOSED)** | Source found: `togethercomputer/together-smg` branch `xorl` @ `a83f1a6f` "smg: per-row sampling_params + routed-experts passthrough for RL on-policy" — committed AND pushed (local checkout `~/together-smg-xorl`, clean) | `smg-isr3k3-bin` = build of that branch; still don't overwrite the shared binary. |

> Companion to `HANDOFF.md`. Scoping map for moving validated session work out of
> session-local clones and onto the `apanda-dev` branches of the three repos.
> Authored 2026-06-30. **Breadth = core code only** (engine `src/` + client library +
> SGLang RL patches); wordle experiment scripts and handoff `.md` docs are deliberately
> left on the science branch / `/shared`.

> **⏩ 2026-07-02 UPDATE (for the incoming code-audit agent):** two changes since authoring.
> (1) **The engine HEAD advanced `98eb289e` → `be40a275c`** (`~/xorl-qwen-k3-reconciliation`):
> commits `49720d816` "dense-K3 roadmap execution" + `be40a275c` docs added a **further 25-file /
> +3,442-line `src/` diff** (dense-K3 work, from the co-active k3/dense agent). So §3's
> "98eb289e / ~3,250 insertions" is a LOWER BOUND — re-scope the engine diff against the current
> HEAD: `git -C ~/xorl-qwen-k3-reconciliation diff origin/apanda-dev..be40a275c -- src/`.
> (2) A **4th area exists that this doc originally excluded — the orchestration scripts** (`/shared/apanda/wordle-sft-runs`);
> the incoming agent asked to audit them, so they're now mapped in **§4 below**. Notably they include a
> real eval-infra bug FIX (`engine_connect_host`) worth folding into the eval tooling.

## Context

A long Wordle RL training effort produced validated changes spread across several
working trees. This document scopes **what should be upstreamed where** so the work
isn't stranded. It is a map + recommended sequence + ready-to-use commit/PR
descriptions. The mechanical git work is left for a follow-up (none done yet).

## Corrected mental model (findings that change the approach)

Three repos are involved (the working dirs are clones/worktrees of these):

| Repo (remote) | Upstream target branch | Where the work currently lives |
|---|---|---|
| `xorl-internal` (engine) | `apanda-dev` (clone: `~/xorl-apanda-dev`, **1 behind** `origin/apanda-dev`) | `~/xorl-qwen-k3-reconciliation` @ `98eb289e`, branch `k3-recon-r3-hangfix-merge-20260630` |
| `xorl-client-internal` (client) | `apanda-dev` (tracks `internal/apanda-dev`; not checked out in either worktree) | session worktree `~/xorl-client-wordle-science-20260614`, branch `science/wordle-retrieval-sft-20260614` |
| `xorl-sglang-internal` | `apanda-dev` (currently checked out) | same repo, **11 uncommitted files** |

Two corrections to the working assumptions in `HANDOFF.md`:

1. **The engine live-path fixes are COMMITTED, not dirty.** `git status` in
   `xorl-qwen-k3-reconciliation` shows **zero dirty files under `src/`** — the ~20 dirty
   files are post-snapshot experiment artifacts (`experiments/k3_tests/`
   configs/results, `docs/notes/`, a `return_original_logprob` removal in the k3 test
   harness). The real engine work is the **committed** `src/` diff at `98eb289e`. So a
   fresh clone at `98eb289e` is self-sufficient for the engine runtime; the dirty files
   are NOT needed to run it. (HANDOFF's "a clone needs the dirty changes, not just
   98eb289e" is true for **SGLang**, false for the **engine** — verify per below.)
2. **`xorl_client.rl` does not exist on `internal/apanda-dev` or `internal/main`.** So
   `grpo_rl_shim.py` is a genuine vendored fallback, not dead code — but it only matters
   if the experiment scripts are upstreamed (they are out of scope here).

`~/xorl-internal` is on `feature/on-policy-distillation` with massive divergence — it's
only the compiled-deps venv, **not** an upstreaming target. Ignore it.

---

## The map (core-code breadth)

### 1. SGLang → `xorl-sglang-internal` `apanda-dev`  *(do first — easiest, already on branch)*

11 uncommitted files, ~146 insertions. Already on the correct branch; just needs
committing. Group into 3 logical commits:

- **`feat: return per-request expert logits in OpenAI completions (RL)`** —
  `entrypoints/openai/protocol.py` (`return_expert_logits` req field + `expert_logits`
  on `SglExt`), `serving_chat.py`, `serving_completions.py`, `entrypoints/openai/utils.py`
  (`process_expert_logits_from_ret`), `layers/moe/topk.py` (`topk_weights` export).
- **`feat: selective batch-invariant op mode via SGLANG_BATCH_INVARIANT_OPS`** —
  `batch_invariant_ops/__init__.py`, `batch_invariant_ops/batch_invariant_ops.py`,
  `layers/layernorm.py` (gate rms_norm on selective mode), `model_executor/model_runner.py`
  (startup logging).
- **`feat: weight-version-keyed request routing`** —
  `managers/tokenizer_manager.py` (prepend `weight_version` to `extra_key`),
  `managers/scheduler.py` (`extra_key` pass-through).

Pairs conceptually with the engine R3 work (expert logits ↔ routing replay) but is an
independent repo/commit.

### 2. Client → `xorl-client-internal` `apanda-dev`  *(do second — clean, isolated)*

Exactly **one** file in scope: `xorl_client/client/chunked_helpers.py` (+23 lines).
Adds `_estimate_r3_payload()` and folds `routed_experts` / `routed_expert_logits` byte
estimates into `estimate_datum_bytes()` for both dict and Datum codepaths. Verified: the
`apanda-dev` blob is identical to the science-branch committed blob (`fdd04ea`), so this
is a clean, conflict-free +23 on top of `apanda-dev`.

- **`feat: account for R3 routed-expert payloads in chunking byte estimate`** —
  cherry-pick or re-apply this single working-tree hunk onto a branch off
  `internal/apanda-dev`.

**Out of scope (stays on `science/wordle-retrieval-sft-20260614`):** everything under
`experiments/wordle/` — `tasks/wordle.py` (+119, `wordle_retrieval_reward`),
`train_grpo_wordle.py` (+597), `train_opsd_baseline.py` (+35), `grpo_rl_shim.py`,
`warmstart_zorl_pool.py`, `test_wordle_retrieval_reward.py`, and all handoff `.md`s, plus
the 3 science-branch docs commits (`876c204`, `5f767f0`, `cb24f20`). (`zorl_config`
gating is already on `apanda-dev`.)

### 3. Engine → `xorl-internal` `apanda-dev`  *(do last — largest, needs review)*

The substantive work. `git diff origin/apanda-dev..98eb289e -- src/` = **~3,250
insertions across 28 files**. Largest pieces:

- `ops/batch_invariant_ops.py` (+1117) — batch-invariant kernels (BI variant)
- `models/layers/moe/experts.py` (+743), `moe/moe_block.py`, `moe/activations.py`
- `server/runner/model_runner.py` (+341), `runner_dispatcher.py` (+158),
  `utils/routing_replay_handler.py` (+206 — R3 replay), `request_processor.py` (+124)
- losses: `importance_sampling_loss.py` (+53), `per_token_ce.py`, `policy_loss.py`,
  `causallm_loss.py`
- `server/server_arguments.py` (+58), `launcher.py`, `orchestrator.py`
- `layers/attention/backend/flash_attention.py` (+135), `multi_head_attention.py`,
  `layers/normalization.py`, `layers/rope.py`, gpt_oss + qwen3 modeling files

**Approach:** this is an integration effort, not a one-line cherry-pick:
1. In `~/xorl-apanda-dev`, `git pull` (local is 1 behind `origin/apanda-dev`); integrate
   against the *freshest* `origin/apanda-dev`.
2. Create `apanda-dev-k3recon-integration`; apply the `src/`-scoped diff from `98eb289e`
   (the `-- src/` scope naturally excludes `experiments/k3_tests/` artifacts and `docs/`).
3. **Review for overlap** — some parity/BI knobs may already have landed on `apanda-dev`
   via other PRs; reconcile rather than blindly overwrite. Split into reviewable commits
   roughly along the groups above (BI ops; MoE; R3 routing/replay; losses; server args).
4. Open a PR to `apanda-dev`; the eventual path to `internal/main` is a separate review.

**Out of scope:** `experiments/k3_tests/` configs/results/manifests, `docs/notes/*`, the
`return_original_logprob` test-harness removal, and the speed-attribution scripts — all
experiment artifacts.

### 4. Orchestration scripts → `/shared/apanda/wordle-sft-runs`  *(NEW area, 2026-07-02; `/shared`-local, not a repo)*

The stack is launched/evaluated by scripts in `/shared/apanda/wordle-sft-runs` (under no repo). The incoming
agent asked to audit these — they're the "how runs were launched + evaluated" layer. Files + this-session changes:

- **Trainer-stack builders** (each emits `launch/<name>-{sampler,smg,smg-svc,trainer}.yaml`):
  - `build_k3diag.py` (no-BI, low-k3 ~3e-4) / `build_k3bi.py` (batch-invariant, lowest-k3 ~2.6e-4) — validated on-policy GRPO stacks (pre-existing).
  - `build_k3pipe.py` **(new)** — k3diag recipe + `--pipeline-rl --no-weight-sync-flush-cache` (pipeline-RL, low-k3).
  - `build_k3pnr3.py` **(new)** — k3pipe with **routing replay PEELED** (`--return-routed-experts`/`--return-expert-logits` removed) → high-k3 base (~3-7e-3); `save-interval 25`.
  - Knobs: `WORDLE_Q36_SERVER_OUTPUT_DIR` (must be unique per stack — gotcha #2), `SAMPLER_INDICES`, `--wandb-name`, `XORL_SRC` pin (k3-recon).
- **Eval tooling:**
  - `prepare_checkpoint_eval_serving.py` — serves a DCP ckpt (private TP2 sglang + EP8 sync). **⭐ FIX this session: `config.pop("engine_connect_host")`** — the copied *training* config's `engine_connect_host` forced the eval orchestrator to connect `:5556` while the worker binds `:25670` → ZMQ "engine-0 not routable" that silently killed EVERY eval 06-26→06-30. The one orchestration change that's a genuine BUG FIX (not just a recipe) — worth folding into the eval tooling wherever it's versioned.
  - `eval_ckpt_generic.sh` — sync 1 ckpt → 1 `shard_eval` → teardown (honest `EVAL_INVALID_RETRIES=0` gate).
  - `eval_ckpt_multi.sh` **(new)** — sync ONCE → many `shard_eval.py` (offset:count:suffix spec) → teardown; avoids the ~7min re-sync per shard. For big-N / multi-try paired evals.
  - `run_bigeval_serial.sh` **(new)** — runs eval_ckpt_multi for N models SERIALLY (concurrent EP8 syncs crash).
  - `shard_eval.py` — splits held-out games across serving replicas.
- **Manifests** (`launch/`): per-stack sampler StatefulSet (SGLang parity flags), SMG Deploy+svc (`smg-isr3k3-bin`), trainer Job+head-svc. Hardcoded bad-node `nodeAffinity NotIn` lists were intentionally CLEARED (node health is transient → handle reactively).

These are `/shared`-scratch. If any deserve versioning — esp. the **`engine_connect_host` eval fix** and the multi/serial eval drivers — the client repo's `experiments/wordle/` is the natural home (but that's outside the 3-repo core-upstreaming above).

---

## Out of scope / gaps to flag

- **SMG patched binary** (`/shared/apanda/wordle-sft-runs/smg-isr3k3-bin`, accepts per-row
  `sampling_params`): the source for this is **not in any of the three named repos**. To
  upstream it, the SMG (sampler-multiplexer-gateway) source repo must be located first —
  out of scope until identified.
- **`~/xorl-internal`** (`feature/on-policy-distillation`): venv host only, not a target.

## Verification

- **Engine self-sufficiency claim:** `git -C ~/xorl-qwen-k3-reconciliation status --short`
  should show nothing under `src/` (confirmed); thus `98eb289e` runs without the dirty
  files. Sanity: `git diff origin/apanda-dev..98eb289e -- src/ | wc -l` ≈ 3.4k.
- **Client diff is clean:** `git -C ~/xorl-client merge-base --is-ancestor internal/apanda-dev
  science/wordle-retrieval-sft-20260614` is true; the chunked_helpers hunk applies with no
  conflict (blob `fdd04ea` matches on both sides pre-hunk).
- **SGLang:** `git -C ~/xorl-sglang-internal status --short` lists the 11 files; after the 3
  commits, `git diff` is empty and the branch is `apanda-dev`.
- No build/test run is part of *this* scoping task. Per-repo CI/tests apply when the actual
  PRs are opened.

## Recommended sequence

1. **SGLang** (3 commits) — self-contained, already on `apanda-dev`, lowest risk.
2. **Client `chunked_helpers`** — clean single-file cherry-pick.
3. **Engine `src/` integration** — the big one; branch + review + split + PR.
