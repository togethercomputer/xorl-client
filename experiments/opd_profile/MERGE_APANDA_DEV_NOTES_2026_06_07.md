# apanda-dev merge — handoff (2026-06-07, resolved autonomously overnight)

## TL;DR
- **The merge of `origin/apanda-dev` is resolved and committed on branch `merge-apanda-dev` (commit `43bfe780`).**
- **The working branch `codex/opd-port-20260602` was reset to the pre-merge checkpoint `baf6dcce` so the autoresearch loop keeps running on proven code.** The merge is **untested against the live OPD flow** — adopt it only after testing (see below).
- The loop is running on `baf6dcce` (= the exact working tree it ran on all session, the 22 prior uncommitted src/xorl changes now committed).

## Why not run the loop on the merged code
apanda-dev did a large server-stack refactor (model_runner +675/−398, p2p +435, inference_endpoints +332, opd_loss −345) and its current tree has **no `hidden_match`** (PR #344 added it, a later FP8/rebase removed it). The local OPD client (`xorl-client-chat-completions/examples/on_policy_distillation.py`) + the generator were built against the *local* server API. Running the loop on the merged server overnight risked an API-drift crash-loop on 64 GPUs — which violates the "never idle / loop must run" requirement. So the merge is preserved for review, loop stays on proven code.

## What's in `merge-apanda-dev` (the resolution)
- **Base = apanda-dev** for every conflicted file (keeps FP8, harden-weight-sync, GLM-5, DSv4, etc.).
- **Ported onto apanda-dev:** the OPD **hidden-match** feature in `src/xorl/ops/loss/opd_loss.py` (the `hidden_match_coef` + `hidden_match_weights` params, the cosine-distance loss term, and the 10 `opd_hidden_match_*` metrics) — the experiments depend on it. Compiles clean.
- **Dropped in favor of apanda-dev (best-effort, per "drop what we don't need"):**
  - `p2p.py` — took apanda-dev. **The `XORL_P2P_HANDSHAKE_BASE_PORT` pin is NOT in the merge** (it's in `baf6dcce`). Re-port it if adopting the merge (it prevents the peer-nic stale-registry bursts; see `P2P_CRASH_DIAGNOSIS.md`).
  - `model_runner.py`, `inference_endpoints.py` — took apanda-dev's upstreamed OPD support (local OPD endpoint/runner additions dropped).
  - `runner_dispatcher.py` — took apanda-dev (dropped the local shared-prefix repack block; the OPD loop doesn't use `use_shared_prefix`).
  - All 5 conflicted test files — took apanda-dev.
- 5 non-overlapping local files (collators ×3, batch_utils, weight_sync_stress) merged cleanly and are in `baf6dcce`.

## To adopt the merge (morning, with a GPU)
1. `git checkout merge-apanda-dev` (or merge it into the working branch).
2. Re-port the **P2P handshake-port pin** into apanda-dev's `p2p.py` (`PeerTransferEngine.__init__`, set `MC_HANDSHAKE_PORT` from `XORL_P2P_HANDSHAKE_BASE_PORT`+gpu_id before `transfer_engine_cls()`).
3. Launch ONE OPD candidate and watch step 0 — verify the client's endpoint/arg calls still match apanda-dev's refactored server (`server_arguments`, `inference_endpoints`, `model_runner`). Fix any API drift.
4. Confirm `opd_hidden_match_coef` flows end-to-end (the ported hidden-match runs).
5. If clean, point the loop's working tree at it.

## Branches
- `codex/opd-port-20260602` → `baf6dcce` (pre-merge, loop runs here).
- `merge-apanda-dev` → `43bfe780` (the resolved merge).
- Nothing pushed.
