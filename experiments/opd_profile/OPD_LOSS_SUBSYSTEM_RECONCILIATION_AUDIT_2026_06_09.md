# OPD-loss subsystem reconciliation audit (2026-06-09)

**Question asked:** if our (pre-glm5) OPD-loss subsystem is the one we want, should it be upstream
in `apanda-dev`? Are we the most sophisticated user? Or does upstream have improvements we're missing?

**Verdict:** This is **not** a "two legitimate designs diverged" situation. The glm5 rebase
`955191dd` ("feat(glm5): rebase support stack onto apanda-dev (#211)") **accidentally reverted
~811 lines of already-merged, deliberately-added OPD-loss features**. `apanda-dev` today is a
strict, *regressed subset* of the pre-rebase subsystem. It has **zero** improvements we lack. Our
branch carries the correct (pre-rebase) subsystem + private OPRD research layered on top.

So: yes, the correct content should be upstream — and it's not a matter of taste, it's restoring
work that was merged and then clobbered. The canonical reference is `955191dd^` (the commit
*before* the rebase).

---

## Evidence

- `git log origin/apanda-dev -- src/xorl/ops/loss/opd_loss.py` shows the features were added by
  **deliberate** commits: `22380973` (forward_kl_full + KL estimator family + PG mode), `528cf412`
  (opd_ppo_kl + opd_pg_clipfrac_lower), `0187bf48` (OPD hidden match weights).
- `955191dd` (the glm5 rebase, the *last* commit touching these files) deleted **792 lines** across
  the three files in one shot: `opd_loss.py` 599→214 (forward_kl_full refs 13→0),
  `compiled_cross_entropy.py` 248-effective (all KL kernels gone, fkf refs 22→0), `opd_streaming_kl.py`.
  It is a 1-parent commit (a squashed rebase), not a reviewed refactor PR.
- **Sole consumer = `model_runner.py`.** Nothing else imports `opd_loss_function`/`OPDLossMetrics`/
  the CCE kernels. There are **no** separate FP8/MTP/RL consumers, so the "the refactor was
  intentional to serve other users" hypothesis is **false** — there are no other users.
- `origin/main` is also 214L without the features → the features lived on the apanda-dev feature
  line and were never blessed into main; the rebase dropped them rather than promoting them.

---

## Three-way classification (OURS = working tree, PREREBASE = `955191dd^`, APANDA = `origin/apanda-dev`)

Structure is cleanly layered: **APANDA ⊂ PREREBASE ⊆ OURS** (OURS = PREREBASE + private OPRD).

### RESTORE — dropped by the rebase, belongs upstream (~811 lines)
| file | what | ~lines |
|---|---|---|
| `opd_loss.py` | loss-mode dispatch (`reverse_kl_full`/`forward_kl_full`/estimators), `_kl_penalty_estimator`+`_kl_penalty_forward` (VERL parity), PG/PPO-clip mode (`opd_ppo_kl`/`pg_clipfrac[_lower]`), expanded `OPDLossMetrics` (hidden-match, entropy/top1, loss-range, PG fields) + **always-emit `to_dict`**, `log_prob_min_clamp`/`loss_max_clamp` | ~385 |
| `compiled_cross_entropy.py` | the KL kernels: `_compute_reverse_kl_with_diag`, `_compute_forward_kl_full[_with_diag]`, `_compute_sampled_token_logprobs`, `_full_vocab_diagnostics`, their compile-caches + 4 public `compiled_*_function` APIs | ~354 |
| `opd_streaming_kl.py` | `streaming_full_vocab_diagnostics` + the `vocab_chunk_size: int\|None` **None-guard** (our restored form is safer than even pre-rebase) | ~72 |
| `tests/ops/loss/test_opd_verl_parity.py` | references all four restored CCE functions + `emit_full_vocab_diagnostics` | — |
| `model_runner.py` (call site only) | restore passing `loss_mode`/`use_policy_gradient`/`hidden_match_coef`/clamps to `opd_loss_function` (apanda passes **0** of these) | small |

### OPRD-EXCLUDE — our private research, stays out of the convergence PR (~190–204 lines, opd_loss.py)
`opd_oprd_loss/raw/num_layers`, per-layer/`teacher_layer_hidden_states`/`student_layer_hidden_states`,
the `use_oprd` MSE-per-layer branch, `kl_loss_weight` (hidden-only knob), `hidden_match_mode`,
region/correctness split fields (`opd_kl_*_per_valid`, `opd_frac_*`, `*_answer_correct/wrong_*`),
`streaming_full_vocab_diagnostics` *wiring into the diag branch*. (Plus all the OPRD machinery in
`model_runner.py`/`packing.py`/the models — none of that is in scope for upstream.)

### Nothing to ADOPT from apanda-dev
The only APANDA-unique constructs are all regressions/artifacts, **do not carry forward**:
1. `opd_num_teachers: Optional[int] = None` + conditional `to_dict` emit — **the dict-keyed
   all-reduce desync bug** (PR #352 worked around it with seeding; pre-rebase's always-emit `int=0`
   supersedes it).
2. `dynamic=True` on the reverse-KL `torch.compile` in CCE — a rebase artifact (pre-rebase has no
   such flag; the other getters are chunked/non-dynamic).
3. The narrower `vocab_chunk_size: int` (no None-guard) in streaming_kl — the regression itself.

---

## Answers to the three questions
1. **Should ours be upstream?** Yes — restoring `955191dd^` content. Not opinion; it's un-reverting a bad rebase.
2. **Are we the most sophisticated user?** We are the **sole** consumer (only `model_runner.py` calls it),
   so there is no other-stakeholder risk and no "intentional for FP8/MTP" tradeoff to weigh.
3. **Does upstream have improvements we're missing?** No. Zero. Its only unique lines are the desync
   bug + two rebase artifacts. (The one upstream idea genuinely worth having — PR #352's
   derive-from-`to_dict` empty-rank seeding — lives in `model_runner.py`, and we already adopted it.)

---

## Convergence PR plan (mechanical, low-risk)
Branch off `origin/apanda-dev`. The rebase reverted to a clean point, so restoration is near-mechanical:
1. `git checkout 955191dd^ -- src/xorl/ops/loss/compiled_cross_entropy.py` (== ours byte-identical).
2. Restore `opd_loss.py` to `955191dd^` (the features-only version — this **excludes all OPRD by
   construction**, since OPRD was added after, on our branch only). Do **not** copy OURS opd_loss.py
   (it carries the 204 OPRD lines + ~14 lines of incidental drift).
3. Restore `opd_streaming_kl.py` from **OURS** (it has the None-guard pre-rebase lacked).
4. Restore `tests/ops/loss/test_opd_verl_parity.py` to its pre-rebase form.
5. `model_runner.py`: surgically restore the `opd_loss_function` **call site** to pass the feature
   params (`loss_mode`, clamps, PG, `hidden_match_coef`) — extracted from pre-rebase, **without** the
   OPRD params. This is the only hand-separation needed (model_runner is a mixed file).
6. Verify: `tests/ops/loss/test_opd_verl_parity.py` + `tests/server/runner/test_opd_runner.py`; ruff.
   PR title `fix(opd): restore OPD-loss features dropped by the glm5 rebase (#211)`.

**Payload:** ~811 lines restored upstream; ~190 lines OPRD stay private on `codex/opd-port-20260602`.
After this lands, our branch's only delta vs apanda-dev for these files is the OPRD research — the
fork tax (recurring merge regressions: None-guard, the metric desync, p2p_invalidate_cache) goes away.
