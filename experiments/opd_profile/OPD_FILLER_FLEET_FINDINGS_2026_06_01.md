# Filler-token encoded-reasoning — SMG-fleet dual-regime eval (2026-06-01)

## Infra (efficient, logged — replaces ad-hoc single-replica curl loops)
- `experiments/opd_profile/k8s/build_eval_fleet.py` — generates N sglang replicas +
  one SMG `--policy cache_aware` router (port 8080) per model. cache_aware routes the
  shared system+10-shot+filler prefix to the same shard → KV-cache reuse (~3× per-GPU
  per the SMG runbook).
- `experiments/opd_profile/eval_filler_fleet.py` — hits the router, runs the dual-regime
  grid, logs EVERY per-sample record to `results/filler_fleet/<run-id>/samples.jsonl`
  + aggregate `summary.json`.
- Fleets launched: eval-q35-35b, eval-q36-35b, eval-q3-235b (TP8×2), eval-q3-235bi (TP8×2),
  each behind its own router. A parallel session ran matching `-fleet-eval` run-ids against
  the same routers (cross-validation) + a 2-node 397B-BF16 fleet.

## Methodology (paper-faithful)
- 4dmult 10-shot, prefill methodology: few-shot demos show `[100 pause]\nAnswer: <prod>`;
  test prefills `[100 pause]\nAnswer: ` and the model emits ONLY the number. Baseline = no filler.
- **Two regimes** (the crux of the earlier reproduction failure):
  - `chat_hardoff` = `/v1/chat/completions` + `chat_template_kwargs={"enable_thinking":false}`
    (HARD thinking-off). Genuine-capability regime.
  - `raw_nothink` = `/v1/completions` raw `<|im_start|>` prompt + only `/no_think` TEXT
    (thinking left enabled in the model's mode). This is what the tomi harness
    (`LiteQwen3InstructRenderer`) does — and the regime where the paper's lifts live.
- pass@1 and pass@8 (k=8, T=1.0), n=100 problems/cell.

## Results (b=base, q=pause; pass@1 | pass@8). Mine + parallel agree.

| model | regime | 4×4 b_p1 | 4×4 q_p1 | 4×4 b_p8 | 4×4 q_p8 |
|---|---|---|---|---|---|
| Q3.5-35B-A3B | chat_hardoff | 0.78–0.85 | 0.78 | 0.96–0.97 | 0.96–0.97 |
| Q3.5-35B-A3B | **raw_nothink** | **0.56–0.58** | **0.76 (+.18)** | 0.91–0.96 | 0.95–0.96 |
| Q3.6-35B-A3B | chat_hardoff | 0.82–0.84 | 0.80 | 0.97 | 0.94–0.95 |
| Q3.6-35B-A3B | **raw_nothink** | **0.52–0.60** | **0.69 (+.10)** | 0.93 | 0.93–0.97 |
| Q3-235B-A22B (base) | chat_hardoff | 0.46–0.48 | 0.54–0.59 | 0.79–0.81 | 0.77 |
| Q3-235B-A22B (base) | raw_nothink | 0.59–0.63 | 0.62–0.67 | 0.73–0.78 | 0.80–0.85 |

5×5 / 6×6: small noisy effects everywhere; pass@8 near floor and NOT raised by filler.

## Verdict (robust across 3 working models × 2 regimes × pass@1/pass@8)
1. **The +pause lift reproduces** — Q3.5-35B raw 4×4 **+0.18 pass@1** (= paper's +14.9 cell;
   earlier failed repro was the HARD `enable_thinking=False` toggle vs the harness's text-only
   `/no_think`).
2. **It is format-recovery, not capability** — `enable_thinking=False` base (0.78–0.85 on 35B)
   *exceeds* the raw-regime filler-boosted number (0.76). A well-prompted single pass beats filler.
3. **The pass@8 capability ceiling is FIXED** — at 4×4, pass@8 ≈ 0.96 (35B) / 0.79 (235B-base)
   regardless of regime or filler. Filler moves pass@1 *toward* that ceiling; it never raises it.
4. **The paper's biggest base-235B lifts (+21pp) were reasoning-LEAK** — free-gen lets the base
   model secretly reason in the filler. The prefill method (number-only emit) removes the leak
   → the lift vanishes (±0.06 noise here).

**Encoded-reasoning implication:** distilling teacher reasoning into a filler buffer would have
to RAISE the pass@8 ceiling (let the model solve what it currently cannot). No model shows filler
doing this. The buffer is a pass@1 reliability channel on a fixed capability ceiling, not an
encodable computation channel. Consistent with the paper's own RL finding (pass@8 not pass@1 =
search) and the 235B buffer-distill plateau.

## Status / pending
- ✅ Q3.5-35B, Q3.6-35B, Q3-235B-base: done (logged, cross-validated).
- ✅ Qwen3-235B-A22B-Instruct-2507 (the variant the paper's appendix actually used) — DONE
  (run-id `q3-235bi-instruct`, 0 garbage after fetching its own tokenizer; the earlier
  `q3-235bi-fleet-eval` all-zeros was a broken tokenizer — INVALID, ignore).
  **ZERO filler effect in EITHER regime**: chat_hardoff 4×4 base .740/pause .720; raw_nothink
  4×4 base .710/pause .710 (exactly flat), pass@8 .890 both. chat_hardoff≈raw_nothink (.74 vs .71)
  because 2507 is a PURE NON-THINKING instruct model → no thinking-primed mode for `/no_think`
  to half-suppress → no format-deficit for filler to recover. Strongest confirmation that the
  filler lift is soft-suppression recovery, not capability. pass@8 ceiling untouched.
- ✅ Qwen3.5-397B-A17B BF16 (2-node TP16, served clean — NOT the broken FP8 multimodal) — DONE.
  **The doc's +20.1 ellipsis cell REPRODUCES and is 100% anti-EOS.** raw_nothink 4×4:
  base p@1 0.250 → ellipsis 0.460 (**+0.210**); 5×5 +0.18. Empty-completion rate (premature EOS):
  4×4 base **37.8%** → ellipsis **19.2%**; 5×5 34.2% → 11.5%. The pass@1 lift EQUALS the recovered
  empties — ellipsis just stops the model halting after `Answer:`. In chat_hardoff (capability)
  the 397B does 4×4 0.99 / 5×5 0.96–0.98 / **6×6 0.86–0.89** with 0% empty and filler FLAT —
  capability is fully present; the raw deficit is entirely the EOS pathology. (pause filler also
  reduced empties but less than ellipsis; 6×6 ellipsis noisy/inverts.) Anti-EOS is NOT distillable
  encoded reasoning — a 0-shot student has no EOS pathology to transfer.

## FINAL mechanistic table — every doc "filler benefit" is a generation/format artifact
| model | the doc's "lift" | what it actually is (this study) | pass@8 ceiling raised? |
|---|---|---|---|
| Q3.5-35B-A3B | 4dmult-10 pause +14.9 | format-recovery (raw `/no_think` softly suppresses; hard `enable_thinking=False` base 0.85 > filler 0.76) | NO (0.96 fixed) |
| Q3.6-35B-A3B | 4dmult-10 pause +4.2 | same format-recovery, smaller | NO |
| Q3-235B-A22B base | arith +21 | reasoning-LEAK (free-gen hidden CoT; prefill→±0.06) | NO |
| Q3-235B-Instruct-2507 | (paper's model) | ZERO effect — pure non-thinking, no suppressible regime | NO |
| Q3.5-397B-A17B | 4dmult-10 ellipsis +20.1 | anti-EOS (empty 37.8%→19.2%; capability regime does 6-digit at 0.89) | NO |

**Bottom line:** the filler buffer moves pass@1 toward a FIXED pass@8 capability ceiling via three
generation artifacts (format-recovery / leak / anti-EOS) — none raise the ceiling, none are a
distillable reasoning channel. Distilling teacher reasoning into a filler buffer has no target.

## CORRECTION (contested cell, n=400) — base-235B is a genuine compute-concentration exception
Re-ran Q3-235B base 4×4 at n=400 (SE~0.025) to settle the parallel session's "distillable compute"
claim vs my "format-recovery" claim. Result:
  chat_hardoff: base 0.430 → pause 0.530 (+0.100); pass@8 0.792 → 0.780 (−0.012)
  raw_nothink : base 0.583 → pause 0.657 (+0.075); pass@8 0.818 → 0.828 (+0.010)
- My "pure format-recovery" framing was TOO STRONG: `raw+pause` (0.657) EXCEEDS the model's best
  no-filler baseline `raw+base` (0.583) by +0.075. So on base-235B the filler adds genuine per-sample
  compute IN ITS OWN regime — real "compute concentration" (Think-Dot-by-Dot), not just chat-template
  recovery. The parallel session's mechanism read is correct for this one cell.
- BUT pass@8 is flat in both regimes (ceiling ~0.82 with/without filler) → concentration toward a
  FIXED ceiling, NOT capability expansion. And it's NOT distillable (buffer-distill = no; paper RL =
  pass@8-not-pass@1). base-235B is the only non-instruct large model → only one still compute-limited
  per-token; the instruct models (35B format-recovery, Instruct-2507 saturated, 397B chat 0.99) and
  free-gen leak (base arith) are all artifacts. Net program conclusion unchanged: buffer is a pass@1
  reliability channel on a fixed pass@8 ceiling, not an encodable/distillable reasoning channel.
