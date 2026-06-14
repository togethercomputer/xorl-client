#!/usr/bin/env python3
"""Deep-analysis decomposition of an OPD control eval.

The autopilot verdict leans on `vs_corrupt` + a teacher-forced `answer_logprob`
margin. This tool separates the two questions the user actually cares about:

  (A) Does the filler help GENERATION over just answering directly?
      -> buffer_delta = acc_pause - acc_nopause (the real "prefill compute" claim).
  (B) Is the vs_corrupt margin real compute, or just OOD generation-derailment?
      -> compare the per-arm generation diagnostics (completion length, cap-hit,
         repeated-numeric, stop-seq). If corrupt fails by RAMBLING/CAP-HITTING
         rather than answering, vs_corrupt is a format artifact, not compute.

Usage:
  da_decompose.py PTC-081                 # resolves last_profile from ideas.yaml
  da_decompose.py /path/to/opd_profile.jsonl
  da_decompose.py PTC-088 PTC-089 ...     # several at once
"""
import json
import sys
from pathlib import Path

import yaml

AR = Path(__file__).resolve().parent
IDEAS = AR / "ideas.yaml"


def resolve(arg: str) -> tuple[str, Path | None]:
    if arg.endswith(".jsonl"):
        return arg, Path(arg)
    data = yaml.safe_load(IDEAS.read_text())
    it = next((i for i in data.get("ideas", []) if i.get("id") == arg), None)
    if not it:
        return arg, None
    p = it.get("last_profile")
    return arg, (Path(p) if p else None)


def g(row: dict, key: str, default=float("nan")) -> float:
    v = row.get(f"eval/{key}", default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def control_rows(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    # keep rows that actually ran a control eval (have the 3 arms scored)
    return [r for r in rows if r.get("eval/acc_pause") is not None
            and r.get("eval/control_scored_pause")]


def report(idea: str, path: Path | None) -> None:
    print(f"\n{'='*78}\n{idea}   {path}")
    if not path or not path.exists():
        print("  (no profile available yet)")
        return
    rows = control_rows(path)
    if not rows:
        print("  (no control-eval rows yet — control_start_step not reached)")
        return
    print("  step | acc:  pause  nopause  corrupt | Δpause-nopause (z)  | Δpause-corrupt (z)"
          " | lp-margin(z) | lp-vscorrupt(z)")
    for r in rows:
        s = r.get("step")
        print(f"  {str(s):>4} |       {g(r,'acc_pause'):.3f}  {g(r,'acc_nopause'):.3f}"
              f"   {g(r,'acc_corrupt_pause'):.3f} |"
              f"  {g(r,'buffer_delta'):+.3f} (z={g(r,'buffer_delta_z'):+.2f}) |"
              f"  {g(r,'buffer_vs_corrupt_delta'):+.3f} (z={g(r,'buffer_vs_corrupt_delta_z'):.1f}) |"
              f"  {g(r,'answer_logprob_margin'):+.3f}(z={g(r,'answer_logprob_margin_z'):.1f}) |"
              f"  {g(r,'answer_logprob_vs_corrupt_margin'):+.2f}(z={g(r,'answer_logprob_vs_corrupt_margin_z'):.0f})")
    # generation-derailment diagnostics on the LAST control row
    r = rows[-1]
    print(f"\n  --- generation diagnostics @ step {r.get('step')} (is vs_corrupt just derailment?) ---")
    print(f"  {'arm':<14} {'mean_compl_tok':>14} {'cap_hit_frac':>12} {'rep_numeric':>12}"
          f" {'stop_seq_frac':>13} {'reason_phrase':>13}")
    for arm in ("pause", "nopause", "corrupt_pause"):
        print(f"  {arm:<14} {g(r,arm+'_mean_completion_tokens'):>14.1f}"
              f" {g(r,arm+'_cap_hit_frac'):>12.3f} {g(r,arm+'_repeated_numeric_frac'):>12.3f}"
              f" {g(r,arm+'_stop_sequence_seen_frac'):>13.3f} {g(r,arm+'_reasoning_phrase_frac'):>13.3f}")
    # interpretation
    bd, bdz = g(r, "buffer_delta"), g(r, "buffer_delta_z")
    real = abs(bdz) >= 2.0 and bd > 0
    corrupt_rambles = g(r, "corrupt_pause_mean_completion_tokens") > 2 * max(
        g(r, "pause_mean_completion_tokens"), 1.0)
    print("\n  VERDICT-LENS:")
    print(f"    (A) filler vs direct-answer: buffer_delta={bd:+.3f} z={bdz:+.2f}"
          f"  -> {'REAL gen lift' if real else 'within noise — filler adds nothing over direct'}")
    print(f"    (B) corrupt arm rambling? corrupt_tok={g(r,'corrupt_pause_mean_completion_tokens'):.0f}"
          f" vs pause_tok={g(r,'pause_mean_completion_tokens'):.0f}"
          f"  -> {'YES: vs_corrupt is OOD-derailment, not compute' if corrupt_rambles else 'no'}")


def main() -> None:
    args = sys.argv[1:] or ["PTC-088", "PTC-089"]
    for a in args:
        idea, path = resolve(a)
        report(idea, path)


if __name__ == "__main__":
    main()
