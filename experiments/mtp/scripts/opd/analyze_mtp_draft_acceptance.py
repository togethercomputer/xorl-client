#!/usr/bin/env python3
"""Per-draft-offset acceptance analysis for native-MTP OPD rollouts.

The OPD-MTP throughput goal is to raise ``commit_len`` (speculative-decode draft
acceptance). The per-step throughput JSON only reports AGGREGATE metrics
(``commit_len_steady``, ``top1_agreement``, ``ent_stud``). Those aggregates are
dominated by the hard offset-2..k draft positions and by the verify positions,
so they MISLEAD: a flat aggregate ``top1`` can hide a sharp-and-fine verify side
plus a diffuse, untrained draft side. Mistaking the aggregate for the draft
signal is what repeatedly pointed debugging at the loss mode.

This tool reads ``rollout_samples.jsonl`` (the ``native_mtp_debug_trace`` per
decode step) and reports the thing that actually gates throughput:

  * per-offset DRAFT acceptance  — P(commit_len >= offset+1 | draft emitted).
    offset-1 is the draft that gates commit_len 1->2; it is the headroom.
  * per-offset DRAFT confidence  — the student's own top-prob at the draft slot.
    Diffuse (~conf of 1/effective-vocab) => not sharpened; sharp-but-rejected =>
    the draft argmax disagrees with the AR verify (target/architecture issue).
  * early-vs-late trend           — a positive slope means training is working
    (just slowly / under-powered LR); dead-flat across a long run points at an
    architectural ceiling (mask-conditioned parallel MTP predicts 2+ tokens
    ahead with no intermediate token; cf. EAGLE-style autoregressive drafting).

Trace layout (per steady step): ``pending_token_ids`` = [verify_token, draft1,
draft2, ...] with ``pending_confidences`` aligned; ``commit_len`` accepted tokens
(always >= 1 for the verify token). So draft-j (1-indexed) is accepted iff
``commit_len >= j + 1``.

Usage:
    python scripts/opd/analyze_mtp_draft_acceptance.py <rollout_samples.jsonl> \
        [--early-max-step N] [--late-min-step N] [--max-offset K]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter


def _describe(values: list[float]) -> str:
    if not values:
        return "n=0"
    return (
        f"n={len(values)} median={statistics.median(values):.3f} "
        f"mean={statistics.mean(values):.3f} "
        f">0.5={sum(v > 0.5 for v in values) / len(values):.2f} "
        f">0.7={sum(v > 0.7 for v in values) / len(values):.2f}"
    )


def analyze(path: str, *, early_max_step: int, late_min_step: int, max_offset: int) -> None:
    commit_len_dist: Counter[int] = Counter()
    effective_k_dist: Counter[int] = Counter()
    verify_conf: list[float] = []
    # per draft offset (1-indexed): confidences, emitted count, accepted count
    draft_conf: dict[int, list[float]] = {j: [] for j in range(1, max_offset + 1)}
    draft_emitted: Counter[int] = Counter()
    draft_accepted: Counter[int] = Counter()
    # early/late offset-1 trend
    early_conf: list[float] = []
    late_conf: list[float] = []
    early_acc = [0, 0]
    late_acc = [0, 0]

    steady_steps = 0
    records = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        records += 1
        step = int(rec.get("step", 0))
        for st in rec.get("native_mtp_debug_trace") or []:
            if st.get("phase") != "steady":
                continue
            steady_steps += 1
            commit_len = int(st.get("commit_len_runtime", st.get("commit_len", 1)))
            eff_k = int(st.get("effective_k_runtime", st.get("effective_k", 0)))
            commit_len_dist[commit_len] += 1
            effective_k_dist[eff_k] += 1
            pc = st.get("pending_confidences") or []
            if pc:
                verify_conf.append(pc[0])
            for j in range(1, max_offset + 1):
                if len(pc) >= j + 1:
                    draft_conf[j].append(pc[j])
                    draft_emitted[j] += 1
                    if commit_len >= j + 1:
                        draft_accepted[j] += 1
            # offset-1 early/late trend
            if len(pc) >= 2:
                if step <= early_max_step:
                    early_conf.append(pc[1])
                    early_acc[1] += 1
                    early_acc[0] += commit_len >= 2
                elif step >= late_min_step:
                    late_conf.append(pc[1])
                    late_acc[1] += 1
                    late_acc[0] += commit_len >= 2

    mean_commit = sum(k * v for k, v in commit_len_dist.items()) / max(1, steady_steps)
    mean_eff_k = sum(k * v for k, v in effective_k_dist.items()) / max(1, steady_steps)

    print(f"file: {path}")
    print(f"records={records} steady_steps={steady_steps}")
    print(f"commit_len dist:  {dict(sorted(commit_len_dist.items()))}  (mean {mean_commit:.4f})")
    print(f"effective_k dist: {dict(sorted(effective_k_dist.items()))}  (mean {mean_eff_k:.3f})")
    print(f"verify-token confidence: {_describe(verify_conf)}")
    print()
    print("per-offset DRAFT (the throughput gate; offset-1 lifts commit_len 1->2):")
    for j in range(1, max_offset + 1):
        emitted = draft_emitted[j]
        acc = draft_accepted[j] / emitted if emitted else 0.0
        print(
            f"  offset-{j}: acceptance={draft_accepted[j]}/{emitted}={acc:.3f}  confidence {_describe(draft_conf[j])}"
        )
    print()
    print(f"offset-1 trend (early step<={early_max_step} vs late step>={late_min_step}):")
    print(f"  confidence  early: {_describe(early_conf)}")
    print(f"  confidence  late:  {_describe(late_conf)}")
    e = early_acc[0] / early_acc[1] if early_acc[1] else 0.0
    lt = late_acc[0] / late_acc[1] if late_acc[1] else 0.0
    print(f"  acceptance  early: {early_acc[0]}/{early_acc[1]}={e:.3f}   late: {late_acc[0]}/{late_acc[1]}={lt:.3f}")
    print()
    print("read: diffuse draft confidence + positive early->late slope => UNDER-TRAINED (raise LR / more steps).")
    print("      dead-flat slope across a long run => architectural ceiling (mask-conditioned 2+-ahead draft).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rollout_samples_jsonl")
    ap.add_argument("--early-max-step", type=int, default=12)
    ap.add_argument("--late-min-step", type=int, default=40)
    ap.add_argument("--max-offset", type=int, default=3)
    args = ap.parse_args()
    analyze(
        args.rollout_samples_jsonl,
        early_max_step=args.early_max_step,
        late_min_step=args.late_min_step,
        max_offset=args.max_offset,
    )


if __name__ == "__main__":
    main()
