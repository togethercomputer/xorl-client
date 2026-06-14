"""Clean-region replay PROOF — runs on the loaded trainer model (model_runner).

Triggered by env XORL_CLEAN_REGION_PROOF=1 via a one-shot hook in model_runner's OPD
forward. Forwards a real rollout's replay batch under CURRENT and CLEAN committed-context
orderings through the loaded Qwen3.6 model, captures the student MTP-head argmax at the
DRAFT/mask rows, and compares both to the SAMPLER's recorded argmax_row_token_ids inside
the effective_k=1 collapse runs. Whichever ordering matches the sampler wins (science
agent's Option-2 test). Writes a JSON verdict and (by default) raises to stop the run.
"""
from __future__ import annotations
import json, os, pathlib, sys, importlib.util, collections

import torch

LIVE_SRC = "/home/apanda/xorl-mtp-commitlen-fix-20260612/src"
if LIVE_SRC not in sys.path:
    sys.path.insert(0, LIVE_SRC)
from xorl.mtp import singleshot as ss  # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_clean_plan_gen", _HERE / "_clean_plan_gen.py")
_cpg = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_cpg)
build_clean_plan = _cpg.build_clean_plan

DEFAULT_RUN = ("/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/"
               "er-opd-q36-mtp-ss-0605c/q36mtp-20260613T054455Z-2s1t")
MASK_TOKEN_ID = 248063


def _pick_sample(run_dir: str, max_scan: int = 64):
    """Pick the sample with the longest effective_k=1 run (max current/clean divergence)."""
    p = pathlib.Path(run_dir) / "artifacts" / "rollout_samples.jsonl"
    best = None; best_run = -1
    with p.open() as f:
        for i, line in enumerate(f):
            if i >= max_scan:
                break
            s = json.loads(line)
            run = mx = 0
            for st in s["native_mtp_debug_trace"]:
                run = run + 1 if st.get("effective_k") == 1 else 0
                mx = max(mx, run)
            if mx > best_run:
                best_run, best = mx, s
    return best, best_run


def _build_batch(sample):
    prompt = sample["prompt_tail_token_ids"]; gen = sample["generated_token_ids_full"]; full = prompt + gen
    return ss.prepare_singleshot_mtp_opd_batch(
        {"input_ids": torch.tensor([full[:-1]]), "target_tokens": torch.tensor([full[1:]]),
         "teacher_cache_indices": torch.arange(len(full) - 1).view(1, -1),
         "mtp_loss_start": torch.tensor([len(prompt) - 1]),
         "singleshot_mtp_native_trace": [sample["native_mtp_debug_trace"]]},
        k_toks=sample["mtp_k_toks"], mask_token_id=MASK_TOKEN_ID,
        rollout_replay=True, validate_native_mtp_trace=False)


def _mask_pos_to_step_offset(prepared, sample):
    """Map each dense mask position -> (step_idx, row_offset). Mask positions of step S
    (offsets >= recompute_len_S) appear in dense order; the j-th is offset recompute_len_S+j."""
    block_ids = prepared["attention_mask"].block_ids[0].tolist()
    mask_pos = prepared["_singleshot_mtp_mask_positions"][0].nonzero().flatten().tolist()
    # recompute_len per step from the trace
    trace = sample["native_mtp_debug_trace"]; k = sample["mtp_k_toks"]
    rl_by_step = {}
    for idx, st in enumerate(trace):
        q = ss._trace_q_len(st)
        if q is None:
            continue
        rl_by_step[idx] = ss._trace_recompute_len(st, q, ss._trace_attempt_k(st, k))
    mapping = {}
    per_step_seen = collections.Counter()
    for pos in sorted(mask_pos):
        step = int(block_ids[pos])
        if step < 0 or step not in rl_by_step:
            continue
        off = rl_by_step[step] + per_step_seen[step]
        per_step_seen[step] += 1
        mapping[pos] = (step, off)
    return mapping


def _forward_argmax(runner, prepared):
    """Forward the prepared batch through the loaded model; return {dense_mask_pos: top1_token}."""
    dev = next(runner.model.parameters()).device
    micro = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in prepared.items()}
    if hasattr(prepared["attention_mask"], "to"):
        micro["attention_mask"] = prepared["attention_mask"].to(dev)
    model_inputs = runner._model_inputs_for_loss(micro, "teacher_hidden_cache")
    import contextlib
    fwd_ctx = getattr(runner, "model_fwd_context", None) or contextlib.nullcontext()
    with torch.no_grad(), fwd_ctx:
        outputs = runner.model(**model_inputs, use_cache=False, output_hidden_states=False)
    hidden = outputs.last_hidden_state[0]  # [S, H]
    mask_pos = micro["_singleshot_mtp_mask_positions"][0].nonzero().flatten()
    w = runner._get_effective_lm_head_weight()
    h_sel = hidden[mask_pos]
    logits = (h_sel.float() @ w.float().t()) if runner.lm_head_fp32 else (h_sel @ w.t()).float()
    top1 = logits.argmax(dim=-1).tolist()
    return {int(p): int(t) for p, t in zip(mask_pos.tolist(), top1)}


def run_proof(runner, *, run_dir=None, out_path=None, min_run=4):
    run_dir = run_dir or os.environ.get("XORL_CLEAN_REGION_PROOF_RUN", DEFAULT_RUN)
    out_path = out_path or os.environ.get("XORL_CLEAN_REGION_PROOF_OUT",
                                          "/shared/opd-control/er-opd-q36-mtp-ss-0605c/clean_region_proof_verdict.json")
    sample, run_len = _pick_sample(run_dir)
    prepared = _build_batch(sample)
    pos2so = _mask_pos_to_step_offset(prepared, sample)

    # CURRENT ordering
    cur = _forward_argmax(runner, prepared)
    # CLEAN ordering (swap the committed-context-sorted plan into the same mask)
    m = prepared["attention_mask"]
    clean_plan = build_clean_plan(m.token_kind, m.key_source_indices, m.query_context_source_indices,
                                  m.block_ids, max_sequences=m.linear_plan.sequence_rows.shape[0],
                                  max_outputs=m.linear_plan.output_rows.shape[0])
    prepared_clean = dict(prepared)
    prepared_clean["attention_mask"] = ss.SingleShotRolloutReplayMask(
        token_kind=m.token_kind, key_source_indices=m.key_source_indices,
        query_context_source_indices=m.query_context_source_indices, block_ids=m.block_ids,
        block_size=m.block_size, linear_plan=clean_plan)
    cln = _forward_argmax(runner, prepared_clean)

    # recorded sampler argmax at draft rows, inside effective_k=1 runs
    trace = sample["native_mtp_debug_trace"]; k = sample["mtp_k_toks"]
    run = 0; ek1_run = {}
    for idx, st in enumerate(trace):
        run = run + 1 if st.get("effective_k") == 1 else 0
        ek1_run[idx] = run
    cur_hit = cur_tot = cln_hit = cln_tot = 0
    examples = []
    for pos, (step, off) in pos2so.items():
        st = trace[step]
        if st.get("effective_k") != 1 or ek1_run.get(step, 0) < min_run:
            continue
        amax = st.get("argmax_row_token_ids") or []
        if off >= len(amax):
            continue
        samp_tok = amax[off]
        if pos in cur:
            cur_tot += 1; cur_hit += int(cur[pos] == samp_tok)
        if pos in cln:
            cln_tot += 1; cln_hit += int(cln[pos] == samp_tok)
            if len(examples) < 8:
                examples.append({"step": step, "off": off, "sampler": samp_tok,
                                 "current": cur.get(pos), "clean": cln.get(pos)})
    cur_a = cur_hit / cur_tot if cur_tot else None
    cln_a = cln_hit / cln_tot if cln_tot else None
    verdict = {
        "sample_idx": sample.get("sample_idx"), "longest_ek1_run": run_len, "min_run": min_run,
        "draft_rows_compared": cur_tot,
        "current_top1_agreement": cur_a, "clean_top1_agreement": cln_a,
        "winner": (None if cur_a is None or cln_a is None else
                   "clean" if cln_a > cur_a else "current" if cur_a > cln_a else "tie"),
        "both_low_escalate": (cur_a is not None and cln_a is not None and cur_a < 0.5 and cln_a < 0.5),
        "examples": examples,
    }
    rank = int(getattr(runner, "rank", 0) or 0)
    if rank == 0:
        pathlib.Path(out_path).write_text(json.dumps(verdict, indent=2))
        print(f"[CLEAN-REGION-PROOF] verdict -> {out_path}\n{json.dumps(verdict, indent=2)}", flush=True)
    return verdict
