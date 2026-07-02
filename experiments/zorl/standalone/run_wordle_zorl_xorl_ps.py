"""Wordle ZORL ES driver against the xorl-trainer PARAMETER SERVER (Muon fold).

The PS-as-xorl-trainer pivot (2026-06-30, R1 PASS): the parameter server is a
`python -m xorl.server.launcher` process that holds the fp32 LoRA-B parent
master and folds the reward-weighted ES update G=Σ c_i ΔW_i through the SERVER
Muon optimizer (bit-for-bit == the sglang fold in fp32; the fp32 master is what
lets the small updates land). This driver:

  1. create_model on the PS  — LoRA parent (rank 16, B=0 cold start) + Muon ES
     optimizer (momentum0, match_rms_adamw, gram-NS no-restart, full_gradient) +
     ZORL session (b_only parent-perturb probe).
  2. per step (default --candidate-transport seeds): PS
     /api/v1/zorl/start_generation with materialization=specs returns explicit
     per-candidate SEED SPECS {b_seed, a_seed, direction, b_sigma,
     perturbation_mode, rank} plus ONE exported parent checkpoint -> the driver
     loads the parent + POSTs the spec list ONCE per replica
     (/register_zorl_candidates; replicas materialize candidates from the seeds
     via the sglang virtual-candidate machinery — zero candidate weight bytes)
     -> score with the REUSED Wordle scoring (zorl_client.score_candidates) ->
     PS /api/v1/zorl/apply_rewards folds G via Muon -> one
     /abort_zorl_generation per replica unloads everything.
     Fallback --candidate-transport path: PS exports every candidate adapter
     (filesystem) and preloads each onto every replica (the 585s/step path).

It REUSES the standalone harness (zorl_client) for the arg surface + Wordle task
+ candidate scoring (so the Wordle reward/rollout logic is identical to the
sglang path), and xorl_ps_client for the PS endpoints. The existing sglang-PS
path in zorl_client is untouched.

Example:
  python run_wordle_zorl_xorl_ps.py --task wordle --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --xorl-ps-url http://127.0.0.1:26040 --infer-url "http://r0:30000 http://r1:30000" \
    --steps 50 --num-pairs 8 --b-sigma 1.5e-4 --muon-lr 2.5e-5 --lora-rank 16 \
    --train-size 16 --eval-size 64 --update-strategy raw --candidate-routing round_robin \
    --train-model-id wordle-zorl-ps
"""

import copy
import re
import importlib.util
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


_HERE = Path(__file__).resolve().parent


def _load(mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, str(_HERE / f"{mod_name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


zc = _load("zorl_client")
ps = _load("xorl_ps_client")


def build_parser():
    parser = zc.build_arg_parser()
    # These are required only by the sglang-native flow (parent adapter on the
    # serving replicas). The xorl PS creates the LoRA parent itself via
    # create_model, so relax them to optional defaults for this driver.
    _unused = {"adapter_dir": "/tmp/zorl-xorl-ps-unused", "parent_lora_name": "unused", "session_id": "unused"}
    for action in parser._actions:
        if action.dest in _unused:
            action.required = False
            action.default = _unused[action.dest]
    g = parser.add_argument_group("xorl-PS backend")
    g.add_argument("--xorl-ps-url", required=True, help="xorl trainer PS API base (e.g. http://127.0.0.1:26040)")
    g.add_argument("--train-model-id", default="wordle-zorl-ps", help="model_id created on the PS")
    # The PS creates the LoRA parent (rank from here); the sglang flow instead
    # carries rank in the init adapter, so build_arg_parser has no --lora-rank.
    g.add_argument("--lora-rank", type=int, default=4, help="LoRA rank of the ES parent created on the PS")
    g.add_argument("--lora-alpha", type=int, default=4, help="LoRA alpha of the ES parent created on the PS")
    g.add_argument("--lora-target-modules", nargs="+", default=["gate_proj", "up_proj", "down_proj"],
                   help="Explicit LoRA target set for the ES parent. MUST stay within what the "
                        "sglang scorers actually apply (leaf-name matching never attaches GDN "
                        "linear_attn.* — unserved noise gets Muon-folded into the base). Default "
                        "is MoE-experts-only, matching the GRPO reference recipe (xorl-infra 1143917).")
    g.add_argument("--muon-lr", type=float, default=2.5e-5, help="Muon LR (match_rms_adamw scale; sglang recipe)")
    g.add_argument("--lr-warmup-steps", type=int, default=0,
                   help="linear LR warmup steps before the schedule (GRPO recipe: 8)")
    g.add_argument("--sync-quantization", default="fp8",
                   help="fresh_ab post-apply sync quantization ('fp8' for the FP8 scorer fleet; '' = server default)")
    g.add_argument("--muon-momentum", type=float, default=0.0, help="Muon momentum (live recipe is 0)")
    g.add_argument("--muon-distributed-mode", default="full_gradient", choices=["shard_local", "full_gradient"])
    g.add_argument("--muon-gram-ns-num-restarts", type=int, default=0, help="0 matches the sglang fold exactly")
    g.add_argument("--candidate-load-timeout", type=float, default=180.0)
    g.add_argument("--candidate-transport", default="seeds", choices=["seeds", "path"],
                   help="How candidates reach the scorer replicas. 'seeds' (default): the PS returns "
                        "per-candidate seed SPECS (materialization=specs, no disk export) and the driver "
                        "POSTs the spec list once per replica (/register_zorl_candidates); replicas "
                        "materialize candidates from the seeds (virtual candidates, zero weight bytes). "
                        "'path' (fallback): the PS exports every candidate as an on-disk adapter and "
                        "preloads it onto every replica (N_candidates x N_replicas loads).")
    g.add_argument("--eval-interval", type=int, default=0,
                   help="Every N steps, score the current candidates on the held-out set; the "
                        "candidate MEAN is a sigma^2-accurate parent proxy (antithetic pairs cancel "
                        "the linear term) -> the honest generalization curve. 0 = off.")
    g.add_argument("--eval-rollouts", type=int, default=1,
                   help="rollouts/puzzle for the periodic held-out eval (cheap: 1)")
    g.add_argument("--smg-url", default="",
                   help="SMG router endpoint for /generate (compiled router, non-bottlenecking). "
                        "When set, scoring routes through SMG (use --candidate-routing owner_via_smg so "
                        "each candidate is pinned to its owner worker via X-SMG-Target-Worker). LoRA "
                        "load/unload still go DIRECT to worker pods (SMG doesn't proxy them).")
    return parser


def main():
    args = build_parser().parse_args()
    if args.task != "wordle":
        print(f"[warn] this driver is tuned for --task wordle (got {args.task!r}); proceeding anyway", flush=True)

    infer_urls = zc._url_list(args.infer_url)
    if not infer_urls:
        raise RuntimeError("--infer-url (sglang serving replicas) is required for scoring")
    ps_url = args.xorl_ps_url.rstrip("/")
    model_id = args.train_model_id
    # score_candidates / the wordle rollout read args.infer_url etc.; mirror main()'s setup.
    args.infer_url = infer_urls
    args.control_infer_urls = infer_urls
    # /generate (scoring) routes through SMG when --smg-url is set (compiled router,
    # won't bottleneck like the Python round-robin); LoRA load/unload stay DIRECT to
    # the worker pods in infer_urls (SMG does not proxy /load_lora_adapter).
    score_urls = zc._url_list(args.smg_url) or infer_urls
    reward_infer_urls = zc._url_list(args.reward_infer_url) or infer_urls
    args.reward_infer_urls = reward_infer_urls
    reward_infer_url = reward_infer_urls[0] if len(reward_infer_urls) == 1 else reward_infer_urls

    print(f"[init] xorl-PS={ps_url} model_id={model_id} replicas={infer_urls}", flush=True)
    print(
        f"[init] muon: lr={args.muon_lr} momentum={args.muon_momentum} "
        f"distributed_mode={args.muon_distributed_mode} gram_ns_restarts={args.muon_gram_ns_num_restarts} | "
        f"zorl: pairs={args.num_pairs} b_sigma={args.b_sigma} perturbation_mode={args.perturbation_mode} "
        f"lora_rank={args.lora_rank} update_strategy={args.update_strategy} "
        f"candidate_transport={args.candidate_transport}",
        flush=True,
    )

    task = zc.load_task(args.task)
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_pool, eval_examples = task.build_examples(
        tokenizer, train_size=args.train_pool_size or args.train_size, eval_size=args.eval_size, seed=args.seed
    )
    print(
        f"[init] train_pool={len(train_pool)} eval={len(eval_examples)} "
        f"multi_turn={getattr(task, 'is_multi_turn', False)}",
        flush=True,
    )

    # 1. create the ZORL LoRA-parent model + Muon ES optimizer on the PS.
    print("[step 1] create_model (LoRA parent + Muon ES optimizer + ZORL session) on the PS...", flush=True)
    cm = ps.create_zorl_model(
        ps_url,
        model_id=model_id,
        base_model=args.model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        muon_lr=args.muon_lr,
        b_sigma=args.b_sigma,
        num_perturbation_pairs=args.num_pairs,
        seed=args.seed,
        perturbation_mode=args.perturbation_mode,
        muon_momentum=args.muon_momentum,
        muon_gram_ns_num_restarts=args.muon_gram_ns_num_restarts,
        muon_distributed_mode=args.muon_distributed_mode,
    )
    print(f"  create_model -> {cm.get('model_id', model_id)} (ok)", flush=True)

    # 1b. Register the sglang scorers so the PS can preload candidate adapters onto
    #     them (sampling-only: sync_weights=False -> no NCCL weight-sync handshake).
    from urllib.parse import urlparse  # noqa: PLC0415

    print(f"[step 1b] register {len(infer_urls)} scorers on the PS (preload targets)...", flush=True)
    for url in infer_urls:
        p = urlparse(url)
        # fresh_ab needs the replicas as p2p (Mooncake) weight-sync receivers so the
        # post-apply base push lands; b_only stays sampling-only (no NCCL handshake).
        # Registration failure in fresh_ab mode is FATAL: without receivers the base
        # sync silently no-ops and every fold after step 1 probes a stale base.
        _sync_recv = args.perturbation_mode == "fresh_ab"
        reg = None
        for attempt in range(5):
            reg = ps.register_inference_endpoint(ps_url, host=p.hostname, port=int(p.port or 30000), sync_weights=_sync_recv)
            if reg.get("success", True) or "already registered" in str(reg.get("message", "")):
                break
            print(f"  register {p.hostname} attempt {attempt+1}/5 failed: {reg.get('message')}; retrying in 30s", flush=True)
            time.sleep(30.0)
        print(f"  registered {p.hostname}:{p.port} -> {reg.get('message', 'ok')}", flush=True)
        if _sync_recv and not (reg.get("success", True) or "already registered" in str(reg.get("message", ""))):
            raise RuntimeError(
                f"fresh_ab requires all replicas registered as sync receivers; {p.hostname} failed after 5 attempts"
            )

    # 2. cold base+think gate: probe the FROZEN BASE on the seed-777 held-out set.
    #    GRPO's honest base+think = 0.00 (retries=0); we expect the same. The
    #    client now omits lora_path when None, so this genuinely measures the
    #    base (previously str(None)/null 400'd per example and reported 0.00
    #    as "all errors").
    #    The TRAINED-parent held-out eval is still approximated by the candidate
    #    mean (a BIASED O(sigma^2) proxy — see faithfulness doc §8.1); serving a
    #    zero-perturbation parent-B adapter for a true parent eval is still TODO.
    print(f"[step 2] cold base+think probe on {len(eval_examples)} eval examples (lora_path=None)...", flush=True)
    try:
        cold = zc.probe_parent(
            reward_infer_url, parent_lora_name=None, examples=eval_examples, task=task, args=args, tokenizer=tokenizer
        )
        print(f"  cold base: reward_mean={cold.get('reward_mean', 0.0):.4f} "
              f"exact_rate={cold.get('exact_match_mean', 0.0):.4f}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  cold probe skipped ({e})", flush=True)

    def lr_for_step(step: int) -> float:
        warmup = int(getattr(args, "lr_warmup_steps", 0) or 0)
        if warmup > 0 and step < warmup:
            return float(args.muon_lr) * (step + 1) / warmup
        if args.lr_schedule == "constant":
            return float(args.muon_lr)
        horizon = int(args.lr_decay_steps) if args.lr_decay_steps > 0 else int(args.steps)
        frac = min(1.0, step / max(1, horizon))
        floor = float(args.muon_lr) * float(args.lr_min_frac)
        return floor + 0.5 * (float(args.muon_lr) - floor) * (1.0 + math.cos(math.pi * frac))

    # 3. main ES loop.
    t0 = time.time()
    for step in range(args.steps):
        if args.max_runtime_seconds and (time.time() - t0) >= args.max_runtime_seconds:
            print(f"[max-runtime] {args.max_runtime_seconds}s reached after {step} steps; stopping", flush=True)
            break

        # Candidate distribution.
        #   seeds (default): PS plans the generation and returns SPECS only
        #     (materialization=specs; one parent checkpoint export, no candidate
        #     exports). The driver POSTs the spec list once per replica and the
        #     replicas materialize candidates from the seeds (virtual candidates).
        #   path (fallback): preload_sampling=True — the PS exports candidate
        #     adapters and loads them on the registered scorers itself
        #     (create_sampling_session), serving each under its API lora_name.
        seeds_transport = args.candidate_transport == "seeds"
        load_t0 = time.time()
        # Robust generation: a transient scorer hiccup (e.g. a pod restart) makes the
        # PS preload throw -> start_generation returns no generation_id. Retry a few
        # times, then skip the step rather than crashing the whole Job.
        gen = None
        for attempt in range(5):
            try:
                cand = ps.start_generation(
                    ps_url, model_id=model_id, num_pairs=args.num_pairs,
                    preload_sampling=not seeds_transport,
                    materialization={"mode": "specs"} if seeds_transport else None,
                )
                if cand.get("generation_id") and cand.get("candidates"):
                    gen = cand
                    break
                print(f"  WARN step {step+1}: start_generation returned no generation_id/candidates "
                      f"(attempt {attempt+1}/5): {str(cand)[:200]}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  WARN step {step+1}: start_generation failed (attempt {attempt+1}/5): {e}", flush=True)
            time.sleep(20)
        if gen is None:
            print(f"  step {step+1}: SKIP — could not start a generation after retries", flush=True)
            continue
        gen_id = gen["generation_id"]
        candidates = gen.get("candidates", [])
        # Each candidate gets an owner_url. We SCORE via SMG pinned to that owner
        # (owner_via_smg), so a candidate's requests co-locate on one worker ->
        # RadixCache reuses the shared prompt prefix.
        for i, c in enumerate(candidates):
            c["lora_name"] = c.get("lora_name") or c["candidate_id"]
            c["owner_url"] = c.get("owner_url") or infer_urls[i % len(infer_urls)]

        parent_lora_name = None
        if seeds_transport:
            # 1) Load the PARENT once per replica (32 loads, not 32xN_cands):
            #    b_only candidates perturb around its LoRA-B (the PS folds into
            #    its own copy each apply, so it is re-exported fresh every
            #    generation); fresh_ab only needs its shapes/scaling.
            # 2) POST the explicit specs once per replica; the replicas
            #    materialize candidates from the seeds (zero weight bytes).
            parent_path = gen.get("parent_path")
            if not parent_path:
                raise RuntimeError(
                    "seeds transport requires the PS to return parent_path "
                    "(xorl PS too old for materialization=specs?)"
                )
            parent_lora_name = f"zorl-parent/{gen_id}"
            specs = [
                {
                    "candidate_id": str(c["candidate_id"]),
                    "lora_name": str(c["lora_name"]),
                    "perturbation_index": int(c["perturbation_index"]),
                    "direction": str(c["direction"]),
                    "b_seed": int(c["b_seed"]),
                    "a_seed": None if c.get("a_seed") is None else int(c["a_seed"]),
                    "b_sigma": float(c.get("b_sigma") or gen["b_sigma"]),
                    "perturbation_mode": str(
                        c.get("perturbation_mode") or gen.get("perturbation_mode") or args.perturbation_mode
                    ),
                    "rank": int(c.get("rank") or gen.get("lora_rank") or args.lora_rank),
                }
                for c in candidates
            ]

            def _register_on(url: str) -> None:
                # Idempotent vs crash residue and partial-failure retries: generation
                # names are deterministic (family/g indices), so a dead run's leftovers
                # collide with ours. Parent: unload-then-reload on conflict (content may
                # differ across runs). Candidates: abort the stale active generation
                # named in the 400 and retry once.
                try:
                    zc.load_lora_adapter(url, parent_lora_name, parent_path)
                except Exception as le:  # noqa: BLE001
                    if "already loaded" not in str(le):
                        raise
                    try:
                        zc.unload_lora_adapter(url, parent_lora_name)
                    except Exception:  # noqa: BLE001
                        pass
                    zc.load_lora_adapter(url, parent_lora_name, parent_path)
                for attempt in range(2):
                    try:
                        ps.register_zorl_candidates(
                            url,
                            session_id=model_id,
                            generation_id=gen_id,
                            parent_lora_name=parent_lora_name,
                            candidates=specs,
                            timeout=args.candidate_load_timeout,
                        )
                        return
                    except Exception as re_:  # noqa: BLE001
                        m = re.search(r"already has active generation '([^']+)'", str(re_))
                        if attempt == 0 and m:
                            try:
                                ps.abort_replica_generation(url, session_id=model_id, generation_id=m.group(1))
                            except Exception:  # noqa: BLE001
                                pass
                            continue
                        raise

            try:
                with ThreadPoolExecutor(max_workers=min(32, len(infer_urls))) as pool:
                    list(pool.map(_register_on, infer_urls))
            except Exception as e:  # noqa: BLE001
                print(f"  step {step+1}: SKIP — seed-spec registration failed: {e}", flush=True)
                # Best-effort teardown so the next step starts clean everywhere.
                with ThreadPoolExecutor(max_workers=min(32, len(infer_urls))) as pool:
                    def _teardown(url: str) -> None:
                        try:
                            ps.abort_replica_generation(url, session_id=model_id, generation_id=gen_id)
                        except Exception:  # noqa: BLE001
                            pass
                        zc.unload_lora_adapter(url, parent_lora_name)
                    list(pool.map(_teardown, infer_urls))
                try:
                    ps.abort_generation(ps_url, model_id=model_id, generation_id=gen_id)
                except Exception as abort_error:  # noqa: BLE001
                    print(f"  WARN step {step+1}: PS abort failed: {abort_error}", flush=True)
                continue
        load_s = time.time() - load_t0

        train_examples = zc._select_train_examples_for_step(
            train_pool, train_size=args.train_size, seed=args.seed, step=step,
            resample=args.resample_train_each_step,
        )
        score_t0 = time.time()
        candidate_rewards, _ = zc.score_candidates(
            score_urls, candidates=candidates, examples=train_examples, task=task, args=args, tokenizer=tokenizer
        )
        score_s = time.time() - score_t0

        rewards_for_update, _ = zc.transform_rewards_for_update(candidate_rewards, strategy=args.update_strategy)
        apply_t0 = time.time()
        res = ps.apply_rewards(
            ps_url, model_id=model_id, generation_id=gen_id,
            candidate_rewards=rewards_for_update, learning_rate=lr_for_step(step),
            sync_after_apply=(args.perturbation_mode == "fresh_ab"),
            sync_quantization=(args.sync_quantization or None),
        )
        apply_s = time.time() - apply_t0

        # Periodic HONEST held-out eval (the generalization gate vs GRPO's 0.67-0.68).
        # Reuse this generation's already-loaded candidate adapters to score the reserved
        # seed-777 held-out set. The candidate MEAN is a sigma^2-accurate estimate of the
        # parent's held-out reward (antithetic +/- pairs cancel the O(sigma) term), so we
        # get the honest parent trajectory without separately serving the parent.
        held_line = ""
        if args.eval_interval and (step % args.eval_interval == 0) and eval_examples:
            eval_args = copy.copy(args)
            eval_args.rollouts_per_puzzle = int(args.eval_rollouts)
            try:
                held_rewards, _ = zc.score_candidates(
                    score_urls, candidates=candidates, examples=eval_examples,
                    task=task, args=eval_args, tokenizer=tokenizer,
                )

                # SOLVE RATE (exact_match = target among guesses) is the GRPO-comparable
                # metric (GRPO reports held-out solve rate ~0.67-0.68). Aggregate it from
                # each candidate's per-puzzle project_metrics; the candidate MEAN is the
                # parent-proxy. Composite reward is kept as a secondary signal.
                def _cand_solve(c):
                    pms = c.get("project_metrics", {}) or {}
                    ev = [m.get("exact_match") for m in pms.values()
                          if isinstance(m, dict) and isinstance(m.get("exact_match"), (int, float))]
                    return sum(ev) / len(ev) if ev else None

                solves = [s for s in (_cand_solve(c) for c in held_rewards) if s is not None]
                hmeans = [float(c["reward_mean"]) for c in held_rewards if c.get("reward_mean") is not None]
                comp = sum(hmeans) / len(hmeans) if hmeans else 0.0
                if solves:
                    held_line = (
                        f"  [held-out@{step+1}] parent_solve_rate={sum(solves) / len(solves):.4f} "
                        f"best_cand_solve={max(solves):.4f} composite_mean={comp:.4f} "
                        f"n_held={len(eval_examples)} retries={args.invalid_retries} "
                        f"eval_rollouts={eval_args.rollouts_per_puzzle}"
                    )
                elif hmeans:
                    held_line = (
                        f"  [held-out@{step+1}] composite_mean={comp:.4f} (no exact_match metric) "
                        f"n_held={len(eval_examples)} retries={args.invalid_retries}"
                    )
            except Exception as e:  # noqa: BLE001
                held_line = f"  [held-out@{step+1}] skipped ({e})"

        # Cleanup (parallel, best-effort; gen-unique names so no reload collision).
        if seeds_transport:
            # One abort per replica unloads ALL of the generation's virtual
            # candidates; then drop this generation's parent copy.
            def _cleanup(url: str) -> None:
                try:
                    ps.abort_replica_generation(url, session_id=model_id, generation_id=gen_id)
                except Exception as e:  # noqa: BLE001
                    print(f"  WARN cleanup: abort on {url} failed: {e}", flush=True)
                zc.unload_lora_adapter(url, parent_lora_name)

            with ThreadPoolExecutor(max_workers=min(32, len(infer_urls))) as pool:
                list(pool.map(_cleanup, infer_urls))
        else:
            unload_jobs = [(url, str(c["lora_name"])) for c in candidates for url in infer_urls]
            with ThreadPoolExecutor(max_workers=min(64, len(unload_jobs) or 1)) as pool:
                list(pool.map(lambda j: zc.unload_lora_adapter(j[0], j[1]), unload_jobs))

        means = [float(c["reward_mean"]) for c in candidate_rewards if c.get("reward_mean") is not None]
        best = max(means) if means else 0.0
        mean = sum(means) / max(len(means), 1)
        metrics = (res or {}).get("metrics", {}) or {}
        print(
            f"[step {step+1}/{args.steps}] cands={len(candidates)} best={best:.4f} mean={mean:.4f} "
            f"update_norm={(res or {}).get('update_norm', metrics.get('update_norm', '?'))} "
            f"used_pairs={(res or {}).get('used_pairs', '?')} dropped={(res or {}).get('dropped_pairs', '?')} "
            f"| load={load_s:.1f}s score={score_s:.1f}s apply={apply_s:.1f}s",
            flush=True,
        )
        if held_line:
            print(held_line, flush=True)

    print(f"[done] {args.steps} steps in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
