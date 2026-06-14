# Handoff: OPD-MTP commit_len — LR experiment + sampler-scale gate (2026-06-13)

> ## ✅ VERDICT (2026-06-13 03:21): H3 — ARCHITECTURAL CEILING. The LR is NOT the lever.
> A 100× muon_lr sweep (1e-5 → 1e-4 → 1e-3) leaves offset-1 draft acceptance **dead-flat at ~0.03** and
> confidence pinned at **~0.29–0.30**. At 1e-3 (100× the original), 65 steps: conf settled 0.293, accept
> 0.033 (the brief 0.045 at step 111 was a resume-jolt transient, not a climb). Tell-tale: ~13% of drafts
> are confident (>0.5) but only ~3% accepted ⇒ the draft predicts *plausible-but-wrong* tokens confidently
> (it predicts t+1 from mask slots with context ending t−1; the AR verify with full context disagrees).
> **⇒ commit_len cannot be lifted by LR / data / more samplers / loss-mode on this architecture.** The fix
> is **EAGLE-style autoregressive drafting** (sampler/model change — SGL/model agent's domain). DO NOT
> scale samplers (the gate failed). The replay/emit fix (b0cfc6de) is still correct; just not sufficient.
> Caveat: runs were ≤165 steps; the flatness ACROSS 100× LR (acceptance not creeping even at 1e-3) is the
> evidence — if it were step-limited, a 100× rate would show *some* slope. It does not.

**Mission:** lift OPD-MTP draft acceptance (`commit_len`) off the ~1.0 floor on the Qwen3.6-35B-A3B
SingleShot-MTP on-policy-distillation run. This is the whole point of MTP-OPD: distill speculation so
decode is cheap. Loss descends but `commit_len` has been pinned ~1.03 the whole project.

**Stack:** `er-opd-q36-mtp-ss-0605c` — 4 trainer nodes / 32 GPUs (FSDP2 dp_shard, EP=32 deepep sms48,
triton MoE, bf16, flex_attention), 2 SGLang student samplers (2 GPU each), 1 teacher + teacher-smg.
Bare pods run a controller daemon polling `desired.sha256` in
`/shared/opd-control/er-opd-q36-mtp-ss-0605c/<slot>/`; a control-hash change re-execs `run.sh`.
WandB project `singleshot`. Client = `scripts/opd/run_opd_pipeline.py`; trainer = `xorl.server.launcher`.

---

## The scientific state — READ THIS FIRST

The question is **under-trained draft vs architectural ceiling (H3)**, and an experiment is LIVE to
settle it. History:

1. A real **replay-visibility + emit-window fix** (branch `fix/mtp-replay-visibility-emit-supervision`
   @ `b0cfc6de`, worktree `/home/apanda/xorl-mtp-commitlen-fix-20260612`) was A/B-tested at scale (50
   steps). It fixed two genuine bugs but **did NOT lift commit_len** (flat ~1.03, top1 ~0.26).
2. My first read ("switch the loss mode") was **overturned by the user's per-offset analysis**
   (`scripts/opd/analyze_mtp_draft_acceptance.py`). Decomposing the sampler trace by draft offset:
   - verify (committed) positions: conf median **0.76** — sharp, trained, fine.
   - **offset-1 draft** (gates commit_len 1→2): conf median **0.30**, acceptance **~3.4%** — diffuse, the gate.
   - The active loss is **already `hard_teacher_ce`** (one-hot CE on the teacher argmax = a sharpening
     objective); teacher argmax == verify token **~88.7%** (target is correct). So "add reverse-KL /
     hard-target CE / entropy penalty" is CIRCULAR/contraindicated. The lever is **draft LR + data + steps**,
     judged by **offset-1 acceptance/confidence**, NOT aggregate top1/ent_stud (which mislead).
3. **Optimizer is muon.** The MTP draft head is a 2D matrix → governed by **`muon_lr`**, not `lr`.
   Original config: `lr=1e-6`, `muon_lr=1e-5`. **Calibration: `muon_lr=1e-4` is the STANDARD** across
   ~94 xorl configs; the OPD config's `1e-5` was the anomaly (2 configs). 29 configs run `0.02`.

### Experiments run (judge by offset-1, via `analyze_mtp_draft_acceptance.py <rollout_samples.jsonl>`)

| LR (muon_lr) | run | offset-1 conf | offset-1 accept | read |
|---|---|---|---|---|
| 1e-5 (orig) | baseline 095624Z / fix 204932Z | ~0.30 flat | ~0.034 flat | stuck |
| 1e-4 (10×, =standard) | 235803Z (cold) | plateaued ~0.29 | dead-flat ~0.027 | leaning H3 |
| **1e-3 (10× above std)** | **014150Z (resume@100) — LIVE** | 0.116→0.284 (recovering jolt) | 0.000→**0.045** @ step111 | **early, slightly +** |

The 1e-3 run (resume@100) is the **decisive test**, IN PROGRESS. Post-resume jolt dropped conf to
0.116 at step 102; by step 111 conf 0.284 / **accept 0.045** (first time above the stuck 0.027). NOT
yet conclusive — confounded by the resume jolt, only ~10 steps in. **Watch to ~step 150–160.**

### THE DECISION GATE (what to do with the verdict)

- **offset-1 acceptance climbs clearly past ~0.05–0.07 and keeps rising / conf breaks 0.40** ⇒ the draft
  IS learnable, was under-trained ⇒ **scale samplers + data (the user's ask) and run hundreds–thousands
  of steps** at the working LR. See "Sampler scale-up" below.
- **offset-1 stays flat (~0.027 accept / ~0.30 conf) even at 1e-3** ⇒ **architectural ceiling (H3):**
  single-shot/parallel MTP drafts from mask slots and predicts t+1 from context ending t−1, so it may
  never match the AR verify. The fix is then **EAGLE-style autoregressive drafting** (a sampler/model
  change — hand to the SGL/model agent), NOT a loss term and NOT more data. Report this; don't burn
  GPUs on samplers.

If 1e-3 is ambiguous after ~60 steps, the next escalation is `muon_lr=0.005` or `0.02` (29 configs run
0.02; muon tolerates it). Watch grad_norm/loss for blowup (revert if NaN).

---

## Live state (2026-06-13 ~02:09)

- **Run:** `q36mtp-20260613T014150Z-2s1t`, resume@100, **`muon_lr=1e-3`** (`lr=1e-5`), prompts/step=64
  (2× the original 32), 2 samplers. Step ~114, healthy. grad_norm ~270 (no blowup).
- **Supervisor:** armed, PID in `$CTL/supervisor.pid`, env carries `OPD_XORL_REPO=fix` +
  `OPD_FULL_FT_LR=1e-5` + `OPD_MUON_LR=1e-3` (so auto-recovery reuses the experiment LR). 1 recovery
  total tonight (a P2P crash, below). **Resume-robust fix applied** (see below).
- **Offset-1 watcher:** background task `b1w1ph9r0` runs `analyze_mtp_draft_acceptance.py` every 4 min;
  exits on conf>0.40, step≥165, crash, or 4h timeout (~03:52). **Re-arm it if it times out before the
  verdict** (same loop; threshold conf>0.40 OR watch acceptance climbing past ~0.06).

---

## Infra changes made tonight (all in the DEFAULT worktree `/home/apanda/xorl-mtp-singleshot-port-20260602`)

1. **Generator `q36_singleshot_reprogrammable_slots.py`:**
   - Added **`OPD_MUON_LR` knob**: `cfg['muon_lr']=float(os.environ.get('OPD_MUON_LR', str(cfg.get('muon_lr',1e-5))))`
     in the config-override block (~line 1005).
   - **Fixed the LR-bake bug**: `OPD_FULL_FT_LR`/`OPD_MUON_LR` were emitted as pod-runtime
     `${VAR:-default}` → always resolved to the default on the pod (no such env) → `lr` was ALWAYS 1e-6
     and NEVER env-settable. Now BAKED as render-time literals via `shell_value(full_ft_lr_val/muon_lr_val)`
     (defined ~line 1018, like the resume vars). So `OPD_MUON_LR=1e-3 ... write-trainer-control` injects 1e-3.
     Verified end-to-end (generated_trainer_config.yaml shows muon_lr:0.001).
2. **Supervisor `opd_trainer_supervisor.sh` — resume-robust fix:** `latest_checkpoint()` now walks run
   dirs **newest-first** (`sorted(..., reverse=True)`) and returns the first with a valid on-disk DCP
   (`break` on found), instead of scanning only the newest dir. Fixes the cold-start-on-recovery bug
   (the supervisor cold-started a resume@100 run tonight because the crashed run had no ckpt of its own).
   Verified it returns step-100, not empty.
3. **`launch_args_...txt`:** `--prompts-per-step` 32 → **64**.

### The venv symlink trick (CRITICAL for OPD_XORL_REPO=fix)
The fix worktree's own `uv sync` venv was DIVERGENT (308 vs 408 pkgs; safetensors/datasets/hf_hub
bumped) AND missing the separate DeepEP wheel → "DeepEP not installed", then nvshmem ABI crash. **Fix:**
`fix/.venv` is a **symlink to the default venv** (`ln -s /home/apanda/xorl-mtp-singleshot-port-20260602/.venv
/home/apanda/xorl-mtp-commitlen-fix-20260612/.venv`). `run.sh` imports xorl via `PYTHONPATH=$XORL_REPO/src`
which WINS over the venv's editable install, so the fix's `src` (incl. the replay/emit fix) runs on the
PROVEN default env. Do NOT `uv sync` a fresh venv for an `OPD_XORL_REPO=<worktree>` A/B — symlink instead.

---

## How to operate (commands)

```
cd /home/apanda/xorl-mtp-singleshot-port-20260602
PY=.venv/bin/python; G=experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py
A="$(cat experiments/opd_profile/k8s/launch_args_er-opd-q36-mtp-ss-0605c.txt)"
CTL=/shared/opd-control/er-opd-q36-mtp-ss-0605c
RB=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c

# Per-offset analysis (THE metric — not aggregate top1):
$PY scripts/opd/analyze_mtp_draft_acceptance.py $(ls -t $RB/q36mtp-*/artifacts/rollout_samples.jsonl|head -1) --max-offset 3

# Relaunch at a new LR (resume@100, supervisor PAUSED first to avoid interference):
touch $CTL/supervisor.pause
$PY $G stop-trainer-control $A
export OPD_XORL_REPO=/home/apanda/xorl-mtp-commitlen-fix-20260612 OPD_FULL_FT_LR=0.00001 OPD_MUON_LR=<LR> \
       OPD_START_STEP=100 OPD_LOAD_CHECKPOINT_PATH=$RB/q36mtp-20260612T204932Z-2s1t/server_output/weights/default/q36mtp-coderforge-v1-step000100
nohup bash experiments/opd_profile/k8s/sync_launch_0605c.sh > /tmp/relaunch.out 2>&1 < /dev/null &
# then re-arm supervisor with the SAME LR envs (kill old PID, restart via `bash`, rm pause)
```

### GOTCHAS (learned the hard way tonight)
- **`pkill -f sync_launch`/`-f opd_trainer_supervisor` SELF-MATCHES your shell → exit 144.** Kill by PID.
- The sync-launch's **"STEP 0 fb PASSED loss=6.x" is a STALE-LOG FALSE MATCH** on resume@100 runs (they
  start at step 100, not 0). Verify the run via the generated config muon_lr + the head log reaching
  `=== OPD step 100`, not the sync-launch's exit message.
- **P2P weight-sync `ret=-1`** (stale Mooncake on day-old samplers after many trainer restarts) crashed a
  run tonight. The supervisor's `recover()` recreates the samplers (clears Mooncake) — that fixed it. If
  it recurs, recreate samplers (`write-student-inference-control`).
- Supervisor must run via **`bash <script>`** (lost its exec bit) and be started with the LR envs in its
  environment or recovery reverts the LR.
- Watch HEAD + ALL 3 worker logs + server.log (a head-only watch misses worker-rank crashes).

---

## Sampler scale-up (GATED on the 1e-3 verdict; the user's explicit ask)
If the draft is confirmed learnable, scale samplers to feed more data + more steps/hour (Plan Phase 2):
render the 4-replica manifest (`render-manifest ... --student-replicas 4`), extract ONLY the sglang-2/3
Service+Pod docs, `kubectl apply` (additive), wait Running+loaded, then `stop-trainer-control` /
`write-student-inference-control` / `write-trainer-control` at `--student-replicas 4`. p2p sync to N
receivers is config-only (untested >2). Then bump `--prompts-per-step` (128 was the prior OOM point at
packing 8193; safe at packing 4608, but ramp + watch). Promote the args file to match so recovery agrees.

---

## Invariants — do NOT re-break (CLAUDE.md + tonight)
expandable_segments OFF; no `NCCL_IB_GID_INDEX/HCA` on the trainer (inits Mooncake); non-privileged RDMA;
`team: turbo`; triton MoE (quack NaNs for training on this branch); the distributed-init timeouts; DeepEP
is a separate wheel (use the venv symlink). No Co-Authored-By/Claude trailers in commits.

Full ledger: memory `mtp-commit-len-floor-root-cause` (`project_mtp_commit_len_floor_root_cause.md`),
`docs/notes/mtp_commit_len_floor_debug_handoff.md` (the REOPENED + CORRECTION notes),
`project_mtp_opd_infra_pilot_state`.
