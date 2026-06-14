# Autoresearch Loop Runbook

> **➡️ SUPERSEDED — see `CANONICAL_RUNBOOK.md`** (same directory). That doc
> consolidates this operational guide + the full experiment ledger (PTC-020→121)
> + the settled conclusion (no filler variant beats no-filler on this stack) +
> the config rules and failure-recovery recipes learned 2026-06-07. Use it as the
> canonical reference; this file is kept for the operational detail it links to.

Canonical handoff target: `experiments/opd_profile/autoresearch/CANONICAL_RUNBOOK.md`.

Use this when starting a fresh agent with: `/loop read @runbook and act on it`.
It is written for the OPD prefill-time-compute autoresearch loop on the
`er-opd-q36-35b-slots` stack.

Last refreshed: `2026-06-05T08:10:00Z`.

## 0a. STANDING OVERNIGHT DIRECTIVE (2026-06-05, user asleep)

The user gave an explicit standing instruction for this overnight session:

- **Never idle the cluster.** Idling the GPU nodes is the worst possible
  outcome. There must always be a run in flight or a runnable queued idea.
- **Never ask the user for permission overnight.** Act autonomously. Do not
  use `AskUserQuestion`. Default to launching runs.
- **Keep the autoresearch loop running.** Keep an `autopilot` process alive and
  keep following this runbook on each loop tick.
- **The promoted recipe is `PTC-044`** (full AM, short filler, hidden matching,
  corrupt negatives, pipeline-RL). The current research program is to validate
  scaleups and single-knob ablations *of `PTC-044`* — not random-filler or
  no-hidden branches.

Per-tick behaviour for a fresh agent (see also §0b, §9, §12):

> **AUTOPILOT LIVENESS IS NOW OWNED BY THE SUPERVISOR (`autopilot_supervisor.py`).
> NEVER manually launch `controller.py autopilot` — doing so races the supervisor
> and creates duplicate autopilots that corrupt `ideas.yaml` (this happened
> 2026-06-06T01:51Z when a concurrent cron-tick relaunched during a crash). The
> supervisor now kills duplicate autopilots automatically, but do not create the
> race in the first place.**

1. **Ensure the SUPERVISOR is alive** (not the autopilot): a genuine *python*
   `autopilot_supervisor.py` proc must exist (check `/proc/<pid>/cmdline`, ignore
   bash wrappers). If — and ONLY if — it is dead, relaunch it (its pidfile guard
   makes a double-launch a safe no-op):
   `cd /home/apanda/xorl-apanda-dev-opd-port && setsid nohup python
   experiments/opd_profile/autoresearch/autopilot_supervisor.py
   >> experiments/opd_profile/autoresearch/logs/autopilot_supervisor.nohup 2>&1 < /dev/null &`
2. The supervisor keeps the autopilot alive and bounded-retries recent crashes.
   Do NOT touch the autopilot yourself. Just monitor + skim
   `logs/autopilot_supervisor.log` for `ESCALATE`.
3. (removed — the supervisor restarts the autopilot; agents must not.)
4. If the queue is fully drained (`controller.py next` returns nothing and no
   `launched` idea): **enqueue a fresh wave** of single-knob `PTC-044`
   ablations/scaleups (see §8a) — editing `ideas.yaml` is still the agent's job.
   The supervisor will then relaunch the autopilot to pick the new work up. Do not
   leave the cluster idle.
5. Only escalate (record + surface) a genuine deterministic break the supervisor's
   bounded retry cannot clear (e.g. an `ESCALATE` line: an idea that exhausted its
   3 infra retries → likely needs an inference-stack recreate, not a relaunch).

### Known infra failure + CRASH-LOOP GUARD (tilelang symbol-lookup crash)

At ~2026-06-05T08:13Z `PTC-052` (byte-identical to `PTC-044`) crashed at step-0:
all trainer ranks `exitcode 127` from
`libtvm_compiler.so: undefined symbol: _ZN3tvm3ffi9ReprPrintERKNS0_3AnyE`
(tilelang / TVM-FFI ABI mismatch that surfaces ONLY when tilelang JIT-*compiles*
a kernel, i.e. on a kernel-cache miss). The tilelang `.so` files are unchanged
for 2 days and `PTC-035/036/044` all ran clean on them, so the compile path is
normally served from a warm kernel cache. Most likely trigger: autopilot
launched `PTC-052` ~30s after `PTC-044` finished, racing `PTC-044`'s GPU/cache
teardown. Recovery taken: reset `PTC-052` -> `queued` with `infra_retry_count: 1`
and relaunched autopilot for ONE clean retry into the (now-idle) warm cache.

CRASH-LOOP GUARD — give any candidate that dies `infra_invalid` with the
`symbol lookup error` / `undefined symbol` / `exitcode 127` tilelang signature
AT MOST ONE clean relaunch (track via `infra_retry_count`). If that retry, or
the next candidate, crashes again with the SAME signature, treat it as a
DETERMINISTIC environment break: do NOT keep relaunching and do NOT enqueue
fresh waves (every flashqla candidate will hit it). Instead: run
`stop-trainer-control`, record it here, and surface it clearly for the user. A
broken shared venv (tilelang) is not something to auto-rebuild overnight; idle
with a loud handoff beats crash-looping 64 GPUs all night.

RESOLUTION TAKEN (2026-06-05T08:54Z): `PTC-052` crashed the SAME way on its one
retry, confirming a deterministic break. Rather than idle, the whole wave was
switched from `gdn_backend: flashqla` to `gdn_backend: fla` — the DEFAULT,
Triton-based, tilelang-free GDN backend (`get_gdn_backend()` defaults to `fla`;
its code path "never requires tilelang"). The valid backends are
`("fla", "flashqla")`. `fla` and `flashqla` compute the same gated-delta-rule
math with different kernels, so the wave still tests the PTC-044 pause-vs-no-pause
recipe — just on a different kernel and likely slower. NOTE for the user:
`flashqla`/tilelang is broken in `/home/apanda/xorl-internal/.venv`
(`libtvm_compiler.so: undefined symbol _ZN3tvm3ffi9ReprPrintERKNS0_3AnyE`;
multiple tilelang versions staged under `/tmp/tilelang-*` point to a version/ABI
mix). `PTC-044` itself ran on flashqla, so for an exact-kernel match re-run this
wave on `flashqla` once tilelang is repaired. Until then, **keep all candidates
on `gdn_backend: fla`** — do NOT use flashqla.

## 0b. DEEP-ANALYSIS BATTERY + SELF-HEALING SUPERVISOR (2026-06-06)

The user is offline for several hours and is **suspicious that the step-10
signal is a fast-emerging artifact with a simpler explanation** ("overfitting to
phenomena that emerge within ~10 steps"). Two things were set up; both must keep
running and be acted on each tick.

### Self-healing supervisor (keeps GPUs non-idle without the operator)
`autopilot_supervisor.py` runs detached (pidfile `logs/supervisor.pid`, log
`logs/autopilot_supervisor.log`). It polls every 60s and, **if no autopilot is
alive but work is queued**, re-queues any *recently* (<3h) crashed `infra_invalid`
idea (bounded `infra_retry_count<3`) and relaunches the autopilot. This closes
the gap that the autopilot **exits/`return`s on any P2P/health failure** (see
`P2P_CRASH_DIAGNOSIS.md` §1, controller.py ~L869-876) — which would otherwise
idle every GPU until a manual relaunch.

**Per-tick now:**
1. `pgrep -f autopilot_supervisor.py` must show a live **python** proc (not just
   the bash wrapper); confirm via `/proc/<pid>/cmdline`. If dead, relaunch:
   `cd /home/apanda/xorl-apanda-dev-opd-port && setsid nohup python
   experiments/opd_profile/autoresearch/autopilot_supervisor.py
   >> experiments/opd_profile/autoresearch/logs/autopilot_supervisor.nohup 2>&1 < /dev/null &`
   (the pidfile guard makes a double-launch a no-op).
2. Skim `logs/autopilot_supervisor.log` for `ESCALATE:` — that means an idea
   exhausted its 3 infra retries; a bare relaunch isn't clearing it.
   **RESOLVED 2026-06-06 04:33: the 02:34–04:00 sync degradation was a TRANSIENT
   BURST and SELF-CLEARED — no intervention needed.** PTC-082's 4th supervisor-retry
   (03:50) hit a good Mooncake handshake window and COMPLETED at 04:06 (weak_signal);
   PTC-083 completed 04:21 (strong_signal); PTC-084 running. So the supervisor's
   bounded retry rode out the bursty window. LESSON: this matches the diagnosis doc's
   "intermittent/bursty, HCA varies, recovers on relaunch" characterization — do NOT
   force the §7c sampler-restart for a burst; it's disruptive (35B reload) and the
   auto-mode classifier correctly DENIED it at 04:03. Only escalate to §7c if a future
   burst truly does NOT clear after many supervisor retries (e.g. an actual `ESCALATE`
   line — an idea exhausting all 3 retries with NO completion in between), and surface
   to the user rather than auto-run (it needs authorization).
   **DISTINGUISH ESCALATE CAUSES (07:03):** the supervisor re-queues ANY recent
   `infra_invalid` by recency, but only the SYNC-CRASH signature (`trainer-head rc=1` +
   receiver `Peer nic not found`/`504`/`batch_transfer ret=-1`) is what §7c fixes. A
   DETERMINISTIC eval-artifact reason — `cap_hit_frac_max=…` or `request_failure_frac…`
   (e.g. PTC-087 "100 random numbers" rambled to 72% cap-hit) — recurs on every retry
   and ESCALATEs too, but is NOT a stack problem: leave it `infra_invalid`, let the
   autopilot advance, do NOT run §7c. Check the ESCALATE reason string before acting.
3. The supervisor owns autopilot liveness now, so §0a steps 1-3 are automatic;
   the agent's job each tick is the analysis below + queue refill (§8a) when low.

   **CLEAN-IDLE END-STATE (2026-06-06 17:37): EXPECTED, not a fault.** The effective
   queue is DRAINED — PTC-090 ESCALATEd (sync burst) and the 6 remaining queued ideas
   (PTC-040–045, 051) are permanently gated on `PTC-035: strong_signal` (unsatisfied),
   so NONE are runnable. The supervisor's `has_work()` was upgraded to use
   `controller.py next` as the runnability oracle, so it now logs **"queue drained …
   idle, not relaunching"** and stays QUIET (no autopilot churn) while remaining alive
   to auto-resume if a runnable idea is added. So a tick seeing "supervisor alive +
   no autopilot running + 'idle, not relaunching'" is the CORRECT drained state — do
   NOT treat the missing autopilot as a fault, and do NOT manufacture artifact-confirming
   busywork to fill it. **The deep-analysis mission is COMPLETE; the right next step is
   the user's redirect to a new question.** To resume: add a runnable idea to
   `ideas.yaml` (the supervisor auto-relaunches the autopilot within ~60s), or the user
   redirects. Only if the stack's PERSISTENT sync degradation needs clearing should §7c
   (user-authorized) be run.

### The two decisive controls (run NEXT, pri 130/129)
- **PTC-088** (`eval_control_start_step:0`, `num_steps:31`, every 5, fla):
  untrained **step-0 baseline + full 0/5/.../30 trajectory** of the promoted
  recipe in one run.
- **PTC-089** (same + `learning_rate:0`): frozen-weights control.

### What to compute when they land (do NOT just read the autopilot verdict)
Read the profile jsonl (`last_profile` in the idea) for ALL control rows, and
answer — these directly test the user's hypotheses:
1. **Is it there at step 0?** If `acc_pause-acc_corrupt` and
   `answer_logprob_margin` are already ~their step-10 values at step 0
   (untrained), the "signal" is a **base-model + prompt-format artifact**, not
   learned prefill compute. (PTC-089 lr=0 corroborates: same numbers at 0/5/10 →
   not gradient-driven.)
2. **Manufactured delta?** Compare `acc_nopause@0` vs `acc_nopause@10`. If
   training DROPS the direct-answer baseline, the pause-vs-no-pause delta is
   manufactured by **damaging the baseline**, not by improving pause. (We already
   know part of the delta is no-pause OOD-collapse.)
3. **Persist or decay?** Track `delta`/`vs_corrupt`/`logprob_margin` across
   steps 0→30. A peak-then-decay says the step-10 verdict cherry-picks early
   dynamics.

### Replication-magnitude collapse (already-strong evidence) — a confound to resolve
The promoted **PTC-044 (flashqla)** had `delta=+0.175 (z=8.27)`,
`answer_logprob_margin=+0.474 (z=59)`. The fla replicates **PTC-079/080** show
`delta=+0.007 (z≈0.33, noise)`, `logprob_margin=+0.034 (z≈6.7)` — a ~25× drop.
So either flashqla≠fla numerically in this regime, **or** PTC-044 was a noise
outlier. PTC-088/089 are on fla, so they characterize the fla regime; flag this
backend confound in any writeup and do not quote the PTC-044 flashqla magnitudes
as the replicated effect. Headline `delta` is the metric the user cares about
(generation accuracy); `answer_logprob_margin` is **teacher-forced** and may be a
positional artifact of "Answer: " preceding the gold answer.

Write findings to `experiments/opd_profile/` (e.g. extend the autoresearch
findings doc) and update priors in MEMORY when a control is decisive.

**Tooling + first result (2026-06-06):** use
`autoresearch/da_decompose.py <PTC-id|profile.jsonl>` to print the per-step
pause/nopause/corrupt trajectory + the generation-derailment diagnostics. First
result is in `OPD_STEP10_ARTIFACT_DECOMPOSITION_2026_06_06.md`: across PTC-079/080/081
the pause-vs-no-pause generation delta is NOISE (z=0.27–0.90), and the verdict's
load-bearing `vs_corrupt` metric is an OOD-derailment artifact (corrupt arm
rambles 44 tok / 62% cap-hit in PTC-081 → vs_corrupt +0.69; behaves normally in
079/080 → +0.07). When PTC-088/089 land, run `da_decompose.py PTC-088 PTC-089` and
APPEND their step-0 / lr=0 rows to that doc (confirm the step-10 numbers ≈ the
untrained model's). Score real signal on `buffer_delta`, NOT `vs_corrupt`.
DONE 2026-06-06: PTC-088 (delta collapse-dominated + confounded pause gain 0.64→0.85)
+ PTC-089 lr=0 (no signal, Δ=−0.044 = untrained) → effect is 100% training-induced.

**RESOLVED-as-best-possible 2026-06-06: PTC-088's acc_pause gain (0.64→0.85) is
most parsimoniously FORMAT ADAPTATION, not compute and not primarily leakage.**
The log-grep leakage check is INFEASIBLE: the control eval logs only *progress*
("sample N/3072"), not the individual eval problems — only *training* samples log
their `Calculate:` strings. So eval/train overlap can't be measured from trainer
logs; it would need the client's eval-pool construction (pod image only) or a
fair no-filler-trained baseline. But it isn't needed: the same format-adaptation
that raises acc_pause (model learns to emit a clean answer in the
`filler+"Answer:"` format) is exactly what collapses acc_nopause (it over-fits to
that scaffold and rambles without it). Both arms move for one format reason;
neither requires latent compute. Leakage is not load-bearing for the conclusion
(the lift is 100% training-induced per PTC-089 lr=0). Thread closed.

## 0c. FAIR NO-FILLER BASELINE WAVE (2026-06-06, user-requested) — read accuracy, NOT the verdict

PTC-091/092/093 = the missing control from the artifact review: train the student to
answer DIRECTLY (no filler) and see if plain accuracy converges to the filler-trained
`acc_pause` (~0.85 at step30, PTC-088). Config (base AM, fla, 31 steps): `student_prefill_count=0`,
empty filler, `opd_hidden_match_coef=0`, `opd_contrastive_corrupt_buffer/answer=0`,
`opd_positive_answer_weight=1.0`, **`eval_control_start_step=30`**. LR sweep: 091=3e-6,
092=1e-5, 093=3e-5.

- **`eval_control_start_step=30` is deliberate:** with no filler, pause≈nopause so the
  control delta≈0 → the autopilot will mark these **`science_reject`** at step 30. THAT
  IS EXPECTED, not a failure (`science_reject` is in `--launch-next-verdicts`, advances
  immediately — putting control at step 30 avoids an early step-0 kill so the run trains
  the full 31 steps).
- **The result is the `eval/accuracy` trajectory** (logged every step) + the step-30
  control accuracy. Read via `da_decompose.py` / the profile's `eval/accuracy`. Compare
  to PTC-088 filler-trained `acc_pause`. Converges to ~0.85 ⇒ filler = format adaptation
  (not compute); lags ⇒ filler may buy compute. **Do NOT read the science_reject verdict
  as "baseline failed."**
- LESSON (race): editing `ideas.yaml`/candidates while the supervisor is live auto-launches
  the in-progress version (PTC-091 briefly launched a v1 before the v2 rewrite; it
  self-healed on the next poll). For multi-step edits, pause the supervisor first.
- **RULE — trajectory runs MUST use `eval_control_start_step = num_steps-1` (control only
  at the final step).** `science_reject` (delta≤0) is in `--launch-next-verdicts` and
  advances IMMEDIATELY (not deferred). At step 0 the model is UNTRAINED so `delta<0`
  (pause ≤ nopause) for BOTH filler and no-filler — so a step-0 control → science_reject →
  the run is killed at step 0. This bit PTC-089, PTC-098 (died at step 5), and nearly
  PTC-099. The per-step `eval/accuracy` is logged every step regardless, so you still get
  the full convergence trajectory; put the control at the end only. (Filler runs can
  *appear* to survive `control_start=0` by luck — if the autopilot's 300s poll first
  catches a step where delta>0 — but it's a race; don't rely on it.)
- **LR sweep retargeted (2026-06-06, user):** 3e-6 is the DEFAULT for both filler and
  no-filler (it's the LR that converges; higher destabilizes — no-filler 1e-5→0.64,
  3e-5→0.20). Sweep LOWER, not higher: filler 094=1e-6, 095=5e-7 (+ 3e-6 = PTC-088).

## 0. Immediate State

Current scientific state:

- The strongest current recipe is full AM, short filler, hidden matching kept,
  corrupt buffer negative kept, corrupt answer negative kept, pipeline-RL on.
- `PTC-044` is the latest decisive result and is `strong_signal`.
- `PTC-044` scorecard:
  `experiments/opd_profile/autoresearch/scorecards/20260605T072940Z-PTC-044-step10-strong_ptc_signal.json`
- `PTC-044` metrics:
  - `acc_pause=0.705078125`
  - `acc_nopause=0.5302734375`
  - `acc_corrupt=0.626953125`
  - `delta=+0.1748046875`
  - `delta_z=8.274321091624453`
  - `vs_corrupt_delta=+0.078125`
  - `vs_corrupt_z=3.7610961813683548`
  - `answer_logprob_margin=+0.4739779793744555`
  - `answer_logprob_z=59.416112583938336`

Current loop state (2026-06-05T08:10Z):

- `PTC-044` finished `strong_signal` and is the promoted recipe.
- An **overnight ablation/scaleup wave of `PTC-044` is now queued and running**.
  Twelve single-knob candidates are queued, each gated on `PTC-044: strong_signal`
  (already satisfied), so they chain through autopilot regardless of individual
  outcomes. In priority order:
  - `PTC-052` (pri 90) exact `PTC-044` replicate / promotion proof — LAUNCHED first.
  - `PTC-050` (pri 89) hidden-match coef 0.5.
  - `PTC-053` (pri 86) hidden-match coef 1.0; `PTC-054` (pri 84) hidden coef 4.0.
  - `PTC-055` (pri 82) corrupt-answer 0.25; `PTC-056` (pri 80) corrupt-answer 0.0625.
  - `PTC-057` (pri 78) corrupt-buffer 0.5; `PTC-058` (pri 76) corrupt-buffer 2.0.
  - `PTC-059` (pri 74) lr 5e-6; `PTC-060` (pri 72) lr 2e-6.
  - `PTC-061` (pri 70) step-15 stability (num_steps 16); `PTC-062` (pri 68)
    prompts/step 256 scaleup.
- A `controller.py autopilot` process IS live (launched 2026-06-05T08:07Z,
  `--max-polls 288`, log under `/tmp/ptc-autopilot-overnight-*.log`; current
  log path also in `/tmp/ptc-autopilot-current-log`).
- While a run is `launched`, `controller.py next` returns nothing and
  `next_idea` returns `None` by design — that is NOT "queue empty", it means a
  run is in flight. Check `launched` status, not just `next`.
- The OPD pods for `stack=er-opd-q36-35b-slots` are Running/Ready with 0 restarts.

When this wave drains, do NOT idle — enqueue another wave per §0a step 4 / §8a.

Current process detail:

- A detached hourly reminder process may exist:
  `python experiments/opd_profile/autoresearch/hourly_wakeup.py --interval-seconds 3600 --first-delay-seconds 3600`
- That process only emits check-due markers. It does not inspect or mutate the
  queue. It is not the autoresearch autopilot.

## 1. Non-Negotiables

Do not touch anything with `MTP` in the name.

All GPU workloads must use `team: turbo` on the pod template labels. Do not
manually set the Volcano scheduler fields unless there is a specific reason.

Use only the OPD stack label for Kubernetes checks:

```bash
kubectl -n apanda get pods -l stack=er-opd-q36-35b-slots -o wide
```

All W&B runs should be in:

```text
together-research/xorl-prefill-time-compute
```

Do not spend the next experiments on random filler/token variants unless the
user explicitly asks for that branch. The high-capacity random-symbol runs
already rejected, and the current positive signal is not a filler-surface sweep.

UPDATE 2026-06-05 (filler-surface branch explicitly REOPENED by user): the user
asked to run a filler-type sweep, so this gate is lifted for the current wave.
Findings so far on the fla stack: varied short 9-symbol = strong (+0.437);
9 dots ≈ 0 (+0.006); 100 dots = reject (−0.059); NL "think" = inert (0). So the
effect tracks symbol VARIETY, not count/capacity ("more" hurts). Now running a
high-capacity (~100-token) random-filler-by-type sweep (PTC-071 symbols / 072
words / 073 numbers / 074 token-ids) to map which content type, if any, works at
capacity. Also: the student sampler eval is now FAST (~90s, was ~30min) — see
`experiments/opd_profile/LESSON_sglang_sampler_flags.md`. Do NOT revert the
sampler to `--max-running-requests 1` / `--disable-cuda-graph` / `--disable-overlap-schedule`.

## 2. Important Paths

```bash
REPO=/home/apanda/xorl-apanda-dev-opd-port
cd "$REPO"

STACK=er-opd-q36-35b-slots
NS=apanda
CONTROL_ROOT=/shared/opd-control/er-opd-q36-35b-slots
RESULT_ROOT=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots

GENERATOR=experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py
CONTROLLER=experiments/opd_profile/autoresearch/controller.py
IDEAS=experiments/opd_profile/autoresearch/ideas.yaml
CANDIDATES=experiments/opd_profile/autoresearch/candidates
SCORECARDS=experiments/opd_profile/autoresearch/scorecards
RUNS=experiments/opd_profile/autoresearch/runs.jsonl
```

Primary files:

- `experiments/opd_profile/autoresearch/controller.py`
- `experiments/opd_profile/autoresearch/ideas.yaml`
- `experiments/opd_profile/autoresearch/candidates/PTC-*.yaml`
- `experiments/opd_profile/autoresearch/runs.jsonl`
- `experiments/opd_profile/autoresearch/scorecards/*.json`
- `experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py`
- `experiments/opd_profile/monitor_live_opd.py`

Primary data:

```text
PROMPTS_JSON=/shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
COT_JSON=/shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
NUM_PROMPTS=8185
```

## 3. What The Loop Does

The controller is a small YAML-driven experiment loop:

1. `ideas.yaml` holds candidate IDs, statuses, priorities, dependency gates,
   hypotheses, and scorecard references.
2. Each candidate has a YAML file under `candidates/`.
3. The controller renders the candidate into trainer control scripts by calling
   `q36_35b_reprogrammable_slots.py`.
4. Trainer pods consume those scripts and write an `opd_profile.jsonl` under
   `RESULT_ROOT`.
5. The controller scores the profile, writes a scorecard, advances idea status,
   stops trainer control, and optionally launches the next runnable idea.

Important controller commands:

```bash
python experiments/opd_profile/autoresearch/controller.py next
python experiments/opd_profile/autoresearch/controller.py monitor --json
python experiments/opd_profile/autoresearch/controller.py score --idea-id PTC-044 --json
python experiments/opd_profile/autoresearch/controller.py launch --id PTC-050 --dry-run
python experiments/opd_profile/autoresearch/controller.py advance --id PTC-044
```

Autopilot command used for the latest loop:

```bash
LOG=/tmp/ptc-autopilot-monitor-12h-opd-only-$(date -u +%Y%m%dT%H%M%SZ).log
setsid python experiments/opd_profile/autoresearch/controller.py autopilot \
  --launch-next \
  --poll-seconds 300 \
  --max-polls 144 \
  --sampler-replicas 2 \
  --sampler-layout spare-teacher1 \
  --terminal-action stop_advance_launch_next \
  --terminal-only-after-final-control \
  --launch-grace-seconds 1800 \
  > "$LOG" 2>&1 &
```

Use `--terminal-only-after-final-control` for step-10 stability candidates. It
prevents an early positive from terminating before the final configured control.

Do not start autopilot if `controller.py next` says `no queued ideas` unless you
have just edited `ideas.yaml` to make a new candidate runnable.

## 4. Scoring Semantics

The main operational metric is:

```text
delta = acc_pause - acc_nopause
```

Current branch uses `score_mode: pause_vs_nopause`.

Promotion-level signal:

- `delta >= 0.06`
- `delta_z >= 3.0`
- positive answer-logprob support
- clean controls and no request-failure/cap-hit artifact large enough to explain
  the result

Retest-level signal:

- `delta >= 0.04`
- `delta_z >= 2.0`
- positive answer-logprob support

Weak signal:

- positive answer-logprob support but operational accuracy below the promotion
  or retest gate

Rejected:

- pause does not beat no-pause accuracy, or answer-logprob support is absent

The current science goal is operational prefill-time compute:

```text
prompt + filler + "Answer: "
```

should answer better than:

```text
prompt + "Answer: "
```

This is narrower than proving a complete prompt-specific memory mechanism.
Mechanism probes matter, but they should not veto a clean operational
pause-vs-no-pause win.

## 5. Current Best Recipe

The current best recipe is full AM with the short visible filler.

Visible student pause:

```text
" ! | ~ _ * ^ # @ " + "Answer: "
```

Core settings:

```yaml
student_prefill_text: " ! | ~ _ * ^ # @ "
student_prefill_count: 1
student_prefill_suffix: "Answer: "
opd_hidden_match_coef: 2.0
opd_contrastive_corrupt_buffer_weight: 1.0
opd_contrastive_corrupt_answer_weight: 0.125
opd_contrastive_corrupt_buffer_mode: rotate
opd_contrastive_corrupt_buffer_span: memory_only
opd_positive_answer_weight: 0.0
opd_loss_max_clamp: 5.0
learning_rate: 3e-6
eval_answer_logprob_control: true
eval_corrupt_pause_control: true
sync_method: p2p
serial_endpoint_sync: false
```

The successful pipeline-RL version adds:

```yaml
opd_pipeline_rl: true
sampler_quiesce_before_sync: false
sampler_layout: spare-teacher1
sampler_replicas: 2
```

The strongest artifact is `PTC-044`:

```text
candidate: experiments/opd_profile/autoresearch/candidates/PTC-044.yaml
profile: /shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260605T063908Z-configPTC-044-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl
scorecard: experiments/opd_profile/autoresearch/scorecards/20260605T072940Z-PTC-044-step10-strong_ptc_signal.json
```

## 6. Evidence Ledger

Decision-critical results:

| ID | Recipe | Result | Interpretation |
| --- | --- | --- | --- |
| `PTC-020` | Full AM, non-pipeline, step 5 | `delta=+0.0459`, `z=2.119`, answer-logprob `+0.1110`, `z=23.76` | Current-code reproduction positive; promoted for retest. |
| `PTC-021` | Full AM, non-pipeline, step 10 | `delta=+0.2285`, `z=11.161`, answer-logprob `+0.5558`, `z=64.99` | Full AM is stable and strong without pipeline-RL. |
| `PTC-022` | Full AM larger batch/screen | `science_reject` | Do not assume bigger prompt batch automatically helps. |
| `PTC-023` | No-hidden, corrupt negatives, non-pipeline, step 5 | `delta=+0.0791`, `z=3.599`, answer-logprob `+0.0804`, `z=15.10` | Hidden matching is not required for an early operational lift. |
| `PTC-024` | Hidden kept, no corrupt negative | near-zero accuracy delta, negative answer-logprob | Corrupt negative/counterfactual pressure looks necessary. |
| `PTC-025` | No hidden, no corrupt negative | `science_reject` | Positive-answer/no-hidden alone is not enough. |
| `PTC-026` | No-hidden, corrupt negatives, non-pipeline, step 10 | `delta=+0.0010`, answer-logprob positive | No-hidden early lift is unstable by step 10. |
| `PTC-029` | No-hidden, corrupt, high-capacity random-symbol filler, pipeline | `delta=-0.1299`, answer-logprob negative | High-capacity random-symbol filler rejected. |
| `PTC-030` | Hidden kept, no corrupt, high-capacity random-symbol filler, pipeline | `delta=-0.0869`, answer-logprob negative | No-corrupt high-capacity branch rejected. |
| `PTC-035` | No-hidden, corrupt, short filler, pipeline, step 5 | `delta=+0.0029`, answer-logprob positive | Pipeline no-hidden branch is weak; do not unlock its ablations. |
| `PTC-036` | Full AM, short filler, pipeline, step 5 | `delta=+0.0635`, `z=2.909`, answer-logprob `+0.2020`, `z=35.01` | Full AM survives pipeline-RL; justified `PTC-044`. |
| `PTC-044` | Full AM, short filler, pipeline, step 10 | `delta=+0.1748`, `z=8.274`, answer-logprob `+0.4740`, `z=59.42` | Best current promotion-quality result. |

Core interpretation:

- Hidden matching is not absolutely required for an early step-5 effect
  (`PTC-023`), but the no-hidden branch is not stable (`PTC-026`) and does not
  survive the pipeline setting cleanly (`PTC-035`).
- Corrupt negative/counterfactual pressure is necessary. Removing it produced
  weak or rejected results (`PTC-024`, `PTC-025`, `PTC-030`).
- Pipeline-RL is not a behavioral confound for full AM. `PTC-036` was positive
  and `PTC-044` was strong.
- The recipe to promote is full AM, not random filler variants and not the
  no-hidden ablation branch.

## 7. Queue And Gating Map

The current `ideas.yaml` has no runnable idea.

Finished branch:

- `PTC-035`: `weak_signal`
- `PTC-036`: `promote_retest`
- `PTC-044`: `strong_signal`

Queued but not runnable:

- `PTC-040`: corrupt-buffer-only no-hidden pipeline ablation
- `PTC-041`: corrupt-answer-only no-hidden pipeline ablation
- `PTC-042`: no-hidden corrupt plus positive-answer term
- `PTC-043`: no-hidden corrupt lower-lr step-10 stability
- `PTC-045`: no-hidden rotate-preserve-whitespace corrupt variant
- All five require `PTC-035` to be `strong_signal` or `promote_retest`.
- `PTC-035` is only `weak_signal`, so do not run these by default.

Conditional fallback:

- `PTC-051`: non-pipeline full-AM current-code reproduction.
- It requires both `PTC-035` and `PTC-036` to be weak/rejected.
- `PTC-036` promoted, so `PTC-051` is intentionally not runnable.
- Keep it as a fallback only if `PTC-036`/`PTC-044` are later invalidated.

Deferred random/filler surface branch:

- `PTC-029` and `PTC-030` rejected the high-capacity random-symbol variants.
- `PTC-031` through `PTC-034` and `PTC-038`/`PTC-039` are not primary next
  steps. They answer filler-surface questions, not the current promotion path.

Deferred useful existing candidate:

- `PTC-050` exists and is not currently runnable.
- It is a full-AM-ish pipeline step-10 candidate with lower hidden coefficient:
  `opd_hidden_match_coef: 0.5`.
- This is a good mechanism/efficiency follow-up after `PTC-044`, but it is not
  a replication of the exact best recipe.

## 8. What To Promote Next

Promote the full-AM pipeline branch, anchored by `PTC-044`.

Recommended next actions, in priority order:

1. If the user wants one more promotion-proof run, add a new exact or near-exact
   replicate of `PTC-044`.
   - Suggested ID: `PTC-052`.
   - Purpose: independent confirmation that full-AM pipeline step-10 is robust.
   - Start from `candidates/PTC-044.yaml`.
   - Keep hidden matching coefficient `2.0`, corrupt buffer `1.0`, corrupt
     answer `0.125`, short filler, pipeline-RL, P2P sync, and final 1k control.
   - Do not change filler type.

2. If the user is comfortable promoting from the current evidence and wants the
   next mechanism/efficiency test, run existing `PTC-050`.
   - Purpose: test whether full hidden matching strength is required.
   - It uses `opd_hidden_match_coef: 0.5` with the full corrupt-negative recipe,
     pipeline-RL, and step-10 control.
   - Add a `requires: [PTC-044]` gate with `PTC-044: [strong_signal]` before
     making it runnable, so the queue documents why it follows `PTC-044`.

3. If `PTC-050` stays strong, promote the lower-hidden-coef recipe as a cheaper
   or less invasive variant.

4. If `PTC-050` weakens or rejects, keep full hidden matching (`coef=2.0`) as
   part of the promoted recipe.

5. Do not unlock `PTC-040` through `PTC-045` unless the user explicitly wants
   to re-open the no-hidden branch despite `PTC-035`.

6. Do not run `PTC-051` unless the full-AM pipeline branch is invalidated. It
   was a fallback for the case where both short-filler pipeline reproductions
   failed; that did not happen.

Concrete queue edit for `PTC-050`, if choosing option 2:

```yaml
- id: PTC-050
  status: queued
  priority: 82
  candidate: candidates/PTC-050.yaml
  requires:
  - PTC-044
  requires_statuses:
    PTC-044:
    - strong_signal
  hypothesis: If PTC-044's full-AM pipeline signal does not require the full hidden-match coefficient, a lower coefficient should preserve most of the step-10 pause-vs-no-pause lift while reducing hidden anchoring pressure.
  rationale: PTC-044 made the full-AM pipeline recipe promotion-quality. PTC-050 tests the smallest immediate mechanism/efficiency change without changing filler, corrupt negatives, pipeline scheduling, or the step-10 final control.
```

Then dry-run:

```bash
python experiments/opd_profile/autoresearch/controller.py next
python experiments/opd_profile/autoresearch/controller.py launch --id PTC-050 --dry-run
```

If the dry-run is sane, start autopilot with the command from section 3.

## 8a. Enqueue The Next Wave When The Queue Drains (overnight, do not idle)

When the current wave finishes (autopilot prints `no queued idea to launch`),
do not stop. Enqueue another wave of single-knob `PTC-044` ablations/scaleups
and restart autopilot. Procedure:

1. Read the latest scorecards first and let the data steer the next knobs:

   ```bash
   ls -t experiments/opd_profile/autoresearch/scorecards/*.json | head -n 24
   python experiments/opd_profile/autoresearch/controller.py score --idea-id PTC-052 --json
   ```

   Extend in the direction that helped, retreat from what hurt. Stay single-knob
   for attribution unless deliberately building one tuned composite of the
   best-so-far knobs. Stay in the full-AM + hidden-match + corrupt-negative +
   pipeline-RL family. Never random-filler, never no-hidden.

2. Knobs already swept by the first wave (PTC-050, PTC-052..062): hidden-match
   coef {0.5, 1.0, 2.0, 4.0}; corrupt-answer {0.0625, 0.125, 0.25};
   corrupt-buffer {0.5, 1.0, 2.0}; lr {2e-6, 3e-6, 5e-6}; steps {10, 15};
   prompts/step {128, 256}. Pick NEW points (e.g. step-20 stability, hidden
   coef 3.0, corrupt-answer 0.5, lr 7e-6 / 1.5e-6, batch 64 or 384, or a tuned
   composite) — do not re-run identical knobs.

3. Generate each candidate as a single-knob edit of an existing fla wave
   candidate such as `candidates/PTC-052.yaml` (NOT `PTC-044.yaml`, which ships
   the broken `gdn_backend: flashqla`). Use the line-replace pattern: copy the
   file, change `id`, `slug`, `buffer_label`, and the one knob; for longer runs
   also set `default_num_steps` and `eval_control_start_step = num_steps - 1`.
   Keep `gdn_backend: fla` until tilelang/flashqla is repaired (see §0a).

4. Append idea entries to `ideas.yaml` with `status: queued`, distinct
   priorities, and the satisfied gate:

   ```yaml
   - id: PTC-0NN
     status: queued
     priority: <distinct>
     candidate: candidates/PTC-0NN.yaml
     requires:
     - PTC-044
     requires_statuses:
       PTC-044:
       - strong_signal
     hypothesis: <single-knob hypothesis>
     rationale: <why this point>
     default_num_steps: 11
     default_prompts_per_step: 128
   ```

5. Validate then relaunch:

   ```bash
   python experiments/opd_profile/autoresearch/controller.py next        # should be your new head
   python experiments/opd_profile/autoresearch/controller.py launch --id <head> --dry-run
   # then the §10 autopilot command
   ```

## 9. How To Act From A Fresh Agent

Start with a read-only status pass:

```bash
cd /home/apanda/xorl-apanda-dev-opd-port
date -u +%Y-%m-%dT%H:%M:%SZ
python experiments/opd_profile/autoresearch/controller.py next
python experiments/opd_profile/autoresearch/controller.py monitor --json
ls -t experiments/opd_profile/autoresearch/scorecards/* | head -n 20
tail -n 80 experiments/opd_profile/autoresearch/runs.jsonl
kubectl -n apanda get pods -l stack=er-opd-q36-35b-slots -o wide
```

Check for live autopilot:

```bash
ps -eo pid,ppid,stat,etime,cmd | rg 'experiments/opd_profile/autoresearch/controller.py autopilot' || true
```

If there is a launched idea:

```bash
python experiments/opd_profile/autoresearch/controller.py monitor --json
python experiments/opd_profile/autoresearch/controller.py score --idea-id <ID> --json
```

If the score is terminal and the autopilot is not running, advance manually only
after checking that the profile belongs to the launched idea:

```bash
python experiments/opd_profile/autoresearch/controller.py advance --id <ID> --profile <PROFILE>
```

If there is no runnable idea:

1. Do not restart autopilot blindly.
2. Choose the next branch from section 8.
3. Edit `ideas.yaml` and/or add a candidate YAML.
4. Dry-run launch.
5. Start autopilot.

## 10. Launch And Monitoring Details

Dry-run a candidate:

```bash
python experiments/opd_profile/autoresearch/controller.py launch --id PTC-050 --dry-run
```

Launch through autopilot, preferred after queue edit:

```bash
LOG=/tmp/ptc-autopilot-monitor-12h-opd-only-$(date -u +%Y%m%dT%H%M%SZ).log
setsid python experiments/opd_profile/autoresearch/controller.py autopilot \
  --launch-next \
  --poll-seconds 300 \
  --max-polls 144 \
  --sampler-replicas 2 \
  --sampler-layout spare-teacher1 \
  --terminal-action stop_advance_launch_next \
  --terminal-only-after-final-control \
  --launch-grace-seconds 1800 \
  > "$LOG" 2>&1 &
echo "$LOG"
```

Watch the loop:

```bash
tail -f "$LOG"
python experiments/opd_profile/autoresearch/controller.py monitor --json
kubectl -n apanda logs er-opd-q36-35b-slots-trainer-head --tail=120
```

Pod health:

```bash
kubectl -n apanda get pods -l stack=er-opd-q36-35b-slots \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.labels.team}{"\t"}{.status.phase}{"\t"}{range .status.containerStatuses[*]}{.ready}{"/"}{.restartCount}{" "}{end}{"\n"}{end}'
```

Expected GPU pods have `team=turbo`. Some non-GPU service pods may not show the
team label. Do not edit MTP resources.

## 11. Common Failure Modes

Stale PID files:

- `experiments/opd_profile/autoresearch/autopilot.pid` and
  `experiments/opd_profile/autoresearch/logs/autopilot.pid` may be stale.
- Prefer `ps ... controller.py autopilot` over trusting pid files.

No queued ideas:

- `ideas.yaml` can contain `status: queued` entries that are not runnable because
  `requires_statuses` are not satisfied.
- Trust `controller.py next` for runnable queue head.

Wrong branch after `PTC-044`:

- Do not force no-hidden ablations open.
- `PTC-035` was weak, and `PTC-026` showed no-hidden step-10 collapse.
- The strongest path is full AM with hidden matching and corrupt negatives.

Premature scoring:

- Step-10 candidates can show an earlier incomplete profile or early rows.
- Use `--terminal-only-after-final-control` in autopilot.
- Do not advance a step-10 candidate until the final control row exists.

Control artifacts:

- A clean operational win uses pause-vs-no-pause and positive answer-logprob.
- Corrupt-control results are useful diagnostics, but the current promotion
  standard is operational improvement plus sanity checks, not a strict causal
  memory proof.

Random filler distraction:

- Random-symbol high-capacity variants rejected.
- Do not add random token/number variants as the next default branch.

## 12. Minimal Decision Tree

Use this when deciding what to do next:

```text
Is an autopilot process alive (ps | rg 'controller.py autopilot')?
  yes -> just monitor (monitor --json + tail log); do nothing else this tick.
  no  -> continue.

Is there a launched idea (status: launched)?
  yes -> a run is in flight but autopilot died; restart autopilot (§10) to resume
         monitoring/advancing it. Do not hand-advance unless terminal+final-control.
  no  -> continue.

Does controller.py next return an idea?
  yes -> dry-run launch, then start autopilot (§10).
  no  -> continue.

Queue is drained (no launched idea, next returns nothing).
  OVERNIGHT: do NOT stop and do NOT idle. Enqueue the next wave of single-knob
  PTC-044 ablations/scaleups (§8a) and start autopilot (§10).
  Only outside the overnight directive does "the promoted recipe is full-AM
  pipeline from PTC-044; stop" apply.

Should no-hidden ablations run?
  default no -> PTC-035 was weak and PTC-026 collapsed by step 10.
  only yes if user explicitly reopens no-hidden branch.

Should fallback PTC-051 run?
  default no -> PTC-036 promoted and PTC-044 was strong.
  only yes if the full-AM pipeline branch is later invalidated.
```

## 13. One-Sentence Handoff

The loop has established full-AM short-filler with hidden matching plus corrupt
negatives as the promotion-quality recipe, including on pipeline-RL at step 10
via `PTC-044`; there is currently no runnable queued idea, so the next agent
should either add a `PTC-044` replicate for one more promotion proof or make
existing `PTC-050` runnable behind `PTC-044` to test whether lower hidden-match
strength preserves the effect.
