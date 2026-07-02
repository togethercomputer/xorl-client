# Reconciliation: OPD/filler working set vs apanda-dev upstreams — 2026-07-02

Full audit + executed consolidation of the four stacks used by the OPD-prefill /
filler-RL line against their upstreams. Convention enforced: **core code → PRs
onto apanda-dev (or main); experiments/docs/data stay on science branches;
infra-shaped assets live in xorl-infra.** Scope was OUR branches only — the
wordle and zorl consolidations ran in parallel under their own agents (see
§Coordination).

## Executed state (all PRs open as of 2026-07-02)

| Repo | Our branch | Upstream | Action taken | PR |
|---|---|---|---|---|
| xorl-client-internal | `exp/opd-prefill` (this worktree) | `apanda-dev` | dirty tree committed (experiments snapshot 63f5667 + docs + example knob 152d5ba); apanda-dev `bcd037d` merged in (491adc8); branch pushed | — (science branch) |
| xorl-client-internal | `opd-battery-consolidation` (hub `/home/apanda/xorl-client`) | `apanda-dev` | 130-file dirty tree committed as 3 commits (core k3 field `daba41e`, GSM8K example `2a27833`, experiments `c490b89`); apanda-dev merged; pushed | — |
| xorl-client-internal | `up/client-core-20260702` | `apanda-dev` | k3 field + merged on_policy_distillation.py (GSM8K/expansion-lock ∪ teacher-layer-cache knob) | [#9](https://github.com/togethercomputer/xorl-client-internal/pull/9) |
| xorl-internal | `up/opd-vpkl-core-20260702` (from Line A) | `apanda-dev` | 49-commit line curated into 5 themed core commits, 12-file conflict set resolved, 394 tests green | [#434](https://github.com/togethercomputer/xorl-internal/pull/434) |
| xorl-internal | `k3-recon-r3-mooncake-20260630` (Line B) | `apanda-dev` | dirty experiments committed (`a0811b67f`); core extraction DEFERRED — superseded by wordle agent's PR #430 (see §Ownership) | #430 (theirs) |
| xorl-sglang-internal | `apanda-dev` | origin | NO push by this agent — wordle agent rewrote+pushed (`e9593887f`) | — |
| xorl-infra-internal | `opd-battery-consolidation` | `main` (no apanda-dev exists) | launchers + sweep configs + watchdog committed & pushed; main is PR-protected | [#2](https://github.com/togethercomputer/xorl-infra-internal/pull/2) |

## What had actually changed (audit summary)

### Client (xorl-client-internal)
- Committed divergence of both working branches was **experiments/-only** (runbooks,
  AMDAHL candidates, microbench scripts). Zero committed core drift.
- The only core-code drift was **uncommitted**: `xorl_client/types/forward_backward_output.py`
  (optional `k3` diagnostic on LossFnOutput) and `examples/on_policy_distillation.py`
  (+119 GSM8K gold-answer scoring + expansion-lock filler filter; +2 teacher-layer-cache
  knob in this worktree). Same file, two divergent uncommitted edits in two checkouts —
  merged in PR #9.
- Discovery: `/home/apanda/xorl-client-chat-completions` is a **symlink** to
  `/home/apanda/xorl-client` — the client was already single-checkout; the "twin with
  130 identical dirty files" was one directory seen twice.

### Training engine (xorl-internal) — the real engine divergence
Two divergent lines from the same 06-13 fork point (`609bed763`), NEITHER a superset:

- **Line A** `codex/opd-repeat2-diagnostics-20260615` — the OPD-prefill throughput line.
  49 local commits, +3,295/−249 core src: VP-KL kernel (`ops/loss/vocab_parallel_reverse_kl.py`),
  lm-head TP under FSDP/HSDP, packed-row replay batching, teacher-cache device caching +
  streamed layer slices, GDN backend selection, optim-cleanup metrics; +2.3k test lines.
  → **Upstreamed as PR #434.** One commit (`afde6df10`) was already upstream (#374).
- **Line B** `k3-recon-r3-mooncake-20260630` — the filler/K3 line. apanda-dev + ONE
  12.5M-line snapshot commit (`49720d816`) mixing +6.6k core src (batch-invariant ops,
  MoE experts, side_payloads/R3 Mooncake, routing_replay_handler, attention/norm
  numerics) with ~1,900 data artifacts. → core content owned/upstreamed by the wordle
  agent (PR #430); dense-K3 remainder stays on the branch. Upstream had meanwhile
  landed the R3 externalization + Mooncake teacher-cache themselves (#405 line).
- **12-file collision set** (both lines edited): model_runner, runner_dispatcher,
  batch_utils, request_processor, moe_block, gated_deltanet, server_arguments,
  module_utils, arguments, trainer, training_utils, qwen3_5_moe modeling.
- Also preserved: the primary checkout `/home/apanda/xorl-internal` sat on
  `feature/on-policy-distillation` with 351 dirty files stale since 06-03 (OPD-MVP era,
  superseded) → snapshot branch `snap/opd-mvp-era-worktree-20260702` via temp-index
  commit, working tree untouched.

### Inference engine (xorl-sglang-internal)
- `apanda-dev` was 20 ahead of `main` in 4 themes: LoRA/virtual-experts MoE (6), MTP +
  OPD teacher-hidden serving (6), ZORL fresh_ab ES (7), K3 numerics knobs #52 (1); plus
  2 unpushed commits — one of which (`06cb9323d`) carried a WRONG message (actually a
  12-file wordle RL patch set). The wordle agent reset+relabeled+pushed (`e9593887f`).
- Stranded sibling worktrees (zorl-ps-fp32 PS layer, duplicated K3 numerics diffs in
  dense-k3-mm ≡ pr52-branch, mtp, target-ablation, isr3-betafp32) belong to the zorl/k3
  streams — untouched per scope; zorl agent's plan covers the PS layer.
- No live sgl-project remote; `main-upstream` snapshot is stale (2026-04-23) — true
  upstream drift unmeasurable from local refs.

### Infra (xorl-infra-internal)
- No apanda-dev exists; base = `main` (PR-protected). Divergence was 4 commits + 34
  dirty (3 slot launchers: q235 variant, stack/sampler/python-bin overrides, fb-capture
  + gdn-backend CLI args, expandable_segments toggle; 30 OPD-battery sweep yamls).
  All committed; watchdog script copied in from client experiments (infra home);
  PR #2 → main. The k8s/zorl manifests referenced by client `k8s` symlinks already
  live here (symlinks locally excluded from the client repo).

## Notable merge-resolution decisions (PR #434)
1. Line A's `load_checkpoint_optimizer` dropped — upstream landed identical semantics
   as `load_optimizer` at both TrainArguments and server layers; tests renamed.
2. `request_processor`: row-batching incompatibility guard moved BEFORE R3 payload
   externalization (fail-fast, no orphaned payload dirs); unioned with upstream's
   routing-payload validation/externalize flow.
3. `model_runner`: per-teacher VP-KL dispatch restored intact; zero-anchors unified on
   upstream's `_fp32_zero_anchor`/`_lm_head_forward_anchor` helpers.
4. `teacher_cache`: upstream removed the file-backed safetensors path (Mooncake-only,
   #405). Line A's device-cache + `get_layer_slice` grafted on top; tests ported to the
   in-memory Mooncake fixture. ⚠️ **Recipe impact**: OPD clients passing file paths as
   `teacher_hidden_caches` must switch to the Mooncake producer path when running
   against apanda-dev.

## UPDATE (later 2026-07-02): 15-PR engine landscape + merge queue

The engine reconciliation grew to three stacks: marin parity (#431–#433), wordle
k3-recon delta restacked ON #433 (#430), the k3 production decomposition
(#435–#444, atomic \`k3/*\` PRs incl. #441 model-runner BI gate + loss TP group),
and this line's #434. Dry-run merge-trees from #434:

- #434 × #441 (incl. #435/#436): **clean auto-merge** despite shared files.
- #434 × #430: **2 files / 4 hunks** — all in the diagnostics-capture region
  (moe_block `_diagnostic_capture_routing` vs `_capture_diagnostic_component`;
  3 matching hunks in model_runner). VP-KL / packed-row / teacher-cache do not conflict.

Human merge queue (per wordle agent, confirmed): #433 → #430 → #431/#432 → #434
(rebase = the 4 hunks above; map posted as a comment on #434). Client #9 and
infra #2 are independent.

## Coordination (three parallel consolidations, 2026-07-02)
- **wordle agent** (bubbly-shell plan): owns sglang apanda-dev push, engine PR #430
  (wordle-validated src ⊂ Line B snapshot), infra `wordle-consolidation`, client
  science-branch push. EXECUTED per memory `wordle-upstreaming-scope`.
- **zorl agent** (mellow-piglet plan): owns xorl-sglang-zorl PS layer commits, a clean
  `zorl-ps` engine branch, client/infra `zorl-consolidation` branches.
- **this agent**: everything in §Executed state. Collisions avoided: skipped the
  sglang push (would have enshrined the mislabeled commit + broken their rewrite);
  deferred Line B extraction (dup of #430). Residual risk: PR #430 and #434 both touch
  the 12-file collision set — whichever merges second will need a rebase; #434's
  resolutions document the intended semantics.

## Verification
- Client PR branch: chunked f/b 42 passed, types 81 passed, k3 roundtrip OK.
- Engine PR branch: 394 passed / 31 GPU-gated skips across all theme suites; zero
  experiments/docs/configs leakage.
- Every repo in the working set ends `git status` clean on a pushed branch.
