# Encoded-Reasoning / Filler-Token OPD — Master Runbook & Source of Truth (2026-06-02)

This is the single consolidated record of the encoded-reasoning investigation: the
scientific question, every eval/experiment run (this session + the parallel "tomi"
session), the methodology and its corrections, the OPD distillation experiment now in
flight, all infrastructure + bring-up hazards, and how to reproduce/continue. If you are
picking this up cold, read §0 then §6 (the live experiment) and §9 (how to run things).

---

## 0. TL;DR / Bottom line

**The question:** can a model's *filler/pause "thinking buffer"* be made to carry
distilled teacher reasoning — i.e., is the buffer a trainable *encoded-reasoning* channel
(OPD: distill a CoT-equipped teacher into a buffer-equipped student)?

**What we now know (high confidence):**
1. **The paper's filler-token effect is REAL and reproduces** — but only on **Qwen3-235B**
   (both `base` and `Instruct-2507`), and only when the direct-answer baseline sits in a
   **"responsive zone" (~0.3–0.7)**. The full **type-inversion** pattern (word-based fillers
   win at 0-shot, numerical/ellipsis win at 10-shot, `pi_digits`/`random_numbers`
   catastrophic at 0-shot) reproduces cell-for-cell.
2. **It does NOT extend down-scale.** Q3.5/Q3.6-35B-A3B, Q3-30B-Instruct-2507: under a
   *clean* prompt the filler is flat-to-negative on every task. Their paper-style "lift" is
   **format-recovery** (the filler partially undoes the `/no_think` soft-suppression; a
   properly-prompted baseline beats it).
3. **Q3.5-397B is saturated** (clean 4×4 baseline 0.98) — no headroom; its original
   "+20.1pp ellipsis" was format-recovery + an **anti-EOS** artifact (and an FP8-multimodal
   serving-garbage artifact in one session).
4. **The single viable cell = Q3-235B-A22B `base`, 4-digit mult.** Native filler shows a
   **compute-concentration signature** (pause **+0.10–0.13 pass@1 / −0.04 pass@8** under the
   clean `chat_hardoff` prompt) — pass@1↑ without pass@8↑ = *compute*, not *search*. This is
   the paper's own headline model+task.

**The load-bearing OPEN question (the experiment now running):** can that compute be
*trained into a buffer* via distillation? The only prior attempt (`nohm`) was **confounded**
(truncated CoT, suppressed baseline, no hidden-match) and went flat. The clean retest
(`er-opd-235b-clean4d`, config A) is in flight on Q3-235B base. **Prior is uncertain:** the
paper's *RL* on filler tokens yields pass@8 (search), not pass@1 (compute) — but distillation
targets the teacher's compute-concentrated buffer *directly*, a different objective, so it is
genuinely untested.

**Practical guidance:** only the **Qwen3-235B** setting is worth further compute. Everything
≤35B is a dead end for this premise under a clean prompt.

---

## 1. The eval findings (6 models, fully logged)

Two harnesses produced these; both live under `experiments/opd_profile/` and
`/old-data/apanda/tomi/outputs/`. Numbers are cross-validated between this session's runs and
the parallel session's independent samplings.

### 1.1 The three prompt regimes (this distinction is the crux of the whole investigation)
- **`raw_nothink`** (paper-faithful): V3 system prompt `/no_think\n<task>\n\nFormat:\nAnswer:
  [number]`, K-shot examples with filler in the assistant turn, **filler + `Answer:`
  prefilled**, model emits only the number. This is the paper's intended methodology — it
  *deliberately* soft-suppresses CoT so the filler effect is isolated.
- **`chat_hardoff`** (clean reference): same V3 block prompt but **`chat_template_kwargs=
  {"enable_thinking": false}`** (hard chat-template toggle, not the `/no_think` text). Higher,
  leak-free baseline = what the model can *actually* do with CoT genuinely off.
- **Diagnostic rule:** a lift that exists under `raw_nothink` but is **beaten by the
  `chat_hardoff` baseline** = **format-recovery** (not real capability). A lift that beats the
  `chat_hardoff` baseline = **genuine compute**.

### 1.2 pass@1 vs pass@8 (compute vs search)
- pass@1↑ **without** pass@8↑ = **compute concentration** (the buffer helps the model land the
  answer it could already reach — *distillable in principle*).
- pass@8↑ **without** pass@1↑ = **search expansion** (diversity; the paper's RL section's
  finding — *not* compute, *not* distillable into a deterministic forward pass).

### 1.3 Per-model verdicts (chat_hardoff 4×4 pass@1 = the clean-prompt test)

| Model | base | pause | Δp@1 | Δp@8 | baseline zone | verdict |
|---|---:|---:|---:|---:|---|---|
| Q3.5-35B-A3B | 0.85 | 0.78 | −0.07 | −0.01 | saturated | pause hurts |
| Q3.6-35B-A3B | 0.84 | 0.80 | −0.04 | −0.03 | saturated | flat |
| **Q3-235B-A22B base** | **0.46** | **0.59** | **+0.13** | **−0.04** | **responsive** | **distillable compute** ✦ |
| Q3-235B-A22B-Instruct-2507 | 0.73 | 0.71 | −0.02 | −0.03 | saturated-by-instruct | flat (pause); see §1.5 |
| Q3.5-397B-A17B BF16 | 0.98 | 0.99 | +0.01 | 0.00 | saturated | flat |

### 1.4 Format-recovery diagnostic (raw_nothink lift vs chat_hardoff baseline)

| Model | raw base | +pause | paper-style lift | chat_hardoff base | format-recovery only? |
|---|---:|---:|---:|---:|---|
| Q3.5-35B-A3B | 0.58 | 0.76 | +0.18 (=paper +14.9pp) | 0.85 | **yes** (clean > lifted) |
| Q3.6-35B-A3B | 0.60 | 0.69 | +0.09 (=paper +4.2pp) | 0.84 | **yes** |
| Q3-235B-A22B base | 0.59 | 0.67 | +0.08 | 0.46 | **NO — pause beats clean** ✦ |
| Q3-235B-Instruct-2507 | 0.75 | 0.71 | −0.04 | 0.73 | yes (no lift either; see §1.5) |
| Q3.5-397B BF16 | 0.30 | 0.30 | 0.00 (vs paper +20.1) | 0.98 | yes (unreproducible) |

### 1.5 Type-inversion — the paper's headline, reproduced on Q3-235B (17-filler V3 sweep, N=1000)
Single-filler probes (pause) are **insufficient** — pause is among the weakest fillers
(paper: |σ|≤2.0). The effect requires the *right* filler per regime. Full sweeps:

**Q3-235B-A22B-Instruct-2507** (paper's exact target variant; fs0 baseline 0.412, fs10 0.650):
- 0-shot winners (word-based): `random_tokens` **+9.7pp**, `counting` +8.0, `fruits` +7.2,
  `nato` +4.8; `pause` +2.0.
- 0-shot losers (numerical, catastrophic): `pi_digits` **−8.6**, `random_numbers` **−8.3**.
- 10-shot inversion: `random_numbers` **+2.1** (best), `fibonacci` +1.8, `ellipsis` +1.5;
  word-based go negative (`fruits` −3.8, `nato` −3.1).
- Every directional prediction of the paper reproduces.

**Q3-235B-A22B base** (fs0 baseline 0.123 — lower → larger magnitudes):
- 0-shot: `nato` **+13.8pp** (best), `states` +11.7, `lorem` +11.2, `counting` +5.1.
- 10-shot: `nato` **+9.5**, `random` +8.8, `fibonacci` +7.9.

### 1.6 The 397B anti-EOS mechanism (ellipsis, raw_nothink)
397B's raw-regime "lift" is **premature-EOS recovery**, not reasoning: ~33% of raw-regime
completions are *empty* (the model EOSes immediately after `Answer:`); ellipsis halves the
empty rate (4×4: 37.8% → 19.2%) and the pass@1 gain (0.25→0.46, +0.21) **equals** the
recovered empties. In `chat_hardoff` the 397B does 6-digit mult at 0.89 with 0% empty and
filler flat — the capability is fully there; the raw deficit is the EOS pathology.

### 1.7 Mechanistic conclusion for the eval side
Every "filler benefit" is a **generation/format artifact** — format-recovery (35B),
instruction-saturation (Instruct-2507 on pause), capability-saturation + anti-EOS (397B),
free-gen reasoning-leak (235B base under the old free-gen harness). The **one exception** is
the Q3-235B base compute-concentration cell (§1.3 ✦), which is the only genuine, non-saturated,
search-free signal — and exactly the paper's headline model+task.

---

## 2. The viable cell in detail — Q3-235B-A22B base, 4-digit mult

- **Native filler (untrained, inference-time):** clean `chat_hardoff` 4×4 pause **+0.13 pass@1
  / −0.04 pass@8** (other agent), **+0.10** at N=400 (this session's contested-cell re-run),
  +0.06 (this session's first N=100). Truth ≈ **+0.10–0.13**, robustly nonzero (~4 SE at N=400).
- **It beats the clean baseline** (raw+pause 0.66 > raw+base 0.59; and chat_hardoff pause 0.59
  > chat_hardoff base 0.46) → **not** format-recovery (unlike 35B).
- **pass@8 unmoved/down** → compute concentration, **not** search → *the signature compatible
  with distillation*.
- **Why this model and not 35B:** base-235B's V3 baseline (0.123) sits in the responsive zone;
  35Bs are saturated (0.84) or floored. Mechanism conjecture (unproven): per-shot output
  entropy is highest in the 0.3–0.7 zone, so extra forward-pass compute has the most marginal
  utility.

**Caveat on "distillable compute":** the label means the *signature* is compute-not-search
(so distillation is *not ruled out*). It does **not** mean distillation has succeeded — that
is the open question §6 tests.

---

## 3. The OPD distillation experiment (the load-bearing test)

### 3.1 What OPD self-distillation does here (teacher / student / masking / KL)
Self-distillation: teacher and student are the **same** base-235B weights; the teacher just
gets more context.
- **Teacher** = *frozen* base-235B (served by `teacher-sglang`). `teacher_cot_mode=insert`
  sequence: `prompt + CoT + nato_buffer + answer`. The per-prompt **CoT** (precomputed
  reasoning) is the teacher's "filler" (`teacher_cot_json` overrides `teacher_filler_text`).
  The teacher's hidden states *at the buffer positions are CoT-informed*.
- **Student** = base-235B *being trained* (the 8-node trainer). Sampled sequence:
  `prompt + nato_buffer + "Answer:" + answer`. **No CoT.**
- **Masking / KL** (config A, see §3.3): `supervise_student_cot=true` keeps the buffer in the
  teacher cache; `opd_supervise_buffer_only=true` → `mask_answer=true`. So **prompt, CoT, and
  answer are all masked; only the ~102 nato-buffer positions are supervised.** The loss = buffer
  logit-KL **+ hidden-match** (`opd_hidden_match_coef`) pushing the student's buffer hidden
  states toward the teacher's CoT-informed ones.
- **Key code refs** (`xorl-client-chat-completions/examples/on_policy_distillation.py`):
  insert/mask logic L170–303; `mask_answer=config.opd_supervise_buffer_only` L2196; `k_filler =
  len(prefill_tokens)` L1826/L1940 → **multi-token fillers (nato) are correctly aligned, no
  patch needed** (the old "multi-token NATO unsafe" memo is stale).

### 3.2 Why `nohm` (the only prior 235B distill run) is NOT a valid negative
`er-opd-235b-nohm` (2026-05-31, 39 steps; `experiments/encoded_reasoning/results/
qwen3_235b_self_distill/er-opd-235b-nohm/`): `buffer_delta` flat ~0.25–0.28. **Three confounds:**
1. **Suppressed baseline:** 0-shot + `</think>Answer:` cue → `acc_nopause` 0.13 (this is the
   real base-235B 0-shot capability, but the buffer "lift" is measured against it, and even
   `acc_pause` 0.40 stays *below* the model's true ability with a good prompt).
2. **Truncated CoT:** CoT median 518 tokens > `sample_packing_sequence_len=512` → the teacher
   saw only ~175 CoT tokens (often not reaching the answer) → garbage distillation target.
3. **No hidden-match:** `opd_hidden_match_coef=0.0` → buffer supervised by *logit-KL only*; the
   code itself documents that buffer-only "Pairs with `opd_hidden_match_coef>0`". The intended
   signal was off.
Plus the student reward-hacked (emit answer then ramble: "…buffer overflow attack…").

### 3.3 The clean retest — `er-opd-235b-clean4d` (config A)
Fixes all three confounds:

| dimension | nohm (confounded) | clean4d (this run) |
|---|---|---|
| shots | 0-shot + `</think>Answer:` | **0-shot** + clean `Answer:` cue (max headroom; nato strongest here) |
| CoT | truncated to ~175 tok (packing 512) | **full CoT** (`sample_packing_sequence_len=1024`, microbatch 64 → same memory) |
| buffer | ` pause`×100 (weakest filler) | **nato** ×3 ≈ **102 tokens** (base-235B's strongest filler) |
| supervision | buffer logit-KL only (`coef=0`) | **buffer logit-KL + hidden-match (`coef=1.0`)** |
| answer | masked (buffer-only) | masked (buffer-only) — **config A** |

**Config A vs B (decided A, with B as fallback):**
- **A** (`opd_supervise_buffer_only=true`, chosen): answer masked → **forces the buffer to be
  the only channel**. If accuracy rises it is *unambiguously* the buffer. Clean test.
- **B** (`opd_supervise_buffer_only=false`, fallback): supervise buffer **+ answer**. Gives the
  student an easy direct-answer path → it would learn input→answer in weights and ignore the
  buffer → `buffer_delta` flat = *false negative for the buffer*. Use only if A is inconclusive,
  attributing via `buffer_delta`.

**Success criteria (watch during training + a final pass@8 checkpoint eval):**
- `opd_hidden_match_loss` > 0 and decreasing → hidden-match engaging (sanity).
- **`buffer_delta` (acc_pause − acc_nopause) GROWS beyond the native ~+0.13, and/or pass@8
  rises** → the trained buffer carries distilled reasoning = **encoded reasoning works**.
- Flat `buffer_delta` → the buffer can't carry it even when forced (a *clean* negative this
  time — baseline de-suppressed, CoT full, hidden-match on). Then try B.

### 3.4 Live status (as of 2026-06-02, write time)
`er-opd-235b-clean4d` relaunching on the curated **nccl** pool (head + 2 workers + 2 samplers
Running, workers 3–7 scheduling into the freed nccl nodes; teachers + dispatch on `default`).
Watcher task `b37jt8qe2` armed (reports step-0 de-confound checks or any crash).

---

## 4. Methodology corrections (chronological honesty — do not repeat these)
1. **"No genuine lift cross-model" (early) — RETRACTED.** Was based on the RL trainer's
   *free-generation* prompt builder (`filler_tokens_rl.py`), miscategorized as the eval harness.
   Free-gen ≠ the paper's V3+prefill.
2. **V3 + `--prefill-answer` is the paper's *intended* method**, not a confound. Filler and
   no-filler both suppress CoT; the contrast is filler-vs-no-filler under matched suppression.
3. **`chat_hardoff` (hard `enable_thinking=False`) ≠ `/no_think` text.** The whole 0.82-vs-0.556
   "I can't reproduce it" saga was using the hard toggle while the paper uses the text-only soft
   suppression. Same model, that one knob flips the result.
4. **Single-filler (pause) probes are insufficient.** Pause is the weakest filler; "Instruct-2507
   kills the effect" was wrong — the full 17-filler sweep reproduces the type-inversion.
5. **0-shot 0.13 on base-235B is the REAL capability, not a bug.** Adding few-shot to "fix" it
   *reduces* the responsive-zone headroom; 0-shot is where nato is strongest.
6. **397B FP8 is BROKEN here** (multimodal `Qwen3_5MoeForConditionalGeneration`, emits `!`
   garbage). Use BF16 / TP=16. The "+20.1 ellipsis" was anti-EOS + possibly FP8 garbage.
7. **`nohm` "distillation says no" is NOT a valid negative** (§3.2). The distillation question is
   genuinely open → `clean4d`.

---

## 5. Infrastructure

### 5.1 Eval harness (built this session; reused by the parallel session)
- `experiments/opd_profile/k8s/build_eval_fleet.py` — generates **N sglang replicas + 1 SMG
  `--policy cache_aware` router** per model (cache_aware routes the shared prefix → KV reuse,
  ~3× per-GPU vs the old dispatch proxy).
- `experiments/opd_profile/eval_filler_fleet.py` — dual-regime (`chat_hardoff` + `raw_nothink`)
  × {base, pause/`--filler`} × {4×4,5×5,6×6} × pass@1/pass@8, **per-sample JSONL** to
  `results/filler_fleet/<run-id>/{samples.jsonl,summary.json}`. `--filler {pause,ellipsis,
  counting}`.
- Parallel session's wide sweeps: `/old-data/apanda/tomi/eval_chat_hardoff_filler_sweep.py`
  (17-filler chat_hardoff, N=1000) and `examples/run_fresh_filler0_eval_passk.py` +
  `run_task_all_fillers.sh` (paper-faithful V3 17-filler, N=1000). Driver default endpoint
  `research-common-08:12345` is NOT reachable from this cluster; the `LiteHF`/`LiteQwen3`
  renderers run it against any sglang.

### 5.2 OPD training stack (the 235B distill run)
~12 pods: 8-node **trainer** (head Job + 7 worker Pods, EP=8 intra-node + shard=64, adamw bf16,
p2p sync) + 2 **samplers** (sglang + Mooncake P2P weight-sync) + **dispatch** (SMG router) + 2
**teachers** (sglang prefill + CoT) + `teacher-smg`. Manifest: `experiments/opd_profile/k8s/
generated/er-opd-235b-clean4d.yaml`. Trainer config: `experiments/opd_profile/configs/
qwen3_235b_a22b_opd_8node_clean4d.yaml` (packing 1024). OPD client:
`xorl-client-chat-completions/examples/on_policy_distillation.py`.

### 5.3 Node-groups (LEARNED THE HARD WAY)
- **nccl** = 10 curated, clean-IB H100 nodes. The 8-node 235B trainer + 2 samplers (Mooncake
  IB) MUST run here; it's the only pool where the gang rendezvous + 235B init reliably succeeds
  (`nohm` proved it). All 10 needed → the whole pool.
- **default** = 42 nodes, but **uncurated** — has dead/flaky-IB nodes (`h100-113` is dead; both
  sessions' pods died on it). Teachers + dispatch (no Mooncake) run here fine. **Moving the
  8-node trainer here FAILED twice** (dead node, rank dropped mid-rendezvous). Do NOT put the
  trainer or Mooncake samplers on default.
- Known-bad nodes to exclude everywhere: **`h100-014`, `h100-050`, `h100-113`**.

### 5.4 Bring-up hazards + fixes (this session hit all of these)
1. **Dispatch on dead node `113`** → trainer-head hangs "Waiting for SMG router" → workers
   rendezvous-timeout. Fix: `nodeAffinity NotIn [113,014,050]` on the dispatch.
2. **Rendezvous timeout** ("6/8 clients joined, 901s"): workers scheduled late (nccl
   contention). Fix: free nodes so all 8 schedule fast; launch the gang **synchronized**.
3. **Engine-init timeout (1800s)** / **rank dropped mid-rendezvous**: gang fragility on
   slow/flaky nodes — use curated nccl, not default.
4. **`delete job … && apply` RACE** → the head Job silently isn't recreated (apply runs while
   old Job terminating). Fix: `kubectl delete job --wait=true`, poll until gone, *then* apply.
5. **Malformed `nodeAffinity` in the head *Job* template** → k8s "cannot unmarshal string into
   NodeSelectorTerm" → Job silently not created. Python `yaml.safe_load` accepts it but k8s
   rejects. Fix: don't hand-indent affinity into Job templates; the head is 1 node, run it
   without the exclusion + force-delete if it lands badly.
6. **Capacity:** the 235B stack needs all 10 nccl nodes; can't coexist with other nccl GPU
   work. Free idle fleets first.
7. **`pkill -f <pattern>` kills the calling shell** if the pattern is in its own argv (exit
   144). Use `pgrep … | guard $$` or just delete the backing pod.
8. **Instruct-2507 tokenizer bug:** snapshot ships only `merges.txt`; the base-tokenizer swap
   → `!`-garbage. Fix: `hf_hub_download` its own `tokenizer.json`/`vocab.json`.

### 5.5 Watcher
`/home/apanda/watch_clean4d.sh` (background): polls trainer-head for OOM/crash, then reports the
first OPD steps' `acc_nopause / acc_pause / buffer_delta / opd_kl` from `opd_profile.jsonl`.

---

## 6. How to run / continue

### 6.1 Eval a new model (filler sweep)
```
python experiments/opd_profile/k8s/build_eval_fleet.py --name eval-<m> \
  --model-path <snapshot> --served-name <name> --tp <N> --replicas 2 > /tmp/fleet.yaml
kubectl apply -n apanda -f /tmp/fleet.yaml
kubectl port-forward -n apanda svc/eval-<m>-router 18080:8080 &
python experiments/opd_profile/eval_filler_fleet.py --port 18080 --model <name> \
  --run-id <m> --filler pause --nprob 100 --ksamp 8
```
Protocol: calibrate `chat_hardoff` baseline first; if 0.3–0.7, sweep all fillers; if >0.8
saturated / <0.1 floored → wrong task (neither is "no effect").

### 6.2 The 235B OPD distill run
1. **Free the nccl pool** (the stack needs all 10): tear down any other nccl GPU pods.
2. `kubectl apply -n apanda -f experiments/opd_profile/k8s/generated/er-opd-235b-clean4d.yaml`
   (trainer+samplers → `node-group: nccl`; teachers+dispatch → `default`).
3. If a relaunch is needed: `kubectl delete job …trainer-head --wait=true`, poll until the Job
   is gone, force-delete trainer+sampler pods, **then** re-apply (avoid the race §5.4.4).
4. `bash /home/apanda/watch_clean4d.sh &` and watch `buffer_delta` + `opd_hidden_match_loss`.
5. If A's `buffer_delta` is flat after ~50+ steps → flip to **B**: set
   `opd_supervise_buffer_only=false` in the manifest's OPD-client args, relaunch the trainer.
6. Final verdict: serve the saved student checkpoint and run `eval_filler_fleet.py` pass@1 +
   pass@8, with vs without the nato buffer, against the native baseline.

### 6.3 Key knobs (OPD client args, in the manifest's trainer-head command)
`model_name=Qwen/Qwen3-235B-A22B` · `teacher_cot_json_path / prompts_json_path` ·
`student_prefill_text=<nato> student_prefill_count=3` · `student_prefill_suffix="Answer: "` ·
`teacher_cot_mode=insert supervise_student_cot=true` ·
`opd_supervise_buffer_only=true` (A) / `false` (B) · `opd_hidden_match_coef=1.0` ·
`num_steps=400 prompts_per_step=64 learning_rate=1e-5` · `sync_method=p2p`.

---

## 7. Open questions / next steps
1. **clean4d verdict** (in flight): does training move `buffer_delta` / pass@8 on Q3-235B base?
2. If A flat → **B** (unmask answer, attribute via `buffer_delta`).
3. **Clean-prompt 17-filler sweep on Q3-235B base** (only pause + V3 done) — close the
   responsive-zone loop on the clean prompt.
4. **Other tasks on Q3-235B** (arithmetic, var-counting) where the responsive zone may be wider
   than 4-digit mult.
5. The paper's **RL gives search (pass@8) not compute (pass@1)** — does distillation differ?
   That is exactly clean4d's bet.

---

## 8. File / data / config index
- **This master doc:** `experiments/opd_profile/ENCODED_REASONING_MASTER_RUNBOOK_2026_06_02.md`
- **My findings docs:** `experiments/opd_profile/OPD_FILLER_FLEET_FINDINGS_2026_06_01.md`,
  `OPD_ENCODED_REASONING_FINDINGS_2026_06_01.md`, `HANDOFF_OPD_ENCODED_REASONING.md`
- **Parallel session docs:** `/old-data/apanda/tomi/outputs/encoded_reasoning_final_
  reconciliation_20260601.md` (authoritative cross-model), `encoded_reasoning_final_summary_
  20260601.md`
- **Eval results:** `experiments/opd_profile/results/filler_fleet/<run>/` (samples.jsonl +
  summary.json); 17-filler sweeps under `/old-data/apanda/tomi/outputs/q3_235b*_v3_allfillers_*`,
  `q36_35b_chathardoff_*`, `q3_30b_instr_chathardoff_*`
- **235B OPD runs:** `experiments/encoded_reasoning/results/qwen3_235b_self_distill/{er-opd-235b-
  nohm,er-opd-235b-icl,er-opd-235b-clean4d}/` (opd_profile.jsonl + trainer logs)
- **Configs:** `experiments/opd_profile/configs/qwen3_235b_a22b_opd_8node{,_clean4d}.yaml`
- **Manifests:** `experiments/opd_profile/k8s/generated/er-opd-235b-clean4d.yaml`,
  `q3-235b-teach.yaml`, `q3-235bi-teach.yaml`
- **Eval scripts:** `eval_filler_fleet.py`, `/home/apanda/{paper_v3_prefill,passk_filler,
  tomi_raw_repro,raw_passk,dual_regime_passk}.py`, `build_fewshot_opd_data.py`
- **CoT data (Q3-235B, 4-digit):** `/shared/opd-coord/randnum_4digit_q3_235b_{cot,prompts}_
  filtered.json` (15979); fs5 variants `_fs5.json`
- **Models:** `/shared/huggingface/hub/models--Qwen--{Qwen3-235B-A22B (8efa617), Qwen3-235B-A22B-
  Instruct-2507 (ac9c66c, needs own tokenizer fetched), Qwen3.5-35B-A3B (59d61f3), Qwen3.6-35B-
  A3B (995ad96), Qwen3.5-397B-A17B (BF16; FP8 BROKEN)}`
- **SMG router:** `/home/apanda/smg-together-thunderagent-port/target/debug/smg`; runbook
  `experiments/opd_profile/runbooks/smg_router_swap.md`
- **Watcher:** `/home/apanda/watch_clean4d.sh`
