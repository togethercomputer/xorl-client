# Full-vocab OPSD-via-ZORL inside SGLang

**Status:** design + build in progress (2026-06-07)
**Goal:** score ZORL/ES candidates by the **true full-vocabulary** OPSD KL against a
hinted teacher, computed **server-side on the GPU** inside the SGLang inference engine,
returning only a scalar reward per candidate. Replaces the abandoned top-k-logprob
approximation (which structurally cannot inject the teacher's answer — see below).

## Why SGLang (not the xorl trainer)

The full-vocab OPD loss (`forward_kl_full` / `reverse_kl_full`, `opd_loss.py`,
`opd_streaming_kl.py`) and the ZORL candidate machinery both already exist in the xorl
server `model_runner`, but they are **not wired together** — `start_zorl_generation`
exports candidate LoRAs and `apply_zorl_rewards` consumes *external* rewards; `opd_loss`
is only called by the gradient trainer. So a full-vocab ES reward is a build either way.

SGLang is the better home:
1. **Multi-LoRA batched forward is exactly the 1024-candidate workload** — SGLang's
   optimized strength; the trainer isn't built to forward 1024 distinct LoRAs.
2. **Full-logit / hidden-state plumbing already exists** (`LogitsProcessor(return_full_logits=True)`,
   `LogitsProcessorOutput.full_logits`, propagated through cuda-graph capture).
3. **ZORL lifecycle is native** here (`create_zorl_lora_candidates`, `start_zorl_generation`,
   `apply_zorl_rewards`).
4. Stays on the **one running 32-GPU pool**; no second system to stand up.

## The two crux decisions

### 1. Full vocab is necessary but not sufficient — direction matters
- **Reverse KL `KL(student‖teacher)`** (OPD-port default) is **mode-seeking**: a token the
  student gives ~0 mass contributes ~0 to the KL regardless of teacher mass, so it gives
  almost no pull toward the teacher's answer token. Full vocab does **not** fix this.
- **Forward KL `KL(teacher‖student)`** is **mode-covering**: it penalizes the student for
  *not* covering the teacher's answer → this is the one that **injects** the hint.
- **Default `kl_mode = forward_kl_full`** for the ES arm. Configurable to reverse.

### 2. Single-shot injects but memorizes; multi-turn generalizes
Forward-KL full-vocab on **single-shot** hint-free Wordle will push the student to put mass
on the answer for *train* puzzles (memorize) but cannot generalize (no information to find a
new hidden word). The **multi-turn feedback game** is what gives the student information so
the injected behavior generalizes. Single-shot is therefore run first only as a *diagnostic*
("can full-vocab inject at all"); the real target is multi-turn.

## Architecture

Teacher-forced / **shared-CoT** ES (fast: forward-only, no per-candidate generation):

```
1. Parent (current student LoRA) samples its hint-free CoT once → token suffix C.
   (multi-turn: C = the concatenated per-turn replies played against feedback.)
2. Teacher forward: base model (no LoRA) on [hinted-prompt + C], capture hidden states
   at the C positions → H_teacher. Computed ONCE; reused for every candidate.
3. Per candidate c:
     student forward: candidate-c LoRA on [hint-free-prompt + C], capture hidden at C → H_c
     reward_c = − Σ_{pos∈C} KL_full(teacher ‖ student)        # streamed over vocab chunks
4. build_zorl_update_from_rewards(rewards)  → parent update  (existing path)
```

Why shared-CoT: matches the existing ZORL `teacher_forced` structure; candidates are small
perturbations of the parent so scoring on the parent's on-policy CoT is a sound local
approximation; no per-candidate generation = forward-only = fast + batchable via multi-LoRA.

## Memory: stream the KL, never materialize `[pos, vocab]`

Ported from `opd_streaming_kl.py`: work on **hidden states** `[tok, hidden]` + the LM-head
weight `[vocab, hidden]`, chunk the **vocab** dimension, online-logsumexp for both
normalizers, then a second pass accumulating the KL. Qwen3 vocab = 151936; a 300-token CoT
would be ~180 MB of logits per seq if materialized — the chunked path keeps it to
`[tok, chunk]`. Forward-only (no autograd needed for ES).

## Build steps (tasks)
1. (this doc)
2. Port streaming vocab-chunked KL into the SGLang fork (forward-only, both directions) + unit test.
3. Capture scored-position hidden states in the SGLang forward (reuse pruned_states/lm_head; per-request flag).
4. `/opd_kl_score` endpoint: teacher forward (cached) + per-candidate student forwards → −KL reward; reuse create/apply ZORL lifecycle; multi-LoRA batch.
5. Client `score_mode=opsd_kl_full` + ZORL-WORDLE-017 yaml (forward_kl_full; multi-turn) + smoke + launch with in-loop multi-turn solve probe.

## Open knobs
- `kl_mode`: forward_kl_full (default) | reverse_kl_full
- `vocab_chunk_size`: 32768 default (tune for the 30B forward batch)
- single-shot diagnostic first, then multi-turn (the real target)
- shared-CoT now; per-candidate on-policy CoT later if the local approximation proves loose
