# SingleShot MTP Port Notes

Date: 2026-06-02

This is the first portable slice from `/home/apanda/singleshot` for Workstream 06.

## What Transfers Cleanly

- Batch rewrite: select the interleaved source-token layout, insert MTP special tokens, and mask loss to the predicted suffix positions.
- CPU-verifiable attention semantics: materialized block IDs and dense boolean masks matching the interleaved SingleShot rule.
- RoPE remap indices: the `0 1 2 3 2 3 ...` block-interleaving pattern used by the prototype.
- Logits-level MTP CE/entropy loss for tests and future runner integration.
- OPD micro-batch rewrite: AR teacher hidden-cache indices are gathered with the same SingleShot source indices as the student input, while ignored slots get harmless cache index zero.
- OPD padding hygiene: when the server packer pads OPD batches before runner execution, the MTP rewrite infers the real span from non-ignored labels and drops stale `cu_seq_lens_*`/`max_length_*` packing metadata after changing the sequence layout.
- OPD driver plumbing: `loss_fn_params.singleshot_mtp` enables trainer-side MTP, and `OPD_STUDENT_SAMPLING_PARAMS_JSON` passes runtime-specific sampler knobs through to SGLang.
- Qwen3.6 OPD smoke manifests: `experiments/opd_profile/k8s/qwen36_singleshot_mtp/` starts from the existing OPD pipeline manifest shape and wires the AR teacher plus MTP student trainer.
- Qwen3.6 EP4 k8s smoke: run `q36-mtp-ep4-pin2-041222` completed one OPD step with AR SGLang teacher, SGLang student rollout, XORL MTP student trainer, `loss=7.6357`, and `valid_tokens=8`. Profile artifact: `/shared/opd-coord/q36-mtp-ep4-pin2-041222/artifacts/qwen36_singleshot_mtp_ep4_opd_profile.jsonl`.

## What Is Not Ported Yet

- FlexAttention `BlockMask` construction.
- Sequence-parallel support for dense 4D MTP masks.
- Full Qwen3.5/Qwen3.6 parity: their `linear_attention` layers currently ignore the 4D interleaved mask, so Qwen3.6 OPD MTP is a smoke only unless `allow_linear_attention_smoke=true` is set explicitly. The smoke removes stale packed-sequence metadata so Qwen3.6 linear-attention short-conv does not consume pre-MTP padding boundaries, but those linear-attention layers still do not implement the SingleShot interleaved mask.
- SGLang speculative runtime changes. The `singleshot` runtime uses mask-token generation (`do_mtp`, `k_toks`, `mask_id`) rather than Slime-style MTP auxiliary layers; this repo now only passes custom sampling params through.

## Suggested Next Step

Add real interleaved-mask support to Qwen3.5/Qwen3.6 `GatedDeltaNet` layers or start parity validation on a dense/full-attention model first. For the Qwen3.6 plumbing smoke, use `examples/server/opd_singleshot_mtp_qwen36/`.
