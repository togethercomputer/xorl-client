#!/usr/bin/env python3
"""MECHANISM-SANDBOX: shared instrumented-forward module (2026-07-06).

ONE implementation of every mechanism feature, imported by the trainer
(`mechanism_sandbox_train.py`), the unit gates (`mechanism_sandbox_gates.py`)
and any probe script — the recurrence track's shared-forward parity doctrine
(RECURRENCE_RETROFIT_DESIGN_20260705.md §6.2). The PRODUCTION eval path
(mechanism_export_hf.py -> sglang serve -> comp_bench_eval.py /
eval_math_suite.py) never imports this module.

Backends
--------
- MarinMechModel : marin 9.7B dense Qwen3 (37 layers, hidden 3840, 30 heads,
  no GQA). Custom single-pass forward built from recur_loop.RecurLayer (the
  parity-proven layer math). Every feature is EXACT here (all layers are full
  attention).
- Q35BMech       : Qwen3.6-35B-A3B hybrid (40 layers; full attention ONLY at
  layer idx [3,7,...,39]; the other 30 are GDN linear attention). Uses the
  UNMODIFIED HF forward (Qwen3_5MoeForCausalLM) + read-only capture hooks +
  custom 4D attention masks + inputs_embeds surgery. 4D-causal-mask
  pass-through == default-mask logits at delta 0.0 (tiny-model CPU gate).

  !! HYBRID CAVEAT (load-bearing; MECHANISM_SANDBOX_20260706.md §feature-b):
  attention-mask control constrains ONLY the 10 full-attention layers. The 30
  GDN layers are a causal recurrent scan (gated delta rule + depthwise conv,
  kernel 4) with NO position masking — content from a "blocked" span still
  reaches every later position through the recurrent state. On the 35B the
  bottleneck guarantee is RESTRICTED to full-attn layers; `gdn_leak_probe`
  quantifies the leak. On marin-dense the guarantee is exact (gate U3 asserts
  bit-identical answer logits under problem-content substitution when both
  answer- and slot-side blocks are active).

FSDP discipline (why the code is shaped this way): under FULL_SHARD each
wrapped unit's params exist only during ITS OWN forward call. Therefore
(i) ALL parameter-touching compute (layer forwards, lm_head / embed calls,
    the quantizer's logits metric) happens INSIDE the root module forward;
(ii) everything the loss assembly needs afterwards is captured as ACTIVATIONS
    via read-only module hooks (q_norm / k_norm / v_proj outputs = post-norm
    pre-RoPE q/k/v; decoder-layer inputs/outputs; rotary cos/sin), so
    `assemble_losses` never touches a parameter.

Feature list (all default-off; FeatureConfig.is_off() => parity path):
  (a) slot supervision at arbitrary positions: value targets (input!=label CE
      remap) AND hidden-state targets; SET-matching (Hungarian / Sinkhorn)
      per SCRAMBLED_ALIGNMENT_20260705.md (buffer = unordered reservoir);
      per-example k-sampling in the batch builders (the win condition's
      trained dose-response knob).
  (b) attention-mask control: block (query-region -> key-region) span pairs.
  (c) slot-output quantization: nearest-token-embedding snap with
      straight-through gradients; optional top-k soft mixture.
  (d) K/V matching loss at chosen layers/positions (post-q/k-norm, pre-RoPE
      by default — phase-free; post-RoPE requires position-aligned targets)
      + attention-target loss (answer-row attention over slots vs a target
      distribution, KL).
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass

sys.path.insert(0, "/home/apanda/xorl-client")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from recur_loop import (  # shared, parity-proven marin layer math — DO NOT FORK
    EOS_ID,
    PAUSE_ID,
    RecurLayer,
    _apply_rope,
)

# Qwen3.6-35B-A3B constants (buffer_sft.py / comp_bench_eval.py conventions)
Q35_BASE_SNAPSHOT = ("/shared/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/"
                     "995ad96eacd98c81ed38be0c5b274b04031597b0")
Q35_PAUSE_ID = 1873                         # "…" single token (verified in buffer_sft.py)
Q35_ANS_PREFIX_IDS = [198, 15666, 25, 220]  # "\nAnswer: " incl. trailing-space 220
Q35_SYSP = ("/no_think\nYou will be given an arithmetic expression and must "
            "compute its value. Answer immediately using the format "
            "'Answer: [ANSWER]' where [ANSWER] is just the numerical value, "
            "nothing else.")                # == buffer_sft ops6 == comp_bench_eval SYSP

REGIONS = ("problem", "prompt", "slots", "answer")


# =====================================================================
# Feature configuration
# =====================================================================

@dataclass
class FeatureConfig:
    # (a) slot supervision
    slot_value_coef: float = 0.0       # CE at slot positions with provided labels
    slot_hidden_coef: float = 0.0      # hidden-state targets at slot positions
    slot_hidden_layer: int = -1        # -1 = final post-norm hidden; else output of layer idx
    slot_hidden_metric: str = "mse"    # mse | cosine
    set_match: str = "none"            # none | hungarian | sinkhorn (order-free)
    sinkhorn_tau: float = 0.1
    sinkhorn_iters: int = 50
    # per-example k sampling (consumed by batch builders; recorded in run config)
    k_mode: str = "fixed"              # fixed | sample
    k_choices: str = ""                # "1x,2x,4x" (multiples of base k) or "8,16,32"
    k_p_first: float = 0.0             # P(choices[0]); 0 = uniform over all choices
    #   (attempt-1 common frame: k in {1x,2x,4x} with P(1x)=0.5, rest uniform)
    k_fixed_mult: int = 1              # k = mult * base_k when k_mode == "fixed"
    #   (arm-B warmup: fixed 4x so digit labels cover every intermediate)
    # (b) attention-mask control
    mask_blocks: str = ""              # "answer->problem;slots->problem" (query->key), blocked
    # (c) slot-output quantization
    quantize_slots: str = "off"        # off | hard | topk
    quantize_topk: int = 8
    quantize_tau: float = 1.0
    quantize_metric: str = "logits"    # logits (FSDP-safe, default) | cosine | dot
    quantize_passb_blocks: str = ""    # EXTRA mask blocks for the reader pass only:
    # the strict discrete bottleneck = mask_blocks="answer->problem" +
    # quantize_passb_blocks="slots->problem" (writer pass reads the problem,
    # reader-pass slots see ONLY the snapped tokens => the answer's entire
    # problem channel is the discrete token choice; exact on marin, GDN caveat
    # on the 35B)
    # (d) K/V matching + attention-target
    kv_match_coef: float = 0.0
    kv_layers: str = ""                # comma-separated layer indices
    kv_rope: str = "pre"               # pre | post   (pre = post-q/k-norm, pre-RoPE)
    kv_what: str = "kv"                # k | v | kv
    kv_metric: str = "mse"
    attn_target_coef: float = 0.0
    attn_layers: str = ""
    attn_target_kind: str = "uniform"  # uniform | file (per-item distributions)

    def is_off(self) -> bool:
        return (self.slot_value_coef == 0 and self.slot_hidden_coef == 0
                and self.mask_blocks == "" and self.quantize_slots == "off"
                and self.kv_match_coef == 0 and self.attn_target_coef == 0)

    @staticmethod
    def _parse_blocks(spec: str) -> list[tuple[str, str]]:
        out = []
        for part in spec.split(";"):
            part = part.strip()
            if not part:
                continue
            q, k = part.split("->")
            q, k = q.strip(), k.strip()
            assert q in REGIONS and k in REGIONS, (q, k)
            out.append((q, k))
        return out

    def blocks(self) -> list[tuple[str, str]]:
        return self._parse_blocks(self.mask_blocks)

    def blocks_passb(self) -> list[tuple[str, str]]:
        return self._parse_blocks(self.mask_blocks) + self._parse_blocks(
            self.quantize_passb_blocks)

    def layer_list(self, s: str) -> list[int]:
        return [int(x) for x in s.split(",") if x.strip() != ""]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @staticmethod
    def from_json(s: str) -> "FeatureConfig":
        return FeatureConfig(**json.loads(s))


def need_from_feats(feats: FeatureConfig) -> dict:
    need = {"kv_layers": set(feats.layer_list(feats.kv_layers)) if feats.kv_match_coef else set(),
            "attn_layers": set(feats.layer_list(feats.attn_layers)) if feats.attn_target_coef else set(),
            "hidden_layers": set()}
    if feats.slot_hidden_coef and feats.slot_hidden_layer >= 0:
        need["hidden_layers"].add(feats.slot_hidden_layer)
    return need


# =====================================================================
# (c) nearest-token-embedding snap with straight-through gradients
# =====================================================================

def snap_to_vocab(h: torch.Tensor, mode: str, topk: int, tau: float, metric: str,
                  lm_head_mod=None, embed_mod=None, embed_weight: torch.Tensor | None = None):
    """h [N, H] -> (out [N, H], idx). FSDP-safe when metric='logits':
    similarity via the lm_head MODULE call and rows via the embedding MODULE
    call (both re-gather their shards). metric='cosine'/'dot' need the raw
    embedding matrix — only legal on unwrapped models (gates/CPU); guarded.

    mode='hard': out = exact embedding row of the argmax token (bit-level
        equal to embed(idx)); straight-through: backward = identity into h.
    mode='topk': out = sum_topk softmax(sim/tau) * rows (fully differentiable).
    """
    if metric == "logits":
        assert lm_head_mod is not None and embed_mod is not None
        sim = lm_head_mod(h).float()
    else:
        assert embed_weight is not None, "cosine/dot need the raw embedding matrix"
        vocab = embed_mod.num_embeddings if embed_mod is not None else embed_weight.shape[0]
        assert embed_weight.shape[0] == vocab, \
            "embedding weight is FSDP-sharded — use metric='logits' under FSDP"
        hf = h.float()
        if metric == "cosine":
            sim = F.normalize(hf, dim=-1) @ F.normalize(embed_weight.float(), dim=-1).t()
        elif metric == "dot":
            sim = hf @ embed_weight.float().t()
        else:
            raise ValueError(metric)

    def rows(idx):
        if metric == "logits" or embed_weight is None:
            return embed_mod(idx)
        return embed_weight[idx]

    if mode == "hard":
        idx = sim.argmax(-1)                                   # [N]
        e = rows(idx).to(h.dtype)                              # exact embedding rows
        # STE: forward value is BIT-EXACTLY e (h - h.detach() == exact 0 forward);
        # backward is identity into h. (NOT h + (e-h).detach(), which is not
        # bit-exact in floating point.)
        out = e + (h - h.detach())
        return out, idx
    if mode == "topk":
        val, idx = sim.topk(topk, dim=-1)                      # [N, topk]
        w = F.softmax(val / tau, dim=-1)
        e = rows(idx).to(torch.float32)                        # [N, topk, H]
        out = (w.unsqueeze(-1).to(torch.float32) * e).sum(1).to(h.dtype)
        return out, idx
    raise ValueError(mode)


@torch.no_grad()
def _snap_stat_accum(capture: dict, h_slots: torch.Tensor, out_v: torch.Tensor,
                     idx: torch.Tensor) -> None:
    """Accumulate per-batch quantizer telemetry into capture['snap_stats']:
    token-id histogram (for snap-entropy / modal-fraction — arm B's degenerate-
    collapse detector) and sum/count of cos(h_slot, snapped_out) (the eval-time
    train/serve mismatch proxy: production serving does NOT snap)."""
    st = capture.setdefault("snap_stats", {"hist": {}, "cos_sum": 0.0, "n": 0})
    flat = idx.reshape(-1)
    for t in flat.tolist():
        st["hist"][t] = st["hist"].get(t, 0) + 1
    cos = F.cosine_similarity(h_slots.float(), out_v.float(), dim=-1)
    st["cos_sum"] += float(cos.sum())
    st["n"] += cos.numel()


def snap_stats_summary(capture: dict) -> dict | None:
    """entropy (bits) over the batch snap-token histogram, modal fraction,
    n_unique, mean cos(h, snap)."""
    st = capture.get("snap_stats")
    if not st or st["n"] == 0:
        return None
    import math as _m
    tot = sum(st["hist"].values())
    ent = -sum((c / tot) * _m.log2(c / tot) for c in st["hist"].values())
    return {"snap_entropy_bits": round(ent, 3),
            "snap_modal_frac": round(max(st["hist"].values()) / tot, 4),
            "snap_unique": len(st["hist"]),
            "snap_mean_cos": round(st["cos_sum"] / st["n"], 4)}


# =====================================================================
# (a) set matching: rectangular Hungarian (pure python) + Sinkhorn
# =====================================================================

def hungarian_assign(cost: torch.Tensor) -> list[int]:
    """Min-cost assignment of each of m rows (teacher targets) to a DISTINCT
    column of k >= m (student slots). Returns col index per row. O(m^2 k)
    potentials algorithm — pure python, no scipy (marin venv has none);
    verified against brute force in gate U1."""
    c = cost.detach().float().cpu()
    m, k = c.shape
    assert m <= k, f"need m<=k, got {m}x{k}"
    INF = float("inf")
    u = [0.0] * (m + 1)
    v = [0.0] * (k + 1)
    p = [0] * (k + 1)
    way = [0] * (k + 1)
    for i in range(1, m + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (k + 1)
        used = [False] * (k + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, k + 1):
                if used[j]:
                    continue
                cur = float(c[i0 - 1][j - 1]) - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(k + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    ans = [0] * m
    for j in range(1, k + 1):
        if p[j]:
            ans[p[j] - 1] = j - 1
    return ans


def sinkhorn_plan(cost: torch.Tensor, tau: float = 0.1, iters: int = 50) -> torch.Tensor:
    """Entropic-OT transport plan [m,k] (row marginals 1/m, col 1/k). DETACHED:
    the plan picks the pairing; gradients flow through the pair costs only
    (DETR-style)."""
    with torch.no_grad():
        m, k = cost.shape
        K = torch.exp(-cost.float() / max(tau, 1e-8))
        r = torch.full((m,), 1.0 / m, device=cost.device)
        c = torch.full((k,), 1.0 / k, device=cost.device)
        u = torch.ones_like(r)
        v = torch.ones_like(c)
        for _ in range(iters):
            u = r / (K @ v).clamp_min(1e-30)
            v = c / (K.t() @ u).clamp_min(1e-30)
        return torch.diag(u) @ K @ torch.diag(v)


def _pair_cost(student: torch.Tensor, teacher: torch.Tensor, metric: str) -> torch.Tensor:
    s = student.float()
    t = teacher.float()
    if metric == "mse":
        return torch.cdist(t, s, p=2).pow(2) / s.shape[-1]
    if metric == "cosine":
        return 1.0 - F.normalize(t, dim=-1) @ F.normalize(s, dim=-1).t()
    raise ValueError(metric)


def set_match_loss(student: torch.Tensor, teacher: torch.Tensor, metric: str = "mse",
                   matcher: str = "hungarian", tau: float = 0.1, iters: int = 50):
    """Order-free supervision of k student slot states with m<=k teacher
    targets: every target lands on a DISTINCT slot; extra slots are free."""
    cost = _pair_cost(student, teacher, metric)       # [m,k], differentiable
    if matcher == "hungarian":
        with torch.no_grad():
            assign = hungarian_assign(cost)
        loss = cost[torch.arange(len(assign), device=cost.device),
                    torch.tensor(assign, device=cost.device)].mean()
        return loss, assign
    if matcher == "sinkhorn":
        plan = sinkhorn_plan(cost, tau, iters)        # [m,k] detached
        loss = (plan.to(cost.dtype) * cost).sum() * cost.shape[0] / plan.sum().clamp_min(1e-30)
        return loss, plan
    raise ValueError(matcher)


def aligned_loss(student: torch.Tensor, teacher: torch.Tensor, metric: str = "mse"):
    """Position-paired slot supervision (slot i <-> teacher row i)."""
    assert student.shape[0] == teacher.shape[0], (student.shape, teacher.shape)
    s, t = student.float(), teacher.float()
    if metric == "mse":
        return F.mse_loss(s, t)
    if metric == "cosine":
        return (1.0 - F.cosine_similarity(s, t, dim=-1)).mean()
    raise ValueError(metric)


# =====================================================================
# (d) K/V matching + attention-target losses (pure activation math)
# =====================================================================

def kv_match_loss(student_k, student_v, teacher_k, teacher_v, metric: str = "mse"):
    parts = []
    for s, t in ((student_k, teacher_k), (student_v, teacher_v)):
        if s is None or t is None:
            continue
        assert s.shape == t.shape, (s.shape, t.shape)
        if metric == "mse":
            parts.append(F.mse_loss(s.float(), t.float()))
        elif metric == "cosine":
            parts.append((1 - F.cosine_similarity(s.float(), t.float(), dim=-1)).mean())
        else:
            raise ValueError(metric)
    assert parts, "kv_match_loss called with nothing to match"
    return torch.stack(parts).mean()


def attn_target_kl(probs_over_slots: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """KL(target || student). probs_over_slots [n_rows, k] renormalized over
    slot cols; target [k] or [n_rows, k]."""
    p = probs_over_slots.clamp_min(1e-9)
    t = target.to(p.dtype)
    if t.dim() == 1:
        t = t.unsqueeze(0).expand_as(p)
    t = t.clamp_min(1e-9)
    return (t * (t.log() - p.log())).sum(-1).mean()


# =====================================================================
# region spans / mask helpers
# =====================================================================

def template_frame_span(ids_with: list[int], ids_empty: list[int]) -> tuple[int, int]:
    """Token span of the variable content inside a rendered template, by
    longest-common-prefix/suffix diff against the SAME template rendered with
    empty content. Boundary BPE merges can shave one token at each edge —
    acceptable; the leak probe substitutes using the SAME span so the
    instrument is self-consistent."""
    p = 0
    while p < min(len(ids_with), len(ids_empty)) and ids_with[p] == ids_empty[p]:
        p += 1
    s = 0
    while (s < min(len(ids_with), len(ids_empty)) - p
           and ids_with[len(ids_with) - 1 - s] == ids_empty[len(ids_empty) - 1 - s]):
        s += 1
    return p, len(ids_with) - s


def region_cols(region: str, spans: dict, L: int) -> tuple[int, int]:
    if region == "problem":
        return spans["problem"]
    if region == "prompt":
        return (0, spans["prompt_end"])
    if region == "slots":
        return spans["slots"]
    if region == "answer":
        return spans["answer"]
    raise ValueError(region)


def apply_mask_blocks_bool(mask: torch.Tensor, blocks: list[tuple[str, str]],
                           spans_per_row: list[dict]) -> torch.Tensor:
    """mask [B,1,L,L] bool (True=attend); blocked rectangles set False per row.
    Diagonal forced True (never an empty softmax row)."""
    m = mask.clone()
    B, _, L, _ = m.shape
    for b in range(B):
        for qr, kr in blocks:
            q0, q1 = region_cols(qr, spans_per_row[b], L)
            k0, k1 = region_cols(kr, spans_per_row[b], L)
            m[b, :, q0:q1, k0:k1] = False
    idx = torch.arange(L, device=m.device)
    m[:, :, idx, idx] = True
    return m


def bool_mask_to_float(mask: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    return torch.zeros_like(mask, dtype=dtype).masked_fill(~mask, torch.finfo(dtype).min)


def full_bool_mask(real: torch.Tensor) -> torch.Tensor:
    B, L = real.shape
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=real.device))
    allow = causal.unsqueeze(0) & real[:, None, :]
    idx = torch.arange(L, device=real.device)
    allow = allow.clone()
    allow[:, idx, idx] = True
    return allow.unsqueeze(1)


# =====================================================================
# capture mixin (hook-based, read-only; shared by both backends)
# =====================================================================

class CaptureMixin:
    """Read-only activation capture via module hooks, gated by an active spec
    so features-off forwards are bit-identical (gate U4). Capture points:
      ('hin', l)  decoder-layer input        [B,T,H]
      ('h', l)    decoder-layer output       [B,T,H]
      ('qn', l)   q_norm output = post-norm PRE-RoPE q   [B,T,heads,hd]
      ('kn', l)   k_norm output = post-norm PRE-RoPE k   [B,T,kv_heads,hd]
      ('v', l)    v_proj output              [B,T,kv_heads*hd]
      'rope'      (cos, sin) from the rotary module
    """

    def _cap_init(self):
        self._spec: dict = {}
        self._cap: dict = {}
        self._handles = []

    def _mk_out_hook(self, key, idx):
        def hook(_mod, _inp, out):
            if idx in self._spec.get(key, ()):
                t = out[0] if isinstance(out, tuple) else out
                self._cap[(key, idx)] = t
        return hook

    def _mk_pre_hook(self, key, idx):
        def hook(_mod, args, kwargs):
            if idx in self._spec.get(key, ()):
                self._cap[(key, idx)] = args[0] if args else kwargs["hidden_states"]
        return hook

    def _hook_attn_modules(self, layer_idx, attn_mod, want_qn, want_kn, want_v):
        if want_qn:
            self._handles.append(attn_mod.q_norm.register_forward_hook(
                self._mk_out_hook("qn", layer_idx)))
        if want_kn:
            self._handles.append(attn_mod.k_norm.register_forward_hook(
                self._mk_out_hook("kn", layer_idx)))
        if want_v:
            self._handles.append(attn_mod.v_proj.register_forward_hook(
                self._mk_out_hook("v", layer_idx)))

    def _hook_rotary(self, rotary_mod):
        def hook(_mod, _inp, out):
            if self._spec.get("rope"):
                self._cap["rope"] = out
        self._handles.append(rotary_mod.register_forward_hook(hook))

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    @contextmanager
    def capturing(self, hidden_layers=(), kv_layers=(), attn_layers=(), rope=False):
        self._spec = {"h": set(hidden_layers),
                      "hin": set(),
                      "qn": set(attn_layers),
                      "kn": set(list(kv_layers) + list(attn_layers)),
                      "v": set(kv_layers),
                      "rope": rope or bool(attn_layers)}
        self._cap = {}
        try:
            yield self._cap
        finally:
            self._spec = {}


# =====================================================================
# marin backend
# =====================================================================

class MarinMechModel(nn.Module, CaptureMixin):
    """Single-pass instrumented forward over the marin dense stack. Layer math
    = recur_loop.RecurLayer (shared with the recurrence track). NO adapter /
    no loop — this is the mechanism sandbox, not the retrofit."""

    def __init__(self, hf_model):
        super().__init__()
        self._cap_init()
        self.config = hf_model.config
        m = hf_model.model
        self.n_layers = len(m.layers)   # 37 real; tiny models allowed in gates
        self.embed_tokens = m.embed_tokens
        self.layers = nn.ModuleList(RecurLayer(l) for l in m.layers)
        self.norm = m.norm
        self.rotary = m.rotary_emb
        self.lm_head = hf_model.lm_head
        self.head_scaling = None  # filled by install_hooks per attn layer

    def install_hooks(self, hidden_layers=(), kv_layers=(), attn_layers=()):
        """Call BEFORE FSDP wrapping. Hooks are inert unless a spec is active."""
        for l in set(hidden_layers):
            self._handles.append(self.layers[l].register_forward_hook(
                self._mk_out_hook("h", l)))
        for l in set(list(kv_layers) + list(attn_layers)):
            a = self.layers[l].inner.self_attn
            self._hook_attn_modules(l, a, want_qn=(l in set(attn_layers)),
                                    want_kn=True, want_v=(l in set(kv_layers)))
        # rotary is called once in forward; capture is direct (no hook needed)

    # ---------- core ----------
    def _mask(self, real, feats: FeatureConfig, spans_per_row, passb: bool = False):
        mask = full_bool_mask(real)
        blocks = (feats.blocks_passb() if passb else feats.blocks()) if feats is not None else []
        if blocks:
            mask = apply_mask_blocks_bool(mask, blocks, spans_per_row)
        return mask

    def _run_layers(self, h, cos, sin, mask, need_kv_all=False):
        caches = [] if need_kv_all else None
        for i, layer in enumerate(self.layers):
            h, kv = layer(h, cos, sin, mask, need_kv=need_kv_all)
            if need_kv_all:
                caches.append(kv)
        return h, caches

    def forward(self, batch: dict, feats: FeatureConfig, need: dict | None = None,
                split_backward: bool = False):
        """ROOT forward (call through the FSDP wrapper). Returns dict:
        h_final [B,L,H], ce, ce_ntok, slot_value (if labels present), capture
        (activation-only), mask/cos/sin (for the assembly step).
        split_backward: cut the graph at the snapped embeddings (see
        Q35BMech.forward docstring) — out gains h1_graph/h1_leaf and the
        trainer runs the two-backward schedule."""
        need = need or {}
        input_ids, real, pos = batch["input_ids"], batch["real"], batch["pos"]
        spans = batch["spans"]
        B, L = input_ids.shape
        mask = self._mask(real, feats, spans)
        h0 = self.embed_tokens(input_ids)
        cos, sin = self.rotary(h0, pos)
        h1_graph = h1_leaf = None

        with self.capturing(hidden_layers=need.get("hidden_layers", ()),
                            kv_layers=need.get("kv_layers", ()),
                            attn_layers=need.get("attn_layers", ())) as cap:
            capture: dict = {}
            if feats.quantize_slots != "off":
                # pass A (writer): slot states == single-pass states (causal)
                hA, _ = self._run_layers(h0, cos, sin, mask)
                hA_final = self.norm(hA)
                capture.update({("passA",) + k: v for k, v in cap.items()
                                if isinstance(k, tuple)})
                capture[("passA", "h_final")] = hA_final
                cap.clear()
                h1 = h0.clone()
                snap_idx = []
                for b in range(B):
                    s0, s1 = spans[b]["slots"]
                    if s1 <= s0:
                        snap_idx.append(None)
                        continue
                    out_v, idx = snap_to_vocab(
                        hA_final[b, s0:s1], mode=feats.quantize_slots,
                        topk=feats.quantize_topk, tau=feats.quantize_tau,
                        metric=feats.quantize_metric,
                        lm_head_mod=self.lm_head, embed_mod=self.embed_tokens,
                        embed_weight=(self.embed_tokens.weight
                                      if feats.quantize_metric != "logits" else None))
                    h1[b, s0:s1] = out_v.to(h1.dtype)
                    snap_idx.append(idx)
                    _snap_stat_accum(capture, hA_final[b, s0:s1], out_v, idx)
                capture["snap_idx"] = snap_idx
                mask = self._mask(real, feats, spans, passb=True)  # reader-pass blocks
                if split_backward:
                    h1_graph = h1                       # pass-A side of the cut
                    h1 = h1.detach().requires_grad_(True)
                    h1_leaf = h1                        # pass-B side (leaf)
                h, _ = self._run_layers(h1, cos, sin, mask)        # pass B (reader)
            else:
                h, _ = self._run_layers(h0, cos, sin, mask)
            capture.update(dict(cap))
        h_final = self.norm(h)
        out = {"h_final": h_final, "capture": capture, "mask": mask,
               "cos": cos, "sin": sin,
               "scaling": self.layers[0].inner.self_attn.scaling, "n_rep": 1}
        if h1_graph is not None:
            out["h1_graph"] = h1_graph
            out["h1_leaf"] = h1_leaf
        labels = batch.get("labels")
        if labels is not None:
            sel = labels != -100
            logits = self.lm_head(h_final[sel]).float()
            out["ce"] = F.cross_entropy(logits, labels[sel], reduction="mean")
            out["ce_ntok"] = sel.sum()
        sl = batch.get("slot_labels")
        if sl is not None and (sl != -100).any():
            selv = sl != -100
            hw = capture.get(("passA", "h_final"), h_final)      # writer states
            logits_v = self.lm_head(hw[selv]).float()
            out["slot_value"] = F.cross_entropy(logits_v, sl[selv], reduction="mean")
        return out

    def rope_for_assembly(self, out):
        return out["cos"], out["sin"], "marin"

    @torch.no_grad()
    def generate_greedy_masked(self, batch: dict, feats: FeatureConfig,
                               max_new_tokens: int = 16, eos_id: int = EOS_ID):
        """Offline greedy decode honoring the TRAINING-time features that
        production serving cannot apply: mask blocks maintained during decode
        AND (W2-2) the two-pass slot QUANTIZER at prefill — pass A writes,
        snap, pass B (reader-pass blocks) builds the KV the decode reads.
        Diagnostic instrument only (the sanctioned sandbox exception: it
        measures the sandbox-vs-production gap itself). Unwrapped models only."""
        input_ids, real, pos = batch["input_ids"], batch["real"], batch["pos"]
        spans = batch["spans"]
        B, L = input_ids.shape
        mask = self._mask(real, feats, spans)
        h0 = self.embed_tokens(input_ids)
        cos, sin = self.rotary(h0, pos)
        if feats.quantize_slots != "off":
            with torch.no_grad():
                hA, _ = self._run_layers(h0, cos, sin, mask)      # pass A (writer)
                hA_final = self.norm(hA)
                h1 = h0.clone()
                for b in range(B):
                    s0, s1 = spans[b]["slots"]
                    if s1 <= s0:
                        continue
                    out_v, _idx = snap_to_vocab(
                        hA_final[b, s0:s1], mode=feats.quantize_slots,
                        topk=feats.quantize_topk, tau=feats.quantize_tau,
                        metric=feats.quantize_metric,
                        lm_head_mod=self.lm_head, embed_mod=self.embed_tokens,
                        embed_weight=(self.embed_tokens.weight
                                      if feats.quantize_metric != "logits" else None))
                    h1[b, s0:s1] = out_v.to(h1.dtype)
            h0 = h1
            mask = self._mask(real, feats, spans, passb=True)     # reader-pass blocks
        h, caches = self._run_layers(h0, cos, sin, mask, need_kv_all=True)
        h_final = self.norm(h)
        tok = self.lm_head(h_final[:, -1]).float().argmax(-1)
        gen = torch.full((B, max_new_tokens), eos_id, dtype=torch.long, device=input_ids.device)
        done = torch.zeros(B, dtype=torch.bool, device=input_ids.device)
        real_run = real.clone()
        pos_next = pos[:, -1] + 1
        blocked_cols = []
        for b in range(B):
            colmask = torch.zeros(L, dtype=torch.bool, device=input_ids.device)
            for qr, kr in feats.blocks():
                if qr == "answer":
                    k0, k1 = region_cols(kr, spans[b], L)
                    colmask[k0:k1] = True
            blocked_cols.append(colmask)
        for step in range(max_new_tokens):
            tok = torch.where(done, torch.full_like(tok, eos_id), tok)
            gen[:, step] = tok
            done = done | (tok == eos_id)
            if bool(done.all()) or step == max_new_tokens - 1:
                break
            h = self.embed_tokens(tok[:, None])
            c2, s2 = self.rotary(h, pos_next[:, None])
            real_run = torch.cat([real_run, (~done)[:, None]], dim=1)
            Tkv = real_run.shape[1]
            m = real_run[:, None, None, :].clone()
            for b in range(B):
                m[b, :, :, :L] &= ~blocked_cols[b][None, None, :]
            m[:, :, :, Tkv - 1] = True
            for i in range(self.n_layers):
                h, kv = self.layers[i](h, c2, s2, m, kv_prefix=caches[i], need_kv=True)
                caches[i] = (torch.cat([caches[i][0], kv[0]], dim=2),
                             torch.cat([caches[i][1], kv[1]], dim=2))
            tok = self.lm_head(self.norm(h[:, -1])).float().argmax(-1)
            pos_next = pos_next + 1
        return gen


# =====================================================================
# Qwen3.6-35B-A3B backend (HF forward + hooks; hybrid caveat in header)
# =====================================================================

class Q35BMech(nn.Module, CaptureMixin):
    """Root module for FSDP: forward(batch, feats, need) -> out dict (same
    schema as MarinMechModel.forward). Uses the unmodified HF TextModel."""

    def __init__(self, causal_lm):
        super().__init__()
        self._cap_init()
        self.lm = causal_lm
        cfg = causal_lm.config
        self.full_attn_layers = [i for i, t in enumerate(cfg.layer_types)
                                 if t == "full_attention"]
        self.gdn_layers = [i for i, t in enumerate(cfg.layer_types)
                           if t == "linear_attention"]
        # SDPA requires additive-mask dtype == query dtype ("invalid dtype for
        # bias" on the real bf16 model — caught by gate U6, 2026-07-06). Masks
        # are built INSIDE forward, so FSDP's root input-cast never touches
        # them: unwrapped models derive the dtype from the embed weight; the
        # FSDP trainer MUST override mask_dtype to MixedPrecision.param_dtype.
        self.mask_dtype: torch.dtype | None = None

    def _mask_dtype(self) -> torch.dtype:
        return self.mask_dtype or self.lm.model.embed_tokens.weight.dtype

    def install_hooks(self, hidden_layers=(), kv_layers=(), attn_layers=()):
        """Register capture hooks ONCE, BEFORE FSDP wrap / checkpoint wrap.
        kv/attn layers MUST be full-attention layers (GDN has no K/V)."""
        for l in list(kv_layers) + list(attn_layers):
            assert l in self.full_attn_layers, \
                f"layer {l} is GDN (linear attention) — no K/V or attention map exists"
        layers = self.lm.model.layers
        for l in set(hidden_layers):
            self._handles.append(layers[l].register_forward_hook(self._mk_out_hook("h", l)))
        for l in set(list(kv_layers) + list(attn_layers)):
            self._hook_attn_modules(l, layers[l].self_attn,
                                    want_qn=(l in set(attn_layers)),
                                    want_kn=True, want_v=(l in set(kv_layers)))
        self._hook_rotary(self.lm.model.rotary_emb)

    # ---------- mask ----------
    @staticmethod
    def build_mask4d(real: torch.Tensor, blocks, spans_per_row, dtype) -> torch.Tensor:
        """Causal + right-pad + span blocks, additive float [B,1,L,L].
        GDN NOTE: the HF GDN path ignores 4D masks entirely (its padding fix
        requires a 2D mask), so this constrains only full-attention layers.
        Batch layouts must be right-pad-ONLY: trailing pads cannot affect real
        positions in a causal scan, so GDN stays exact without a pad mask."""
        allow = full_bool_mask(real)
        if blocks:
            allow = apply_mask_blocks_bool(allow, blocks, spans_per_row)
        return bool_mask_to_float(allow, dtype=dtype)

    def _plain(self, input_ids=None, inputs_embeds=None, attention_mask=None,
               position_ids=None):
        out = self.lm.model(input_ids=input_ids, inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask, position_ids=position_ids,
                            use_cache=False)
        return out.last_hidden_state

    def forward(self, batch: dict, feats: FeatureConfig, need: dict | None = None,
                split_backward: bool = False):
        """split_backward (D3 spike, 2026-07-07): cut the autograd graph at the
        snapped embeddings (pass-A output -> pass-B input). The trainer then
        runs TWO backwards — pass B first, then pass A seeded with the leaf's
        grad. The STE identity makes the composition exact for the hard snap;
        the point is FSDP1 memory: in the JOINT graph every layer's params are
        used twice (pass A + pass B), so reduce-scatter fires only after the
        pass-A backward and up to ~40 unsharded bf16 layer-grads (~1.56 GiB
        each) pile up — the measured 35B OOM (ATTEMPT1_PREREG_20260706.md:
        206-223). With the cut, each backward sees each layer once and
        reduce-scatters promptly. Gradient equality vs the joint graph is
        gated by ms_split_grad_gate.py BEFORE any 35B use."""
        need = need or {}
        input_ids, real = batch["input_ids"], batch["real"]
        spans = batch["spans"]
        pos2d = batch["pos"]
        B, L = input_ids.shape
        blocks = feats.blocks()
        if blocks:
            attn_mask = self.build_mask4d(real, blocks, spans, self._mask_dtype())
        else:
            attn_mask = real.to(torch.long)   # 2D padding mask -> HF default causal
        h1_graph = h1_leaf = None
        with self.capturing(hidden_layers=need.get("hidden_layers", ()),
                            kv_layers=need.get("kv_layers", ()),
                            attn_layers=need.get("attn_layers", ())) as cap:
            capture: dict = {}
            if feats.quantize_slots != "off":
                hA = self._plain(input_ids=input_ids, attention_mask=attn_mask,
                                 position_ids=pos2d)
                capture.update({("passA",) + k: v for k, v in cap.items()
                                if isinstance(k, tuple)})
                capture[("passA", "h_final")] = hA        # post-final-norm
                if "rope" in cap:
                    capture[("passA", "rope")] = cap["rope"]
                cap.clear()
                e0 = self.lm.model.embed_tokens(input_ids)
                h1 = e0.clone()
                snap_idx = []
                for b in range(B):
                    s0, s1 = spans[b]["slots"]
                    if s1 <= s0:
                        snap_idx.append(None)
                        continue
                    out_v, idx = snap_to_vocab(
                        hA[b, s0:s1], mode=feats.quantize_slots,
                        topk=feats.quantize_topk, tau=feats.quantize_tau,
                        metric=feats.quantize_metric,
                        lm_head_mod=self.lm.lm_head, embed_mod=self.lm.model.embed_tokens,
                        embed_weight=(self.lm.model.embed_tokens.weight
                                      if feats.quantize_metric != "logits" else None))
                    h1[b, s0:s1] = out_v.to(h1.dtype)
                    snap_idx.append(idx)
                    _snap_stat_accum(capture, hA[b, s0:s1], out_v, idx)
                capture["snap_idx"] = snap_idx
                blocks_b = feats.blocks_passb()
                if blocks_b:
                    attn_mask = self.build_mask4d(real, blocks_b, spans, self._mask_dtype())
                if split_backward:
                    h1_graph = h1                       # pass-A side of the cut
                    h1 = h1.detach().requires_grad_(True)
                    h1_leaf = h1                        # pass-B side (leaf)
                h = self._plain(inputs_embeds=h1, attention_mask=attn_mask,
                                position_ids=pos2d)
            else:
                h = self._plain(input_ids=input_ids, attention_mask=attn_mask,
                                position_ids=pos2d)
            capture.update(dict(cap))
        a0 = self.lm.model.layers[self.full_attn_layers[0]].self_attn
        out = {"h_final": h, "capture": capture,
               "mask": attn_mask if isinstance(attn_mask, torch.Tensor) and attn_mask.dim() == 4 else None,
               "rope": capture.get("rope"),
               "scaling": a0.scaling,
               "n_rep": a0.num_key_value_groups}
        if h1_graph is not None:
            out["h1_graph"] = h1_graph
            out["h1_leaf"] = h1_leaf
        labels = batch.get("labels")
        if labels is not None:
            sel = labels != -100
            logits = self.lm.lm_head(h[sel]).float()
            out["ce"] = F.cross_entropy(logits, labels[sel], reduction="mean")
            out["ce_ntok"] = sel.sum()
        sl = batch.get("slot_labels")
        if sl is not None and (sl != -100).any():
            selv = sl != -100
            hw = capture.get(("passA", "h_final"), h)
            logits_v = self.lm.lm_head(hw[selv]).float()
            out["slot_value"] = F.cross_entropy(logits_v, sl[selv], reduction="mean")
        return out

    def rope_for_assembly(self, out):
        rope = out.get("rope") or out["capture"].get("rope") \
            or out["capture"].get(("passA", "rope"))
        assert rope is not None, "rotary capture missing (attn feature not active?)"
        return rope[0], rope[1], "q35b"


def fsdp_split_backward_save(model) -> list:
    """Split-backward support (D3 spike, 2026-07-07; torch 2.10 FSDP1).

    The split schedule runs TWO backwards over ONE root forward. FSDP1's
    first-backward teardown destroys three things the second backward needs;
    call this right AFTER the forward (captures state while it is alive) and
    `fsdp_split_backward_rearm` between the backwards (restores it):

      1. `handle._needs_pre_backward_unshard` — reset False by the
         post-backward final callback, so bwd#2's pre-backward hooks would
         skip re-gathering (dangling flat-param views: "setStorage ...
         storage size of 0", measured gate pod 2026-07-07 ~14:00Z).
      2. `flat_param._tensors[i]` — the forward-saved view Tensors that
         BACKWARD_PRE re-points (`tensor.data = view`); cleared to None in
         BACKWARD_POST `_use_sharded_views` ("unneeded now"), so bwd#2's
         unshard dies on `NoneType.data` (measured ~14:08Z).
      3. `flat_param._post_backward_hook_state` — the reduce-scatter hook on
         the AccumulateGrad, REMOVED by `_finalize_params`; without
         re-registration bwd#2 grads would neither reduce-scatter nor reach
         the sharded grad accumulator (silent-wrong gradients). We capture
         the forward-registered AccumulateGrad object and re-hook it.

    End-to-end gradient equality of the whole surgery vs the joint graph is
    what ms_split_grad_gate.py gates (tiny-FSDP fp32 exact; marin fp32 w8;
    bf16-MP report). No-op (returns []) for non-FSDP models."""
    states = getattr(model, "_all_fsdp_states", None) or []
    saved = []
    for st in states:
        h = getattr(st, "_handle", None)
        if h is None:
            continue
        fp = h.flat_param
        tensors = getattr(fp, "_tensors", None)
        pbhs = getattr(fp, "_post_backward_hook_state", None)
        saved.append((st, h,
                      list(tensors) if tensors is not None else None,
                      pbhs[0] if pbhs else None))
    return saved


def _walk_accumulate_grads(roots: list) -> list:
    """BFS the autograd graph from root TENSORS; return AccumulateGrad nodes.
    !! Hold a strong reference to EVERY visited node wrapper: grad_fn python
    wrappers are created on demand and freed when dropped, so their id()s
    get recycled and poison the seen-set (measured 2026-07-07 ~15:0xZ: the
    walk visited 2 nodes of a multi-thousand-node graph, found 0
    accumulators, and silently disabled every reduce-scatter hook)."""
    stack = [t.grad_fn for t in roots if t is not None and t.grad_fn is not None]
    seen: set = set()
    keep: list = []          # strong refs -> stable id()s
    accs = []
    while stack:
        fn = stack.pop()
        if fn is None or id(fn) in seen:
            continue
        seen.add(id(fn))
        keep.append(fn)
        if type(fn).__name__ == "AccumulateGrad":
            accs.append(fn)
            continue
        for nxt, _ in fn.next_functions:
            if nxt is not None and id(nxt) not in seen:
                stack.append(nxt)
    return accs


_SPLIT_JIT = {"active": False, "hooked": set(), "handles": [], "acc_refs": [],
              "fired": 0}


def _split_jit_prehook(st, h, _mod, _args):
    """Just-in-time accumulator hook for ACTIVATION-CHECKPOINTED FSDP units:
    NO_REENTRANT checkpoint builds the inner graph at RECOMPUTE time (during
    backward), so the layer's AccumulateGrad does not exist in the forward
    graph (tinyadj: split grads n=1 of 140) and, under MP, its identity is
    only knowable after the backward's own unshard (.data dtype swaps create
    a fresh TensorImpl -> fresh accumulator; the v2 post-forward capture
    changed nothing). The recompute re-runs the module forward, so a
    forward-pre hook that fires while the handle is in BACKWARD_* state sees
    the post-unshard impl and can hook the accumulator the recomputed edges
    will bind to."""
    J = _SPLIT_JIT
    if not J["active"]:
        return
    from torch.distributed.fsdp._flat_param import HandleTrainingState
    if h._training_state not in (HandleTrainingState.BACKWARD_PRE,
                                 HandleTrainingState.BACKWARD_POST):
        return
    fp = h.flat_param
    if not fp.requires_grad:
        return
    with torch.enable_grad():
        gf = fp.expand_as(fp).grad_fn
    acc = gf.next_functions[0][0] if gf is not None else None
    if acc is None or id(acc) in J["hooked"]:
        return
    pbhs = getattr(fp, "_post_backward_hook_state", None)
    if pbhs is not None and pbhs[0] is acc:
        J["hooked"].add(id(acc))     # torch's own hook already covers it
        return
    import functools as _ft
    from torch.distributed.fsdp._runtime_utils import _post_backward_hook
    J["handles"].append(acc.register_hook(_ft.partial(_post_backward_hook, st, h)))
    J["hooked"].add(id(acc))
    J["acc_refs"].append(acc)
    J["fired"] += 1


def fsdp_split_install_jit_hooks(model) -> int:
    """Install the just-in-time forward-pre hooks once per model (idempotent).
    Hooks are inert unless armed. Registered on the handle's fully-sharded
    module AND its checkpoint-wrapped inner module (the NO_REENTRANT
    recompute calls the inner forward); double-fires dedup via the registry."""
    if getattr(model, "_d3_split_jit_installed", False):
        return 0
    import functools as _ft
    states = getattr(model, "_all_fsdp_states", None) or []
    n = 0
    for st in states:
        h = getattr(st, "_handle", None)
        if h is None:
            continue
        mods = []
        m = getattr(h, "_fully_sharded_module", None)
        if m is not None:
            mods.append(m)
            inner = getattr(m, "_checkpoint_wrapped_module", None)
            if inner is not None:
                mods.append(inner)
        for m in mods:
            m.register_forward_pre_hook(_ft.partial(_split_jit_prehook, st, h))
            n += 1
    model._d3_split_jit_installed = True
    return n


def fsdp_split_jit_arm():
    J = _SPLIT_JIT
    J["active"] = True
    J["hooked"] = set()
    J["handles"] = []
    J["acc_refs"] = []


def fsdp_split_jit_disarm():
    J = _SPLIT_JIT
    for hd in J["handles"]:
        hd.remove()
    J["active"] = False
    J["hooked"] = set()
    J["handles"] = []
    J["acc_refs"] = []


def fsdp_split_backward_hook_graph(model, saved: list, roots: list) -> list:
    """4. MixedPrecision: the graph-leaf AccumulateGrad identity CHURNS
    between forwards/reshards (probe 2026-07-07: registered hook_acc !=
    post-forward acc under bf16 MP; registering on the post-forward
    accumulator changed nothing — delta stayed 7.125 and the 35B pileup was
    bit-identical, so the pass-B edges target yet another object). The only
    identity-proof route is the GRAPH ITSELF: walk the autograd graph from
    the given roots, collect AccumulateGrad nodes whose .variable is one of
    our FlatParameters, and register torch's own _post_backward_hook on
    exactly those. Returns removable hook handles.
    Call once with the pass-B roots BEFORE bwd#1, and once with the pass-A
    roots (h1_graph + any writer-side loss scalars) before bwd#2."""
    import functools as _ft
    from torch.distributed.fsdp._runtime_utils import _post_backward_hook
    fp_map = {}
    for st, h, _tensors, _acc in saved:
        fp_map[id(h.flat_param)] = (st, h)
    handles = []
    dbg = os.environ.get("D3_SPLIT_DEBUG") == "1"
    for acc in _walk_accumulate_grads(roots):
        var = getattr(acc, "variable", None)
        entry = fp_map.get(id(var)) if var is not None else None
        if entry is None:
            if dbg and var is not None:
                print(f"[d3-split-dbg] walk: unmatched acc var shape={tuple(var.shape)} dtype={var.dtype}", flush=True)
            continue
        st, h = entry
        if id(acc) in _SPLIT_JIT["hooked"]:
            continue
        pbhs = getattr(h.flat_param, "_post_backward_hook_state", None)
        if pbhs is not None and pbhs[0] is acc:
            _SPLIT_JIT["hooked"].add(id(acc))
            if dbg:
                print(f"[d3-split-dbg] walk: torch-hook covers {getattr(h, '_fully_sharded_module', type(h)).__class__.__name__}", flush=True)
            continue
        handles.append(acc.register_hook(
            _ft.partial(_post_backward_hook, st, h)))
        _SPLIT_JIT["hooked"].add(id(acc))
        if dbg:
            print(f"[d3-split-dbg] walk: hooked {type(getattr(h, '_fully_sharded_module', None)).__name__} numel={h.flat_param.numel()}", flush=True)
    return handles


def fsdp_joint_backward_arm_hooks(model, roots: list) -> tuple[list, list]:
    """JOINT-graph (single-backward) counterpart of
    `fsdp_split_backward_hook_graph` — fixes the FSDP1 + MixedPrecision
    SILENT-GRADIENT-LOSS for any FSDP unit that forwards >= 2x in one step
    (found 2026-07-07 by the D3 oracle adjudication: the joint baseline
    matched 140/141 tensors with `lm_head.weight` missing ENTIRELY,
    `d3_spike/gates/split_adj_tiny.json`).

    Mechanism: under MixedPrecision, FSDP1's dtype-changing `.data` swaps
    (`_flat_param.py:1360` -> _mp_shard on unshard; `:1848` -> _local_shard
    on reshard, torch 2.10) reset the flat param's grad accumulator
    (Variable.set_data drops it on dtype change), so EVERY forward of a unit
    binds a FRESH AccumulateGrad node. torch registers its reduce-scatter
    post-backward hook only in the FIRST forward
    (`_runtime_utils.py:1428-1436`: "prefer the *first* forward ... If we
    instead prefer the *last* forward, then the hook runs early") — a
    heuristic that silently assumes the first-forward node EXECUTES in the
    backward. The two-pass quantizer violates it: lm_head's first forward is
    the snap-logits call whose only consumer is `sim.argmax(-1)`
    (grad-severed), so its node never fires; the answer-CE gradient arrives
    on the second forward's hookless node, accumulates into the unsharded
    flat grad, and is dropped without error at `_finalize_params` (only
    `_saved_grad_shard`, written by `_post_backward_hook`, reaches the
    optimizer views).

    Fix: walk the ACTUAL autograd graph from `roots` (strong refs held —
    grad_fn wrapper ids recycle otherwise, see `_walk_accumulate_grads`);
    group the AccumulateGrad nodes per flat param. Wherever the executed set
    is not exactly {torch's hooked node}, remove torch's hook and register a
    COUNTING hook on every executed node that calls torch's own
    `_post_backward_hook` exactly once — when the LAST of them fires — so
    reduce-scatter still sees the fully-accumulated grad and double-forwarded
    decoder layers keep their current (late) fire point. fp32/no-MP: the
    accumulator is stable across forwards ("Without parameter mixed
    precision, the AccumulateGrad objects are the same"), the walk finds
    exactly the hooked node, and this is a structural no-op — hence no
    opt-out flag: the fix self-neutralizes wherever the bug cannot occur.
    Units whose accumulators the walk cannot see (none observed; e.g. a
    hypothetical checkpointed region materializing accumulators only at
    recompute) keep torch's hook — identical to current behavior.

    Call AFTER the forward and BEFORE `.backward()`; `.remove()` the
    returned handles after the backward and keep the returned refs alive
    until then. No-op (returns ([], [])) for non-FSDP models. The SPLIT
    schedule does NOT use this (its per-backward re-arm is
    `fsdp_split_backward_hook_graph` — unchanged)."""
    states = getattr(model, "_all_fsdp_states", None) or []
    fp_map = {}
    for st in states:
        h = getattr(st, "_handle", None)
        if h is None or not h.flat_param.requires_grad:
            continue
        fp_map[id(h.flat_param)] = (st, h)
    if not fp_map:
        return [], []
    from torch.distributed.fsdp._runtime_utils import _post_backward_hook
    per_fp: dict = {}
    for acc in _walk_accumulate_grads([r for r in roots if r is not None]):
        var = getattr(acc, "variable", None)
        if var is not None and id(var) in fp_map:
            per_fp.setdefault(id(var), []).append(acc)
    handles: list = []
    refs: list = []
    dbg = os.environ.get("D3_SPLIT_DEBUG") == "1"
    for fp_id, acc_list in per_fp.items():
        st, h = fp_map[fp_id]
        fp = h.flat_param
        pbhs = getattr(fp, "_post_backward_hook_state", None)
        hooked = pbhs[0] if pbhs else None
        if hooked is not None and len(acc_list) == 1 and acc_list[0] is hooked:
            continue  # single executed node == torch's own hook: leave it
        if pbhs is not None:
            # torch's hook sits on a node that either never fires (dead
            # first-forward node — the lm_head drop) or fires possibly
            # before the last contribution (early reduce-scatter); the
            # counting hook below replaces it. _finalize_params re-removes
            # the handle (idempotent) and delattrs the state as usual.
            pbhs[1].remove()
        counter = {"n": len(acc_list)}

        def _mk_hook(st=st, h=h, counter=counter):
            def hook(*_unused):
                counter["n"] -= 1
                if counter["n"] == 0:
                    _post_backward_hook(st, h, None)
            return hook

        for acc in acc_list:
            handles.append(acc.register_hook(_mk_hook()))
            refs.append(acc)
        if dbg:
            print(f"[d3-joint-dbg] re-armed {type(getattr(h, '_fully_sharded_module', None)).__name__} "
                  f"numel={fp.numel()} n_accs={len(acc_list)} torch_hook_in_graph="
                  f"{any(a is hooked for a in acc_list)}", flush=True)
    return handles, refs


def fsdp_split_backward_rearm(saved: list) -> int:
    """Second half of the split-backward FSDP1 surgery (see
    `fsdp_split_backward_save`): restore the forward-saved view tensors and
    re-arm the pre-backward unshard flags. Call between the two backwards.
    Post-backward (reduce-scatter) hooks are handled separately by
    `fsdp_split_backward_hook_graph` on each backward's own graph roots."""
    n = 0
    for _st, h, tensors, _acc in saved:
        fp = h.flat_param
        cur = getattr(fp, "_tensors", None)
        if tensors is not None and cur is not None:
            for i, t in enumerate(tensors):
                if cur[i] is None and t is not None:
                    cur[i] = t
        h._needs_pre_backward_unshard = True
        n += 1
    return n


# =====================================================================
# attention rows / pre-RoPE K/V from CAPTURED activations (no params)
# =====================================================================

def rows_probs_from_capture(out: dict, capture: dict, layer: int, rows_b: int,
                            rows: torch.Tensor, backend_kind: str,
                            passA: bool = False) -> torch.Tensor:
    """Post-softmax attention probs [n_rows, n_q_heads, L] (fp32) for query
    rows of batch row rows_b at `layer`, recomputed from captured post-norm
    pre-RoPE q/k. Blocked columns are EXACT zeros (softmax over -inf).
    Verified against HF eager attn_weights in gate U4b."""
    pref = ("passA",) if passA else ()
    qn = capture[pref + ("qn", layer)]
    kn = capture[pref + ("kn", layer)]
    q = qn[rows_b:rows_b + 1].transpose(1, 2)         # [1,Hq,T,D]
    k = kn[rows_b:rows_b + 1].transpose(1, 2)         # [1,Hkv,T,D]
    if backend_kind == "marin":
        cos, sin = out["cos"], out["sin"]
        q, k = _apply_rope(q, k, cos[rows_b:rows_b + 1], sin[rows_b:rows_b + 1])
    else:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            apply_rotary_pos_emb, repeat_kv)
        rope = capture.get(pref + ("rope",)) if passA else capture.get("rope")
        if rope is None:
            rope = capture["rope"]
        cos, sin = rope
        cos_b = cos[rows_b:rows_b + 1] if cos.dim() == 3 else cos
        sin_b = sin[rows_b:rows_b + 1] if sin.dim() == 3 else sin
        q, k = apply_rotary_pos_emb(q, k, cos_b, sin_b)
        k = repeat_kv(k, out["n_rep"])
    T = q.shape[2]
    qr = q[0, :, rows]                                 # [H, n_rows, D]
    scores = torch.einsum("hnd,hld->hnl", qr.float(), k[0].float()) * out["scaling"]
    mask = out.get("mask")
    if mask is not None:
        if mask.dtype == torch.bool:
            bias = torch.zeros(1, rows.numel(), T, device=scores.device)
            bias.masked_fill_(~mask[rows_b, :, rows, :], float("-inf"))
        else:
            bias = mask[rows_b, :, rows, :].float()
    else:
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=scores.device))
        bias = torch.zeros(1, rows.numel(), T, device=scores.device)
        bias.masked_fill_(~causal[rows].unsqueeze(0), float("-inf"))
    probs = F.softmax(scores + bias, dim=-1)
    return probs.transpose(0, 1)                       # [n_rows, H, T]


def kv_prerope_from_capture(capture: dict, layer: int, rows_b: int, cols,
                            head_dim: int, passA: bool = False):
    """(k, v) at `cols`: k = k_norm output (post-norm PRE-RoPE) [n, Hkv, D];
    v = v_proj output reshaped [n, Hkv, D]."""
    pref = ("passA",) if passA else ()
    kn = capture[pref + ("kn", layer)]
    vp = capture[pref + ("v", layer)]
    k = kn[rows_b, cols]
    v = vp[rows_b, cols]
    n = k.shape[0]
    return k.reshape(n, -1, head_dim), v.reshape(n, -1, head_dim)


# =====================================================================
# loss assembly (activation math only — FSDP-safe outside the forward)
# =====================================================================

def assemble_losses(backend, out: dict, batch: dict, feats: FeatureConfig) -> dict:
    losses = {"ce": out["ce"]}
    total = out["ce"]
    cap = out["capture"]
    spans = batch["spans"]
    B = batch["input_ids"].shape[0]
    is_q35 = isinstance(backend, Q35BMech)
    passA = feats.quantize_slots != "off"   # writer-side captures when quantizing

    if "slot_value" in out and feats.slot_value_coef:
        losses["slot_value"] = out["slot_value"]
        total = total + feats.slot_value_coef * out["slot_value"]

    def writer_hidden(layer_idx: int):
        if layer_idx == -1:
            return cap[("passA", "h_final")] if passA else out["h_final"]
        key = ("passA", "h", layer_idx) if passA else ("h", layer_idx)
        return cap[key]

    if feats.slot_hidden_coef:
        h = writer_hidden(feats.slot_hidden_layer)
        parts = []
        for b in range(B):
            t = batch["teacher_hiddens"][b]
            if t is None:
                continue
            t = t.detach()  # teacher targets are CONSTANTS — a grad-carrying
            # target routes backward into whatever produced it (the armA P3
            # crash: an undetached embed slice vs the FSDP flat shard)
            s0, s1 = spans[b]["slots"]
            s = h[b, s0:s1]
            if feats.set_match == "none":
                parts.append(aligned_loss(s[: t.shape[0]], t, feats.slot_hidden_metric))
            elif t.shape[0] <= s.shape[0]:
                # m targets onto k>=m distinct slots; extra slots free
                l, _ = set_match_loss(s, t, feats.slot_hidden_metric,
                                      matcher=feats.set_match,
                                      tau=feats.sinkhorn_tau, iters=feats.sinkhorn_iters)
                parts.append(l)
            else:
                # TRANSPOSE matching (m > k, e.g. 4*n_ops digit targets at
                # sampled k = 1x/2x n_ops): every SLOT gets a distinct target,
                # extra targets uncovered — cost is differentiable in both
                # args so gradients still flow into the slots
                l, _ = set_match_loss(t, s, feats.slot_hidden_metric,
                                      matcher=feats.set_match,
                                      tau=feats.sinkhorn_tau, iters=feats.sinkhorn_iters)
                parts.append(l)
        if parts:
            losses["slot_hidden"] = torch.stack(parts).mean()
            total = total + feats.slot_hidden_coef * losses["slot_hidden"]

    if feats.kv_match_coef:
        head_dim = (backend.lm.model.layers[backend.full_attn_layers[0]].self_attn.head_dim
                    if is_q35 else backend.layers[0].inner.self_attn.head_dim)
        parts = []
        for b in range(B):
            tk = batch.get("teacher_kv", [None] * B)[b]
            if tk is None:
                continue
            s0, s1 = spans[b]["slots"]
            cols = torch.arange(s0, s1, device=batch["input_ids"].device)
            for l in feats.layer_list(feats.kv_layers):
                assert feats.kv_rope == "pre", \
                    "post-RoPE matching requires position-aligned targets; only 'pre' is implemented"
                sk, sv = kv_prerope_from_capture(cap, l, b, cols, head_dim, passA=passA)
                tk_l = tk.get(l) if isinstance(tk, dict) else None
                if tk_l is None:
                    continue
                m = tk_l[0].shape[0]
                parts.append(kv_match_loss(
                    sk[:m] if "k" in feats.kv_what else None,
                    sv[:m] if "v" in feats.kv_what else None,
                    tk_l[0] if "k" in feats.kv_what else None,
                    tk_l[1] if "v" in feats.kv_what else None,
                    metric=feats.kv_metric))
        if parts:
            losses["kv"] = torch.stack(parts).mean()
            total = total + feats.kv_match_coef * losses["kv"]

    if feats.attn_target_coef:
        kind = "q35b" if is_q35 else "marin"
        parts = []
        for b in range(B):
            s0, s1 = spans[b]["slots"]
            a0, a1 = spans[b]["answer"]
            if s1 <= s0:
                continue
            rows = torch.arange(a0, a1, device=batch["input_ids"].device)
            tgt = batch.get("attn_targets", [None] * B)[b]
            if tgt is None:
                tgt = torch.full((s1 - s0,), 1.0 / (s1 - s0),
                                 device=batch["input_ids"].device)
            for l in feats.layer_list(feats.attn_layers):
                probs = rows_probs_from_capture(out, cap, l, b, rows, kind, passA=False)
                p_slots = probs[:, :, s0:s1].mean(1)
                p_slots = p_slots / p_slots.sum(-1, keepdim=True).clamp_min(1e-9)
                parts.append(attn_target_kl(p_slots, tgt))
        if parts:
            losses["attn"] = torch.stack(parts).mean()
            total = total + feats.attn_target_coef * losses["attn"]

    losses["total"] = total
    return losses


# =====================================================================
# batch builders
# =====================================================================

def sample_k(base_k: int, feats: FeatureConfig, seed: int, step: int, row: int) -> int:
    """Per-example k sampling (RECONCILIATION §3 win-condition clause 1: a
    model trained at FIXED k cannot exhibit a dose curve). Deterministic in
    (seed, step, row)."""
    if feats.k_mode != "sample" or not feats.k_choices:
        return base_k * max(1, feats.k_fixed_mult)
    import random as _r
    rng = _r.Random(seed * 1000003 + step * 977 + row)
    choices = []
    for c in feats.k_choices.split(","):
        c = c.strip()
        if c.endswith("x"):
            choices.append(max(1, int(round(float(c[:-1]) * base_k))))
        else:
            choices.append(int(c))
    if feats.k_p_first > 0 and len(choices) > 1:
        # attempt-1 common frame: P(choices[0]) = k_p_first, remainder uniform
        if rng.random() < feats.k_p_first:
            return choices[0]
        return rng.choice(choices[1:])
    return rng.choice(choices)


def marin_annotate_spans(tok, batch: dict, items: list[dict], arm: str) -> None:
    """Add per-row spans to a recur_loop.build_batch batch (in place).
    Layout there: [left-pad | prefix | slots [P,P+k_i) | tail at P+k].
    'answer' region = the whole tail (think-close lead + 'Answer: ...' + eos).
    problem span via template-frame diff."""
    from experiments.marin.standalone.prompts import render_chat_prompt
    P, k = batch["P"], batch["k"]
    if not hasattr(tok, "_ms_empty_ids"):
        tok._ms_empty_ids = tok.encode(
            render_chat_prompt(tok, "", add_boxed_instruction=True),
            add_special_tokens=False)
    empty_ids = tok._ms_empty_ids
    spans = []
    for b, it in enumerate(items):
        prompt_ids = None
        if hasattr(tok, "_recur_cache"):
            prompt_ids = tok._recur_cache.get(it.get("sig"))
        if prompt_ids is None:
            prompt_ids = tok.encode(
                render_chat_prompt(tok, it["prompt"], add_boxed_instruction=True),
                add_special_tokens=False)
        p0, p1 = template_frame_span(prompt_ids, empty_ids)
        plen = len(prompt_ids) + (2 if arm == "r1" else 4)
        lp = P - plen
        ki = batch["ks"][b]
        treal = int(batch["real"][b, P + k:].sum())
        spans.append({"problem": (lp + p0, lp + p1), "prompt_end": P,
                      "slots": (P, P + ki), "answer": (P + k, P + k + treal)})
    batch["spans"] = spans


def build_batch_q35(tok, items: list[dict], device, feats: FeatureConfig | None = None,
                    seed: int = 0, step: int = 0, with_answer: bool = True,
                    slot_labels_fn=None, k_base_fn=None):
    """Qwen3.6-35B batch, CONTIGUOUS per-row layout (right-pad ONLY — GDN
    layers cannot mask mid-sequence pads, so none may exist):
        [sys+user chat prompt | PAUSE x k_i | "\\nAnswer: " | answer | <|im_end|> | pad]
    Bit-compatible with buffer_sft.build_datum (arm p3/c0) and with
    comp_bench_eval.py --pause-slots prefills."""
    feats = feats or FeatureConfig()
    eos_id = tok.encode("<|im_end|>", add_special_tokens=False)[0]
    if not hasattr(tok, "_ms_empty_ids"):
        msgs = [{"role": "system", "content": Q35_SYSP}, {"role": "user", "content": ""}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        tok._ms_empty_ids = tok.encode(text, add_special_tokens=False)
    rows = []
    for r_i, it in enumerate(items):
        msgs = [{"role": "system", "content": Q35_SYSP},
                {"role": "user", "content": it["problem"]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        prompt_ids = tok.encode(text, add_special_tokens=False)
        p0, p1 = template_frame_span(prompt_ids, tok._ms_empty_ids)
        base_k = k_base_fn(it) if k_base_fn is not None else it.get("n_ops", 0)
        ki = sample_k(base_k, feats, seed, step, r_i) if base_k > 0 else 0
        ans_ids = tok.encode(str(it["answer"]), add_special_tokens=False)
        seq = prompt_ids + [Q35_PAUSE_ID] * ki + Q35_ANS_PREFIX_IDS
        a0 = len(seq)
        if with_answer:
            seq = seq + ans_ids + [eos_id]
        sl = slot_labels_fn(it, ki) if slot_labels_fn is not None else None
        rows.append({"seq": seq, "prompt_len": len(prompt_ids), "k": ki,
                     "problem": (p0, p1), "a0": a0, "gold": it["answer"],
                     "slot_labels": sl, "idx": it.get("idx")})
    L = max(len(r["seq"]) for r in rows)
    B = len(rows)
    input_ids = torch.full((B, L), eos_id, dtype=torch.long)
    real = torch.zeros(B, L, dtype=torch.bool)
    pos = torch.zeros(B, L, dtype=torch.long)
    labels = torch.full((B, L), -100, dtype=torch.long)
    slot_labels = torch.full((B, L), -100, dtype=torch.long)
    spans = []
    for b, r in enumerate(rows):
        n = len(r["seq"])
        input_ids[b, :n] = torch.tensor(r["seq"])
        real[b, :n] = True
        pos[b, :n] = torch.arange(n)
        pl, ki = r["prompt_len"], r["k"]
        spans.append({"problem": r["problem"], "prompt_end": pl,
                      "slots": (pl, pl + ki), "answer": (r["a0"], n)})
        if with_answer:
            for c in range(r["a0"] - 1, n - 1):     # buffer_sft CE shift convention
                labels[b, c] = input_ids[b, c + 1]
        if r["slot_labels"]:
            for j, lab in enumerate(r["slot_labels"][:ki]):
                slot_labels[b, pl + j] = lab
    return {"input_ids": input_ids.to(device), "real": real.to(device),
            "pos": pos.to(device), "labels": labels.to(device),
            "slot_labels": slot_labels.to(device), "spans": spans,
            "golds": [r["gold"] for r in rows], "ks": [r["k"] for r in rows],
            "idxs": [r["idx"] for r in rows]}


def digit_slot_labels_from_trace(tok, item: dict, k: int, digits: int = 4) -> list[int] | None:
    """P2-style digit-decomposed value targets from the comp-bench generator's
    `trace` (innermost-first intermediate values): slot (i,d) = d-th char of
    sign-char + |v_i| left-zero-padded to digits-1. Requires k == digits *
    n_ops; returns None otherwise."""
    trace = item.get("trace")
    if not trace:
        return None
    if k != digits * len(trace):
        return None
    out = []
    for _, v in trace:
        s = str(abs(int(v))).rjust(digits - 1, "0")
        s = ("-" if int(v) < 0 else "0") + s
        assert len(s) == digits, (v, s)
        for ch in s:
            out.append(tok.encode(ch, add_special_tokens=False)[0])
    return out


DIGIT_CHARS = "-0123456789"   # index 0 = sign char, 1..10 = digits 0..9


def trace_digit_char_indices(item: dict, digits: int = 4) -> list[int] | None:
    """Same digit-decomposition as digit_slot_labels_from_trace but returned as
    indices into DIGIT_CHARS (tokenizer-free) — used to build EMBEDDING-vector
    hidden targets for arm A ('digit-decomposed intermediate-value supervision
    at early-mid layers': target for each code position = the input-embedding
    row of its digit token, prescribed at a mid layer, SET-matched, annealed).
    Always digits * n_ops entries (independent of the sampled k — the matcher
    handles k < m via transpose assignment)."""
    trace = item.get("trace")
    if not trace:
        return None
    out = []
    for _, v in trace:
        s = str(abs(int(v))).rjust(digits - 1, "0")
        s = ("-" if int(v) < 0 else "0") + s
        for ch in s:
            out.append(DIGIT_CHARS.index(ch))
    return out


# =====================================================================
# GDN leak probe (35B bottleneck quantification)
# =====================================================================

@torch.no_grad()
def gdn_leak_probe(backend, tok, items: list[dict], device,
                   feats: FeatureConfig, scramble_seed: int = 777) -> dict:
    """With the FULL bottleneck blocks active (answer->prompt;slots->prompt or
    the problem-scoped variants), substitute the problem-span CONTENT and
    measure the answer-position response:
      - marin-dense: exactly invariant (asserted in gate U3);
      - 35B: any delta == information reaching the answer through unmaskable
        paths (30 GDN layers + conv kernel-4 edge bleed). Reports final-hidden
        cosine/L2 delta and next-token-logit KL at the primary answer query
        (the trailing-space col that predicts the first answer digit)."""
    import random as _r
    rng = _r.Random(scramble_seed)
    is_q35 = isinstance(backend, Q35BMech)
    deltas = []
    for it in items:
        b1 = build_batch_q35(tok, [it], device, feats=feats, with_answer=False)
        ids2 = b1["input_ids"].clone()
        p0, p1 = b1["spans"][0]["problem"]
        perm = list(range(p0, p1))
        rng.shuffle(perm)
        ids2[0, p0:p1] = b1["input_ids"][0, perm]
        b2 = {**b1, "input_ids": ids2}
        o1 = backend(b1, feats)
        o2 = backend(b2, feats)
        q = b1["spans"][0]["answer"][1] - 1
        h1, h2 = o1["h_final"][0, q], o2["h_final"][0, q]
        head = backend.lm.lm_head if is_q35 else backend.lm_head
        l1 = head(h1.unsqueeze(0)).float()[0]
        l2 = head(h2.unsqueeze(0)).float()[0]
        kl = F.kl_div(F.log_softmax(l2, -1), F.log_softmax(l1, -1),
                      log_target=True, reduction="sum")
        deltas.append({"idx": it.get("idx"),
                       "h_cos": F.cosine_similarity(h1.float(), h2.float(), dim=0).item(),
                       "h_l2": (h1.float() - h2.float()).norm().item(),
                       "logit_kl": kl.item(),
                       "argmax_same": bool(l1.argmax() == l2.argmax())})
    return {"n": len(deltas),
            "mean_h_cos": sum(d["h_cos"] for d in deltas) / len(deltas),
            "mean_logit_kl": sum(d["logit_kl"] for d in deltas) / len(deltas),
            "argmax_flip_rate": 1.0 - sum(d["argmax_same"] for d in deltas) / len(deltas),
            "per_item": deltas}


# =====================================================================
# 35B strict text-only load (the "Unexpected key" doctrine: NO silent drops)
# =====================================================================

def load_q35b_text(snapshot_dir: str = Q35_BASE_SNAPSHOT, dtype=torch.float32,
                   device: str = "cpu", meta_ok: bool = False):
    """Build Qwen3_5MoeForCausalLM (TEXT-only) and load the multimodal
    snapshot with an EXPLICIT key map: model.language_model.* -> model.*,
    lm_head.weight kept; model.visual.* / mtp.* skipped BY NAME (copied back
    verbatim at export). STRICT audit: missing == [] and unexpected == [] or
    raise — the Qwen3 merge_qkv lesson (silent WARNING-only drops served
    random init)."""
    import json as _json
    from pathlib import Path
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    cfg = AutoConfig.from_pretrained(snapshot_dir)
    text_cfg = cfg.text_config
    text_cfg.dtype = dtype
    with torch.device("meta"):
        model = Qwen3_5MoeForCausalLM(text_cfg)
    if meta_ok:
        return model, None
    index = _json.load(open(Path(snapshot_dir) / "model.safetensors.index.json"))["weight_map"]
    wanted = {}
    for key, shard in index.items():
        if key.startswith("model.visual.") or key.startswith("mtp."):
            continue
        tgt = "lm_head.weight" if key == "lm_head.weight" else key.replace(
            "model.language_model.", "model.", 1)
        wanted[tgt] = (shard, key)
    sd = {}
    open_shards: dict = {}
    for tgt, (shard, key) in wanted.items():
        if shard not in open_shards:
            open_shards[shard] = safe_open(Path(snapshot_dir) / shard, framework="pt",
                                           device="cpu")
        sd[tgt] = open_shards[shard].get_tensor(key).to(dtype)
    load_res = model.load_state_dict(sd, strict=False, assign=True)
    missing = list(load_res.missing_keys)
    unexpected = list(load_res.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(f"STRICT LOAD AUDIT FAILED missing={missing[:8]}({len(missing)}) "
                           f"unexpected={unexpected[:8]}({len(unexpected)})")
    # the meta-device build leaves NON-PERSISTENT buffers (rotary inv_freq) on
    # meta; re-materialize them, then assert nothing meta remains
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as _M
    model.model.rotary_emb = _M.Qwen3_5MoeTextRotaryEmbedding(config=text_cfg)
    meta_left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
                 if t.device.type == "meta"]
    if meta_left:
        raise RuntimeError(f"meta tensors remain after load: {meta_left[:8]}")
    if device != "cpu":
        model = model.to(device)
    return model, sd
