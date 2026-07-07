#!/usr/bin/env python3
"""RECURRENCE-STAGE0: shared reference implementation of the slot-looped P-R-C
forward for the marin 9.7B dense Qwen3 (37 layers, hidden 3840, 30 heads, no GQA).

Imported by BOTH the trainer and the eval harness (design §6.2 parity doctrine:
one implementation of the loop forward, no train/serve fork).

Semantics (RECURRENCE_RETROFIT_DESIGN_20260705.md §3.2-3.4, Scope S, final-KV
Jacobi updates):
  - P-R-C = layers [0,12) | [12,24) | [24,37); NO dropped layers.
  - Sequence layout per example: [left-pad | prefix (prompt+<think>+\n) |
    slots (PAUSE x k) | tail (\n</think>\n\n + answer...) | right-pad].
  - Pass 1 = ordinary forward (== loop iteration 1 for slots; NO adapter).
  - Iterations i>=2 re-run R over ONLY the slot positions:
        x_i = A([p ; RMSNorm(s_{i-1})]),   s_i = R(x_i)
    where p = prelude output at the slots, A: 2h->h. Slot queries attend to the
    FROZEN prefix KV (from pass 1) at layers 12-23 plus the slots' own
    iteration-i KV under the causal mask (synchronous/Jacobi). Slot KV at
    layers 12-23 is OVERWRITTEN each iteration (final-KV discipline).
  - Tail positions run layers 12-23 once, attending prefix KV + final-iter
    slot KV. Coda (24-36) runs once over the full sequence. Slot KV at
    prelude/coda layers comes from their single pass.
  - Adapter init [I | N(0, sigma)] (sigma=1e-3 for training; sigma=0 gives the
    EXACT identity: every iteration computes R(p), so the model == base at
    every r). r=1 never touches the adapter (honest in-model control).
  - Full BPTT through all slot iterations (no truncation; k<=32 positions only).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SPAN = (12, 24)  # loop band [12,24); prelude [0,12); coda [24,37)
PAUSE_ID = 1981
START_THINK_ID = 128002
END_THINK_ID = 128003
NL_ID = 198
NL2_ID = 271
EOS_ID = 128009


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin):
    # q,k: [B,H,T,D]; cos,sin: [B,T,D]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class RecurLayer(nn.Module):
    """Wraps one Qwen3DecoderLayer; custom forward with explicit KV-prefix
    support. Math is identical to Qwen3DecoderLayer.forward (sdpa path)."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, h, cos, sin, attn_mask, kv_prefix=None, need_kv=False):
        # h [B,T,Hd]; cos/sin [B,T,D]; attn_mask bool [B,1,T,Tkv] (True=attend)
        a = self.inner.self_attn
        B, T, _ = h.shape
        residual = h
        x = self.inner.input_layernorm(h)
        q = a.q_norm(a.q_proj(x).view(B, T, -1, a.head_dim)).transpose(1, 2)
        k = a.k_norm(a.k_proj(x).view(B, T, -1, a.head_dim)).transpose(1, 2)
        v = a.v_proj(x).view(B, T, -1, a.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        if kv_prefix is not None:
            k_all = torch.cat([kv_prefix[0], k], dim=2)
            v_all = torch.cat([kv_prefix[1], v], dim=2)
        else:
            k_all, v_all = k, v
        o = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=attn_mask, scale=a.scaling)
        o = o.transpose(1, 2).reshape(B, T, -1)
        h = residual + a.o_proj(o)
        residual = h
        x = self.inner.post_attention_layernorm(h)
        h = residual + self.inner.mlp(x)
        return h, ((k, v) if need_kv else None)


class RMSNormF(nn.Module):
    """Qwen3-style RMSNorm (fp32 compute, cast back)."""

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        dt = x.dtype
        xf = x.to(torch.float32)
        var = xf.pow(2).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.variance_epsilon)
        return self.weight * xf.to(dt)


class RecurModel(nn.Module):
    """P-R-C slot-loop retrofit of a loaded HF Qwen3ForCausalLM."""

    def __init__(self, hf_model, sigma: float = 1e-3, adapter_seed: int = 1234,
                 init_adapter: bool = True):
        super().__init__()
        self.config = hf_model.config
        m = hf_model.model
        assert len(m.layers) == 37, f"expected 37 layers, got {len(m.layers)}"
        self.embed_tokens = m.embed_tokens
        self.layers = nn.ModuleList(RecurLayer(l) for l in m.layers)
        self.norm = m.norm
        self.rotary = m.rotary_emb
        self.lm_head = hf_model.lm_head
        hs = self.config.hidden_size
        self.adapter = nn.Linear(2 * hs, hs, bias=False)
        self.state_norm = RMSNormF(hs, eps=self.config.rms_norm_eps)
        if init_adapter and self.adapter.weight.device.type != "meta":
            self.init_adapter_identity(sigma, adapter_seed)

    @torch.no_grad()
    def init_adapter_identity(self, sigma: float, seed: int):
        hs = self.config.hidden_size
        w = torch.zeros(hs, 2 * hs, dtype=self.adapter.weight.dtype)
        w[:, :hs] = torch.eye(hs, dtype=w.dtype)
        if sigma > 0:
            g = torch.Generator().manual_seed(seed)
            w[:, hs:] = torch.randn(hs, hs, generator=g, dtype=torch.float32).to(w.dtype) * sigma
        self.adapter.weight.copy_(w)
        self.state_norm.weight.fill_(1.0)

    # ---------------- mask helpers ----------------
    @staticmethod
    def full_mask(real: torch.Tensor) -> torch.Tensor:
        """Causal + only-real-kv mask; diagonal forced True (pad rows attend
        self so softmax rows are never empty; their outputs are unused)."""
        B, L = real.shape
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=real.device))
        allow = causal.unsqueeze(0) & real[:, None, :]
        idx = torch.arange(L, device=real.device)
        allow[:, idx, idx] = True
        return allow.unsqueeze(1)  # [B,1,L,L]

    # ---------------- core forward ----------------
    def _encode(self, input_ids, real, pos, P: int, k: int, r: int, need_kv: bool = False):
        """Run the P-R-C slot-loop forward over a padded batch.

        input_ids/real/pos: [B,L]; prefix = cols [0,P), slots = [P,P+k),
        tail = [P+k,L). k==0 -> plain forward (no loop machinery, identical
        math to the base model). Returns (final_hidden [B,L,Hd], caches|None).
        caches[l] = (k,v) for all 37 layers over all L cols (for decoding).
        """
        B, L = input_ids.shape
        h = self.embed_tokens(input_ids)
        cos, sin = self.rotary(h, pos)
        m_full = self.full_mask(real)
        caches: list = [None] * 37 if need_kv else None

        # Phase A: prelude 0..11, full sequence
        for i in range(SPAN[0]):
            h, kv = self.layers[i](h, cos, sin, m_full, need_kv=need_kv)
            if need_kv:
                caches[i] = kv

        # Phase B1: R block over prefix cols [0,P)
        hp = h[:, :P]
        pkv = []
        for i in range(SPAN[0], SPAN[1]):
            hp, kv = self.layers[i](hp, cos[:, :P], sin[:, :P], m_full[:, :, :P, :P], need_kv=True)
            pkv.append(kv)
        # Phase B2: slot loop
        if k > 0:
            p_slots = h[:, P:P + k]
            cos_s, sin_s = cos[:, P:P + k], sin[:, P:P + k]
            # slot queries attend real prefix kv + causal among slots (all real)
            m_slots = m_full[:, :, P:P + k, :P + k]
            s = None
            skv = None
            for it in range(1, r + 1):
                x = p_slots if it == 1 else self.adapter(
                    torch.cat([p_slots, self.state_norm(s)], dim=-1))
                hcur = x
                skv = []
                for li, i in enumerate(range(SPAN[0], SPAN[1])):
                    hcur, kv = self.layers[i](
                        hcur, cos_s, sin_s, m_slots, kv_prefix=pkv[li], need_kv=True)
                    skv.append(kv)
                s = hcur
            h23_slots = s
            rkv = [(torch.cat([pkv[li][0], skv[li][0]], dim=2),
                    torch.cat([pkv[li][1], skv[li][1]], dim=2)) for li in range(12)]
        else:
            h23_slots = None
            rkv = pkv
        # Phase B3: R block over tail cols [P+k, L) (skipped if tail empty,
        # e.g. the cot-format prefill whose sequence ends at the slot region)
        ht = h[:, P + k:]
        has_tail = ht.shape[1] > 0
        if has_tail:
            m_tail = m_full[:, :, P + k:, :]
            tkv_list = []
            for li, i in enumerate(range(SPAN[0], SPAN[1])):
                ht, kv = self.layers[i](ht, cos[:, P + k:], sin[:, P + k:], m_tail,
                                        kv_prefix=rkv[li], need_kv=True)
                tkv_list.append(kv)
        if need_kv:
            for li, i in enumerate(range(SPAN[0], SPAN[1])):
                if has_tail:
                    caches[i] = (torch.cat([rkv[li][0], tkv_list[li][0]], dim=2),
                                 torch.cat([rkv[li][1], tkv_list[li][1]], dim=2))
                else:
                    caches[i] = rkv[li]
        # stitch layer-23 output
        parts = [hp] + ([h23_slots] if k > 0 else []) + ([ht] if has_tail else [])
        h = torch.cat(parts, dim=1)
        # Phase C: coda 24..36, full sequence
        for i in range(SPAN[1], 37):
            h, kv = self.layers[i](h, cos, sin, m_full, need_kv=need_kv)
            if need_kv:
                caches[i] = kv
        return self.norm(h), caches

    def forward(self, mode: str, input_ids, real, pos, P: int, k: int, r: int,
                labels=None, logits_from: int = 0):
        """mode='ce': mean CE over label positions (labels [B,L], -100 = ignore;
        labels[c] is the target for position c). mode='logits': logits for
        cols >= logits_from (fp32)."""
        hid, _ = self._encode(input_ids, real, pos, P, k, r, need_kv=False)
        if mode == "ce":
            sel = labels != -100
            hsel = hid[sel]                     # [N,Hd]
            logits = self.lm_head(hsel).float()  # [N,V]
            loss = F.cross_entropy(logits, labels[sel], reduction="mean")
            return loss, sel.sum()
        if mode == "logits":
            return self.lm_head(hid[:, logits_from:]).float()
        raise ValueError(mode)

    # ---------------- generation (eval harness; not used under FSDP) ----------------
    @torch.no_grad()
    def generate_greedy(self, input_ids, real, pos, P: int, k: int, r: int,
                        max_new_tokens: int = 10, eos_id: int = EOS_ID):
        """Prefill with the slot loop, then greedy decode. Returns [B,max_new]
        token ids (eos-padded after stop)."""
        B, L = input_ids.shape
        hid, caches = self._encode(input_ids, real, pos, P, k, r, need_kv=True)
        last_logits = self.lm_head(hid[:, -1]).float()  # [B,V]
        real_run = real.clone()
        pos_next = pos[:, -1] + 1  # tail has no right-pad in eval layout
        out = torch.full((B, max_new_tokens), eos_id, dtype=torch.long, device=input_ids.device)
        done = torch.zeros(B, dtype=torch.bool, device=input_ids.device)
        tok = last_logits.argmax(-1)
        for step in range(max_new_tokens):
            tok = torch.where(done, torch.full_like(tok, eos_id), tok)
            out[:, step] = tok
            done = done | (tok == eos_id)
            if bool(done.all()) or step == max_new_tokens - 1:
                break
            h = self.embed_tokens(tok[:, None])
            cos, sin = self.rotary(h, pos_next[:, None])
            real_run = torch.cat([real_run, (~done)[:, None]], dim=1)
            Tkv = real_run.shape[1]
            m = real_run[:, None, None, :].clone()  # [B,1,1,Tkv]
            m[:, :, :, Tkv - 1] = True  # always attend self
            for i in range(37):
                h, kv = self.layers[i](h, cos, sin, m, kv_prefix=caches[i], need_kv=True)
                caches[i] = (torch.cat([caches[i][0], kv[0]], dim=2),
                             torch.cat([caches[i][1], kv[1]], dim=2))
            tok = self.lm_head(self.norm(h[:, -1])).float().argmax(-1)
            pos_next = pos_next + 1
        return out


# ---------------- batch building (shared by trainer + eval + identity) ----------------

def build_batch(tok, items, arm: str, device, with_answer: bool, k_override=None,
                k_mult: int = 1, k_mults=None):
    """Build a padded batch from task items.

    arm='r1': think block = <think>\n [PAUSE]*k_i \n</think>\n\n ; per-item
              k_i = (item['k'] if present else item['band']) x multiplier,
              where multiplier = k_mults[i] (per-item list; the ATTEMPT-1
              clause-1 k-sampling) or the scalar k_mult (eval dose rows).
              The slot region is padded to the batch max k (pad slots
              real=False, masked everywhere; each item's position ids stay
              contiguous across its own real tokens, so per-item semantics
              are identical to the unpadded layout).
    arm='c0': native empty-think <think>\n\n</think>\n\n ; k = 0 (ablated row).
    arm='cot': forced-thinking prefill <think>\n, NO tail — free CoT
              generation (calibration ceiling rows; with_answer must be False).
    with_answer=True  -> tail carries 'Answer: \\boxed{gold}' + eos and labels
                         (CE from the position predicting the first answer token
                         through the position predicting eos — the validated C0
                         convention).
    with_answer=False -> tail ends with the 'Answer:' token ids (eval prefill;
    'Answer:' IS a token prefix of 'Answer: \\boxed{..}' on this tokenizer,
    'Answer: ' with trailing space is NOT — verified 2026-07-06).
    Returns dict of tensors + meta.
    """
    assert arm in ("r1", "c0", "cot")
    assert not (arm == "cot" and with_answer)
    prefixes, tails, golds, ks = [], [], [], []
    for i_it, it in enumerate(items):
        prompt_ids = tok._recur_cache.get(it["sig"]) if hasattr(tok, "_recur_cache") else None
        if prompt_ids is None:
            from experiments.marin.standalone.prompts import render_chat_prompt
            text = render_chat_prompt(tok, it["prompt"], add_boxed_instruction=True)
            prompt_ids = tok.encode(text, add_special_tokens=False)
            if hasattr(tok, "_recur_cache"):
                tok._recur_cache[it["sig"]] = prompt_ids
        if arm == "r1":
            base_k = k_override if k_override is not None else it.get("k", it["band"])
            mult = k_mults[i_it] if k_mults is not None else k_mult
            ki = base_k * mult
            prefix = prompt_ids + [START_THINK_ID, NL_ID]
            tail_lead = [NL_ID, END_THINK_ID, NL2_ID]
        elif arm == "c0":
            ki = 0
            prefix = prompt_ids + [START_THINK_ID, NL2_ID, END_THINK_ID, NL2_ID]
            tail_lead = []
        else:  # cot
            ki = 0
            prefix = prompt_ids + [START_THINK_ID, NL_ID]
            tail_lead = []
        ans_ids = tok.encode(f"Answer: \\boxed{{{it['gold']}}}", add_special_tokens=False)
        if arm == "cot":
            tail = []
            ans_start = None
        elif with_answer:
            tail = tail_lead + ans_ids + [EOS_ID]
            ans_start = len(tail_lead)  # index within tail of first answer token
        else:
            tail = tail_lead + tok.encode("Answer:", add_special_tokens=False)
            ans_start = None
        prefixes.append(prefix)
        tails.append((tail, ans_start))
        golds.append(it["gold"])
        ks.append(ki)
    P = max(len(p) for p in prefixes)
    k = max(ks)  # slot-region width (pad slots masked out per item)
    Tmax = max(len(t) for t, _ in tails)
    B = len(items)
    L = P + k + Tmax
    input_ids = torch.full((B, L), EOS_ID, dtype=torch.long)
    real = torch.zeros(B, L, dtype=torch.bool)
    pos = torch.zeros(B, L, dtype=torch.long)
    labels = torch.full((B, L), -100, dtype=torch.long)
    for b, (prefix, (tail, ans_start), ki) in enumerate(zip(prefixes, tails, ks)):
        lp = P - len(prefix)
        # prefix + real slots at [P, P+ki)
        head = prefix + [PAUSE_ID] * ki
        input_ids[b, lp:P + ki] = torch.tensor(head)
        real[b, lp:P + ki] = True
        pos[b, lp:P + ki] = torch.arange(len(head))
        # tail at [P+k, P+k+len(tail)); positions continue from len(prefix)+ki
        t0 = P + k
        input_ids[b, t0:t0 + len(tail)] = torch.tensor(tail)
        real[b, t0:t0 + len(tail)] = True
        pos[b, t0:t0 + len(tail)] = torch.arange(len(head), len(head) + len(tail))
        if with_answer:
            # CE: positions predicting tokens [first answer token .. eos]
            a0 = t0 + ans_start             # col of first answer token
            aend = t0 + len(tail)           # one past eos col
            for c in range(a0 - 1, aend - 1):
                labels[b, c] = input_ids[b, c + 1]
    return {
        "input_ids": input_ids.to(device), "real": real.to(device),
        "pos": pos.to(device), "P": P, "k": k,
        "labels": labels.to(device), "golds": golds, "ks": ks,
    }


def attach_prompt_cache(tok):
    if not hasattr(tok, "_recur_cache"):
        tok._recur_cache = {}


def load_tokenizer(ckpt_dir: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    tok.chat_template = open(
        "/home/apanda/xorl-client/experiments/marin/standalone/data/delphi_v0.jinja2").read()
    attach_prompt_cache(tok)
    return tok


def build_recur_from_hf(ckpt_dir: str, device, dtype=torch.bfloat16, sigma=1e-3,
                        adapter_seed=1234):
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(
        ckpt_dir, dtype=dtype, low_cpu_mem_usage=True, attn_implementation="sdpa")
    model = RecurModel(hf, sigma=sigma, adapter_seed=adapter_seed)
    model.adapter.to(dtype)
    model.state_norm.to(dtype)
    return model.to(device), hf
