# SMG: add a `/teacher_hidden_cache` passthrough route (for OPD teacher prefill)

**Owner:** (to assign) · **Status:** spec, not yet implemented · **Author context:** 2026-05-29

## Goal

Let SMG load-balance the OPD **teacher prefill** across N sglang teacher replicas,
so the client POSTs to one SMG URL (no client-side round-robin). Today the teacher
prefill is the OPD step bottleneck: one TP=2 sglang teacher replica does ~6,900 tok/s
and a step is ~1.42M tokens (128 prompts × ~11k-token CoTs) → **~206 s/step, teacher-
bound** (fwd_bwd is only ~22 s). Fanning the prefill across ~8 replicas behind SMG
brings the teacher to ≈ fwd_bwd. SMG already fronts the **sampler** sglang pods
(`dispatch` pod, `--policy cache_aware`); we want the same for the teacher.

## Why this needs an SMG code change (the blocker)

SMG has a **fixed route table** (`model_gateway/src/server.rs`, the
`protected_routes`/`Router::new()` block around L1059–L1110: `/generate`,
`/v1/chat/completions`, `/v1/completions`, `/rerank`, `/health`, …) and its
**fallback is `sink_handler` → `StatusCode::NOT_FOUND`** (`server.rs:107`, registered
`.fallback(sink_handler)` at `server.rs:1298`). The OPD teacher uses a **custom
sglang endpoint** `POST /teacher_hidden_cache`
(`xorl-sglang-internal/python/sglang/srt/entrypoints/teacher_hidden_cache.py`,
wired in `http_server.py`) — it is NOT one of SMG's known routes, so SMG 404s it.

It **cannot** be routed through SMG's existing `/generate`: the whole point of the
custom endpoint is that it writes the bulky hidden tensors to a **shared-FS file
inside the sglang process** and returns only metadata; going through `/generate`
with `return_hidden_states` would ship ~5.8 GB/step of hiddens back over HTTP.

So: **add one passthrough route** that picks a healthy worker (load-balanced) and
forwards the POST body verbatim to that worker's `/teacher_hidden_cache`, returning
the worker's response unchanged.

## What to implement

### 1. Handler: raw passthrough, load-balanced, no body parsing
Add `async fn teacher_hidden_cache(...)` in `model_gateway/src/server.rs` modeled on
the shape of `generate` (`server.rs:318`, which delegates to
`state.router.route_generate(...)`), BUT:

- **Do NOT deserialize the body.** The payload is large (input_ids =
  ~16 seqs × ~11k tokens of ints per request). Take the raw bytes
  (`axum::body::Bytes`) and forward them as-is — never parse/re-serialize the
  `input_ids`. Preserve `Content-Type: application/json`.
- **Select a healthy worker with a content-agnostic policy.** The teacher batches
  have no shared prefix beyond a ~6-token template and each is a large prefill, so
  `cache_aware` (prefix routing) gives no benefit. Use **`round_robin`** (simplest;
  the per-step prepare-batches are ~equal-sized so it balances well) — or
  `power_of_two` (least-load) if load tracking is convenient. Get workers from
  `state.context.worker_registry.get_all()` and filter `w.is_healthy()` (see
  `readiness()` in `server.rs` for the registry-access pattern). Reuse a policy from
  `model_gateway/src/policies/` (`round_robin.rs`, `power_of_two.rs`,
  `LoadBalancingPolicy::select_worker`) or a simple `AtomicUsize` round-robin counter
  held in `AppState`.
- **Forward** with the shared `reqwest::Client` (`AppContext.client`,
  `app_context.rs:54`): `client.post(format!("{}/teacher_hidden_cache", worker_url))
  .header(CONTENT_TYPE, "application/json").body(bytes).timeout(LONG).send()`.
- **Return the worker response verbatim** — status code + body bytes. The worker
  returns `200 {cache_indices_by_sample, path, dtype, num_tokens, hidden_size}` on
  success or `400 {error: ...}` on failure; the client relies on both
  (`_teacher_cache_from_sglang` does `raise_for_status()` then `response.json()`).
- **Timeout: generous.** One teacher batch (e.g. 16 seqs × ~11k tok) takes ~20–60 s
  to prefill. Set the upstream reqwest timeout to **≥600 s** and ensure no shorter
  SMG-level/request timeout truncates it.
- **Concurrency:** the client issues several concurrent prepares
  (`opd_prepare_concurrency`); each hits this route independently and must route to
  a (possibly different) worker. Make sure worker selection is per-request and
  thread-safe.
- **No retries that double-execute a file write** — a retry would re-run the prefill
  and re-write the cache file. Prefer: no automatic retry (let the client's own
  error handling surface it), or retry only on connection-refused (worker down)
  before any bytes are sent.

### 2. Register the route
In the same router-builder block (`server.rs` ~L1090, next to `.route("/generate",
post(generate))`):
```rust
.route("/teacher_hidden_cache", post(teacher_hidden_cache))
```
Place it on the **same router that has worker access** as `/generate` (so the
worker registry + client + policy are in `State<Arc<AppState>>`). It does not need
auth/tenant middleware beyond what `/generate` uses; match `/generate`'s middleware.

### 3. (Optional) policy override for this route
Default SMG `--policy` is `cache_aware` (good for samplers). This route should ignore
that and use round_robin/power_of_two internally regardless of the global `--policy`,
since cache_aware is meaningless for an opaque body. Hard-code the load-based
selection inside the handler (don't depend on the CLI `--policy`).

## Request / response schema (what flows through)

Client → SMG → worker, `POST /teacher_hidden_cache`, JSON body:
```json
{ "input_ids":     [[int, ...], ...],     // per-sample teacher token ids (~11k each)
  "target_tokens": [[int, ...], ...],     // per-sample mask; -100 = ignore (not kept)
  "cache_path":    "/shared/.../teacher_hidden_step{N}_mb{M}.safetensors",
  "cache_key":     "hidden_states",
  "dtype":         "bfloat16" }
```
Worker → SMG → client, success `200`:
```json
{ "cache_indices_by_sample": [[int,...], ...], "path": "...",
  "tensor_key": "hidden_states", "dtype": "bfloat16",
  "num_tokens": int, "hidden_size": int }
```
Failure `400`: `{ "error": { "message": "..." } }`. **Pass both through unchanged.**

## Build & deploy

- Repo: `/home/apanda/smg-together-thunderagent-port` (smg 1.4.1). Build the **debug**
  binary (`cargo build` → `target/debug/smg`); debug is fine (routing isn't CPU-bound,
  per `smg_router_swap.md`).
- **The binary is shared with the sampler `dispatch` pod** (same path, mounted via the
  home PVC). The change is **additive** (new route only — does not touch existing
  routes), so rebuilding is safe for the samplers, but verify the existing routes
  still build/serve before relying on it.
- Deploy a **teacher SMG** pod just like the sampler `dispatch` pod (see the dispatch
  container spec in any `experiments/opd_profile/k8s/generated/er-opd5sgl-*.yaml`, and
  `smg_router_swap.md`):
  ```bash
  SMG_BIN=/home/apanda/smg-together-thunderagent-port/target/debug/smg
  WORKER_URLS="http://<run>-teacher-sglang-0:30000 http://<run>-teacher-sglang-1:30000 ..."
  exec "${SMG_BIN}" launch --host 0.0.0.0 --port 8080 \
       --policy round_robin --worker-urls ${WORKER_URLS}
  ```
  The teacher replicas are TP=2 sglang servers launched with
  `--enable-return-hidden-states --disable-radix-cache --chunked-prefill-size 16384`
  (see `opd_complete_runbook.md` §10.11). They need **no IB/Mooncake** (NVLink-only TP,
  write hiddens to /shared), so they go on **default-group nodes** with a free-GPU
  picker (`CUDA_VISIBLE_DEVICES = 2 freest GPUs`), not the nccl pool.

## Client wiring (no client change needed)

The merged canonical client (`/home/apanda/xorl-client-chat-completions/examples/
on_policy_distillation.py`) already POSTs to one `teacher_base_url` via
`_teacher_cache_from_sglang` with `teacher_backend=sglang`. Just point it at the
teacher-SMG service:
```
teacher_base_url=http://<run>-teacher-smg:8080  teacher_backend=sglang
```
To actually fan out across replicas, also shrink the prepare batch so there are
enough concurrent prepares to fill the replicas, e.g.
`opd_prepare_batch_size=16 opd_prepare_concurrency=8` (128 prompts → 8 batches → 8
replicas in parallel).

## Validation (do before trusting it)

1. **Numerical:** run `examples/ab_teacher_cache.py` (canonical client) with
   `--sglang-url http://<teacher-smg>:8080` — cosine of kept-position hiddens vs the
   xorl teacher must stay ~0.998 (same gate as the direct-replica A/B that already
   passed).
2. **Load spread:** grep each teacher replica's sglang log for `Prefill batch` counts
   over a step — requests should be ~evenly distributed across replicas.
3. **End-to-end:** an OPD step's teacher phase should drop from ~206 s (1 replica) to
   ≈ 206/N s; confirm the step becomes fwd_bwd-bound (~22 s) at ~8–10 replicas.

## Notes / gotchas

- Don't `cache_aware`-route this — the body is opaque and the CoTs differ per prompt.
- Don't deserialize `input_ids` in SMG (huge); forward raw bytes.
- Long timeout (≥600 s) — teacher prefills are slow.
- Additive route; shared binary with the sampler dispatch — verify existing routes
  after rebuild.
- This load-balances *necessary* work (each prompt's distinct CoT must be prefilled);
  it does not reduce total teacher FLOPs. The only way to reduce them is cross-epoch
  caching of the fixed per-prompt prefix, which is out of scope here.
