"""CPU unit test for the PS-mode multi-step ZORL generation lifecycle.

Reproduces the bug + validates the fix WITHOUT a GPU or a live server: it mocks
``zorl_client._post`` with a stateful fake that mirrors the real SGLang ZORL
generation-lifecycle invariants (one active generation per URL at a time; the
generation index advances on each ``/start_zorl_generation``; ``begin_generation``
rejects with an HTTP-400-style "already has active generation" error if one is
still active; ``/abort_zorl_generation`` clears it WITHOUT applying an update),
then drives the exact PS-mode lockstep sequence the client step loop performs and
asserts:

  1. each step starts a FRESH generation id on every replica AND the PS;
  2. the PRIOR step's generation is torn down (aborted) on every replica AND the
     PS before the next step starts;
  3. without the teardown (legacy PS loop), step 2 fails with the 400 — i.e. the
     test actually exercises the bug it fixes.

Run: ``python experiments/zorl/standalone/test_ps_generation_lifecycle.py``
(no pytest required; also discoverable as a pytest module).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import zorl_client as zc  # noqa: E402


class _FakeReplica:
    """Stateful per-URL fake of the SGLang ZORL generation lifecycle."""

    def __init__(self, url: str, *, family: str = "session-family-000000"):
        self.url = url
        self.family = family
        self.generation = 0  # next generation index to hand out
        self.active_generation_id: str | None = None
        # Observability for assertions.
        self.started: list[str] = []
        self.aborted: list[str] = []

    def _gen_id(self, gen_index: int) -> str:
        return f"{self.family}-g{gen_index:06d}"

    def start(self) -> dict:
        # Mirrors begin_generation: reject if one is still active (the live 400).
        if self.active_generation_id is not None:
            raise RuntimeError(
                f"/start_zorl_generation on {self.url} → HTTP 400: "
                f"ZORL session 'session' already has active generation "
                f"'{self.active_generation_id}'"
            )
        gen_id = self._gen_id(self.generation)
        self.generation += 1
        self.active_generation_id = gen_id
        self.started.append(gen_id)
        return {
            "generation_id": gen_id,
            "candidates": [
                {"candidate_id": f"{gen_id}-p0000+", "perturbation_index": 0, "direction": "positive", "lora_name": "c+"},
                {"candidate_id": f"{gen_id}-p0000-", "perturbation_index": 0, "direction": "negative", "lora_name": "c-"},
            ],
        }

    def abort(self, generation_id: str) -> dict:
        # Mirrors abort_generation -> complete_generation: must match the active
        # generation; raises the benign error the client is expected to swallow.
        if self.active_generation_id is None:
            raise RuntimeError(
                f"/abort_zorl_generation on {self.url} → HTTP 400: "
                f"No active ZORL generation for model_id=session"
            )
        if self.active_generation_id != generation_id:
            raise RuntimeError(
                f"/abort_zorl_generation on {self.url} → HTTP 400: "
                f"Active ZORL generation mismatch for model_id=session: "
                f"expected {self.active_generation_id}, got {generation_id}"
            )
        self.aborted.append(generation_id)
        self.active_generation_id = None
        return {"success": True, "generation_id": generation_id}


class _FakePS(_FakeReplica):
    """The PS additionally serves /ps_apply_and_broadcast. In this fake the PS
    fold does NOT auto-complete its own generation (so the client-side teardown
    is what clears it) — exercising the defensive PS abort. (If the live server
    DID auto-complete, the client's PS abort would hit the benign 'no active'
    path; that case is covered separately below.)"""

    def __init__(self, url: str, *, complete_on_apply: bool = False, **kw):
        super().__init__(url, **kw)
        self.complete_on_apply = complete_on_apply
        self.applied: list[str] = []

    def apply_and_broadcast(self, generation_id: str) -> dict:
        if self.active_generation_id != generation_id:
            raise RuntimeError(
                f"/ps_apply_and_broadcast on {self.url} → HTTP 400: "
                f"generation mismatch (active={self.active_generation_id}, got={generation_id})"
            )
        self.applied.append(generation_id)
        if self.complete_on_apply:
            self.active_generation_id = None
        return {
            "success": True,
            "stage": "synced",
            "used_pairs": 1,
            "metrics": {"update_norm": 1.0},
            "sync_stats": {"total_nnz": 123, "density": 0.5, "packed_bytes": 999, "tp_size": 1},
        }


class _Servers:
    """Routes mocked _post(url, path, payload) to the right fake by URL."""

    def __init__(self, replicas: dict[str, _FakeReplica], ps: _FakePS):
        self.replicas = replicas
        self.ps = ps

    def _node(self, url: str) -> _FakeReplica:
        if url == self.ps.url:
            return self.ps
        return self.replicas[url]

    def post(self, url: str, path: str, payload: dict, *, timeout: float = 300.0, headers=None) -> dict:
        node = self._node(url)
        if path == "/start_zorl_generation":
            return node.start()
        if path == "/abort_zorl_generation":
            return node.abort(str(payload["generation_id"]))
        if path == "/ps_apply_and_broadcast":
            assert isinstance(node, _FakePS)
            return node.apply_and_broadcast(str(payload["generation_id"]))
        if path == "/flush_cache":
            return {}
        raise AssertionError(f"unexpected path in test: {path}")


def _ps_step_with_teardown(servers: _Servers, replica_urls: list[str], ps_url: str, *, session_id: str) -> str:
    """The PS-mode step sequence AS FIXED in zorl_client: start (replicas+PS) ->
    apply+broadcast (PS) -> abort (replicas + PS)."""
    orig_post = zc._post
    zc._post = servers.post
    try:
        gen = zc.start_zorl_generation_all(replica_urls, session_id=session_id, num_pairs=1)
        gen_id = gen["generation_id"]
        ps_gen = zc.start_zorl_generation_all([ps_url], session_id=session_id, num_pairs=1)
        assert str(ps_gen["generation_id"]) == str(gen_id), (gen_id, ps_gen["generation_id"])
        apply_result = zc.ps_apply_and_broadcast(
            ps_url, replica_urls, session_id=session_id, generation_id=gen_id,
            candidate_rewards=[], lr=0.01,
        )
        assert apply_result.get("success") is True
        # --- the fix under test ---
        if apply_result.get("success"):
            zc.abort_zorl_generation_all(replica_urls, session_id=session_id, generation_id=gen_id)
            zc.abort_zorl_generation_all([ps_url], session_id=session_id, generation_id=gen_id)
        return gen_id
    finally:
        zc._post = orig_post


def _ps_step_without_teardown(servers: _Servers, replica_urls: list[str], ps_url: str, *, session_id: str) -> str:
    """The BUGGY (pre-fix) PS-mode step: start + apply, no per-step teardown."""
    orig_post = zc._post
    zc._post = servers.post
    try:
        gen = zc.start_zorl_generation_all(replica_urls, session_id=session_id, num_pairs=1)
        gen_id = gen["generation_id"]
        zc.start_zorl_generation_all([ps_url], session_id=session_id, num_pairs=1)
        zc.ps_apply_and_broadcast(
            ps_url, replica_urls, session_id=session_id, generation_id=gen_id,
            candidate_rewards=[], lr=0.01,
        )
        return gen_id
    finally:
        zc._post = orig_post


def _build_servers(n_replicas: int = 3, *, complete_on_apply: bool = False) -> tuple[_Servers, list[str], str]:
    replica_urls = [f"http://replica-{i}:30060" for i in range(n_replicas)]
    ps_url = "http://ps-0:30060"
    replicas = {u: _FakeReplica(u) for u in replica_urls}
    ps = _FakePS(ps_url, complete_on_apply=complete_on_apply)
    return _Servers(replicas, ps), replica_urls, ps_url


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fixed_loop_runs_multiple_steps_with_fresh_gen_and_teardown():
    servers, replica_urls, ps_url = _build_servers(n_replicas=3)
    n_steps = 4
    seen_gen_ids: list[str] = []
    for step in range(n_steps):
        gen_id = _ps_step_with_teardown(servers, replica_urls, ps_url, session_id="session")
        seen_gen_ids.append(gen_id)

    # (1) A FRESH generation id per step, monotonically advancing g000000..g000003.
    assert seen_gen_ids == [f"session-family-000000-g{i:06d}" for i in range(n_steps)], seen_gen_ids
    assert len(set(seen_gen_ids)) == n_steps

    # (2) Every replica AND the PS started AND aborted each generation, in order.
    for url, replica in servers.replicas.items():
        assert replica.started == seen_gen_ids, (url, replica.started)
        assert replica.aborted == seen_gen_ids, (url, replica.aborted)
        assert replica.active_generation_id is None, url
    assert servers.ps.started == seen_gen_ids
    assert servers.ps.aborted == seen_gen_ids
    assert servers.ps.applied == seen_gen_ids
    assert servers.ps.active_generation_id is None
    print("PASS test_fixed_loop_runs_multiple_steps_with_fresh_gen_and_teardown")


def test_buggy_loop_step2_hits_already_active_400():
    """Without the per-step teardown, step 2's /start_zorl_generation fails with
    the live 'already has active generation' 400 — confirming the test exercises
    the real bug, and that the teardown is what prevents it."""
    servers, replica_urls, ps_url = _build_servers(n_replicas=3)
    # Step 1 completes start+apply (no teardown).
    _ps_step_without_teardown(servers, replica_urls, ps_url, session_id="session")
    # Step 2 must blow up on start.
    raised = False
    try:
        _ps_step_without_teardown(servers, replica_urls, ps_url, session_id="session")
    except RuntimeError as e:
        raised = True
        assert "already has active generation" in str(e), str(e)
    assert raised, "expected the buggy loop to fail on step 2's /start_zorl_generation"
    print("PASS test_buggy_loop_step2_hits_already_active_400")


def test_teardown_idempotent_on_retry_when_ps_auto_completes():
    """If the live PS auto-completes its own generation inside the fold (so the
    PS abort hits 'no active generation'), the client must still succeed — the
    abort is idempotent and the benign error is swallowed."""
    servers, replica_urls, ps_url = _build_servers(n_replicas=2, complete_on_apply=True)
    gen_id = _ps_step_with_teardown(servers, replica_urls, ps_url, session_id="session")
    assert gen_id == "session-family-000000-g000000"
    # PS generation was cleared by the apply; the client's PS abort was a no-op
    # (swallowed), and the replicas were cleared by the client.
    assert servers.ps.active_generation_id is None
    assert servers.ps.aborted == []  # PS abort hit the benign path, never recorded
    for replica in servers.replicas.values():
        assert replica.aborted == [gen_id]
        assert replica.active_generation_id is None
    # A second step still advances to a fresh gen (no wedge).
    gen_id2 = _ps_step_with_teardown(servers, replica_urls, ps_url, session_id="session")
    assert gen_id2 == "session-family-000000-g000001"
    print("PASS test_teardown_idempotent_on_retry_when_ps_auto_completes")


def test_abort_all_double_call_is_idempotent():
    """A retry that re-enters the teardown (abort the same gen twice) must not
    raise: the second abort hits 'no active generation' and is swallowed."""
    servers, replica_urls, ps_url = _build_servers(n_replicas=2)
    orig_post = zc._post
    zc._post = servers.post
    try:
        gen = zc.start_zorl_generation_all(replica_urls, session_id="session", num_pairs=1)
        gen_id = gen["generation_id"]
        zc.abort_zorl_generation_all(replica_urls, session_id="session", generation_id=gen_id)
        # Re-enter (simulating a retry of the apply/teardown block):
        zc.abort_zorl_generation_all(replica_urls, session_id="session", generation_id=gen_id)
    finally:
        zc._post = orig_post
    for replica in servers.replicas.values():
        assert replica.active_generation_id is None
    print("PASS test_abort_all_double_call_is_idempotent")


def test_abort_all_raises_on_genuine_failure():
    """A non-benign server error (e.g. connection error / unexpected 500) must
    propagate, not be silently swallowed — the step should fail loud."""
    replica_urls = ["http://replica-0:30060", "http://replica-1:30060"]

    def boom(url, path, payload, *, timeout=300.0, headers=None):
        raise RuntimeError(f"{path} on {url} → HTTP 500: internal kaboom")

    orig_post = zc._post
    zc._post = boom
    raised = False
    try:
        zc.abort_zorl_generation_all(replica_urls, session_id="session", generation_id="g")
    except RuntimeError as e:
        raised = True
        assert "failed to clear generation" in str(e), str(e)
    finally:
        zc._post = orig_post
    assert raised, "genuine abort failures must propagate"
    print("PASS test_abort_all_raises_on_genuine_failure")


def _run_all():
    test_fixed_loop_runs_multiple_steps_with_fresh_gen_and_teardown()
    test_buggy_loop_step2_hits_already_active_400()
    test_teardown_idempotent_on_retry_when_ps_auto_completes()
    test_abort_all_double_call_is_idempotent()
    test_abort_all_raises_on_genuine_failure()
    print("\nALL PASS")


if __name__ == "__main__":
    _run_all()
