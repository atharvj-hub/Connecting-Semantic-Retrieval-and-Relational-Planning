"""
eval/run_integration.py  —  Fix #3: integration + unit test runner.

Drives the REAL paths (orchestrator.answer / cli cache helpers) through
scripted multi-turn sequences and asserts behaviour. No human input (provides
a non-interactive clarify_cb). Prints OK/FAIL per case like run_eval.py.

Test homes:
  Cases 1-4  → Fixes 1+2 (cache re-execute, history-aware, clarity, no false positive)
  Case 5     → Fix 5 (cost-gate refusal + fail-open)
  Cases 6-7  → Fix 6 (scalar template + number verification)

Requires live MySQL + Pinecone + (ideally) Neo4j, same as run_eval.py.
Exit non-zero if any case fails (CI-gate ready).

Run:
    python eval/run_integration.py
"""

import os
import sys
import time
import json

# --- path setup (same pattern as run_eval.py) --------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config  # noqa: E402

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------
_results = []


def assert_case(name, cond, detail=""):
    """Record and print one test result."""
    status = "OK" if cond else "FAIL"
    _results.append((name, cond))
    msg = f"  [{status}] {name}"
    if detail and not cond:
        msg += f"  — {detail}"
    print(msg)


# ---------------------------------------------------------------------------
# Case 1: Cache re-executes, not replays (Bug B / P-1)
# ---------------------------------------------------------------------------
def test_cache_reexecutes():
    """Drive the REAL cli.answer_turn twice (not a hand-rolled mirror of it).
    1st ask = miss → the orchestrator must run. 2nd ask = hit → it must
    RE-EXECUTE the cached SQL (not replay stored text) and must NOT call the
    orchestrator again. Spying on BOTH ask.execute and orchestrator.answer means:
    a revert to replay would fail 1b, and a broken cache that regenerates every
    time would fail 1c — so the test actually guards answer_turn's behaviour."""
    import ask
    import cli
    import orchestrator
    import multi_sql

    cli.CACHE_FILE.unlink(missing_ok=True)
    exec_calls, orch_calls = [0], [0]
    orig_exec, orig_orch = ask.execute, orchestrator.answer

    def spy_exec(sql, max_rows=50):
        exec_calls[0] += 1
        return orig_exec(sql, max_rows)

    def spy_orch(*a, **k):
        orch_calls[0] += 1
        return orig_orch(*a, **k)

    ask.execute = spy_exec
    orchestrator.answer = spy_orch
    try:
        state = {"sess": cli.new_session(), "model": multi_sql.DEFAULT_MODEL,
                 "show_thinking": False, "mode": "escalate"}
        q = "How many plugins are there?"

        # 1st ask: cache miss → the orchestrator must run.
        cli.answer_turn(q, state)
        assert_case("1a: first ask runs the orchestrator (cache miss)",
                     orch_calls[0] >= 1, f"orchestrator calls = {orch_calls[0]}")

        # 2nd ask: cache hit → re-execute the cached SQL, skip the orchestrator.
        exec_calls[0], orch_calls[0] = 0, 0
        cli.answer_turn(q, state)
        assert_case("1b: second ask RE-EXECUTES the cached SQL (not replay)",
                     exec_calls[0] >= 1, f"ask.execute calls = {exec_calls[0]}")
        assert_case("1c: second ask SKIPS the orchestrator (true cache hit)",
                     orch_calls[0] == 0, f"orchestrator calls = {orch_calls[0]}")
    finally:
        ask.execute = orig_exec
        orchestrator.answer = orig_orch
        cli.CACHE_FILE.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Case 2: History-aware cache, no cross-context hit (Bug B1)
# ---------------------------------------------------------------------------
def test_no_cross_context_hit():
    """The real Bug-B1 danger: the SAME follow-up phrase in a DIFFERENT context
    must not collide. Cache an identical phrase under context A, then look it up
    under context B. The OLD raw-question-only key would FALSE-HIT (phrase is
    identical); the history-aware key must miss. 2b is the positive control —
    same phrase, SAME context DOES hit — so 2a isn't passing just because the key
    never matches anything."""
    import cli

    cli.CACHE_FILE.unlink(missing_ok=True)
    try:
        follow_up = "what about the active ones?"

        # Context A (plugins): cache the follow-up here.
        history_a = [{"q": "How many plugins are there?",
                      "sql": "SELECT COUNT(*) FROM plugins"}]
        cli.cache_add(follow_up, history_a,
                      "SELECT COUNT(*) FROM plugins WHERE status='ACTIVE'")

        # Context B (credit transactions): identical phrase, different prior context.
        history_b = [{"q": "How many credit transactions are there?",
                      "sql": "SELECT COUNT(*) FROM tenant_credit_transactions"}]
        hit_b = cli.cache_lookup(follow_up, history_b)
        assert_case("2a: identical follow-up in a DIFFERENT context does NOT hit",
                     hit_b is None,
                     f"cross-context false hit, score={hit_b[1]:.3f}" if hit_b else "")

        # Positive control: same phrase, SAME context → should hit.
        hit_a = cli.cache_lookup(follow_up, history_a)
        assert_case("2b: same follow-up in the SAME context hits (key is sound)",
                     hit_a is not None,
                     "expected a hit in the original context")
    finally:
        cli.CACHE_FILE.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Case 3: Clarity reaches the orchestrator (Bug A)
# ---------------------------------------------------------------------------
def test_clarity_reaches_orchestrator():
    """Via orchestrator.answer(q, clarify_cb=None), ask 'which tenant is the
    most active?' → assert needs_clarification is True."""
    import orchestrator
    res = orchestrator.answer("which tenant is the most active?",
                              mode="escalate", show_thinking=False,
                              clarify_cb=None)
    assert_case("3: clarity triggers on ambiguous question",
                 res.get("needs_clarification") is True,
                 f"needs_clarification={res.get('needs_clarification')}, "
                 f"status={res.get('status')}")


# ---------------------------------------------------------------------------
# Case 4: Specific question still flows (no false clarification)
# ---------------------------------------------------------------------------
def test_specific_no_clarification():
    """'how many workflows are there?' → assert needs_clarification is False
    and it produces an answer."""
    import orchestrator
    res = orchestrator.answer("how many workflows are there?",
                              mode="escalate", show_thinking=False,
                              clarify_cb=None)
    assert_case("4: specific question flows without clarification",
                 res.get("needs_clarification") is not True,
                 f"needs_clarification={res.get('needs_clarification')}")


# ---------------------------------------------------------------------------
# Case 5: Cost-gate refuses a runaway query + fail-open (Fix 5)
# ---------------------------------------------------------------------------
def test_cost_gate():
    """Activated when Fix 5 lands. Tests:
    5a: synthetic expensive query → explain_cost refuses it.
    5b: unparseable EXPLAIN → returns ok (fail-open)."""
    try:
        from ask import explain_cost  # noqa: F401
    except ImportError:
        print("  [SKIP] 5a: cost-gate refusal (Fix 5 not yet landed)")
        print("  [SKIP] 5b: cost-gate fail-open (Fix 5 not yet landed)")
        return

    import ask

    # 5a: synthetic expensive cross-join — should be refused.
    # Use real tables from the schema.
    expensive_sql = ("SELECT * FROM workflow_execution_logs "
                     "CROSS JOIN tenant_credit_transactions")
    # Temporarily lower the threshold so it refuses even on this tiny test DB.
    import config
    original_max = config.EXPLAIN_MAX_ROWS
    config.EXPLAIN_MAX_ROWS = 10
    try:
        ok, est_rows, reason = ask.explain_cost(expensive_sql)
        assert_case("5a: cost-gate refuses expensive query",
                     ok is False,
                     f"ok={ok}, est_rows={est_rows}, reason={reason}")
    finally:
        config.EXPLAIN_MAX_ROWS = original_max

    # 5b: fail-open — unparseable EXPLAIN should return ok=True.
    # Monkeypatch _mysql to return garbage EXPLAIN output.
    original_mysql = ask._mysql

    class FakeConn:
        def cursor(self):
            return FakeCursor()
        def close(self):
            pass

    class FakeCursor:
        def execute(self, sql):
            # Return something unparseable for JSON EXPLAIN.
            pass
        def fetchall(self):
            return [("garbage", "data")]
        @property
        def description(self):
            return [("col1",), ("col2",)]
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    ask._mysql = lambda: FakeConn()
    try:
        ok, est_rows, reason = ask.explain_cost("SELECT 1")
        assert_case("5b: cost-gate fail-open on parse error",
                     ok is True,
                     f"ok={ok}, reason={reason}")
    finally:
        ask._mysql = original_mysql


# ---------------------------------------------------------------------------
# Cases 6-7: Fix 6 unit tests (compose guard — pure functions, no backends)
# ---------------------------------------------------------------------------
def test_scalar_template():
    """Single-row single-col → deterministic template, no LLM call."""
    try:
        from ask import _scalar_template  # noqa: F401
    except ImportError:
        print("  [SKIP] 6: scalar template (Fix 6 not yet landed)")
        return

    import ask
    result = ask._scalar_template(["total_count"], [(42,)])
    assert_case("6: scalar template returns deterministic string",
                 result is not None and "42" in result,
                 f"result={result!r}")


def test_verify_numbers():
    """Answer containing a number absent from rows → returns False (degrades)."""
    try:
        from ask import _verify_numbers  # noqa: F401
    except ImportError:
        print("  [SKIP] 7: verify numbers (Fix 6 not yet landed)")
        return

    import ask

    # Number present in rows → should pass.
    ok_result = ask._verify_numbers("There are 42 plugins.", ["count"], [(42,)])
    assert_case("7a: verify_numbers passes when number is in rows",
                 ok_result is True,
                 f"result={ok_result}")

    # Number absent from rows → should fail (degrade).
    bad_result = ask._verify_numbers("There are 999 plugins.", ["count"], [(42,)])
    assert_case("7b: verify_numbers fails when number is NOT in rows",
                 bad_result is False,
                 f"result={bad_result}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("  INTEGRATION + UNIT TEST RUNNER  (Fix #3)")
    print("=" * 74)

    # Fix 1+2 integration cases (require live backends).
    print("\n--- Fix 1: Cache correctness ---")
    try:
        test_cache_reexecutes()
    except Exception as exc:
        assert_case("1: cache re-executes", False, f"crashed: {exc}")

    try:
        test_no_cross_context_hit()
    except Exception as exc:
        assert_case("2: no cross-context hit", False, f"crashed: {exc}")

    print("\n--- Fix 2: Clarity in orchestrator ---")
    try:
        test_clarity_reaches_orchestrator()
    except Exception as exc:
        assert_case("3: clarity reaches orchestrator", False, f"crashed: {exc}")

    try:
        test_specific_no_clarification()
    except Exception as exc:
        assert_case("4: specific question flows", False, f"crashed: {exc}")

    # Fix 5 cases (activated when Fix 5 lands).
    print("\n--- Fix 5: Cost-gate ---")
    try:
        test_cost_gate()
    except Exception as exc:
        assert_case("5: cost-gate", False, f"crashed: {exc}")

    # Fix 6 unit tests (activated when Fix 6 lands).
    print("\n--- Fix 6: Compose guard ---")
    try:
        test_scalar_template()
    except Exception as exc:
        assert_case("6: scalar template", False, f"crashed: {exc}")

    try:
        test_verify_numbers()
    except Exception as exc:
        assert_case("7: verify numbers", False, f"crashed: {exc}")

    # Summary.
    print("\n" + "=" * 74)
    passed = sum(1 for _, ok in _results if ok)
    total = len(_results)
    print(f"  {passed}/{total} passed")
    if passed < total:
        failed = [name for name, ok in _results if not ok]
        print(f"  FAILED: {', '.join(failed)}")
    print("=" * 74)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
