"""
test_semantic_compiler.py — prove the Tier 0 metric compiler is correct.

Two layers of proof, both with NO LLM in the loop:

  Part 1 (always runs, no backends): compile each form and assert it produces the
  exact deterministic SQL we designed. Proves the compiler is stable + predictable.

  Part 2 (best effort, needs MySQL): run the COMPILED sql and the GOLD reference
  sql and assert they return the SAME rows. Proves the compiler's output is
  *semantically equal to the hand-written gold answer* — the real claim.

These four cases (q06/q11/q19/q20) all come from ONE tiny catalog (metrics.yml),
which is the whole point: a small catalog × dimensions/filters covers many
questions, every one guaranteed correct.

Run:
    python eval/test_semantic_compiler.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config                       # noqa: E402
import metrics_layer as ml          # noqa: E402

# (gold id, form, expected compiled SQL, gold reference SQL)
CASES = [
    ("q06",
     {"metric": "total_granted_credits"},
     "SELECT SUM(granted_credits) AS total_granted_credits FROM tenant_credit_grants",
     "SELECT SUM(granted_credits) FROM tenant_credit_grants"),

    ("q11",
     {"metric": "grant_count", "dimensions": ["by_source"]},
     "SELECT source AS by_source, COUNT(*) AS grant_count "
     "FROM tenant_credit_grants GROUP BY source",
     "SELECT source, COUNT(*) FROM tenant_credit_grants GROUP BY source"),

    ("q20",
     {"metric": "total_granted_credits", "dimensions": ["by_tenant"]},
     "SELECT t1.name AS by_tenant, SUM(t0.granted_credits) AS total_granted_credits "
     "FROM tenant_credit_grants t0 JOIN tenants t1 ON t0.tenant_id = t1.tenant_id "
     "GROUP BY t1.name",
     "SELECT t.name, SUM(g.granted_credits) FROM tenant_credit_grants g "
     "JOIN tenants t ON g.tenant_id = t.tenant_id GROUP BY t.name"),

    ("q19",
     {"metric": "total_granted_credits", "dimensions": ["by_metering_key"],
      "filters": [{"name": "tenant_is", "value": "wsee"}]},
     "SELECT t0.metering_key AS by_metering_key, "
     "SUM(t0.granted_credits) AS total_granted_credits "
     "FROM tenant_credit_grants t0 JOIN tenants t1 ON t0.tenant_id = t1.tenant_id "
     "WHERE t1.name = 'wsee' GROUP BY t0.metering_key",
     "SELECT g.metering_key, SUM(g.granted_credits) FROM tenant_credit_grants g "
     "JOIN tenants t ON g.tenant_id = t.tenant_id WHERE t.name = 'wsee' "
     "GROUP BY g.metering_key"),

    ("q01",
     {"metric": "plugin_count"},
     "SELECT COUNT(*) AS plugin_count FROM plugins",
     "SELECT COUNT(*) FROM plugins"),

    ("q05",
     {"metric": "workflow_count"},
     "SELECT COUNT(*) AS workflow_count FROM workflows",
     "SELECT COUNT(*) FROM workflows"),

    ("q07",
     {"metric": "max_granted_credit"},
     "SELECT MAX(granted_credits) AS max_granted_credit FROM tenant_credit_grants",
     "SELECT MAX(granted_credits) FROM tenant_credit_grants"),

    ("q08",
     {"metric": "total_credit_transaction_amount"},
     "SELECT SUM(amount) AS total_credit_transaction_amount "
     "FROM tenant_credit_transactions",
     "SELECT SUM(amount) FROM tenant_credit_transactions"),

    ("q09",
     {"metric": "node_execution_count",
      "filters": [{"name": "node_status_is", "value": "FAILED"}]},
     "SELECT COUNT(*) AS node_execution_count FROM node_execution_logs "
     "WHERE status = 'FAILED'",
     "SELECT COUNT(*) FROM node_execution_logs WHERE status = 'FAILED'"),

    ("q10",
     {"metric": "avg_workflow_duration"},
     "SELECT AVG(duration_ms) AS avg_workflow_duration FROM workflow_execution_logs",
     "SELECT AVG(duration_ms) FROM workflow_execution_logs"),

    ("q12",
     {"metric": "workflow_execution_count", "dimensions": ["by_workflow_status"]},
     "SELECT status AS by_workflow_status, COUNT(*) AS workflow_execution_count "
     "FROM workflow_execution_logs GROUP BY status",
     "SELECT status, COUNT(*) FROM workflow_execution_logs GROUP BY status"),

    ("q13",
     {"metric": "node_execution_count", "dimensions": ["by_node_status"]},
     "SELECT status AS by_node_status, COUNT(*) AS node_execution_count "
     "FROM node_execution_logs GROUP BY status",
     "SELECT status, COUNT(*) FROM node_execution_logs GROUP BY status"),

    ("q14",
     {"metric": "plugin_count", "dimensions": ["by_category"]},
     "SELECT category AS by_category, COUNT(*) AS plugin_count "
     "FROM plugins GROUP BY category",
     "SELECT category, COUNT(*) FROM plugins GROUP BY category"),
]


def main():
    print("=" * 74)
    print("  SEMANTIC COMPILER TEST  (Tier 0 — no LLM)")
    print("=" * 74)
    catalog = ml.load_catalog()
    results = []

    # --- Part 1: deterministic compile (no DB) ---
    print("\n--- Part 1: compiles to the exact SQL we designed ---")
    for cid, form, expected, _gold in CASES:
        got = ml.compile_form(form, catalog)
        ok = got == expected
        results.append(ok)
        print(f"  [{'OK' if ok else 'FAIL'}] {cid} compile")
        if not ok:
            print(f"        expected: {expected}")
            print(f"        got     : {got}")

    # --- Part 2: semantic equivalence vs gold (needs MySQL) ---
    print("\n--- Part 2: compiled SQL == gold result set (live MySQL) ---")
    conn = None
    try:
        import pymysql
        conn = pymysql.connect(
            host=config.MYSQL_HOST, port=config.MYSQL_PORT, user=config.MYSQL_USER,
            password=config.MYSQL_PASSWORD, database=config.MYSQL_DB,
            connect_timeout=5)
    except Exception as exc:
        print(f"  [SKIP] no MySQL connection ({exc})")

    if conn:
        def run(sql):
            with conn.cursor() as cur:
                cur.execute(sql)
                return sorted(str(r) for r in cur.fetchall())
        for cid, form, _expected, gold in CASES:
            try:
                ok = run(ml.compile_form(form, catalog)) == run(gold)
            except Exception as exc:
                ok = False
                print(f"        {cid} error: {exc}")
            results.append(ok)
            print(f"  [{'OK' if ok else 'FAIL'}] {cid} matches gold result set")
        conn.close()

    passed, total = sum(results), len(results)
    print("\n" + "=" * 74)
    print(f"  {passed}/{total} passed")
    print("=" * 74)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
