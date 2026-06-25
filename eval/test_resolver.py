"""
test_resolver.py — prove the NL->form resolver maps questions to the right metric
and that the full pipe (NL -> form -> compiled SQL -> rows) matches the gold answer.

Unlike the compiler test, this one has an LLM in the loop, so it's PROBABILISTIC
(temperature 0 keeps it stable). The point isn't 100.000% determinism — it's that
the resolver is BOUNDED: it can only pick catalog names, an out-of-scope question
falls through (None), and the end-to-end result is checked against the gold rows.

Run:
    python eval/test_resolver.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config                       # noqa: E402
import metrics_layer as ml          # noqa: E402

# (natural-language question, expected metric or None, gold reference SQL or None)
CASES = [
    ("How many credits have been granted in total?",
     "total_granted_credits",
     "SELECT SUM(granted_credits) FROM tenant_credit_grants"),

    ("How many credit grants are there for each source?",
     "grant_count",
     "SELECT source, COUNT(*) FROM tenant_credit_grants GROUP BY source"),

    ("Show each tenant's name and their total granted credits.",
     "total_granted_credits",
     "SELECT t.name, SUM(g.granted_credits) FROM tenant_credit_grants g "
     "JOIN tenants t ON g.tenant_id = t.tenant_id GROUP BY t.name"),

    # Out of scope for the catalog -> must fall through (resolve returns None).
    ("What's the weather in Tokyo today?", None, None),
]


def main():
    print("=" * 74)
    print("  RESOLVER TEST  (NL -> form -> SQL, LLM in the loop)")
    print("=" * 74)
    catalog = ml.load_catalog()
    results = []

    conn = None
    try:
        import pymysql
        conn = pymysql.connect(
            host=config.MYSQL_HOST, port=config.MYSQL_PORT, user=config.MYSQL_USER,
            password=config.MYSQL_PASSWORD, database=config.MYSQL_DB,
            connect_timeout=5)
    except Exception as exc:
        print(f"  [note] no MySQL — end-to-end row checks skipped ({exc})")

    def run(sql):
        with conn.cursor() as cur:
            cur.execute(sql)
            return sorted(str(r) for r in cur.fetchall())

    for q, expected_metric, gold in CASES:
        form = ml.resolve(q, catalog)

        if expected_metric is None:                      # out-of-scope case
            ok = form is None
            results.append(ok)
            print(f"\n  [{'OK' if ok else 'FAIL'}] out-of-scope falls through")
            print(f"        Q: {q!r}  ->  {form}")
            continue

        ok_metric = bool(form) and form.get("metric") == expected_metric
        results.append(ok_metric)
        print(f"\n  [{'OK' if ok_metric else 'FAIL'}] resolves to the right metric")
        print(f"        Q: {q!r}")
        print(f"        form: {form}")

        if ok_metric and conn and gold:
            try:
                sql = ml.compile_form(form, catalog)
                ok_rows = run(sql) == run(gold)
            except Exception as exc:
                ok_rows, sql = False, f"(error: {exc})"
            results.append(ok_rows)
            print(f"  [{'OK' if ok_rows else 'FAIL'}] end-to-end matches gold rows")
            print(f"        SQL: {sql}")

    if conn:
        conn.close()
    passed, total = sum(results), len(results)
    print("\n" + "=" * 74)
    print(f"  {passed}/{total} passed")
    print("=" * 74)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
