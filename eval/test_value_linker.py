"""
test_value_linker.py — prove value-linking fixes the 'Active' vs 'ACTIVE' failure.

Needs live MySQL (it reads each column's real values). The gold data has
plugins.status = 'ACTIVE', so a query written with 'active' (wrong case) must be
corrected to 'ACTIVE' and then return the same rows as the correctly-cased query.

Run:
    python eval/test_value_linker.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config                       # noqa: E402
import value_linker as vl          # noqa: E402

SCHEMA = json.loads(config.SCHEMA_JSON.read_text(encoding="utf-8"))
_results = []


def check(name, cond, detail=""):
    _results.append(cond)
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail and not cond else ""))


def main():
    print("=" * 74)
    print("  VALUE-LINKER TEST")
    print("=" * 74)

    conn = None
    try:
        import pymysql
        conn = pymysql.connect(
            host=config.MYSQL_HOST, port=config.MYSQL_PORT, user=config.MYSQL_USER,
            password=config.MYSQL_PASSWORD, database=config.MYSQL_DB,
            connect_timeout=5)
    except Exception as exc:
        print(f"  [SKIP] needs live MySQL ({exc})")
        return 0

    def run(sql):
        with conn.cursor() as cur:
            cur.execute(sql)
            return sorted(str(r) for r in cur.fetchall())

    # 1. wrong-case value gets corrected, and the corrected query == the right one
    bad = "SELECT plugin_id FROM plugins WHERE status = 'active'"
    fixed, corr = vl.link_values(bad, SCHEMA)
    print(f"\n  input : {bad}")
    print(f"  output: {fixed}")
    print(f"  corrections: {corr}")
    check("1a: a correction was recorded", len(corr) == 1)
    check("1b: it was a CASE fix to 'ACTIVE'",
          bool(corr) and corr[0]["to"] == "ACTIVE" and corr[0]["kind"] == "case")
    try:
        gold = run("SELECT plugin_id FROM plugins WHERE status = 'ACTIVE'")
        check("1c: corrected query returns the same rows as the right-cased one",
              run(fixed) == gold)
    except Exception as exc:
        check("1c: corrected query runs", False, str(exc))

    # 2. an already-correct value is left untouched (no churn, no false correction)
    good = "SELECT plugin_id FROM plugins WHERE status = 'ACTIVE'"
    out, corr2 = vl.link_values(good, SCHEMA)
    check("2: already-correct value is unchanged", out == good and corr2 == [])

    # 3. a value that doesn't exist and isn't close → no correction, no crash
    nonsense = "SELECT plugin_id FROM plugins WHERE status = 'zzzznotathing'"
    out3, corr3 = vl.link_values(nonsense, SCHEMA)
    check("3: unknown value is not force-corrected", corr3 == [])

    conn.close()
    passed, total = sum(_results), len(_results)
    print("\n" + "=" * 74)
    print(f"  {passed}/{total} passed")
    print("=" * 74)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
