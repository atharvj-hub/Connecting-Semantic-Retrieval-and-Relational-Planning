"""
value_linker.py — fix filter VALUES to the real values in the database.

THE #1 SILENT FAILURE
---------------------
The model writes  WHERE status = 'Active'  but the column actually stores
'ACTIVE'. The SQL is perfectly valid, it runs fine, and it returns ZERO rows —
so the agent confidently answers "0" to a question whose real answer is not zero.
Validation can't catch this (the SQL *is* valid); only checking the value against
the real data can.

WHAT THIS DOES
--------------
After a SQL is generated (and validated), we find each string filter value
(`col = 'x'`, `col IN ('a','b')`), look at the column's REAL distinct values, and
correct the value:
  * exact   — already correct, leave it.
  * case    — same word, different casing ('Active' -> 'ACTIVE'). SAFE: auto-fix.
  * fuzzy   — close spelling ('Activ' -> 'ACTIVE'). A JUDGEMENT call — we apply it
              but record it, and only when we can see the column's FULL domain.

PRODUCTION NOTE (worth internalising): case-correction is safe to auto-apply (it's
the same value). Fuzzy-correction is *not* obviously safe — silently changing a
value the user typed can hide a real "that value doesn't exist" signal. A
production system would usually SURFACE a fuzzy suggestion for confirmation rather
than apply it blindly. We apply it here (small project) but every change is recorded
in the returned `corrections` so it's auditable.
"""

import difflib

try:
    import sqlglot
    from sqlglot import exp
    _HAVE_SQLGLOT = True
except ModuleNotFoundError:                  # pragma: no cover
    _HAVE_SQLGLOT = False

DISTINCT_LIMIT = 200     # categorical columns fit well under this; a column that
                         # hits the cap is high-cardinality, so we skip fuzzy on it
                         # (our view of its domain is only partial).
FUZZY_CUTOFF = 0.85      # how close a spelling must be to auto-correct


def _distinct_values(table, column, limit=DISTINCT_LIMIT):
    """Up to `limit` distinct non-null values of a column. For low-cardinality
    categoricals (status, category, source...) this is the whole domain."""
    import ask                              # reuse the one MySQL connector
    try:
        conn = ask._mysql()
    except Exception:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT `%s` FROM `%s` WHERE `%s` IS NOT NULL "
                        "LIMIT %d" % (column, table, column, limit))
            return [r[0] for r in cur.fetchall()]
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _best_match(value, domain):
    """Return (corrected_value, kind) or (None, None). kind in exact|case|fuzzy.
    Fuzzy only fires when we likely have the FULL domain (domain smaller than the
    fetch cap) — otherwise a 'no match' might just be beyond our sample."""
    strs = [str(c) for c in domain]
    if value in strs:
        return value, "exact"
    low = value.lower()
    for c in strs:
        if c.lower() == low:
            return c, "case"
    if len(domain) < DISTINCT_LIMIT:                 # we can trust "not present"
        near = difflib.get_close_matches(value, strs, n=1, cutoff=FUZZY_CUTOFF)
        if near:
            return near[0], "fuzzy"
    return None, None


def link_values(sql, schema):
    """Correct string filter values in `sql` to the real DB values.

    Returns (sql, corrections). `corrections` is a list of
    {column, from, to, kind}. If nothing is corrected (or sqlglot/MySQL aren't
    available), returns the ORIGINAL sql unchanged and an empty list."""
    if not _HAVE_SQLGLOT:
        return sql, []
    try:
        tree = sqlglot.parse_one(sql, dialect="mysql")
    except Exception:
        return sql, []
    if tree is None:
        return sql, []

    cols = {t["table_name"]: {c["name"] for c in t["columns"]}
            for t in schema["tables"]}

    # alias / table-name -> real table (mirror of ask.validate's resolver)
    alias2t = {}
    for tn in tree.find_all(exp.Table):
        if tn.name in cols:
            alias2t[tn.alias or tn.name] = tn.name
            alias2t[tn.name] = tn.name

    def resolve(colnode):
        q = colnode.table
        if q:
            return alias2t.get(q)
        owners = [r for r in set(alias2t.values())
                  if colnode.name in cols.get(r, set())]
        return owners[0] if len(owners) == 1 else None

    corrections = []

    def fix(colnode, litnode):
        if not isinstance(litnode, exp.Literal) or not litnode.is_string:
            return
        table = resolve(colnode)
        if not table:
            return
        value = litnode.this
        domain = _distinct_values(table, colnode.name)
        if not domain:
            return
        match, kind = _best_match(value, domain)
        if match is not None and kind != "exact" and str(match) != value:
            corrections.append({"column": f"{table}.{colnode.name}",
                                "from": value, "to": str(match), "kind": kind})
            litnode.set("this", str(match))

    for eq in tree.find_all(exp.EQ):
        l, r = eq.this, eq.expression
        if isinstance(l, exp.Column) and isinstance(r, exp.Literal):
            fix(l, r)
        elif isinstance(r, exp.Column) and isinstance(l, exp.Literal):
            fix(r, l)
    for in_node in tree.find_all(exp.In):
        col = in_node.this
        if isinstance(col, exp.Column):
            for lit in in_node.expressions:
                fix(col, lit)

    if not corrections:
        return sql, []                       # no churn if nothing changed
    return tree.sql(dialect="mysql"), corrections
