"""
ask.py  —  Task 4: the ASK phase (question -> validated MySQL)

PHASE A (this file, so far): retrieve + assemble the focused context.
    1. embed the question (Ollama nomic-embed-text)
    2. Pinecone -> candidate tables/columns (similarity THRESHOLD + table CAP)
    3. column-hits map back to their tables -> candidate set
    4. Neo4j -> join keys among candidates (direct FK + shared-parent sibling
       joins), each tagged with fk_type (declared / inferred_*)
    5. assemble a focused context string (only those tables: columns, types,
       description, sample values + the join keys)

Later phases (B, C) add: qwen2.5-coder writes SQL -> validate vs schema -> repair.

Run:
    python src/ask.py "how many orders did each customer place"
"""

import json
import re
import sys
import time

import config

try:
    import ollama
    from pinecone import Pinecone
    from neo4j import GraphDatabase
except ModuleNotFoundError as exc:
    sys.exit(f"ERROR: missing package ({exc.name}). Run: pip install -r requirements.txt")

try:
    import sqlglot
    from sqlglot import exp
    _HAVE_SQLGLOT = True
except ModuleNotFoundError:
    _HAVE_SQLGLOT = False        # deep validation degrades to table-existence only

try:
    import pymysql
    _HAVE_MYSQL = True
except ModuleNotFoundError:
    _HAVE_MYSQL = False          # no live DB -> stop at validated SQL (Phase A-C)

# tunables for retrieval (the threshold + cap decision).
# Cosine scores never reach 0 even for nonsense (an unrelated question still tops
# out ~0.47), so a single low absolute threshold can't detect out-of-scope. We use
# a FLOOR (top hit must beat it, else out-of-scope) + a RELATIVE band off the top
# hit (keeps only tables close to the best match), capped.
SCORE_FLOOR = 0.55            # best hit below this => question not about this DB
CONFIDENCE_BAR = 0.62         # best hit below this (but above floor) => low-confidence
RELATIVE_MARGIN = 0.15        # keep tables scoring within this of the top hit
MAX_TABLES = 6                # cap candidate tables to keep the prompt focused
PINECONE_TOPK = 25            # pull enough hits before filtering

# P6: a read-only system never runs these; if the QUESTION asks for one we warn.
_DESTRUCTIVE_INTENT = re.compile(
    r"\b(delete|remove|drop|truncate|update|insert|modify|wipe|erase)\b", re.I)


# --- token accounting (--count-tokens) --------------------------------------
# Ollama returns the model's OWN exact token counts on every call:
#   prompt_eval_count = input (prompt) tokens, eval_count = output tokens.
# We accumulate them so a demo can show REAL usage instead of an estimate.
TOKEN_STATS = {"enabled": False, "events": []}


def _record_tokens(label, resp):
    if not TOKEN_STATS["enabled"]:
        return
    resp = resp or {}
    TOKEN_STATS["events"].append({"label": label,
                                  "in": resp.get("prompt_eval_count"),
                                  "out": resp.get("eval_count")})


def _print_token_report():
    ev = TOKEN_STATS["events"]
    if not ev:
        return
    print("\n" + "-" * 64 + "\nTOKEN USAGE (exact, reported by Ollama):\n" + "-" * 64)
    w = max(len(e["label"]) for e in ev)
    ti = to = 0
    for e in ev:
        i, o = e["in"] or 0, e["out"] or 0
        ti, to = ti + i, to + o
        print(f"  {e['label']:<{w}}   in {i:>6}   out {o:>5}")
    print("  " + "-" * (w + 24))
    print(f"  {'TOTAL':<{w}}   in {ti:>6}   out {to:>5}   ({ti + to} tokens)")
    if any(e["in"] in (None, 0) for e in ev):
        print("  (in=0 means Ollama served that prompt from cache - re-run cold "
              "for a true input count)")


def _embed(text):
    # Note: the embeddings endpoint doesn't report token counts, and the cost is
    # negligible (one short question, no generation), so it's left out of the report.
    return ollama.embeddings(model=config.OLLAMA_EMBED_MODEL, prompt=text)["embedding"]


# --- Step 2-3: retrieve candidate tables ------------------------------------

def retrieve(question):
    """Return (tables, hits) where tables is an ordered candidate-table list and
    hits is the raw top matches (for display). A table's score = the best score
    among its own table-doc and any of its column-docs."""
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index(config.PINECONE_INDEX)
    res = index.query(vector=_embed(question), top_k=PINECONE_TOPK,
                      include_metadata=True)
    matches = res["matches"]

    best = {}            # table -> {score, cols:[(col,score)]}
    for m in matches:
        md = m["metadata"]
        tbl, score = md["table"], m["score"]
        slot = best.setdefault(tbl, {"score": 0.0, "cols": []})
        slot["score"] = max(slot["score"], score)
        if md.get("kind") == "column":
            slot["cols"].append((md.get("column"), score))

    # rank tables by their best hit; apply floor (out-of-scope) + relative band
    ranked = sorted(best.items(), key=lambda kv: kv[1]["score"], reverse=True)
    if not ranked or ranked[0][1]["score"] < SCORE_FLOOR:
        return [], matches                      # out of scope
    top = ranked[0][1]["score"]
    tables = [(t, d) for t, d in ranked
              if d["score"] >= top - RELATIVE_MARGIN][:MAX_TABLES]
    return tables, matches


# --- Step 3b: graph expansion (Neo4j adds tables Pinecone missed) ------------

GRAPH_EXPANSION = 2           # max FK-parent tables to pull into the candidate set


def expand_candidates(table_names):
    """The core of the two-DB design: Pinecone finds tables by meaning but can
    MISS an entity/bridge table the question needs (e.g. it retrieves orders but
    not `customers`, which holds the customer NAME). Neo4j knows the candidates all
    REFERENCE `customers`, so we pull the most-referenced parent tables in.
    Returns (added_table_names, error)."""
    if not table_names:
        return [], None
    try:
        driver = GraphDatabase.driver(
            config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    except Exception as exc:
        return [], f"Neo4j unavailable ({exc})"
    added = []
    try:
        with driver.session() as s:
            for r in s.run(
                "MATCH (ta:Table)-[:HAS_COLUMN]->(:Column)-[:REFERENCES]->"
                "(:Column)<-[:HAS_COLUMN]-(p:Table) "
                "WHERE ta.name IN $t AND NOT p.name IN $t "
                "RETURN p.name AS name, count(DISTINCT ta.name) AS refs "
                "ORDER BY refs DESC LIMIT $k", t=table_names, k=GRAPH_EXPANSION):
                added.append(r["name"])
    except Exception as exc:
        return [], f"Neo4j unavailable ({exc})"
    finally:
        driver.close()
    return added, None


# --- Step 4: join discovery (Neo4j) -----------------------------------------

def find_joins(table_names):
    """Join keys among the candidate tables: direct FK links AND shared-parent
    sibling joins (e.g. two tables that both reference customers.customer_id). Each
    carries its confidence so the prompt can prefer declared joins.

    Returns (joins, error): if Neo4j is unreachable (e.g. Aura paused) we degrade
    gracefully — single-table questions still work, multi-table ones lose join
    hints but don't crash."""
    if len(table_names) < 2:
        return [], None
    try:
        driver = GraphDatabase.driver(
            config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    except Exception as exc:
        return [], f"Neo4j unavailable ({exc})"
    joins = []
    try:
        with driver.session() as s:
            # direct: tableA.col -> tableB.col
            for r in s.run(
                "MATCH (ta:Table)-[:HAS_COLUMN]->(ca:Column)-[rel:REFERENCES]->"
                "(cb:Column)<-[:HAS_COLUMN]-(tb:Table) "
                "WHERE ta.name IN $t AND tb.name IN $t "
                "RETURN ta.name AS a, ca.name AS ak, tb.name AS b, cb.name AS bk, "
                "rel.fk_type AS conf", t=table_names):
                joins.append({"a": r["a"], "ak": r["ak"], "b": r["b"],
                              "bk": r["bk"], "conf": r["conf"], "via": None})
            # sibling: tableA and tableB both reference the SAME parent column
            for r in s.run(
                "MATCH (ta:Table)-[:HAS_COLUMN]->(ca:Column)-[:REFERENCES]->"
                "(p:Column)<-[:REFERENCES]-(cb:Column)<-[:HAS_COLUMN]-(tb:Table) "
                "WHERE ta.name IN $t AND tb.name IN $t AND ta.name < tb.name "
                "RETURN ta.name AS a, ca.name AS ak, tb.name AS b, cb.name AS bk, "
                "p.name AS shared", t=table_names):
                joins.append({"a": r["a"], "ak": r["ak"], "b": r["b"],
                              "bk": r["bk"], "conf": "shared-key", "via": r["shared"]})
    except Exception as exc:
        return [], f"Neo4j unavailable ({exc})"
    finally:
        driver.close()
    # de-dup (a==b mirror pairs from the direct query)
    seen, out = set(), []
    for j in joins:
        key = tuple(sorted([f"{j['a']}.{j['ak']}", f"{j['b']}.{j['bk']}"]))
        if key not in seen:
            seen.add(key)
            out.append(j)
    return out, None


# --- Step 5: assemble the focused context -----------------------------------

def build_context(tables, joins, schema):
    by_name = {t["table_name"]: t for t in schema["tables"]}
    lines = []
    for tname, meta in tables:
        t = by_name[tname]
        lines.append(f"TABLE {tname}")
        if t["description"].strip():
            lines.append(f"  purpose: {t['description'].strip()}")
        cols = ", ".join(f"{c['name']} {c['type']}" for c in t["columns"])
        lines.append(f"  columns: {cols}")
        if t["primary_key"]:
            lines.append(f"  primary key: {', '.join(t['primary_key'])}")
        # sample values help the model get WHERE-clause casing right
        samples = {k: [v for v in vs if v is not None][:3]
                   for k, vs in t.get("sample_values", {}).items()}
        samples = {k: v for k, v in samples.items() if v}
        if samples:
            shown = "; ".join(f"{k}={v}" for k, v in list(samples.items())[:6])
            lines.append(f"  sample values: {shown}")
        lines.append("")

    if joins:
        lines.append("JOIN KEYS (how these tables connect):")
        for j in joins:
            tag = f"[{j['conf']}]"
            via = f"  (both reference {j['via']})" if j["via"] else ""
            lines.append(f"  {j['a']}.{j['ak']} = {j['b']}.{j['bk']}  {tag}{via}")
    else:
        lines.append("JOIN KEYS: (none found among these tables)")
    return "\n".join(lines)


# --- Step 6: generate SQL (Phase B) -----------------------------------------

def _sql_prompt(question, context, prior_error=None):
    p = (
        "You are a MySQL expert. Write ONE MySQL SELECT query that answers the "
        "question, using ONLY the tables, columns and join keys provided.\n\n"
        f"{context}\n\n"
        "Rules:\n"
        "- Use ONLY the tables and columns listed above; never invent names.\n"
        "- Join tables using ONLY the JOIN KEYS listed. Prefer [declared] keys.\n"
        "- MySQL dialect. A read-only SELECT only (no INSERT/UPDATE/DELETE/DDL).\n"
        "- Match string values exactly as shown in sample values (e.g. 'ALLOCATION').\n"
        "- Add LIMIT 100 unless the query is an aggregate.\n"
    )
    if prior_error:
        p += f"\nYour previous attempt was invalid: {prior_error}\nFix it.\n"
    p += f"\nQuestion: {question}\n\nReturn ONLY the SQL, no explanation."
    return p


def generate(question, context, prior_error=None, model=None):
    model = model or config.OLLAMA_SQL_MODEL
    resp = ollama.generate(model=model,
                           prompt=_sql_prompt(question, context, prior_error))
    _record_tokens(f"SQL generation ({model})", resp)
    return _extract_sql(resp["response"])


def _extract_sql(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1)
    # keep from the first SELECT/WITH onward
    m = re.search(r"\b(select|with)\b", text, re.IGNORECASE)
    if m:
        text = text[m.start():]
    return text.strip().rstrip(";").strip()


# --- Step 6: validate SQL (Phase C) -----------------------------------------

_FORBIDDEN = ["insert", "update", "delete", "drop", "alter", "truncate",
              "create", "grant", "replace", "merge"]


def _type_family(t):
    """Coarse type family so join endpoints can be compared (BIGINT vs INT = int)."""
    base = re.split(r"[ (]", (t or "").strip())[0].upper()
    if base in {"INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT", "MEDIUMINT"}:
        return "int"
    if base in {"VARCHAR", "CHAR", "TEXT", "LONGTEXT", "MEDIUMTEXT", "TINYTEXT"}:
        return "text"
    return base


def validate(sql, schema):
    """(ok, error). Layered: (1) SELECT-only safety, (2) deep checks via sqlglot —
    alias resolution, column existence, join-key type match (P1/P2/P3). Falls back
    to plain table-existence if sqlglot is unavailable or can't parse."""
    if not sql:
        return False, "empty SQL"
    low = sql.lower()
    if not re.match(r"^\s*(select|with)\b", low):
        return False, "query must start with SELECT (or WITH)"
    for w in _FORBIDDEN:
        if re.search(r"\b" + w + r"\b", low):
            return False, f"forbidden keyword '{w.upper()}' - only SELECT is allowed"

    cols = {t["table_name"]: {c["name"]: c["type"] for c in t["columns"]}
            for t in schema["tables"]}

    if not _HAVE_SQLGLOT:
        used = set(re.findall(r"(?:from|join)\s+`?([A-Za-z_]\w*)`?", sql, re.I))
        unknown = [t for t in used if t not in cols]
        return (False, f"unknown table(s): {', '.join(unknown)}") if unknown else (True, None)

    try:
        tree = sqlglot.parse_one(sql, dialect="mysql")
    except Exception as exc:
        return False, f"could not parse SQL: {exc}"
    if tree is None:
        return False, "unparseable SQL"

    # alias/table-name -> real table; reject unknown tables
    alias2t = {}
    for tn in tree.find_all(exp.Table):
        if tn.name not in cols:
            return False, f"unknown table: {tn.name}"
        alias2t[tn.alias or tn.name] = tn.name
        alias2t[tn.name] = tn.name

    def col_type(c):                       # resolve a Column node -> its sql type
        q = c.table
        if q:
            return cols.get(alias2t.get(q), {}).get(c.name)
        for real in set(alias2t.values()):
            if c.name in cols[real]:
                return cols[real][c.name]
        return None

    # P1 undefined alias + P3 hallucinated column
    for c in tree.find_all(exp.Column):
        if c.name == "*":
            continue
        q = c.table
        if q and q not in alias2t:
            return False, f"undefined table/alias '{q}' (in {q}.{c.name})"
        if q:
            if c.name not in cols[alias2t[q]]:
                return False, f"column '{c.name}' does not exist in table '{alias2t[q]}'"
        elif not any(c.name in cols[r] for r in set(alias2t.values())):
            return False, f"column '{c.name}' not found in any table in scope"

    # P2 wrong join key: both sides of an '=' are columns with mismatched type family
    for eq in tree.find_all(exp.EQ):
        l, r = eq.this, eq.expression
        if isinstance(l, exp.Column) and isinstance(r, exp.Column):
            lt, rt = col_type(l), col_type(r)
            if lt and rt and _type_family(lt) != _type_family(rt):
                return False, (f"type-mismatched join/filter: {l.sql()} ({lt}) = "
                               f"{r.sql()} ({rt})")
    return True, None


# --- Step 7: execute on MySQL + compose answer (Phase D) --------------------

def _mysql():
    return pymysql.connect(
        host=config.MYSQL_HOST, port=config.MYSQL_PORT, user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD, database=config.MYSQL_DB,
        cursorclass=pymysql.cursors.Cursor, connect_timeout=5)


def explain(sql):
    """Final validation gate: ask MySQL to plan the query without running it.
    Catches anything sqlglot missed (bad functions, ambiguous columns, ...)."""
    try:
        conn = _mysql()
    except Exception as exc:
        return None, f"MySQL unavailable ({exc})"
    try:
        with conn.cursor() as cur:
            cur.execute("EXPLAIN " + sql)
            cur.fetchall()
        return True, None
    except Exception as exc:
        return False, str(exc).split("\n")[0]
    finally:
        conn.close()


def execute(sql, max_rows=50):
    """Run the validated SELECT, return (columns, rows). Safe: validation already
    guaranteed SELECT-only; we also cap rows."""
    conn = _mysql()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(max_rows)
        return cols, rows
    finally:
        conn.close()


def compose_answer(question, cols, rows):
    """Turn result rows into a short English answer (llama3.1)."""
    if not rows:
        return "The query ran successfully but returned no rows (no matching data)."
    preview = [dict(zip(cols, r)) for r in rows[:10]]
    prompt = (
        "Answer the user's question in 1-2 plain sentences using ONLY these SQL "
        "results. Be specific with numbers. Do not invent anything.\n\n"
        f"Question: {question}\n"
        f"Columns: {cols}\n"
        f"Rows (up to 10 shown): {json.dumps(preview, default=str)}\n\n"
        "Answer:")
    try:
        resp = ollama.generate(model=config.OLLAMA_ANSWER_MODEL, prompt=prompt)
        _record_tokens("answer compose (llama3.1)", resp)
        return re.sub(r"<think>.*?</think>", "", resp["response"], flags=re.DOTALL).strip()
    except Exception as exc:
        return f"(could not compose a sentence: {exc}) - see rows above."


# --- orchestrate Phase A -----------------------------------------------------

def answer(question, sql_model=None, compose=True):
    """Run the full pipeline and RETURN a structured trace (no printing).

    This is the machine-readable twin of ask(): the eval harness and the model
    bake-off call THIS, then read the dict; the CLI ask() is a thin printer over
    it. `sql_model` overrides which Ollama model writes the SQL (None = the
    configured default) — that single knob is what makes the model comparison a
    fair, one-variable swap.
    """
    model = sql_model or config.OLLAMA_SQL_MODEL
    schema = json.loads(config.SCHEMA_JSON.read_text(encoding="utf-8"))
    tok_start = len(TOKEN_STATS["events"])     # so trace["tokens"] = just THIS call
    t = {}                                     # per-stage wall-clock seconds
    tr = {
        "question": question, "sql_model": model,
        "destructive_intent": bool(_DESTRUCTIVE_INTENT.search(question)),
        "in_scope": False, "low_confidence": False, "matches": [],
        "candidate_tables": [], "graph_added": [], "tables": [],
        "joins": [], "join_error": None, "context": None,
        "attempts": [], "sql": None, "valid": False, "valid_attempt": None,
        "executed": False, "cols": [], "rows": [], "exec_error": None,
        "answer_text": None, "timings": t, "tokens": [],
    }

    # 1-3. retrieve candidate tables
    s = time.time()
    tables, matches = retrieve(question)
    t["retrieve"] = round(time.time() - s, 3)
    tr["matches"] = [{"score": m["score"], "kind": m["metadata"].get("kind"),
                      "table": m["metadata"]["table"],
                      "column": m["metadata"].get("column")} for m in matches[:8]]
    if not tables:                              # out of scope (below the floor)
        tr["tokens"] = TOKEN_STATS["events"][tok_start:]
        return tr
    tr["in_scope"] = True
    tr["candidate_tables"] = [tn for tn, _ in tables]
    tr["low_confidence"] = tables[0][1]["score"] < CONFIDENCE_BAR

    # 3b. graph expansion (Neo4j adds entity/bridge tables Pinecone missed)
    s = time.time()
    added, _ = expand_candidates([tn for tn, _ in tables])
    t["expand"] = round(time.time() - s, 3)
    if added:
        tr["graph_added"] = added
        tables += [(a, {"score": None, "cols": []}) for a in added]

    # 4. join discovery
    s = time.time()
    joins, join_err = find_joins([tn for tn, _ in tables])
    t["joins"] = round(time.time() - s, 3)
    tr["join_error"] = join_err
    tr["joins"] = joins
    tr["tables"] = [tn for tn, _ in tables]

    # 5. focused context
    context = build_context(tables, joins, schema)
    tr["context"] = context

    # 6-7. generate -> validate (sqlglot) -> EXPLAIN -> execute, with repair loop
    error, sql = None, None
    s = time.time()
    for attempt in range(1, 4):
        sql = generate(question, context, prior_error=error, model=model)
        ok, error = validate(sql, schema)                       # static (sqlglot)
        explained = None
        if ok and _HAVE_MYSQL:
            explained, exp_err = explain(sql)
            if explained is False:
                ok, error = False, f"MySQL rejected the query: {exp_err}"
        tr["attempts"].append({"sql": sql, "valid": ok, "error": error,
                               "explained": explained})
        if ok:
            tr["valid"], tr["valid_attempt"] = True, attempt
            break
    t["generate_validate"] = round(time.time() - s, 3)
    tr["sql"] = sql

    if tr["valid"] and _HAVE_MYSQL:
        try:
            s = time.time()
            cols, rows = execute(sql)
            t["execute"] = round(time.time() - s, 3)
            tr["executed"], tr["cols"] = True, list(cols)
            tr["rows"] = [list(r) for r in rows]
            tr["answer_text"] = compose_answer(question, cols, rows) if compose else None
        except Exception as exc:
            tr["exec_error"] = str(exc)

    tr["tokens"] = TOKEN_STATS["events"][tok_start:]
    return tr


def ask(question):
    """CLI view: run answer() and print the same human-readable report as before."""
    tr = answer(question)

    print(f"\nQUESTION: {question}\n" + "=" * 64)
    if tr["destructive_intent"]:                # P6
        print("NOTE: this is a READ-ONLY system - it can only answer with SELECT "
              "queries. Showing the matching data instead of changing anything.\n")

    print("Top Pinecone hits:")
    for m in tr["matches"]:
        lbl = m["table"] if m["kind"] == "table" else f"{m['table']}.{m['column']}"
        print(f"  {m['score']:.3f}  [{m['kind']}] {lbl}")

    if not tr["in_scope"]:
        print("\n** No table scored above the floor - this question does not "
              "appear answerable from this database. **")
        return

    print(f"\nCandidate tables ({len(tr['candidate_tables'])}): "
          f"{', '.join(tr['candidate_tables'])}")
    if tr["low_confidence"]:                    # P4
        print("WARNING: no table is a strong match - the question may be too vague "
              "or broad. The SQL below is a best guess; consider being more specific.")
    if tr["graph_added"]:
        print(f"Graph-expanded (referenced by candidates, added by Neo4j): "
              f"{', '.join(tr['graph_added'])}")
    if tr["join_error"]:
        print(f"WARNING: join discovery skipped - {tr['join_error']}\n"
              "  (single-table answers still work; resume Neo4j Aura for joins.)")

    print("\n" + "-" * 64 + "\nFOCUSED CONTEXT (sent to the SQL model):\n" + "-" * 64)
    print(tr["context"])

    print("\n" + "-" * 64 + "\nGENERATED SQL:\n" + "-" * 64)
    for i, a in enumerate(tr["attempts"], 1):
        if a["valid"]:
            print(a["sql"])
            print(f"\n[VALID on attempt {i}]")
        else:
            print(f"[attempt {i} invalid: {a['error']}]")

    if not tr["valid"]:
        print("\n** Could not produce valid SQL after 3 attempts. Last try:\n"
              + (tr["sql"] or ""))
        return
    if not _HAVE_MYSQL:
        return
    if tr["exec_error"]:
        print(f"\nExecution error: {tr['exec_error']}")
        return

    print("\n" + "-" * 64 + f"\nRESULT ({len(tr['rows'])} row(s)):\n" + "-" * 64)
    print(" | ".join(str(c) for c in tr["cols"]))
    for r in tr["rows"][:20]:
        print(" | ".join(str(v) for v in r))
    print("\n" + "-" * 64 + "\nANSWER:\n" + "-" * 64)
    print(tr["answer_text"])


def main():
    args = sys.argv[1:]
    if "--count-tokens" in args:
        TOKEN_STATS["enabled"] = True
        args = [a for a in args if a != "--count-tokens"]
    if not args:
        sys.exit('Usage: python src/ask.py [--count-tokens] "your question here"')
    ask(" ".join(args))
    _print_token_report()


if __name__ == "__main__":
    sys.exit(main())
