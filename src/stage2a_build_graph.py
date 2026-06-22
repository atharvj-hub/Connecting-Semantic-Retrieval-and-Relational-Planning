"""
stage2a_build_graph.py  —  Task 2: schema.json -> graph.json (Neo4j-ready)

LOCKED mapping-only graph model (+ FK confidence tag)
---------------------------------------------------------------------------
Every node has a SURROGATE id (an opaque counter value) and a single property,
its name. Every relationship is an id -> id edge.

    Node:  { "id": "t1" | "c1", "properties": { "name": "..." } }
    Edge:  { "source": "...", "target": "..." [, "fk_type": "..."] }

    id prefix   't' = a table,  'c' = a column   (this is how we tell kind apart)
    edge kind   source 't' -> ownership (table owns column)   [no fk_type]
                source 'c' -> foreign key (column references column) [has fk_type]

FK CONFIDENCE (the only data added to the locked shape)
    Real databases built on JPA/Hibernate often DON'T declare foreign keys at the
    DB level (they enforce links in app code), so the dump under-reports relations
    (here: 24 declared, ~98 implicit). We recover the implicit ones in Pass 3 and
    tag every column->column edge so declared fact stays separable from guesswork:

        fk_type = "declared"        -> written as a FOREIGN KEY in the dump (certain)
        fk_type = "inferred_rule"   -> a column with the SAME name is declared as an
                                       FK elsewhere; we copied that target (high conf)
        fk_type = "inferred_naming" -> pure "<x>_id -> <x-table>.id" name+type guess
                                       (medium conf)

    Ownership edges carry NO fk_type (their kind is fully derived from the prefix).

Everything else is still DERIVED, never stored (a column's table by walking its
ownership edge; name/type/PK from schema.json).

Output (data/output/graph.json):
{
  "database": "shop_db",
  "nodes": [ { "id": "t1", "properties": { "name": "regions" } }, ... ],
  "edges": [
    { "source": "t1", "target": "c1" },                          // ownership
    { "source": "c5", "target": "c1", "fk_type": "declared" },    // declared FK
    { "source": "c9", "target": "c1", "fk_type": "inferred_rule" } // recovered FK
  ]
}

Run:
    python src/stage2a_build_graph.py
"""

import json
import re
import sys

import config


# --- helpers for inference --------------------------------------------------

def _type_family(t):
    """Collapse a SQL type to a coarse family so FK endpoints can be compared
    (BIGINT vs INT = same; VARCHAR vs CHAR = same; DECIMAL/TIMESTAMP/JSON exact)."""
    base = re.split(r"[ (]", (t or "").strip())[0].upper()
    if base in {"INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT", "MEDIUMINT"}:
        return "int"
    if base in {"VARCHAR", "CHAR", "TEXT", "LONGTEXT", "MEDIUMTEXT", "TINYTEXT"}:
        return "text"
    return base


def _match_table(base, table_set):
    """Find the table a '<base>_id' column likely points to, trying simple
    singular->plural variants. Returns the table name or None (None = skip,
    we never force a guess)."""
    for cand in (base, base + "s", base + "es"):
        if cand in table_set:
            return cand
    return None


def build_graph():
    """Return (graph_dict, unresolved_fks, stats).

    Pass 1: nodes + ownership edges + lookup maps (id, type, pk).
    Pass 2: DECLARED foreign-key edges (fk_type='declared').
    Pass 3: INFERRED edges — Tier 1 rules learned from declared FKs, then Tier 2
            '<x>_id -> table' name+type guesses. All type-checked, all tagged.
    """
    data = json.loads(config.SCHEMA_JSON.read_text(encoding="utf-8"))
    tables = data["tables"]
    database = data.get("database")

    nodes = []
    ownership_edges = []
    fk_edges = []

    col_id_by_natural = {}                 # (table, col) -> surrogate id
    col_type = {}                          # (table, col) -> sql type
    pk_col = {}                            # table -> single PK column (or None)
    table_set = {t["table_name"] for t in tables}
    t_count = c_count = 0

    # --- Pass 1 -------------------------------------------------------------
    for t in tables:
        t_count += 1
        tid = f"t{t_count}"
        tname = t["table_name"]
        nodes.append({"id": tid, "properties": {"name": tname}})
        pk = t.get("primary_key") or []
        pk_col[tname] = pk[0] if len(pk) == 1 else None
        for col in t["columns"]:
            c_count += 1
            cid = f"c{c_count}"
            cname = col["name"]
            col_id_by_natural[(tname, cname)] = cid
            col_type[(tname, cname)] = col["type"]
            nodes.append({"id": cid, "properties": {"name": cname}})
            ownership_edges.append({"source": tid, "target": cid})

    # --- Pass 2: declared FKs ----------------------------------------------
    unresolved = []
    declared_cols = set()                  # (table, col) already linked -> never infer
    learned = {}                           # col name -> set of (ref_table, ref_col)
    for t in tables:
        tname = t["table_name"]
        for fk in t["foreign_keys"]:
            src = col_id_by_natural.get((tname, fk["column"]))
            tgt = col_id_by_natural.get(
                (fk["references_table"], fk["references_column"]))
            if src is None or tgt is None:
                unresolved.append((tname, fk["column"],
                                   fk["references_table"], fk["references_column"]))
                continue
            fk_edges.append({"source": src, "target": tgt, "fk_type": "declared"})
            declared_cols.add((tname, fk["column"]))
            learned.setdefault(fk["column"], set()).add(
                (fk["references_table"], fk["references_column"]))

    # --- Pass 3: inferred FKs ----------------------------------------------
    stats = {"declared": len(fk_edges), "inferred_rule": 0, "inferred_naming": 0}
    inferred_samples = []                  # (child.col -> parent.col, kind) for report
    for t in tables:
        tname = t["table_name"]
        for col in t["columns"]:
            n = col["name"]
            if (tname, n) in declared_cols or n == "id":
                continue                   # already declared, or own key

            target = kind = None

            # Tier 1 — copy a rule learned from a real declared FK of the same name
            if n in learned and len(learned[n]) == 1:
                rt, rc = next(iter(learned[n]))
                if (rt, rc) in col_id_by_natural and \
                        _type_family(col_type[(tname, n)]) == _type_family(col_type[(rt, rc)]):
                    target, kind = (rt, rc), "inferred_rule"

            # Tier 2 — pure '<x>_id -> <x-table>.<pk>' name + type guess
            if target is None and n.endswith("_id"):
                ct = _match_table(n[:-3], table_set)
                rc = pk_col.get(ct) if ct else None
                if ct and rc and (ct, rc) in col_id_by_natural and \
                        _type_family(col_type[(tname, n)]) == _type_family(col_type[(ct, rc)]):
                    target, kind = (ct, rc), "inferred_naming"

            if target:
                src_id = col_id_by_natural[(tname, n)]
                tgt_id = col_id_by_natural[target]
                if src_id != tgt_id:       # no self-loop
                    fk_edges.append({"source": src_id, "target": tgt_id, "fk_type": kind})
                    stats[kind] += 1
                    if len(inferred_samples) < 15:
                        inferred_samples.append(
                            (f"{tname}.{n} -> {target[0]}.{target[1]}", kind))

    graph = {"nodes": nodes, "edges": ownership_edges + fk_edges}
    if database:
        graph = {"database": database, **graph}
    stats["samples"] = inferred_samples
    return graph, unresolved, stats


def _report(graph, unresolved, stats):
    """Human-readable summary. Stays NAME-based so you can verify by eye."""
    nodes = graph["nodes"]
    edges = graph["edges"]
    name_of = {n["id"]: n["properties"]["name"] for n in nodes}

    n_tables = sum(1 for n in nodes if n["id"].startswith("t"))
    n_cols = sum(1 for n in nodes if n["id"].startswith("c"))
    own = [e for e in edges if e["source"].startswith("t")]
    fks = [e for e in edges if e["source"].startswith("c")]

    print(f"Database : {graph.get('database', '(unknown)')}")
    print(f"Nodes    : {n_tables} Table + {n_cols} Column = {len(nodes)}")
    print(f"Edges    : {len(own)} ownership + {len(fks)} foreign-key "
          f"= {len(edges)}")
    print(f"FK detail: {stats['declared']} declared + "
          f"{stats['inferred_rule']} inferred_rule + "
          f"{stats['inferred_naming']} inferred_naming\n")

    if unresolved:
        print(f"WARNING: {len(unresolved)} declared FK(s) point to a column not "
              f"in the dump:")
        for a, b, c, d in unresolved:
            print(f"  ! {a}.{b} -> {c}.{d}")
        print()

    if stats["samples"]:
        print("Sample of RECOVERED (inferred) links - eyeball these:")
        for txt, kind in stats["samples"]:
            print(f"  [{kind:15}] {txt}")
        print()

    table_of_col = {e["target"]: name_of[e["source"]] for e in own}
    inbound = {}
    for e in fks:
        parent = table_of_col.get(e["target"], "?")
        child = table_of_col.get(e["source"], "?")
        inbound.setdefault(parent, []).append(child)
    print("Backtracking summary - who points AT each hub (inbound, derived):")
    top = sorted(inbound.items(), key=lambda kv: len(kv[1]), reverse=True)[:5]
    for name, srcs in top:
        print(f"  {name}: {len(srcs)} inbound  <- {sorted(set(srcs))}")


def main():
    graph, unresolved, stats = build_graph()

    output_path = config.GRAPH_JSON
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")

    _report(graph, unresolved, stats)
    print(f"\nWrote {output_path.relative_to(config.BASE_DIR)}")


if __name__ == "__main__":
    sys.exit(main())
