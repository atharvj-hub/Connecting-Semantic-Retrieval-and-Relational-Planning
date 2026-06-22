"""
stage4a_build_semantic.py  —  Stage 4a: schema.json -> semantic.json (Pinecone input)

WHAT
    Turn the described schema.json into retrieval documents — ONE per table AND ONE
    per column (decision: table + column vectors) — each carrying the exact string
    Stage 5a will embed.

OPTIMIZATION TARGET = RETRIEVAL RECALL (not documentation)
    Users rarely type schema names. They ask "customer credits", "available
    balance", "how are refunds tracked". So each embedding_text is enriched with:
      * the clean description (from schema.json)
      * KEYWORDS  — domain concepts the entity is about
      * ALIASES   — alternative names users actually type
      * real FK RELATIONSHIPS as plain sentences (DERIVED from foreign_keys —
        never invented; Neo4j still owns actual joins)
      * a representative sample value (for columns)
    Keywords/aliases are LLM-generated HERE and stored ONLY in semantic.json — the
    truth file (schema.json) stays clean for SQL generation + human validation.

GROUNDING (avoid the hallucination we saw earlier)
    The keyword/alias prompt is told to base terms ONLY on the name + description,
    and NOT to invent systems/workflows/topics. Relationships come from real FKs.

INCREMENTAL
    Re-uses keywords/aliases already in an existing semantic.json (keyed by id) so a
    re-run is cheap. Use --force to regenerate them all.

Output (data/output/semantic.json) — a flat list of mixed docs:
  table doc : { id:"table:X", kind:"table", table, description, keywords, aliases,
                columns, relationships, embedding_text }
  column doc: { id:"col:X.c", kind:"column", table, column, datatype, description,
                keywords, aliases, fk, example, embedding_text }

Run:
    python src/stage4a_build_semantic.py
    python src/stage4a_build_semantic.py --force
"""

import json
import re
import sys

import config

try:
    import ollama
except ModuleNotFoundError:
    sys.exit("ERROR: the 'ollama' package is not installed. "
             "Run: pip install -r requirements.txt")


# --- Keyword / alias generation (grounded) ----------------------------------

def _terms_prompt(kind, name, context, description):
    return (
        "You generate SEARCH TERMS that help a user find a database "
        f"{kind} by meaning.\n\n"
        f"{context}\n"
        f"Description: {description}\n\n"
        "List the words and short phrases a user might TYPE when looking for this "
        f"{kind}. Output EXACTLY two lines and nothing else:\n"
        "KEYWORDS: <comma-separated domain concepts / nouns it is about>\n"
        "ALIASES: <comma-separated alternative names a user might call it>\n\n"
        "Rules: base every term ONLY on the name and description above. Do NOT "
        "invent business systems, workflows, products, or topics not implied by "
        "them. 5-9 lowercase items per line, no duplicates."
    )


def _parse_terms(text):
    """Pull the KEYWORDS: / ALIASES: lines into two clean lists."""
    kw, al = [], []
    for line in text.splitlines():
        low = line.strip().lower()
        if low.startswith("keywords:"):
            kw = _split(line.split(":", 1)[1])
        elif low.startswith("aliases:"):
            al = _split(line.split(":", 1)[1])
    return kw, al


def _split(s):
    items = [re.sub(r"[\"'.]", "", x).strip().lower() for x in s.split(",")]
    seen, out = set(), []
    for it in items:                       # dedupe, drop blanks
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _gen_terms(kind, name, context, description):
    if not description.strip():
        return [], []
    try:
        resp = ollama.generate(
            model=config.OLLAMA_DESC_MODEL,
            prompt=_terms_prompt(kind, name, context, description))
        text = re.sub(r"<think>.*?</think>", "", resp["response"], flags=re.DOTALL)
        return _parse_terms(text)
    except Exception as exc:
        print(f"    ! term generation failed for {name}: {exc}")
        return [], []


# --- embedding_text builders ------------------------------------------------

def _table_embed_text(table, rel_sentences, keywords, aliases):
    lines = [f"Table: {table['table_name']}"]
    if table["description"].strip():
        lines.append(f"Purpose: {table['description'].strip()}")
    lines.append("Columns: " + ", ".join(c["name"] for c in table["columns"]))
    if rel_sentences:
        lines.append("Relationships: " + "; ".join(rel_sentences))
    if keywords:
        lines.append("Keywords: " + ", ".join(keywords))
    if aliases:
        lines.append("Aliases: " + ", ".join(aliases))
    return "\n".join(lines)


def _column_embed_text(tname, col, fk, example, keywords, aliases):
    lines = [f"Column: {col['name']}", f"Table: {tname}", f"Type: {col['type']}"]
    if col["description"].strip():
        lines.append(f"Meaning: {col['description'].strip()}")
    if example is not None:
        lines.append(f"Example value: {example}")
    if fk:
        lines.append(f"Foreign key: links to {fk}")
    if keywords:
        lines.append("Keywords: " + ", ".join(keywords))
    if aliases:
        lines.append("Aliases: " + ", ".join(aliases))
    return "\n".join(lines)


def _first_sample(samples):
    """A representative, non-null, short sample value — or None."""
    for v in samples or []:
        if v is None:
            continue
        s = str(v)
        return s[:60] + "..." if len(s) > 60 else s
    return None


# --- build ------------------------------------------------------------------

def build(force=False):
    data = json.loads(config.SCHEMA_JSON.read_text(encoding="utf-8"))

    # cache previously generated terms (keyed by doc id) so re-runs are cheap
    cache = {}
    if config.SEMANTIC_JSON.exists() and not force:
        for d in json.loads(config.SEMANTIC_JSON.read_text(encoding="utf-8")):
            cache[d["id"]] = (d.get("keywords", []), d.get("aliases", []))

    docs = []
    missing_desc = 0
    tables = data["tables"]

    for i, table in enumerate(tables, 1):
        tname = table["table_name"]
        if tname in config.SKIP_TABLES:           # migration/bookkeeping noise
            print(f"[{i}/{len(tables)}] {tname}  (skipped — not indexed)")
            continue
        if not table["description"].strip():
            missing_desc += 1
        print(f"[{i}/{len(tables)}] {tname}")

        fk_target = {fk["column"]: f"{fk['references_table']}.{fk['references_column']}"
                     for fk in table["foreign_keys"]}
        rel_sentences = [f"links to {fk['references_table']} via {fk['column']}"
                         for fk in table["foreign_keys"]]

        # ----- table doc -----
        tid = f"table:{tname}"
        kw, al = cache.get(tid) or _gen_terms(
            "table", tname,
            f"Table: {tname}\nColumns: " + ", ".join(c["name"] for c in table["columns"]),
            table["description"])
        docs.append({
            "id": tid, "kind": "table", "table": tname,
            "description": table["description"],
            "keywords": kw, "aliases": al,
            "columns": [c["name"] for c in table["columns"]],
            "relationships": rel_sentences,
            "embedding_text": _table_embed_text(table, rel_sentences, kw, al),
        })

        # ----- column docs -----
        for col in table["columns"]:
            cid = f"col:{tname}.{col['name']}"
            fk = fk_target.get(col["name"])
            example = _first_sample(table.get("sample_values", {}).get(col["name"]))
            ckw, cal = cache.get(cid) or _gen_terms(
                "column", col["name"],
                f"Column: {col['name']} ({col['type']}) in table {tname}"
                + (f", foreign key to {fk}" if fk else ""),
                col["description"])
            docs.append({
                "id": cid, "kind": "column", "table": tname, "column": col["name"],
                "datatype": col["type"], "description": col["description"],
                "keywords": ckw, "aliases": cal, "fk": fk, "example": example,
                "embedding_text": _column_embed_text(tname, col, fk, example, ckw, cal),
            })

        _save(docs)                        # save per-table so a crash keeps progress

    n_tab = sum(1 for d in docs if d["kind"] == "table")
    n_col = sum(1 for d in docs if d["kind"] == "column")
    print(f"\nBuilt {n_tab} table + {n_col} column docs -> "
          f"{config.SEMANTIC_JSON.relative_to(config.BASE_DIR)}")
    if missing_desc:
        print(f"WARNING: {missing_desc} table(s) have an EMPTY description — run "
              f"Stage 2 first for best recall.")
    print("\nExample TABLE embedding_text:\n" + "-" * 60)
    print(docs[0]["embedding_text"])
    return docs


def _save(docs):
    config.SEMANTIC_JSON.write_text(
        json.dumps(docs, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    build(force="--force" in sys.argv)


if __name__ == "__main__":
    sys.exit(main())
