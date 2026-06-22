"""
stage2_describe.py  —  Stage 2: Ollama fills the empty descriptions in schema.json

WHAT
    Every table and every column in schema.json has an empty "description" field.
    This script hands each one to a LOCAL Ollama model and writes the 1-2 sentence
    answer back INTO schema.json (in place — one source of truth, no second file).

WHY LOCAL OLLAMA (not a cloud API)
    A 400-table DB is thousands of AI calls. Local = free, private, no rate limits,
    re-runnable offline. We use llama3.1:8b (see config.OLLAMA_DESC_MODEL).

INCREMENTAL (the important habit)
    Before calling the model we SKIP any description that is already non-empty.
    So a crash/timeout on table #300 means a re-run only does the rest, not all
    704 again. Use --force to overwrite everything on purpose.

GRANULARITY (decision: tables AND columns)
    * table description  = what it stores + what questions it answers   (1-2 sentences)
    * column description = what this one value is, using sample values   (1 sentence)

Run:
    python src/stage2_describe.py            # fill only the blanks
    python src/stage2_describe.py --force    # redo everything
"""

import json
import re
import sys

import config

try:
    import ollama
except ModuleNotFoundError:
    sys.exit("ERROR: the 'ollama' package is not installed.\n"
             "       Run:  pip install -r requirements.txt\n"
             "       And make sure the Ollama app/server is running.")


# --- Prompt builders --------------------------------------------------------
# The RULES live in the prompt so all 704 descriptions share one consistent
# style instead of drifting. Samples are included because a column name alone is
# ambiguous (is status the word 'active' or the number 1?) — samples fix that.

def _table_prompt(table):
    cols = ", ".join(c["name"] for c in table["columns"])
    fks = "; ".join(
        f"{fk['column']} -> {fk['references_table']}.{fk['references_column']}"
        for fk in table["foreign_keys"]
    ) or "none"
    # keep only columns that actually have a real value, so an empty table never
    # tempts the model to invent meaning from nulls.
    samples = {k: _usable_samples(v)
               for k, v in table.get("sample_values", {}).items()}
    samples = {k: v for k, v in samples.items() if v}
    samples = dict(list(samples.items())[:6])
    sample_line = (json.dumps(samples, default=str) if samples
                   else "(none available — this table has no sample rows)")
    return (
        "You are writing schema documentation that helps an AI pick the right "
        "tables and write correct SQL.\n\n"
        f"Table: {table['table_name']}\n"
        f"Columns: {cols}\n"
        f"Foreign keys: {fks}\n"
        f"Example rows (a HINT only, not the full data): {sample_line}\n\n"
        "Write 1-2 flowing sentences (no lists, no labels, no preamble) that cover: "
        "what ONE row represents, what the table records, and the questions it "
        "answers or the tables it joins to. Name any foreign-key link.\n"
        "Be specific and use the real domain terms. Only name a linked table that "
        "is listed under Foreign keys above; never guess one. Do NOT assume the "
        "data is about SQL, AI, or any specific industry (e.g. energy, utilities, "
        "transport, healthcare) unless the column names clearly show it. Do NOT "
        "use vague words "
        "(\"various\", \"some information\"), decorative adjectives, or any comment "
        "about how many rows exist or whether values are null. Start directly with "
        "the description (e.g. \"Each row...\") — no preamble like \"Here is\", no "
        "quotes."
    )


def _usable_samples(samples):
    """Real, non-null sample values only. Drives the no-invent branch below:
    an 8B model can't be trusted to 'not invent' on request, so when there are
    NO usable samples we change the RULES instead of pleading with the model."""
    return [v for v in (samples or []) if v is not None and str(v).strip() != ""]


def _column_prompt(table_name, column, samples, fk_hint=None):
    link = (f"\nThis column is a foreign key to {fk_hint}." if fk_hint else "")
    usable = _usable_samples(samples)

    if usable:
        # We HAVE real values: let the model name the ones it actually sees.
        value_rule = (
            "- For a status/type/category column, only list specific values that "
            "actually appear in the example values above — never add others.\n"
        )
        examples = f"Example values: {json.dumps(usable, default=str)}"
    else:
        # No values to anchor on: forbid ALL specific values (this is where the
        # model used to hallucinate 'Rental'/'water consumption'/etc.).
        value_rule = (
            "- You have NO example values. Do NOT list, guess, or invent ANY "
            "specific values, codes, categories, or enum options. Describe only "
            "the column's general role from its name and type.\n"
        )
        examples = "Example values: (none available — describe the role only)"

    return (
        "You are documenting a database column for an AI that writes SQL.\n\n"
        f"Table: {table_name}\n"
        f"Column: {column['name']} ({column['type']}){link}\n"
        f"{examples}\n\n"
        "In ONE concrete sentence, state what this column PERMANENTLY represents "
        "(its role), not the current data. Follow these rules:\n"
        "- If it identifies an entity or is a key, say what it identifies. Only "
        "name a linked table if one is given above as a foreign key; NEVER guess "
        "or invent a table name.\n"
        f"{value_rule}"
        "- Give the unit or format when the type implies it (money for DECIMAL, a "
        "timestamp for TIMESTAMP, a UUID, an email).\n"
        "- Do NOT assume the data is about SQL, AI, or any topic not shown above.\n"
        "- No vague words, no decorative adjectives, do NOT quote the examples "
        "back, and do NOT mention null/empty/duplicate values or data quality.\n"
        "Reply with the sentence only — no preamble, no quotes."
    )


def _ask(prompt):
    """Send one prompt to Ollama, return the trimmed text (or '' on failure)."""
    try:
        resp = ollama.generate(model=config.OLLAMA_DESC_MODEL, prompt=prompt)
        text = resp["response"]
        # qwen3 and other reasoning models may emit a <think>...</think> block
        # before the real answer — strip it so it never lands in schema.json.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()
    except Exception as exc:                       # network/timeout/model-missing
        print(f"    ! Ollama call failed: {exc}")
        return ""


# --- Main loop --------------------------------------------------------------

def describe(force=False, only=None):
    """only: optional set/list of table names to process (forced). When given,
    every OTHER table is left completely untouched — useful for re-doing a few
    tables with an improved prompt without reprocessing the whole DB."""
    data = json.loads(config.SCHEMA_JSON.read_text(encoding="utf-8"))
    tables = data["tables"]
    if only:
        only = set(only)
        force = True                       # targeting a table implies redo it

    filled_t = filled_c = skipped = failed = 0
    total_cols = sum(len(t["columns"]) for t in tables)
    scope = f"{len(only)} selected table(s)" if only \
        else f"{len(tables)} tables + {total_cols} columns"
    print(f"Describing {scope} with '{config.OLLAMA_DESC_MODEL}'  (force={force})\n")

    for i, table in enumerate(tables, 1):
        tname = table["table_name"]
        if only and tname not in only:
            continue                       # leave non-selected tables untouched
        print(f"[{i}/{len(tables)}] {tname}")

        # --- table description ---
        if force or not table["description"].strip():
            text = _ask(_table_prompt(table))
            if text:
                table["description"] = text
                filled_t += 1
                # save after EACH table so a crash never loses prior work
                _save(data)
            else:
                failed += 1
        else:
            skipped += 1

        # column -> "references_table.references_column" for FK columns, so the
        # column prompt can name the real link instead of guessing.
        fk_target = {
            fk["column"]: f"{fk['references_table']}.{fk['references_column']}"
            for fk in table["foreign_keys"]
        }

        # --- column descriptions ---
        for col in table["columns"]:
            if not (force or not col["description"].strip()):
                skipped += 1
                continue
            samples = table.get("sample_values", {}).get(col["name"], [])
            text = _ask(_column_prompt(tname, col, samples,
                                       fk_hint=fk_target.get(col["name"])))
            if text:
                col["description"] = text
                filled_c += 1
            else:
                failed += 1
        _save(data)

    print(f"\nDone. tables filled: {filled_t}, columns filled: {filled_c}, "
          f"skipped (already had text): {skipped}, failed: {failed}")
    if failed:
        print("Some calls failed (blank descriptions left). Re-run to fill them.")
    return data


def _save(data):
    config.SCHEMA_JSON.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    force = "--force" in sys.argv
    only = None
    if "--tables" in sys.argv:             # e.g. --tables customers,orders
        only = [t for t in sys.argv[sys.argv.index("--tables") + 1].split(",") if t]
    describe(force=force, only=only)


if __name__ == "__main__":
    sys.exit(main())
