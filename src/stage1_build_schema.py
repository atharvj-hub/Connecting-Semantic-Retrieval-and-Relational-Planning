"""
stage1_build_schema.py  —  Stage 1 entry point

    raw SQL dump  ->  data/output/schema.json   (descriptions still empty)

This is the ONLY file you run for Stage 1:

    python src/stage1_build_schema.py

What it does, in order:
    1. Read the schema dump and split it into per-table CREATE TABLE blocks.
    2. For each block, parse columns, primary key, and foreign keys.
    3. Read the data dump and pull a few sample rows per table.
    4. Assemble each table into the agreed JSON shape (description left "").
    5. Write data/output/schema.json.

The result is IDEMPOTENT: same dump in -> identical schema.json out, every
time. That is what lets you test on a different database by swapping the dump
files in config.py and re-running.
"""

import json

import config
from sql_parser import (
    find_database_name,
    find_create_table_blocks,
    parse_columns,
    parse_foreign_keys,
    extract_sample_values,
)


def build_schema():
    """Parse the dump files and return the full schema as a Python dict."""
    # --- Step 1: load the dump file(s) -------------------------------------
    # Combined dumps point SCHEMA_SQL and DATA_SQL at the same file; read once.
    print(f"[1/5] Reading schema dump : {config.SCHEMA_SQL.name}")
    schema_sql = config.SCHEMA_SQL.read_text(encoding="utf-8", errors="replace")

    if config.DATA_SQL == config.SCHEMA_SQL:
        print(f"[2/5] Data dump           : same file (combined dump)")
        data_sql = schema_sql
    else:
        print(f"[2/5] Reading data dump   : {config.DATA_SQL.name}")
        data_sql = config.DATA_SQL.read_text(encoding="utf-8", errors="replace")

    # --- Step 2: split into per-table blocks -------------------------------
    database = find_database_name(schema_sql)
    blocks = find_create_table_blocks(schema_sql)
    print(f"[3/5] Database: {database or '(none declared)'} — found {len(blocks)} tables")

    # --- Steps 3-4: parse each table ---------------------------------------
    print("[4/5] Parsing columns, keys, foreign keys, and sample rows ...")
    tables = []
    for table_name, body in blocks:
        columns, primary_key = parse_columns(body)
        foreign_keys = parse_foreign_keys(body)
        sample_values = extract_sample_values(
            data_sql,
            table_name,
            [c["name"] for c in columns],
            k=config.SAMPLE_ROWS,
            max_len=config.MAX_VALUE_LEN,
        )

        tables.append({
            "table_name": table_name,
            "description": "",            # <- filled by Stage 2 (Ollama)
            "primary_key": primary_key,   # <- table-level array (composite-safe)
            "columns": columns,           # <- each has empty "description" too
            "foreign_keys": foreign_keys,
            "sample_values": sample_values,
        })
        print(f"        - {table_name:<32} "
              f"{len(columns):>2} cols, "
              f"{len(foreign_keys):>2} FKs, "
              f"PK={primary_key or '(none)'}")

    return {"database": database, "tables": tables}


def main():
    schema = build_schema()

    # --- Step 5: write the JSON --------------------------------------------
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.SCHEMA_JSON.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[5/5] Wrote {config.SCHEMA_JSON.relative_to(config.BASE_DIR)} "
          f"({len(schema['tables'])} tables)")


if __name__ == "__main__":
    main()
