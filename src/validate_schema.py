"""
validate_schema.py  —  fast STRUCTURAL check on data/output/schema.json

This is NOT Stage 3. Stage 3 is the human reading descriptions for meaning.
This script just catches mechanical breakage so you don't waste eyeball time:
    - column names that aren't valid identifiers (a sign the parser grabbed junk)
    - empty column types
    - foreign keys that point at a table not present in the dump
    - tables with no primary key

Run it after every Stage 1 run, especially when you swap in a new dump:

    python src/validate_schema.py

Exit code is 0 when clean, 1 when problems are found (handy for scripting).
"""

import json
import sys

import config


def validate():
    if not config.SCHEMA_JSON.exists():
        print(f"ERROR: {config.SCHEMA_JSON} not found. Run stage1 first.")
        return 1

    data = json.loads(config.SCHEMA_JSON.read_text(encoding="utf-8"))
    tables = data.get("tables", [])
    table_names = {t["table_name"] for t in tables}

    # ERRORS  = the parser almost certainly grabbed something wrong.
    # WARNINGS = unusual but legitimate (e.g. Liquibase's PK-less changelog table).
    errors = []
    warnings = []

    for t in tables:
        name = t["table_name"]

        if not t.get("primary_key"):
            warnings.append(f"{name}: no primary key (ok for log/changelog tables)")

        for col in t["columns"]:
            if not col["name"].isidentifier():
                errors.append(f"{name}: suspicious column name {col['name']!r}")
            if not col["type"]:
                errors.append(f"{name}.{col['name']}: empty type")

        for fk in t["foreign_keys"]:
            if fk["references_table"] not in table_names:
                errors.append(
                    f"{name}.{fk['column']} -> {fk['references_table']} "
                    f"(referenced table not in dump)"
                )

    n_cols = sum(len(t["columns"]) for t in tables)
    n_fks = sum(len(t["foreign_keys"]) for t in tables)
    print(f"Tables: {len(tables)}   Columns: {n_cols}   Foreign keys: {n_fks}")

    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  ! {w}")

    if errors:
        print(f"\n{len(errors)} ERROR(s) found:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\nStructure OK. Now do the Stage 3 human read for meaning.")
    return 0


if __name__ == "__main__":
    sys.exit(validate())
