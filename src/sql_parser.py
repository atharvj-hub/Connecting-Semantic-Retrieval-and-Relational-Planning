"""
sql_parser.py  —  Stage 1 core logic

Turns a raw MySQL dump (plain text) into structured Python objects.

WHY plain-text + regex instead of a SQL library (e.g. sqlparse):
    A MySQL dump is full of dialect-specific syntax that trips up generic SQL
    parsers: `UNSIGNED`, `AUTO_INCREMENT`, `ENGINE=InnoDB`, backtick quoting,
    `ENUM('a','b')`, and `/*!50705 ... */` conditional comments. Because the
    dump format is small and predictable, targeted regex is more reliable,
    has ZERO dependencies, and is easy to debug line by line.

This module exposes four pure functions (no file I/O, no globals) so each can
be tested in isolation:
    find_create_table_blocks(schema_sql) -> [(name, body), ...]
    parse_columns(body)                  -> (columns, primary_key)
    parse_foreign_keys(body)             -> [fk, ...]
    extract_sample_values(data_sql, ...) -> {column: [values]}
"""

import re

# ---------------------------------------------------------------------------
# Regexes (compiled once at import time)
# ---------------------------------------------------------------------------

# A whole `CREATE TABLE x ( ... ) ...;` block.
#   group 1 = table name (optional surrounding backticks stripped via `?)
#   group 2 = body = everything between the opening "(" and the closing "\n)"
# We end the body at a newline followed by ")" so it works whether the table
# closes with ") ENGINE=InnoDB ...;" or ") DEFAULT CHARSET=utf8mb4;".
CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+`?(\w+)`?\s*\((.*?)\n\)\s*[^;]*;",
    re.IGNORECASE | re.DOTALL,
)

# The type at the start of a column definition. Captures, in order:
#   base type word          -> INT, VARCHAR, DECIMAL, ENUM ...
#   optional (...) group    -> (128), (4,2), ('G','PG',...)   [commas allowed]
#   optional UNSIGNED/etc.  -> SMALLINT UNSIGNED
# The (...) uses [^)]* which is safe here because Sakila's ENUM/SET values
# never contain a ")".
TYPE_RE = re.compile(
    r"^([A-Za-z]+(?:\s*\([^)]*\))?(?:\s+(?:UNSIGNED|ZEROFILL|SIGNED))*)",
    re.IGNORECASE,
)

# PRIMARY KEY (col)  or  PRIMARY KEY (col_a, col_b)   -> capture inside parens.
PRIMARY_KEY_RE = re.compile(r"PRIMARY\s+KEY\s*\(([^)]*)\)", re.IGNORECASE)

# The database name, from either `USE `db`` or `CREATE DATABASE ... `db``.
DATABASE_RE = re.compile(
    r"(?:USE|CREATE\s+DATABASE(?:\s+IF\s+NOT\s+EXISTS)?)\s+`?(\w+)`?",
    re.IGNORECASE,
)

# FOREIGN KEY (local_cols) REFERENCES other_table (other_cols)
FOREIGN_KEY_RE = re.compile(
    r"FOREIGN\s+KEY\s*\(([^)]*)\)\s*REFERENCES\s+`?(\w+)`?\s*\(([^)]*)\)",
    re.IGNORECASE,
)

# Line prefixes that are NOT column definitions (indexes / constraints).
_NON_COLUMN_PREFIXES = (
    "PRIMARY KEY", "UNIQUE KEY", "UNIQUE INDEX", "KEY ", "KEY(",
    "FULLTEXT KEY", "FULLTEXT INDEX", "SPATIAL KEY", "SPATIAL INDEX",
    "INDEX ", "INDEX(", "CONSTRAINT", "FOREIGN KEY", "CHECK",
)


def _strip_backticks(token: str) -> str:
    """Remove surrounding backticks and whitespace from an identifier."""
    return token.strip().strip("`").strip()


# Comment handling inside a CREATE TABLE body.
#   /* ... */   normal block comment           -> deleted entirely
#   /*! ... */  MySQL "executable" comment      -> UNWRAPPED (inner SQL kept),
#               because it can hold a real column, e.g. sakila's address table
#               declares its GEOMETRY `location` column as
#               `/*!50705 location GEOMETRY */ ... /*!50705 NOT NULL,*/`
#   -- ...      line comment                    -> deleted to end of line
_BLOCK_COMMENT_RE = re.compile(r"/\*(?!!).*?\*/", re.DOTALL)  # /* */ but NOT /*!
_EXEC_OPEN_RE = re.compile(r"/\*!\d*\s?")                     # the "/*!#####" opener
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")


def _clean_body(body: str) -> str:
    """Strip comments from a block body, unwrapping executable /*! ... */ ones."""
    body = _BLOCK_COMMENT_RE.sub("", body)   # remove ordinary block comments
    body = _EXEC_OPEN_RE.sub("", body)       # drop the "/*!#####" openers ...
    body = body.replace("*/", "")            # ... and their closing markers
    body = _LINE_COMMENT_RE.sub("", body)    # remove -- line comments
    return body


# ---------------------------------------------------------------------------
# 0. Find the database name (optional; not every dump declares one)
# ---------------------------------------------------------------------------
def find_database_name(sql: str):
    """Return the database name from a USE/CREATE DATABASE statement, or None."""
    m = DATABASE_RE.search(sql)
    return _strip_backticks(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# 1. Split the schema file into per-table blocks
# ---------------------------------------------------------------------------
def find_create_table_blocks(schema_sql: str):
    """Return [(table_name, body_text), ...] for every CREATE TABLE in the dump.

    `body_text` is the raw text between the opening "(" and closing ")" — i.e.
    the column definitions plus the PRIMARY KEY / KEY / CONSTRAINT lines.
    """
    blocks = []
    for match in CREATE_TABLE_RE.finditer(schema_sql):
        table_name = _strip_backticks(match.group(1))
        body = _clean_body(match.group(2))   # strip comments before parsing
        blocks.append((table_name, body))
    return blocks


# ---------------------------------------------------------------------------
# 2. Parse columns + primary key from one block body
# ---------------------------------------------------------------------------
def parse_columns(body: str):
    """Parse a block body into (columns, primary_key).

    columns      -> [{"name": str, "type": str, "description": ""}, ...]
    primary_key  -> [str, ...]   (a list so composite keys are first-class)

    `description` is intentionally left "" — it is filled by the LLM in Stage 2.
    """
    columns = []
    primary_key = []

    for raw_line in body.split("\n"):
        line = raw_line.strip().rstrip(",").strip()
        if not line:
            continue

        upper = line.upper()

        # Primary key line: grab the column list, don't treat it as a column.
        if upper.startswith("PRIMARY KEY"):
            m = PRIMARY_KEY_RE.search(line)
            if m:
                primary_key = [
                    _strip_backticks(c) for c in m.group(1).split(",") if c.strip()
                ]
            continue

        # Any other index / constraint line is not a column.
        if upper.startswith(_NON_COLUMN_PREFIXES):
            continue

        # Otherwise this is a real column definition: "<name> <type> <modifiers>"
        parts = line.split(None, 1)          # split on first run of whitespace
        if len(parts) < 2:
            continue                          # malformed / not a column
        name = _strip_backticks(parts[0])
        rest = parts[1]

        type_match = TYPE_RE.match(rest)
        if not type_match:
            continue
        # Normalise: collapse internal whitespace and uppercase the type syntax.
        col_type = re.sub(r"\s+", " ", type_match.group(1)).strip().upper()

        columns.append({"name": name, "type": col_type, "description": ""})

    return columns, primary_key


# ---------------------------------------------------------------------------
# 3. Parse foreign keys from one block body
# ---------------------------------------------------------------------------
def parse_foreign_keys(body: str):
    """Parse FK constraints into:
        [{"column", "references_table", "references_column"}, ...]

    Composite FKs (multiple columns) are expanded into one entry per
    column pair so each row is a single, graph-ready edge for Neo4j (Task 2).
    """
    foreign_keys = []
    for m in FOREIGN_KEY_RE.finditer(body):
        local_cols = [_strip_backticks(c) for c in m.group(1).split(",") if c.strip()]
        ref_table = _strip_backticks(m.group(2))
        ref_cols = [_strip_backticks(c) for c in m.group(3).split(",") if c.strip()]

        for local, ref in zip(local_cols, ref_cols):
            foreign_keys.append({
                "column": local,
                "references_table": ref_table,
                "references_column": ref,
            })
    return foreign_keys


# ---------------------------------------------------------------------------
# 4. Extract sample values from the data file
# ---------------------------------------------------------------------------
def _scan_tuple(text: str, start: int):
    """Scan one `( ... )` value tuple starting at index `start` (the '(').

    Returns (inside_text, index_after_closing_paren). Respects single-quoted
    strings (with '' and \\' escapes) and /* ... */ comments so that a comma
    or paren INSIDE a string/comment is never mistaken for a delimiter.
    """
    assert text[start] == "("
    i = start + 1
    depth = 1
    in_str = False
    out = []
    n = len(text)
    while i < n and depth > 0:
        c = text[i]
        if in_str:
            if c == "\\" and i + 1 < n:          # backslash escape: keep both
                out.append(c); out.append(text[i + 1]); i += 2; continue
            if c == "'" and i + 1 < n and text[i + 1] == "'":  # '' escaped quote
                out.append("''"); i += 2; continue
            if c == "'":
                in_str = False
            out.append(c); i += 1; continue
        # not in a string
        if c == "'":
            in_str = True; out.append(c); i += 1; continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":      # comment: pass through
            j = text.find("*/", i + 2)
            if j == -1:
                j = n - 2
            out.append(text[i:j + 2]); i = j + 2; continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                i += 1
                break
        out.append(c); i += 1
    return "".join(out), i


def _split_tuple_fields(inside: str):
    """Split the inside of one tuple into raw field strings.

    Same quote/comment awareness as _scan_tuple. MySQL conditional comments
    `/*!##### ... */` are first UNWRAPPED (their inner SQL is the real value,
    e.g. a spatial blob in sakila's `address` table), matching how MySQL itself
    treats them.
    """
    # Unwrap executable comments: /*!50705 0x...,*/  ->  0x...,
    inside = re.sub(r"/\*!\d*\s?", "", inside)
    inside = inside.replace("*/", "")

    fields = []
    cur = []
    in_str = False
    i = 0
    n = len(inside)
    while i < n:
        c = inside[i]
        if in_str:
            if c == "\\" and i + 1 < n:
                cur.append(c); cur.append(inside[i + 1]); i += 2; continue
            if c == "'" and i + 1 < n and inside[i + 1] == "'":
                cur.append("''"); i += 2; continue
            if c == "'":
                in_str = False
            cur.append(c); i += 1; continue
        if c == "'":
            in_str = True; cur.append(c); i += 1; continue
        if c == ",":
            fields.append("".join(cur).strip()); cur = []; i += 1; continue
        cur.append(c); i += 1
    fields.append("".join(cur).strip())
    return fields


def _convert(token: str, max_len: int):
    """Convert one raw SQL value token into a JSON-friendly Python value."""
    if token.upper() == "NULL":
        return None
    if len(token) >= 2 and token[0] == "'" and token[-1] == "'":
        s = token[1:-1].replace("''", "'").replace("\\'", "'").replace("\\\\", "\\")
        return s[:max_len]
    # Bare number?
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token[:max_len]   # hex blobs, etc.


def extract_sample_values(data_sql: str, table_name: str, column_names,
                          k: int = 3, max_len: int = 60):
    """Return {column_name: [up to k sample values]} for one table.

    Reads only the FIRST `INSERT INTO <table> VALUES (...)...` statement and
    stops after k tuples, so huge inserts (e.g. 16k payments) are not fully
    parsed — we only need a hint.
    """
    pattern = re.compile(
        r"INSERT\s+INTO\s+`?" + re.escape(table_name) + r"`?\s+VALUES\s*",
        re.IGNORECASE,
    )
    m = pattern.search(data_sql)
    if not m:
        return {}

    pos = m.end()
    rows = []
    while len(rows) < k and pos < len(data_sql):
        # skip whitespace / newlines / the commas between tuples
        while pos < len(data_sql) and data_sql[pos] in " \t\r\n,":
            pos += 1
        if pos >= len(data_sql) or data_sql[pos] == ";":
            break
        if data_sql[pos] != "(":
            break
        inside, pos = _scan_tuple(data_sql, pos)
        fields = _split_tuple_fields(inside)
        rows.append([_convert(f, max_len) for f in fields])

    # Pivot rows -> {column: [values]}, aligning by position with column_names.
    samples = {}
    for idx, col in enumerate(column_names):
        values = [row[idx] for row in rows if idx < len(row)]
        if values:
            samples[col] = values
    return samples
