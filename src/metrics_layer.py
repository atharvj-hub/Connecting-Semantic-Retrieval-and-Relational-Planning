"""
metrics_layer.py — Tier 0 semantic layer: the deterministic metric compiler.

THE IDEA
--------
Instead of letting an LLM write raw SQL (flexible but can silently lie), we define
the business vocabulary ONCE in a hand-authored catalog (metrics.yml):

  * metrics    — named aggregations  (total_granted_credits = SUM(granted_credits))
  * dimensions — ways to slice them  (by_tenant = tenants.name)
  * filters    — reusable predicates (tenant_is = tenants.name = ?)
  * joins      — how tables connect  (grants.tenant_id = tenants.tenant_id)

`compile_form()` then turns a STRUCTURED request —
    {"metric": "total_granted_credits", "dimensions": ["by_tenant"], "filters": []}
— into a SQL string that is GUARANTEED correct, because *code* assembles it from
verified parts. An LLM's only job (added later, in front of this) is the easy
classification: natural language -> that little form.

WHY THIS IS SAFE
----------------
The compiler is pure and deterministic. If a request references something the
catalog doesn't define, it raises CompileError — the caller then falls through to
the general LLM pipeline. So the semantic layer NEVER emits a wrong query; its
failure mode is the loud "I don't have a metric for that", not the silent wrong
number.

This module is the bulletproof CORE. The NL->form resolver (an LLM step) sits in
front of it as a separate, later piece.
"""

import config

try:
    import yaml
except ModuleNotFoundError:                  # pragma: no cover
    yaml = None


class CompileError(Exception):
    """A form can't be compiled (unknown metric/dimension/filter, or no join
    path). The caller treats this as 'not a known metric' and falls through to
    the general pipeline — the semantic layer fails LOUD, never with wrong SQL."""


def load_catalog(path=None):
    """Load metrics.yml into a dict. Raises CompileError if PyYAML is missing."""
    if yaml is None:
        raise CompileError("PyYAML not installed — run: pip install pyyaml")
    p = path or config.METRICS_YML
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _lit(value):
    """Render a filter value as a SQL literal: numbers bare, strings quoted with
    single-quotes doubled (basic injection-safety for v0)."""
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _find_join(joins, base, other):
    """Return (base_key, other_key) connecting `other` directly to `base`.
    v0 supports single-hop star joins only; multi-hop path-finding (A->B->C) is
    deferred — that's the separate 'graph-pathfinding joins' feature."""
    for j in joins or []:
        if j["left"] == base and j["right"] == other:
            return j["left_key"], j["right_key"]
        if j["left"] == other and j["right"] == base:
            return j["right_key"], j["left_key"]
    raise CompileError(f"no join path from '{base}' to '{other}'")


def compile_form(form, catalog):
    """Compile a structured request into a deterministic SQL string.

    form = {
        "metric":     "<metric name>",                # required
        "dimensions": ["<dim name>", ...],            # optional
        "filters":    [{"name": "<filter>", "value": <v>}, ...],  # optional
    }

    Raises CompileError on anything the catalog doesn't define.
    """
    metrics = catalog.get("metrics", {})
    dims = catalog.get("dimensions", {})
    filters = catalog.get("filters", {})
    joins = catalog.get("joins", [])

    mname = form.get("metric")
    if mname not in metrics:
        raise CompileError(f"unknown metric: {mname!r}")
    metric = metrics[mname]
    base = metric["table"]

    dim_list = form.get("dimensions") or []
    fil_list = form.get("filters") or []

    # Collect tables in a deterministic order: base first, then dims, then filters.
    tables = [base]

    def add(t):
        if t not in tables:
            tables.append(t)

    for d in dim_list:
        if d not in dims:
            raise CompileError(f"unknown dimension: {d!r}")
        add(dims[d]["table"])
    for f in fil_list:
        if f["name"] not in filters:
            raise CompileError(f"unknown filter: {f['name']!r}")
        add(filters[f["name"]]["table"])

    # Aliases only when more than one table is involved (keeps single-table SQL clean).
    multi = len(tables) > 1
    alias = {t: (f"t{i}" if multi else "") for i, t in enumerate(tables)}

    def ref(table, expr):
        if expr == "*":
            return "*"
        a = alias[table]
        return f"{a}.{expr}" if a else expr

    # SELECT: dimensions first (they also drive GROUP BY), then the metric.
    select_parts, group_parts = [], []
    for d in dim_list:
        col = ref(dims[d]["table"], dims[d]["expr"])
        select_parts.append(f"{col} AS {d}")
        group_parts.append(col)
    select_parts.append(f"{metric['agg']}({ref(base, metric['expr'])}) AS {mname}")

    # FROM + JOINs (every non-base table joins to the base table — star schema).
    from_clause = f"{base} {alias[base]}".strip()
    join_clauses = []
    for t in tables[1:]:
        bk, ok = _find_join(joins, base, t)
        join_clauses.append(
            f"JOIN {t} {alias[t]} ON {alias[base]}.{bk} = {alias[t]}.{ok}")

    # WHERE
    where_parts = []
    for f in fil_list:
        fil = filters[f["name"]]
        op = fil.get("op", "=")
        where_parts.append(f"{ref(fil['table'], fil['expr'])} {op} {_lit(f['value'])}")

    sql = f"SELECT {', '.join(select_parts)} FROM {from_clause}"
    if join_clauses:
        sql += " " + " ".join(join_clauses)
    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)
    if group_parts:
        sql += " GROUP BY " + ", ".join(group_parts)
    return sql


# =============================================================================
# Resolver — natural language -> form  (the ONE place an LLM is used)
# =============================================================================
# The compiler above is deterministic. The resolver is the small, BOUNDED LLM
# step that turns a question into a {metric, dimensions, filters} form. Two safety
# properties keep it honest:
#   1. It can only choose names that exist in the catalog (they're in the prompt).
#   2. We RE-VALIDATE by trying to compile the form — so a hallucinated name just
#      makes resolve() return None, and the caller falls through to the general
#      pipeline. The LLM never writes SQL; it only fills in the little form.

import json as _json
import re as _re


def _catalog_menu(catalog):
    """Render the catalog as a compact menu for the resolver prompt (kept in sync
    with metrics.yml automatically — no second place to edit)."""
    def block(title, items):
        if not items:
            return f"{title}: (none)"
        lines = [f"- {name}: {d.get('description', '')}" for name, d in items.items()]
        return f"{title}:\n" + "\n".join(lines)
    return "\n\n".join([
        block("METRICS (choose exactly one, or null)", catalog.get("metrics", {})),
        block("DIMENSIONS (zero or more)", catalog.get("dimensions", {})),
        block("FILTERS (zero or more; include a value)", catalog.get("filters", {})),
    ])


_RESOLVER_PROMPT = """You route a question to a metrics engine. Map it to ONE metric \
plus optional dimensions (groupings) and filters. Use ONLY names from the menu below.

{menu}

Respond with ONLY a JSON object, nothing else:
{{"metric": "<name or null>", "dimensions": ["<name>", ...], "filters": [{{"name": "<name>", "value": "<value>"}}, ...]}}

If the question does not fit any metric, respond {{"metric": null}}.

Question: "{question}"
JSON:"""


def resolve(question, catalog=None, model=None):
    """Natural language -> form, or None if it doesn't map to a known metric.

    Fails SAFE: any parse error, an unknown name, or a form that won't compile
    returns None — the caller then falls through to the general LLM pipeline."""
    import ollama
    catalog = catalog if catalog is not None else load_catalog()
    prompt = _RESOLVER_PROMPT.format(menu=_catalog_menu(catalog), question=question)
    try:
        resp = ollama.generate(model=model or config.OLLAMA_SQL_MODEL, prompt=prompt,
                               options={"temperature": 0})
        text = _re.sub(r"<think>.*?</think>", "", resp["response"], flags=_re.DOTALL)
        m = _re.search(r"\{.*\}", text, _re.DOTALL)
        if not m:
            return None
        form = _json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(form, dict) or not form.get("metric"):
        return None
    # Re-validate by compiling: if it won't compile, it's not a usable form.
    try:
        compile_form(form, catalog)
    except CompileError:
        return None
    return form


def answer_semantic(question, catalog=None, model=None):
    """Tier-0 convenience: resolve -> compile -> SQL string, or None to fall
    through to the general pipeline. (Validation/cost-gate/execution happen in the
    caller, reusing ask.validate / ask.explain_cost / ask.execute.)"""
    catalog = catalog if catalog is not None else load_catalog()
    form = resolve(question, catalog, model)
    return None if form is None else compile_form(form, catalog)
