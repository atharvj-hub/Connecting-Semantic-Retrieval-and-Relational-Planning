"""
mcp_server.py  —  Phase 0: expose the v2 database agent as a LOCAL stdio MCP server.

This is the newest shell over the SAME engine (CLI -> chat -> now MCP). It does
NOT reimplement anything: every tool delegates to the production LangGraph +
Opik path in orchestrator.answer(). See docs/mcp-implementation-plan.md.

WHAT WE EXPOSE (and why) — docs/mcp-implementation-plan.md §3:
  * ask_database(question)  -> the whole agent (Tier-0 semantic layer first, then
                               the escalate->pool LangGraph orchestration). THE product.
  * search_schema(query)    -> the relevant *slice* of the schema (scoped, read-only),
                               so the caller can ask sharper questions.
  * validate_sql(sql)       -> check a query is valid WITHOUT running it.

WHAT WE DO NOT EXPOSE (and why) — §4:
  * run_sql / writes / raw credentials / cross-tenant / full-schema-dump.
    `run_sql` is deliberately ABSENT from this build (not a runtime toggle) — friction L1.5.

PHASE 0 SCOPE (intentionally local + single-tenant):
  * stdio transport only (one machine, one DB, one user).
  * Every answer carries `answered_question` (echo-back, friction L1.6) so a caller
    that silently reworded the question makes the swap visible to the human.
  * Confidence is STRUCTURAL and honest (friction L2.1, crude form): HIGH only when the
    deterministic Tier-0 metric layer answered; otherwise medium/low.
  * NOT yet done (tracked in docs/mcp-frictions-backlog.md, do not assume these exist):
    tenant identity/isolation (L1.1/L1.2), tenant-scoped cache (L1.3), DB-level
    read-only credential (L1.4), per-tenant creds (L2.4), rich assumptions, etc.
    => Do NOT point this at real multi-tenant data until Layer 1 is closed.

PROTOCOL SAFETY: stdio MCP speaks JSON-RPC over stdout, but orchestrator nodes
print()/stream to stdout. We redirect stdout->stderr around every engine call so
stray prints can't corrupt the protocol stream (they still show up in logs/stderr).

RUN IT (local):
    pip install "mcp[cli]"
    python src/mcp_server.py                 # serves on stdio

WIRE IT INTO CLAUDE (Desktop/Code) — add to the MCP servers config:
    {
      "mcpServers": {
        "database-agent": {
          "command": "python",
          "args": ["C:/Users/athar/OneDrive/Desktop/internship/ai_project2/src/mcp_server.py"]
        }
      }
    }
"""

import contextlib
import os
import sys

# The src/ modules use bare imports (`import config`); make that work no matter
# what directory the MCP client launches us from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json

import config
import ask
import orchestrator

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:
    sys.exit('ERROR: missing package (mcp). Run: pip install "mcp[cli]"')

# Schema is the truth source for validation + schema search (loaded once).
SCHEMA = json.loads(config.SCHEMA_JSON.read_text(encoding="utf-8"))

# Cap rows returned over the wire (engine already caps execute() at 50; this is a
# second belt-and-braces guard so a tool result never blows up the caller's context).
MAX_ROWS_OUT = 50

mcp = FastMCP("database-agent")


def _engine_answer(question: str) -> dict:
    """Call the production orchestrator with stdout redirected to stderr so the
    engine's prints/streaming never corrupt the stdio JSON-RPC stream.

    No clarify_cb is passed: MCP is single-shot, so an ambiguous question comes
    back as status='needs_clarification' (the agent 'stepping in') rather than
    blocking — exactly the behaviour we want a caller to relay to the human.
    """
    with contextlib.redirect_stdout(sys.stderr):
        return orchestrator.answer(question, mode="escalate", show_thinking=False)


def _confidence_and_source(res: dict) -> tuple[str, str]:
    """STRUCTURAL confidence (honest, not vibes) — friction L2.1, crude Phase-0 form.

    HIGH only when the deterministic Tier-0 metric layer produced the answer; the
    general LLM path is medium, or low when retrieval confidence was below the bar.
    """
    if res.get("tier0"):
        return "high", "tier0-semantic"
    if res.get("low_confidence"):
        return "low", ("pool-voted" if res.get("escalated") else "single-model")
    return "medium", ("pool-voted" if res.get("escalated") else "single-model")


@mcp.tool()
def ask_database(question: str) -> dict:
    """Answer a natural-language question about the database with a governed, audited result.

    Pass the user's question VERBATIM — do not summarize, expand, reinterpret, or drop
    qualifiers. The returned `answered_question` echoes back exactly what was answered so
    any reinterpretation is visible to the user.

    Returns a receipt:
      status            : "ok" | "needs_clarification" | "out_of_scope" | "invalid" | "too_expensive"
      answered_question : the exact question this answer is for (echo-back)
      answer            : plain-English answer (when status == "ok")
      sql               : the SQL that produced it (provenance; null if none)
      columns, rows     : the actual result data (rows capped)
      confidence        : "high" | "medium" | "low"  (high only via the deterministic metric layer)
      source            : how it was answered (tier0-semantic | single-model | pool-voted)
      clarifying_question : present when status == "needs_clarification" — relay this to the user
      notes             : honest caveats (e.g. truncation, low confidence)

    Read-only. Cannot modify data. Cannot run caller-supplied SQL (use validate_sql for that).
    """
    res = _engine_answer(question)
    status = res.get("status") or ("ok" if res.get("in_scope") else "out_of_scope")

    receipt: dict = {
        "status": status,
        "answered_question": question,   # Phase-0 single-shot: == input. Field is the contract.
        "answer": None,
        "sql": res.get("sql"),
        "columns": [],
        "rows": [],
        "confidence": None,
        "source": None,
        "clarifying_question": res.get("clarifying_question"),
        "notes": [],
    }

    if status == "out_of_scope":
        receipt["notes"].append("No table scored above the retrieval floor — this "
                                "question doesn't look answerable from this database.")
        return receipt

    if status == "needs_clarification":
        receipt["notes"].append("Question is too vague to answer with one definite "
                                "query. Relay clarifying_question to the user.")
        return receipt

    confidence, source = _confidence_and_source(res)
    receipt["confidence"] = confidence
    receipt["source"] = source

    if status != "ok":   # invalid | too_expensive
        receipt["notes"].append(f"Could not produce a runnable answer (status={status}).")
        return receipt

    # status == "ok": attach the data from the winning candidate.
    receipt["answer"] = res.get("answer")
    winner = res.get("winner") or {}
    cols = list(winner.get("cols") or [])
    rows = [list(r) for r in (winner.get("rows") or [])]
    receipt["columns"] = cols
    if len(rows) > MAX_ROWS_OUT:
        receipt["notes"].append(f"Rows truncated to {MAX_ROWS_OUT} of {len(rows)}.")
        rows = rows[:MAX_ROWS_OUT]
    receipt["rows"] = rows
    if confidence != "high":
        receipt["notes"].append("Answer came from the general LLM path, not the "
                                "deterministic metric layer — verify the SQL.")
    return receipt


@mcp.tool()
def search_schema(query: str) -> dict:
    """Search the database schema for tables/columns relevant to a query.

    Returns only the relevant SLICE (scoped, read-only) — never the full catalog.
    Use this to discover what exists before asking a sharper question.

    Returns:
      in_scope : whether anything matched above the retrieval floor
      tables   : candidate table names, best first
      matches  : [{table, column, kind, score}]  — the top schema hits with scores
    """
    tables, matches = ask.retrieve(query)
    return {
        "in_scope": bool(tables),
        "tables": [name for name, _ in tables],
        "matches": [
            {
                "table": m["metadata"].get("table"),
                "column": m["metadata"].get("column"),
                "kind": m["metadata"].get("kind"),
                "score": round(m["score"], 4),
            }
            for m in matches[:15]
        ],
    }


@mcp.tool()
def validate_sql(sql: str) -> dict:
    """Check whether a SQL query is valid WITHOUT executing it.

    Runs the same sqlglot validation (and, if the DB is reachable, a dry EXPLAIN)
    used by the agent's repair loop. Does NOT run the query and returns no data.

    Returns: { valid: bool, error: str | null }
    """
    ok, error = ask.validate(sql, SCHEMA)
    if ok and ask._HAVE_MYSQL:
        with contextlib.redirect_stdout(sys.stderr):
            explained, exp_err = ask.explain(sql)
        if explained is False:
            return {"valid": False, "error": f"MySQL rejected the query: {exp_err}"}
    return {"valid": bool(ok), "error": error}


if __name__ == "__main__":
    # FastMCP defaults to stdio transport.
    mcp.run()
