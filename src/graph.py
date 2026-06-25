"""
graph.py  —  v2: the v1 ASK pipeline, re-expressed as a LangGraph graph.

This is a *1:1 port* of ask.answer(). The logic is unchanged — every node here
delegates to the same battle-tested function in ask.py (retrieve, expand,
find_joins, build_context, validate, explain, execute). Only two things are
genuinely rewritten:

  * SQL generation and answer composition now go through langchain_ollama's
    ChatOllama instead of a raw `ollama.generate` call, so the observability
    layer (Opik, wired in M2) can trace every LLM call natively and we get
    real per-call token usage without the hand-built TOKEN_STATS dict.

What the graph shape buys us (for free, vs the hand-rolled state machine):
  * a CHECKPOINTER (SqliteSaver) saves state after every node -> a failed run
    can be replayed/inspected step by step instead of re-run;
  * the 3-attempt repair loop is now a visible CONDITIONAL EDGE, not a `for`;
  * the same structured trace dict comes out the other end, so eval/run_eval.py
    keeps working unchanged as the regression gate proving this port didn't
    move accuracy.

Design rules (see also memory: v2-langgraph-plan):
  * NON-SERIALIZABLE clients (Pinecone, Neo4j driver, ChatOllama) never enter
    graph state — they live module-level / inside nodes. State holds plain data.
  * Output dict keys match ask.answer()'s trace exactly.

Run a single question:
    python src/graph.py "how many orders did each customer place"
"""

import json
import operator
import sqlite3
import sys
import uuid
from typing import Any, Optional

try:
    from typing import Annotated, TypedDict
except ImportError:                      # pragma: no cover
    from typing_extensions import Annotated, TypedDict

import config
import ask                               # v1 logic, reused verbatim

try:
    from langchain_ollama import ChatOllama
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.types import interrupt, Command
except ModuleNotFoundError as exc:
    sys.exit(f"ERROR: missing package ({exc.name}). "
             f"Run: pip install langgraph langchain-ollama langgraph-checkpoint-sqlite")

# --- Opik observability (M2) -------------------------------------------------
# OpikTracer is a LangChain callback: attached to graph.invoke() it logs every
# node + every LLM call (inputs/outputs/latency/tokens) to the self-hosted Opik
# UI. It is OPTIONAL and FAIL-SAFE: if the package is missing or the local Opik
# server is down, tracing silently no-ops so runs (and the eval gate) never break
# just because observability is offline. Toggle with env OPIK_TRACING=0.
import os

OPIK_PROJECT = os.environ.get("OPIK_PROJECT", "text2sql-v2")
OPIK_TRACING = os.environ.get("OPIK_TRACING", "1") != "0"


def _opik_callbacks():
    if not OPIK_TRACING:
        return []
    try:
        import opik_ollama             # token counts + per-call timing for Ollama
        tracer = opik_ollama.make_tracer(OPIK_PROJECT)
        return [tracer] if tracer is not None else []
    except Exception:
        return []                    # opik not installed / not configured -> skip

# Load the schema once (truth source for validation + context assembly).
SCHEMA = json.loads(config.SCHEMA_JSON.read_text(encoding="utf-8"))

# Where the checkpointer persists every run's per-step state (replayability).
_CKPT_PATH = config.OUTPUT_DIR / "checkpoints.sqlite"


# --- graph state -------------------------------------------------------------
# This is the typed twin of ask.py's `tr` dict. `attempts` and `tokens`
# accumulate across the repair loop, so they use an additive reducer; every
# other field is simply overwritten by whichever node last wrote it.

class AgentState(TypedDict, total=False):
    # inputs / knobs
    question: str
    sql_model: str
    compose: bool
    # retrieval
    destructive_intent: bool
    in_scope: bool
    low_confidence: bool
    needs_clarification: bool
    clarifying_question: Optional[str]
    clarified: bool
    matches: list
    candidate_tables: list
    graph_added: list
    tables: list
    joins: list
    join_error: Optional[str]
    context: Optional[str]
    # generate -> validate -> execute
    attempts: Annotated[list, operator.add]
    sql: Optional[str]
    valid: bool
    valid_attempt: Optional[int]
    executed: bool
    cols: list
    rows: list
    exec_error: Optional[str]
    answer_text: Optional[str]
    tokens: Annotated[list, operator.add]


# --- ChatOllama helpers (the only rewritten LLM calls) -----------------------

_LLM_CACHE: dict = {}


def _llm(model, temperature=None):
    """One ChatOllama client per (model, temperature). SQL generation pins
    temperature=0 for deterministic, reproducible queries (v1 ran at Ollama's
    default ~0.8, which let valid-but-wrong SQL slip through on occasion and made
    the eval non-reproducible). Answer composition keeps the default — it's prose,
    not a query, so a little variety is harmless."""
    key = (model, temperature)
    if key not in _LLM_CACHE:
        kw = {} if temperature is None else {"temperature": temperature}
        _LLM_CACHE[key] = ChatOllama(model=model, **kw)
    return _LLM_CACHE[key]


def _usage(resp, label):
    """Pull the model's own token counts out of a ChatOllama response, in the
    same {label,in,out} shape run_eval.py already reads."""
    um = getattr(resp, "usage_metadata", None) or {}
    return {"label": label, "in": um.get("input_tokens"),
            "out": um.get("output_tokens")}


def _gen_sql(question, context, prior_error, model):
    """Same prompt + same SQL extraction as v1 (ask._sql_prompt / _extract_sql);
    only the call path changes to ChatOllama."""
    prompt = ask._sql_prompt(question, context, prior_error)
    resp = _llm(model, temperature=0).invoke(prompt)
    return ask._extract_sql(resp.content), _usage(resp, f"SQL generation ({model})")


def _compose(question, cols, rows):
    """English answer from result rows (ChatOllama). Mirrors ask.compose_answer:
    no LLM call (and no token usage) when there are no rows."""
    if not rows:
        return "The query ran successfully but returned no rows (no matching data).", None

    # Fix #6 fast-path: if scalar, bypass LLM entirely
    scalar = ask._scalar_template(cols, rows)
    if scalar:
        return scalar, None

    preview = [dict(zip(cols, r)) for r in rows[:10]]
    prompt = (
        "Answer the user's question in 1-2 plain sentences using ONLY these SQL "
        "results. Be specific with numbers. Do not invent anything.\n\n"
        f"Question: {question}\n"
        f"Columns: {cols}\n"
        f"Rows (up to 10 shown): {json.dumps(preview, default=str)}\n\n"
        "Answer:")
    try:
        resp = _llm(config.OLLAMA_ANSWER_MODEL).invoke(prompt)
        import re
        text = re.sub(r"<think>.*?</think>", "", resp.content, flags=re.DOTALL).strip()

        # Fix #6 guard: hallucinated number → degrade to the actual rows (truthful),
        # reusing ask._render_rows so the two compose paths can't drift (P-4).
        if not ask._verify_numbers(text, cols, rows):
            return ask._render_rows(cols, rows), _usage(resp, "answer compose")

        return text, _usage(resp, "answer compose")
    except Exception as exc:
        return f"(could not compose a sentence: {exc}) - see rows above.", None


# --- clarity gatekeeper (M3) -------------------------------------------------
# A question can retrieve the right tables with high confidence yet still be
# impossible to answer with ONE definite query, because it leans on a vague term
# whose metric isn't defined ("most active", "the situation", "a summary"). The
# retrieval score can't catch that — it's a semantic judgement — so a small LLM
# gatekeeper decides: specific enough to write SQL, or ambiguous enough to ask?

_CLARITY_PROMPT = """You are a strict gatekeeper for a text-to-SQL system. Decide \
whether a user's question is SPECIFIC enough to translate into ONE definite SQL \
query, or whether it is AMBIGUOUS — so underspecified that a careful analyst would \
have to ask a clarifying question before writing any SQL.

AMBIGUOUS: relies on a vague/subjective term whose metric is undefined — e.g.
"most active", "best", "how things are doing", "the situation", "a summary",
"performance", "overall" — without saying BY WHAT measure.

SPECIFIC: names a clear quantity, filter, or grouping, even if it needs a join —
e.g. "how many X", "list the Y of every Z", "total A by B", "the status of X".

Respond with ONLY a JSON object, nothing else:
{{"ambiguous": true or false, "clarifying_question": "one short question to ask the user, or empty string if specific"}}

Examples:
Q: "How many plugins are installed?"
{{"ambiguous": false, "clarifying_question": ""}}
Q: "List the display name and version of every plugin."
{{"ambiguous": false, "clarifying_question": ""}}
Q: "How many credits have been granted in total?"
{{"ambiguous": false, "clarifying_question": ""}}
Q: "Which tenant is the most active?"
{{"ambiguous": true, "clarifying_question": "How should I measure 'most active' - by number of workflows, sessions, or credits used?"}}
Q: "What's the overall credit situation?"
{{"ambiguous": true, "clarifying_question": "Which credit figure do you want - total granted, total consumed, or remaining balance?"}}

Question: "{question}"
JSON:"""


def _clarity_check(question, model=None):
    """Return (is_ambiguous, clarifying_question). Fails OPEN: any error -> treat
    as specific, so the gatekeeper can never block an otherwise-answerable question.

    Uses the SAME model as SQL generation by default: the gatekeeper runs right
    before generate, so sharing the model keeps Ollama from unload/reloading a
    second model between the two calls (that swap was ~3-4x slower per question)."""
    try:
        resp = _llm(model or config.OLLAMA_SQL_MODEL, temperature=0).invoke(
            _CLARITY_PROMPT.format(question=question))
        import re
        txt = re.sub(r"<think>.*?</think>", "", resp.content, flags=re.DOTALL)
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if not m:
            return False, ""
        data = json.loads(m.group(0))
        return bool(data.get("ambiguous")), (data.get("clarifying_question") or "").strip()
    except Exception:
        return False, ""


# --- nodes -------------------------------------------------------------------

def retrieve_node(state: AgentState) -> dict:
    """Steps 1-3: embed -> Pinecone -> candidate tables (floor/band/cap).
    Also sets the scope + confidence flags (the floor/bar logic lives in
    ask.retrieve + ask.CONFIDENCE_BAR, kept identical)."""
    question = state["question"]
    tables, matches = ask.retrieve(question)
    upd: dict = {
        "destructive_intent": bool(ask._DESTRUCTIVE_INTENT.search(question)),
        "matches": [{"score": m["score"], "kind": m["metadata"].get("kind"),
                     "table": m["metadata"]["table"],
                     "column": m["metadata"].get("column")} for m in matches[:8]],
    }
    if not tables:                                   # below the floor -> out of scope
        upd["in_scope"] = False
        return upd
    upd["in_scope"] = True
    upd["candidate_tables"] = [tn for tn, _ in tables]
    upd["low_confidence"] = tables[0][1]["score"] < ask.CONFIDENCE_BAR
    return upd


def clarity_node(state: AgentState) -> dict:
    """M3 gatekeeper: flag a vague question before any SQL is written. Skipped
    once the user has already clarified (so we never loop asking forever)."""
    if state.get("clarified"):
        return {"needs_clarification": False}
    ambiguous, cq = _clarity_check(state["question"], state.get("sql_model"))
    return {"needs_clarification": ambiguous, "clarifying_question": cq or None}


def clarify_pause_node(state: AgentState) -> dict:
    """The human-in-the-loop pause. interrupt() suspends the graph and surfaces
    the clarifying question; when the run is resumed with the user's reply, that
    reply is folded into the question and we re-retrieve with the sharper intent.
    In a non-interactive run (eval), invoke() returns at the interrupt instead of
    blocking — answer() reports needs_clarification and stops."""
    reply = interrupt({"clarifying_question": state.get("clarifying_question")})
    refined = f"{state['question']} (clarification: {reply})"
    return {"question": refined, "clarified": True, "needs_clarification": False,
            "clarifying_question": None}


def expand_node(state: AgentState) -> dict:
    """Step 3b: Neo4j pulls in entity/bridge tables Pinecone missed."""
    cands = state["candidate_tables"]
    added, _ = ask.expand_candidates(cands)
    return {"graph_added": added, "tables": cands + added}


def joins_node(state: AgentState) -> dict:
    """Step 4: join keys among the candidate tables (degrades if Neo4j down)."""
    joins, join_err = ask.find_joins(state["tables"])
    return {"joins": joins, "join_error": join_err}


def context_node(state: AgentState) -> dict:
    """Step 5: assemble the focused context string sent to the SQL model.
    build_context wants (name, meta) pairs but only uses the names."""
    pairs = [(tn, {}) for tn in state["tables"]]
    return {"context": ask.build_context(pairs, state["joins"], SCHEMA)}


def generate_node(state: AgentState) -> dict:
    """Steps 6-7 (one attempt): generate -> validate (sqlglot) -> EXPLAIN.
    Re-entered by the conditional edge for each repair attempt; the prior
    attempt's error is fed back in, exactly like v1's repair loop."""
    attempts = state.get("attempts", [])
    prior_error = attempts[-1]["error"] if attempts else None
    sql, usage = _gen_sql(state["question"], state["context"], prior_error,
                          state["sql_model"])
    ok, error = ask.validate(sql, SCHEMA)
    explained = None
    if ok and ask._HAVE_MYSQL:
        explained, exp_err = ask.explain(sql)
        if explained is False:
            ok, error = False, f"MySQL rejected the query: {exp_err}"
    attempt = {"sql": sql, "valid": ok, "error": error, "explained": explained}
    upd = {"attempts": [attempt], "sql": sql, "tokens": [usage]}
    if ok:
        upd["valid"] = True
        upd["valid_attempt"] = len(attempts) + 1
    return upd


def execute_node(state: AgentState) -> dict:
    """Phase D: run the validated SELECT, compose an English answer."""
    if not ask._HAVE_MYSQL:
        return {}
    # Fix #5: cost-gate before execution.
    cost_ok, est_rows, cost_reason = ask.explain_cost(state["sql"])
    if not cost_ok:
        return {"exec_error": cost_reason}
    try:
        cols, rows = ask.execute(state["sql"])
        upd: dict = {"executed": True, "cols": list(cols),
                     "rows": [list(r) for r in rows]}
        if state.get("compose", True):
            text, usage = _compose(state["question"], cols, rows)
            upd["answer_text"] = text
            if usage:
                upd["tokens"] = [usage]
        return upd
    except Exception as exc:
        return {"exec_error": str(exc)}


# --- edges (routing) ---------------------------------------------------------

def _route_after_retrieve(state: AgentState) -> str:
    return "clarity" if state.get("in_scope") else END


def _route_after_clarity(state: AgentState) -> str:
    """Ambiguous (and not already clarified) -> pause and ask; else proceed."""
    if state.get("needs_clarification") and not state.get("clarified"):
        return "clarify_pause"
    return "expand"


def _route_after_generate(state: AgentState) -> str:
    """The repair loop, now a visible conditional edge: valid -> execute;
    else retry generate up to 3 attempts; else give up."""
    if state.get("valid"):
        return "execute"
    if len(state.get("attempts", [])) >= 3:
        return END
    return "generate"


# --- build / compile ---------------------------------------------------------

def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("clarity", clarity_node)
    g.add_node("clarify_pause", clarify_pause_node)
    g.add_node("expand", expand_node)
    g.add_node("joins", joins_node)
    g.add_node("context", context_node)
    g.add_node("generate", generate_node)
    g.add_node("execute", execute_node)

    g.set_entry_point("retrieve")
    g.add_conditional_edges("retrieve", _route_after_retrieve,
                            {"clarity": "clarity", END: END})
    g.add_conditional_edges("clarity", _route_after_clarity,
                            {"clarify_pause": "clarify_pause", "expand": "expand"})
    g.add_edge("clarify_pause", "retrieve")        # re-retrieve with refined intent
    g.add_edge("expand", "joins")
    g.add_edge("joins", "context")
    g.add_edge("context", "generate")
    g.add_conditional_edges("generate", _route_after_generate,
                            {"generate": "generate", "execute": "execute", END: END})
    g.add_edge("execute", END)
    return g.compile(checkpointer=checkpointer)


_GRAPH = None


def _graph():
    global _GRAPH
    if _GRAPH is None:
        conn = sqlite3.connect(str(_CKPT_PATH), check_same_thread=False)
        _GRAPH = build_graph(checkpointer=SqliteSaver(conn))
    return _GRAPH


# --- public entry point (drop-in twin of ask.answer) -------------------------

# All trace keys initialised so the returned dict always has the full shape
# eval/run_eval.py expects, even on early exits (out-of-scope, failed repair).
_TRACE_DEFAULTS = {
    "in_scope": False, "low_confidence": False, "destructive_intent": False,
    "needs_clarification": False, "clarifying_question": None, "clarified": False,
    "matches": [], "candidate_tables": [], "graph_added": [], "tables": [],
    "joins": [], "join_error": None, "context": None, "attempts": [],
    "sql": None, "valid": False, "valid_attempt": None, "executed": False,
    "cols": [], "rows": [], "exec_error": None, "answer_text": None, "tokens": [],
}


def answer(question, sql_model=None, compose=True):
    """Run the graph and return the same structured trace dict as ask.answer().

    `sql_model` overrides which Ollama model writes the SQL (None = configured
    default) — the single knob the model bake-off / eval flips.
    """
    model = sql_model or config.OLLAMA_SQL_MODEL
    init = {"question": question, "sql_model": model, "compose": compose,
            "attempts": [], "tokens": []}
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex}}
    cbs = _opik_callbacks()
    if cbs:
        cfg["callbacks"] = cbs
    final = _graph().invoke(init, config=cfg)
    if cbs:
        try:
            cbs[0].flush()           # push traces before a short-lived run exits
        except Exception:
            pass
    tr = {**_TRACE_DEFAULTS, **{k: v for k, v in final.items()
                                if k in AgentState.__annotations__}}
    # If the graph paused at the clarify interrupt (non-interactive run), invoke()
    # returns with an __interrupt__ payload instead of an answer — surface it.
    intr = final.get("__interrupt__")
    if intr:
        tr["needs_clarification"] = True
        try:
            tr["clarifying_question"] = intr[0].value.get("clarifying_question")
        except Exception:
            pass
    return tr


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit('Usage: python src/graph.py "your question here"')
    tr = answer(" ".join(args))
    print(f"\nQUESTION: {tr['question']}\n" + "=" * 64)
    if not tr["in_scope"]:
        print("** Out of scope - no table scored above the floor. **")
        return
    print(f"Candidate tables: {', '.join(tr['candidate_tables'])}")
    if tr["graph_added"]:
        print(f"Graph-added: {', '.join(tr['graph_added'])}")
    if tr["low_confidence"]:
        print("WARNING: low confidence - the question may be vague.")
    print("\nGENERATED SQL:")
    for i, a in enumerate(tr["attempts"], 1):
        print(a["sql"] if a["valid"] else f"[attempt {i} invalid: {a['error']}]")
        if a["valid"]:
            print(f"[VALID on attempt {i}]")
    if tr["valid"] and tr["executed"]:
        print(f"\nRESULT ({len(tr['rows'])} row(s)):")
        print(" | ".join(str(c) for c in tr["cols"]))
        for r in tr["rows"][:20]:
            print(" | ".join(str(v) for v in r))
        print("\nANSWER:\n" + (tr["answer_text"] or ""))


if __name__ == "__main__":
    main()
