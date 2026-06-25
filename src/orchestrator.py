"""
orchestrator.py  —  M6: multi-model SQL orchestration as a real LangGraph graph.

This is the TRINITY-shaped orchestration, with a TOGGLE between two modes:

  mode="pool"      (Option A)  — always run the FULL model pool (every model ×
                                 every prompt strategy), then vote. Max coverage,
                                 slow. The "show everything" / demo mode.

  mode="escalate"  (Option B, default) — try ONE strong model first; only if it
                                 hits trouble (invalid SQL after repair, or low
                                 retrieval confidence) escalate to the full pool
                                 and vote. Fast + reliable for the common 95%,
                                 full power only when needed.

TRINITY roles, made explicit as graph nodes:
  * coordinator  = the mode router + the escalate-on-trouble decision (edges)
  * worker(s)    = single_node / pool_node  (the models generating candidates)
  * verifier     = vote_node  (+ the sqlglot/EXPLAIN/execute checks inside each
                   candidate)  -> picks the result the candidates agree on

It REUSES the engine from multi_sql.py (prepare / make_candidate / voting) so the
default turn, /compare, and this orchestrator all share one generation+validation
path. Building it as a StateGraph (not a loop) earns checkpointing, tracing and a
visible, conditionally-routed graph — and leaves a clean seam to add a semantic
layer as "Tier 0" in front later.

Graph shape:

    retrieve ─(in scope?)─┬─ pool ───────────► pool_node ─► vote ─► compose ─► END
                          ├─ escalate ─► single_node ─(ok & confident?)
                          │                     ├─ yes ─► vote ─► compose ─► END
                          │                     └─ no  ─► pool_node ─► vote ─► compose ─► END
                          └─ out of scope ─────────────────────────────────► END
"""

import json
import operator
import sqlite3
import sys
import uuid
from collections import Counter

try:
    from typing import Annotated, Optional, TypedDict
except ImportError:                          # pragma: no cover
    from typing_extensions import Annotated, Optional, TypedDict

import config
import ask
import multi_sql
import graph as graph_module  # for _clarity_check (P-4: reuse, don't duplicate)
import metrics_layer          # Tier 0: the deterministic semantic/metrics layer
import value_linker           # fix filter values ('Active' -> 'ACTIVE')

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.types import interrupt, Command
except ModuleNotFoundError as exc:
    sys.exit(f"ERROR: missing package ({exc.name}). Run: pip install langgraph "
             f"langgraph-checkpoint-sqlite")

SCHEMA = json.loads(config.SCHEMA_JSON.read_text(encoding="utf-8"))
_CKPT = config.OUTPUT_DIR / "orchestrator_checkpoints.sqlite"

# one Opik tracer for the whole module (cached; auto-skips if Opik is down)
_TRACER = {"set": False, "obj": None}


def _tracer():
    if not _TRACER["set"]:
        _TRACER["obj"] = multi_sql._tracer()
        _TRACER["set"] = True
    return _TRACER["obj"]


# --- state -------------------------------------------------------------------

class OrchState(TypedDict, total=False):
    question: str
    mode: str                                # "pool" | "escalate"
    history: list
    show_thinking: bool
    sql_model: str                           # default single model (escalate path)
    # filled by nodes
    in_scope: bool
    low_confidence: bool
    needs_clarification: bool                # Fix #2: M3 clarity gatekeeper
    clarifying_question: Optional[str]
    clarified: bool
    names: list
    context: Optional[str]
    candidates: Annotated[list, operator.add]
    escalated: bool
    winner: Optional[dict]
    sql: Optional[str]
    answer: Optional[str]
    status: Optional[str]                    # P-5: typed result contract


# --- nodes -------------------------------------------------------------------

def retrieve_node(state: OrchState) -> dict:
    """Grounding (shared by both modes): retrieve tables, expand via the graph,
    find joins, assemble the focused context. Also flags low retrieval confidence,
    which is one of the escalate triggers."""
    question = state["question"]
    tables, _ = ask.retrieve(question)
    if not tables:
        return {"in_scope": False, "status": "out_of_scope"}
    names = [t for t, _ in tables]
    added, _ = ask.expand_candidates(names)
    names += added
    joins, _ = ask.find_joins(names)
    context = ask.build_context([(t, {}) for t in names], joins, SCHEMA)
    low_conf = tables[0][1]["score"] < ask.CONFIDENCE_BAR
    return {"in_scope": True, "names": names, "context": context,
            "low_confidence": low_conf}


def clarity_node(state: OrchState) -> dict:
    """Fix #2: M3 gatekeeper — flag a vague question before any SQL is written.
    Reuses graph._clarity_check (P-4: single source of truth). Skipped once the
    user has already clarified (so we never loop asking forever)."""
    if state.get("clarified"):
        return {"needs_clarification": False}
    ambiguous, cq = graph_module._clarity_check(
        state["question"], state.get("sql_model"))
    return {"needs_clarification": ambiguous,
            "clarifying_question": cq or None}


def clarify_pause_node(state: OrchState) -> dict:
    """Fix #2: human-in-the-loop pause. interrupt() suspends the graph and
    surfaces the clarifying question; when resumed with the user's reply, the
    reply is folded into the question and we re-retrieve with sharper intent."""
    reply = interrupt({"clarifying_question": state.get("clarifying_question")})
    refined = f"{state['question']} (clarification: {reply})"
    return {"question": refined, "clarified": True,
            "needs_clarification": False, "clarifying_question": None}


def single_node(state: OrchState) -> dict:
    """Escalate mode, step 1: one strong model has a go (shared system prompt)."""
    m = (multi_sql.model_by_name(state.get("sql_model") or multi_sql.DEFAULT_MODEL)
         or multi_sql.MODELS[0])
    cand = multi_sql.make_candidate(
        state["question"], state["context"], SCHEMA, m, "shared",
        multi_sql.TEMPS[0], _tracer(), state.get("history"),
        state.get("show_thinking", True))
    return {"candidates": [cand]}


def pool_node(state: OrchState) -> dict:
    """The full pool: every model × every prompt strategy. Skips any (model,
    strategy) already tried by single_node so escalation doesn't repeat work."""
    done = {(c["model"], c["strategy"]) for c in state.get("candidates", [])}
    out = []
    for m in multi_sql.MODELS:
        if m["provider"] == "openai" and not multi_sql._minimax_ok():
            print(f"\n(skipping {m['label']} — no usable MINIMAX_API_KEY)")
            continue
        for strat in multi_sql.STRATEGIES:
            for temp in multi_sql.TEMPS:
                if (m["label"], strat) in done:
                    continue
                try:
                    out.append(multi_sql.make_candidate(
                        state["question"], state["context"], SCHEMA, m, strat,
                        temp, _tracer(), state.get("history"),
                        state.get("show_thinking", True)))
                except Exception as exc:
                    print(f"  [error running candidate: {exc}]")
    return {"candidates": out, "escalated": True}


def vote_node(state: OrchState) -> dict:
    """Verifier: among candidates that executed, pick the result the most agree on
    (works for 1 candidate or many)."""
    cands = state.get("candidates", [])
    executed = [c for c in cands if c.get("executed")]
    if not executed:
        last = cands[-1] if cands else None
        return {"winner": last, "sql": last["sql"] if last else None}
    vote = Counter(c["rkey"] for c in executed)
    win_key, _ = vote.most_common(1)[0]
    winner = next(c for c in executed if c["rkey"] == win_key)
    return {"winner": winner, "sql": winner["sql"]}


def compose_node(state: OrchState) -> dict:
    """Turn the winning result into a plain-English answer (local model)."""
    w = state.get("winner")
    if not w or not w.get("executed"):
        # P-5: distinguish a cost-gate refusal (too_expensive) from a plain
        # invalid/failed query, so an MCP client can react differently.
        err = ((w or {}).get("error") or "").lower()
        status = "too_expensive" if "too expensive" in err else "invalid"
        return {"answer": None, "status": status}
    return {"answer": ask.compose_answer(state["question"], w["cols"], w["rows"]),
            "status": "ok"}


# --- routing (the coordinator) ----------------------------------------------

def _route_after_retrieve(state: OrchState) -> str:
    if not state.get("in_scope"):
        return END
    return "clarity"  # Fix #2: always check clarity before proceeding


def _route_after_clarity(state: OrchState) -> str:
    """Fix #2: ambiguous (and not already clarified) → pause and ask; else
    route to the selected mode."""
    if state.get("needs_clarification") and not state.get("clarified"):
        return "clarify_pause"
    return "pool" if state.get("mode") == "pool" else "single"


def _route_after_single(state: OrchState) -> str:
    """Escalate-on-trouble: a confident, valid, executed single answer wins;
    anything shaky escalates to the full pool."""
    cands = state.get("candidates", [])
    last = cands[-1] if cands else None
    good = bool(last and last.get("valid") and last.get("executed")
                and not state.get("low_confidence"))
    return "vote" if good else "pool"


# --- build / compile ---------------------------------------------------------

def build_graph(checkpointer=None):
    g = StateGraph(OrchState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("clarity", clarity_node)
    g.add_node("clarify_pause", clarify_pause_node)
    g.add_node("single", single_node)
    g.add_node("pool", pool_node)
    g.add_node("vote", vote_node)
    g.add_node("compose", compose_node)

    g.set_entry_point("retrieve")
    g.add_conditional_edges("retrieve", _route_after_retrieve,
                            {"clarity": "clarity", END: END})
    g.add_conditional_edges("clarity", _route_after_clarity,
                            {"clarify_pause": "clarify_pause",
                             "single": "single", "pool": "pool"})
    g.add_edge("clarify_pause", "retrieve")  # re-retrieve with refined intent
    g.add_conditional_edges("single", _route_after_single,
                            {"vote": "vote", "pool": "pool"})
    g.add_edge("pool", "vote")
    g.add_edge("vote", "compose")
    g.add_edge("compose", END)
    return g.compile(checkpointer=checkpointer)


_GRAPH = None


def _graph():
    global _GRAPH
    if _GRAPH is None:
        conn = sqlite3.connect(str(_CKPT), check_same_thread=False)
        _GRAPH = build_graph(checkpointer=SqliteSaver(conn))
    return _GRAPH


# --- Tier 0: semantic layer (deterministic metric path, tried first) ---------

def _try_semantic(question, model_name=None):
    """If the question maps to a known metric, compile deterministic SQL and run
    it (validate -> cost-gate -> execute -> compose). Returns a normalised result
    dict on success, or None to fall through to the general graph pipeline.

    Fail-safe by design: ANY problem — no metric match, invalid SQL, too
    expensive, execution error — returns None, so Tier 0 can never block or break
    a question, it can only *short-circuit* the easy ones with a guaranteed answer.
    Note: stateless for now (ignores history), so follow-ups simply fall through."""
    if not ask._HAVE_MYSQL:
        return None
    try:
        sql = metrics_layer.answer_semantic(
            question, model=model_name or multi_sql.DEFAULT_MODEL)
    except Exception:
        return None
    if not sql:
        return None                                   # no metric matched
    ok, _err = ask.validate(sql, SCHEMA)
    if not ok:
        return None                                   # compiled SQL didn't validate
    sql, _vl = value_linker.link_values(sql, SCHEMA)  # fix 'failed' -> 'FAILED' etc.
    cost_ok, _est, _reason = ask.explain_cost(sql)
    if not cost_ok:
        return None                                   # too expensive -> let general path decide
    try:
        cols, rows = ask.execute(sql)
    except Exception:
        return None                                   # execution failed -> fall through
    answer_text = ask.compose_answer(question, cols, rows)
    winner = {"sql": sql, "cols": list(cols), "rows": [list(r) for r in rows],
              "executed": True, "valid": True, "error": None,
              "model": "semantic-layer (Tier 0)", "strategy": "metric"}
    return {
        "in_scope": True, "mode": "tier0", "sql": sql, "answer": answer_text,
        "winner": winner, "candidates": [winner], "n_candidates": 1,
        "escalated": False, "low_confidence": False,
        "needs_clarification": False, "clarifying_question": None,
        "status": "ok", "tier0": True,
    }


# --- public entry point ------------------------------------------------------

def answer(question, mode="escalate", history=None, model_name=None,
           show_thinking=True, clarify_cb=None):
    """Run the orchestration graph and return a normalised result dict.

    clarify_cb: optional callback for interactive clarification. Called with
    the clarifying question string; should return the user's reply. When None
    (eval / non-interactive), an ambiguous question surfaces
    needs_clarification=True in the result instead of pausing.

    Thread-id MUST be reused across resume calls — a new thread_id would start
    a fresh run, losing the interrupt state.
    """
    # Tier 0: try the deterministic semantic/metrics path first. A known metric
    # short-circuits the whole pipeline with guaranteed-correct SQL; anything else
    # returns None and falls through to the general graph below.
    tier0 = _try_semantic(question, model_name)
    if tier0 is not None:
        return tier0

    init = {"question": question, "mode": mode, "history": history or [],
            "show_thinking": show_thinking,
            "sql_model": model_name or multi_sql.DEFAULT_MODEL, "candidates": []}
    thread_id = uuid.uuid4().hex
    cfg = {"configurable": {"thread_id": thread_id}}

    # Get/compile graph once — must reuse the same compiled graph + checkpointer
    # across the initial invoke and any resume calls.
    compiled = _graph()
    final = compiled.invoke(init, config=cfg)

    # Fix #2: handle interrupt (clarity pause).
    # Cap at 3 rounds to prevent infinite ask loops.
    for _ in range(3):
        intr = final.get("__interrupt__")
        if not intr:
            break
        # Extract the clarifying question from the interrupt payload.
        try:
            cq = intr[0].value.get("clarifying_question", "")
        except Exception:
            cq = ""
        if clarify_cb is not None:
            # Interactive path: ask the user and resume the graph.
            reply = clarify_cb(cq)
            if reply:
                # Resume on the SAME thread_id (critical for correctness).
                final = compiled.invoke(
                    Command(resume=reply),
                    config={"configurable": {"thread_id": thread_id}})
                continue
        # Non-interactive or no reply: surface needs_clarification and stop.
        cands = final.get("candidates", [])
        return {
            "in_scope": final.get("in_scope", False),
            "mode": mode,
            "sql": final.get("sql"),
            "answer": final.get("answer"),
            "winner": final.get("winner"),
            "candidates": cands,
            "n_candidates": len(cands),
            "escalated": final.get("escalated", False),
            "low_confidence": final.get("low_confidence", False),
            "needs_clarification": True,
            "clarifying_question": cq,
            "status": "needs_clarification",
        }

    cands = final.get("candidates", [])
    return {
        "in_scope": final.get("in_scope", False),
        "mode": mode,
        "sql": final.get("sql"),
        "answer": final.get("answer"),
        "winner": final.get("winner"),
        "candidates": cands,
        "n_candidates": len(cands),
        "escalated": final.get("escalated", False),
        "low_confidence": final.get("low_confidence", False),
        "needs_clarification": final.get("needs_clarification", False),
        "clarifying_question": final.get("clarifying_question"),
        "status": final.get("status", "ok" if final.get("in_scope") else "out_of_scope"),
    }


def main():
    args = sys.argv[1:]
    mode = "pool" if "--pool" in args else "escalate"
    args = [a for a in args if a != "--pool"]
    if not args:
        sys.exit('Usage: python src/orchestrator.py [--pool] "your question"')
    res = answer(" ".join(args), mode=mode)
    print("\n" + "=" * 74)
    if not res["in_scope"]:
        print("** Out of scope. **")
        return
    print(f"MODE: {res['mode']}   escalated={res['escalated']}   "
          f"candidates={res['n_candidates']}")
    print(f"SQL: {res['sql']}")
    print(f"ANSWER: {res['answer']}")


if __name__ == "__main__":
    sys.exit(main())
