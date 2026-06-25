"""
chat.py  —  v2 interactive shell with human-in-the-loop clarification (M3).

This is the runtime surface that the M3 interrupt needs. The one-shot CLI
(graph.py / ask.py) has nowhere to "wait for the human", so a vague question can
only be reported, not resolved. Here, when the graph pauses to ask, we print the
clarifying question, read the user's reply, and RESUME the same run (same
thread_id, so the checkpointer continues exactly where it left off) with the
sharper intent folded in.

Flow per question:
    invoke graph  ->  if it paused (interrupt): ask user, resume  ->  print answer

Run:
    python src/chat.py
"""
import sys
import uuid

import config                                   # noqa: F401 (loads .env)
import graph as engine
from langgraph.types import Command


def _render(tr):
    if not tr["in_scope"]:
        print("\n  This question doesn't look answerable from this database.\n")
        return
    if tr.get("graph_added"):
        print(f"  (tables: {', '.join(tr['tables'])})")
    if not tr["valid"]:
        print("\n  Couldn't produce valid SQL for that one.\n")
        return
    print("\n  SQL: " + (tr["sql"] or ""))
    if tr.get("answer_text"):
        print("\n  " + tr["answer_text"] + "\n")
    elif tr.get("rows") is not None:
        print(f"\n  {len(tr['rows'])} row(s).\n")


def ask_interactive(question):
    """Run one question, handling any clarify-interrupt by asking the user and
    resuming. Returns the final trace dict."""
    g = engine._graph()
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex}}
    cbs = engine._opik_callbacks()
    if cbs:
        cfg["callbacks"] = cbs
    payload = {"question": question, "sql_model": config.OLLAMA_SQL_MODEL,
               "compose": True, "attempts": [], "tokens": []}

    for _ in range(3):                            # cap clarification rounds
        result = g.invoke(payload, config=cfg)
        intr = result.get("__interrupt__")
        if not intr:
            break
        cq = ""
        try:
            cq = intr[0].value.get("clarifying_question") or ""
        except Exception:
            pass
        print(f"\n  I need a bit more detail: {cq}")
        reply = input("  your answer > ").strip()
        payload = Command(resume=reply)           # resume the SAME paused run
    if cbs:
        try:
            cbs[0].flush()
        except Exception:
            pass
    return result                                 # final graph state


def main():
    print("Text-to-SQL chat (v2). Ask a question, or 'exit' to quit.")
    print("Vague questions will be met with a clarifying question.\n")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", ":q"}:
            break
        tr = ask_interactive(q)
        # normalise to the trace shape _render expects
        tr = {**engine._TRACE_DEFAULTS, **{k: v for k, v in tr.items()
                                           if k in engine.AgentState.__annotations__}}
        _render(tr)


if __name__ == "__main__":
    sys.exit(main())
