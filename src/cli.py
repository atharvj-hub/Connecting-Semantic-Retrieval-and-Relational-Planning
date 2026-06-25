"""
cli.py  —  M5: interactive, Claude-Code-style terminal for the SQL agent.

What you get:
  * Ask a question -> ONE fast model answers, streaming its reasoning live.
  * Slash commands for the heavy stuff:
      /compare [question]   run the FULL multi-model pool (all flavours, voted)
      /model [name]         show or switch the default single model
      /history              show this conversation
      /sessions             list saved sessions
      /resume <id>          continue a past session
      /new                  start a fresh session
      /help                 list commands
      /exit                 quit
  * SESSIONS: every conversation is saved and resumable (data/output/sessions/).
  * MULTI-TURN MEMORY: follow-ups ("now group that by month") see prior turns.
  * SEMANTIC CACHE: a near-duplicate question reuses the previous answer instantly.

The engine itself lives in multi_sql.py — this file is just the conversation shell
on top of it, so /compare and the default turn share the exact same generation,
validation, execution and Opik tracing.

Run:
    python src/cli.py
"""

import json
import math
import re
import sys
import time
import uuid

import config
import ask
import multi_sql
import orchestrator

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

console = Console()

SESS_DIR = config.OUTPUT_DIR / "sessions"
CACHE_FILE = config.OUTPUT_DIR / "qcache.json"


# --- semantic cache ----------------------------------------------------------
# Design: cache the validated SQL (not the answer text) and RE-EXECUTE on every
# hit so data is always fresh. TTL guards how long we trust the cached *SQL*
# before regenerating from scratch. See docs/v2-hardening-plan.md Fix #1.

def _load_cache():
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_cache(cache):
    try:
        CACHE_FILE.write_text(json.dumps(cache, default=str), encoding="utf-8")
    except Exception:
        pass


def _cos(a, b):
    s = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return s / (na * nb) if na and nb else 0.0


def _context_key(question, history, identity=""):
    """Build a context key string from conversation history + current question.

    The key folds the last CACHE_HISTORY_TURNS prior questions so a follow-up
    can only match a cache entry that had the *same* conversational context.
    A cold question (empty history) keys on itself alone, exactly as before.

    The identity parameter is reserved for MCP multi-user keying — today it's
    always "" (single-user). When multi-user lands, prepending an identity
    component is a one-liner: _context_key(q, h, identity=user_id).
    """
    prior = [t["q"] for t in (history or [])[-config.CACHE_HISTORY_TURNS:]]
    parts = prior + [question]
    raw_key = "\n".join(parts)
    # Structured so identity prepends cleanly (hash boundary).
    return f"{identity}||{raw_key}" if identity else raw_key


def _volatility(sql):
    """Classify a SQL query's volatility for TTL selection.

    volatile  — references churny tables (logs, transactions, etc.)
    default   — aggregates over non-stable tables
    stable    — pure lookups on catalog/config tables
    """
    tables = re.findall(r"(?:from|join)\s+`?(\w+)`?", sql, re.I)
    tables_lower = [t.lower() for t in tables]

    # Check volatile: any table name contains a volatile pattern substring
    # AND is not overridden by the stable-tables set.
    has_volatile = False
    for tbl in tables_lower:
        if tbl in config.CACHE_STABLE_TABLES:
            continue
        for pat in config.CACHE_VOLATILE_PATTERNS:
            if pat in tbl:
                has_volatile = True
                break
        if has_volatile:
            break

    if has_volatile:
        return "volatile"

    # Aggregates over non-stable tables get default TTL.
    if re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\b", sql, re.I):
        return "default"

    return "stable"


def _ttl_for(vol):
    """Return the TTL in seconds for a volatility class."""
    return {"volatile": config.CACHE_TTL_VOLATILE,
            "default": config.CACHE_TTL_DEFAULT,
            "stable": config.CACHE_TTL_STABLE}.get(vol, config.CACHE_TTL_DEFAULT)


def cache_lookup(question, history):
    """Find a matching cache entry: context-aware key, TTL enforcement.

    Returns (entry, score) on hit, or None on miss.
    """
    cache = _load_cache()
    if not cache:
        return None
    ctx = _context_key(question, history)
    try:
        emb = ask._embed(ctx)
    except Exception:
        return None
    best, best_s = None, 0.0
    now = time.time()
    for item in cache:
        s = _cos(emb, item.get("emb", []))
        if s > best_s:
            best, best_s = item, s
    if not best or best_s < config.CACHE_THRESHOLD:
        return None
    # TTL check: is the cached SQL still trusted?
    age = now - best.get("ts", 0)
    ttl = _ttl_for(best.get("volatility", "default"))
    if age > ttl:
        return None  # expired → regenerate
    return (best, best_s)


def cache_add(question, history, sql):
    """Store a validated SQL in the cache (no answer text — we recompose on hit).

    Applies FIFO cap at CACHE_MAX_ENTRIES.
    """
    ctx = _context_key(question, history)
    try:
        emb = ask._embed(ctx)
    except Exception:
        return
    vol = _volatility(sql)
    cache = _load_cache()
    cache.append({"key": ctx, "sql": sql, "emb": emb,
                  "ts": time.time(), "volatility": vol})
    # FIFO cap: evict oldest entries.
    if len(cache) > config.CACHE_MAX_ENTRIES:
        cache = cache[-config.CACHE_MAX_ENTRIES:]
    _save_cache(cache)


# --- sessions ----------------------------------------------------------------

def _sess_path(sid):
    return SESS_DIR / f"{sid}.json"


def new_session():
    SESS_DIR.mkdir(parents=True, exist_ok=True)
    return {"id": uuid.uuid4().hex[:8],
            "created": time.strftime("%Y-%m-%d %H:%M"), "turns": []}


def save_session(s):
    SESS_DIR.mkdir(parents=True, exist_ok=True)
    _sess_path(s["id"]).write_text(json.dumps(s, indent=2, default=str),
                                   encoding="utf-8")


def list_sessions():
    SESS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(SESS_DIR.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return out


def load_session(sid):
    p = _sess_path(sid)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# --- rendering ---------------------------------------------------------------

def show_result(cand, answer, names, cached=False, score=None):
    if cand is None:
        console.print("\n[yellow]That question doesn't look answerable from this "
                      "database.[/yellow]\n")
        return
    tag = f"  [dim](cached, {score:.2f} match)[/dim]" if cached else ""
    console.print(Panel(Markdown(f"```sql\n{cand['sql']}\n```"),
                        title=f"SQL{tag}", border_style="cyan"))
    if cand.get("executed"):
        title = ("Answer  (from cache)" if cached
                 else f"Answer  ({len(cand['rows'])} row(s))")
        console.print(Panel(answer or "(no answer composed)",
                            title=title, border_style="green"))
    elif not cached:
        console.print(f"[red]Could not produce valid SQL: {cand.get('error')}[/red]")


# --- commands ----------------------------------------------------------------

HELP = """[bold]Commands[/bold]
  [cyan]/mode [escalate|pool][/cyan] B=escalate (fast, default) · A=pool (all models)
  [cyan]/compare [question][/cyan]   run ALL models (full pool, voted winner)
  [cyan]/model [name][/cyan]         show or set the default single model
  [cyan]/think [on|off][/cyan]       show or hide live reasoning (default on)
  [cyan]/history[/cyan]              show this conversation
  [cyan]/sessions[/cyan]             list saved sessions
  [cyan]/resume <id>[/cyan]          continue a past session
  [cyan]/new[/cyan]                  start a fresh session
  [cyan]/help[/cyan]                 this help
  [cyan]/exit[/cyan]                 quit
Anything else is a question, answered by the default model with live reasoning."""


def handle_command(cmd, state):
    parts = cmd.strip().split(maxsplit=1)
    name = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    sess = state["sess"]

    if name in ("/exit", "/quit", "/q"):
        return False
    if name == "/help":
        console.print(Panel(HELP, border_style="blue"))
    elif name == "/compare":
        q = arg or (sess["turns"][-1]["q"] if sess["turns"] else "")
        if not q:
            console.print("[yellow]Usage: /compare <question>[/yellow]")
        else:
            multi_sql.run(q, history=sess["turns"],
                          show_thinking=state["show_thinking"])
    elif name == "/mode":
        a = arg.lower()
        if a in ("pool", "a"):
            state["mode"] = "pool"
        elif a in ("escalate", "b"):
            state["mode"] = "escalate"
        elif a:
            console.print("[yellow]usage: /mode escalate|pool[/yellow]")
        label = "A: full pool" if state["mode"] == "pool" else "B: escalate"
        console.print(f"mode: [cyan]{state['mode']}[/cyan]  ({label})")
    elif name == "/think":
        if arg.lower() in ("on", "off"):
            state["show_thinking"] = (arg.lower() == "on")
        else:
            state["show_thinking"] = not state["show_thinking"]
        console.print(f"live thinking: [cyan]"
                      f"{'on' if state['show_thinking'] else 'off'}[/cyan]")
    elif name == "/model":
        if not arg:
            opts = ", ".join(m["name"] for m in multi_sql.MODELS)
            console.print(f"default model: [cyan]{state['model']}[/cyan]\n"
                          f"available: {opts}")
        elif multi_sql.model_by_name(arg):
            state["model"] = arg
            console.print(f"default model set to [cyan]{arg}[/cyan]")
        else:
            console.print(f"[red]unknown model '{arg}'[/red]")
    elif name == "/history":
        if not sess["turns"]:
            console.print("[dim]no turns yet[/dim]")
        for i, t in enumerate(sess["turns"], 1):
            console.print(f"[bold]{i}.[/bold] {t['q']}\n   [dim]{t.get('sql', '')}[/dim]")
    elif name == "/sessions":
        tbl = Table("id", "created", "turns", "first question")
        for s in list_sessions():
            first = s["turns"][0]["q"] if s["turns"] else "—"
            tbl.add_row(s["id"], s.get("created", "?"), str(len(s["turns"])), first[:50])
        console.print(tbl)
    elif name == "/resume":
        s = load_session(arg)
        if s:
            state["sess"] = s
            console.print(f"[green]resumed session {arg} "
                          f"({len(s['turns'])} turns)[/green]")
        else:
            console.print(f"[red]no session '{arg}'[/red]")
    elif name == "/new":
        state["sess"] = new_session()
        console.print(f"[green]new session {state['sess']['id']}[/green]")
    else:
        console.print(f"[yellow]unknown command {name} — try /help[/yellow]")
    return True


# --- main loop ---------------------------------------------------------------

def answer_turn(question, state):
    sess = state["sess"]
    # 1) semantic cache — re-execute cached SQL for fresh data (Fix #1).
    #    The cache stores SQL, not answer text. On hit we re-run the query
    #    and recompose the English answer, so data is always current.
    hit = cache_lookup(question, sess["turns"])
    if hit:
        item, score = hit
        try:
            # Fix #5 × Fix #1 seam: cost-gate the cached SQL before re-executing.
            # A query that was cheap when cached can grow expensive as its table
            # grows. On cost-gate failure, surface the message (don't fall through
            # to regen — it would produce the same expensive SQL).
            cost_ok, est_rows, cost_reason = ask.explain_cost(item["sql"])
            if not cost_ok:
                console.print(f"\n[red]{cost_reason}[/red]\n")
                return
            cols, rows = ask.execute(item["sql"])
            answer = ask.compose_answer(question, cols, rows)
            cand_view = {"sql": item["sql"], "executed": True,
                         "rows": rows, "error": None}
            show_result(cand_view, answer, None, cached=True, score=score)
            sess["turns"].append({"q": question, "sql": item["sql"],
                                  "answer": answer, "model": "cache"})
            save_session(sess)
            return
        except Exception:
            # Re-execution failed (schema drift, SQL no longer valid) →
            # treat as miss and fall through to the full path.
            pass

    # 2) run the orchestrator in the current mode (escalate=B default, pool=A).
    #    Reasoning streams live inside the graph's worker nodes.
    #    Fix #2: pass a clarify_cb so vague questions pause interactively.
    def _clarify(cq):
        console.print(f"\n[bold yellow]Clarification needed:[/bold yellow] {cq}")
        try:
            return console.input("  [bold green]answer[/bold green] > ").strip()
        except (EOFError, KeyboardInterrupt):
            return ""

    res = orchestrator.answer(question, mode=state["mode"], history=sess["turns"],
                              model_name=state["model"],
                              show_thinking=state["show_thinking"],
                              clarify_cb=_clarify)
    if not res["in_scope"]:
        console.print("\n[yellow]That question doesn't look answerable from this "
                      "database.[/yellow]\n")
        return
    if res.get("needs_clarification"):
        # Non-interactive fallback (shouldn't happen in the CLI, but be safe).
        console.print(f"\n[yellow]Could not resolve ambiguity: "
                      f"{res.get('clarifying_question', '')}[/yellow]\n")
        return
    w = res.get("winner") or {}
    cand_view = {"sql": res["sql"], "executed": bool(res["answer"]),
                 "rows": w.get("rows", []), "error": "could not produce valid SQL"}
    show_result(cand_view, res["answer"], None)
    if res["n_candidates"] > 1:
        label = ("escalated to pool" if state["mode"] == "escalate"
                 else "pool mode")
        console.print(f"[dim]({label} — {res['n_candidates']} candidates "
                      f"compared, voted)[/dim]")
    sess["turns"].append({"q": question, "sql": res["sql"],
                          "answer": res["answer"], "model": state["mode"]})
    save_session(sess)
    if res["answer"] and res["sql"]:
        cache_add(question, sess["turns"], res["sql"])


def main():
    state = {"sess": new_session(), "model": multi_sql.DEFAULT_MODEL,
             "tracer": multi_sql._tracer(), "show_thinking": True,
             "mode": "escalate"}
    console.print(Panel(
        "[bold]Text-to-SQL Agent — v2[/bold]\n"
        f"model: [cyan]{state['model']}[/cyan]   |   mode: [cyan]escalate[/cyan] "
        "(B)   |   type a question, or [cyan]/help[/cyan] for commands",
        border_style="magenta"))
    while True:
        try:
            q = console.input("\n[bold green]you[/bold green] > ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not q:
            continue
        if q.startswith("/"):
            if not handle_command(q, state):
                break
        else:
            answer_turn(q, state)
    console.print(f"[dim]session {state['sess']['id']} saved.[/dim]")


if __name__ == "__main__":
    sys.exit(main())
