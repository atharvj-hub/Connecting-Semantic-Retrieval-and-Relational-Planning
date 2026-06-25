"""
multi_sql.py  —  M4: multi-model SQL generation with real-time visible reasoning.

What this adds on top of the single-model pipeline (graph.py / ask.py):
  * A MODEL POOL: several models each take a crack at the SAME question —
    qwen3 (local, has a real thinking channel), qwen2.5-coder (local, fast),
    and MiniMax-M3 (cloud, reasoning). Each is a "flavour".
  * A DIVERSITY MATRIX: every model is run under each PROMPT STRATEGY
    (a shared prompt for all + a curated per-model prompt), and the temps/styles
    are knobs you can widen later. Default = models x {shared, per_model}.
  * REAL-TIME THINKING: we stream each model's reasoning as it happens — the
    [thinking] channel for reasoning models, or reason-out-loud content for the
    fast model — so you watch it think, ChatGPT-style.
  * SELECTION: every candidate is validated + executed; we group by the result
    they return and the majority answer wins (execution voting). All candidates
    stay visible so you (and the senior) can compare the SQL + reasoning.

It REUSES the existing pieces: ask.retrieve / expand_candidates / find_joins /
build_context for the focused context, and ask.validate / explain / execute /
compose_answer for checking and answering. Only the generation layer is new.

Notes for THIS machine (8GB VRAM): local models can't fit in memory together, so
they run one-at-a-time (Ollama swaps them) — we group all of a model's candidates
back-to-back to swap as little as possible. The cloud model uses no VRAM.

Run:
    python src/multi_sql.py "how many tenants are there"
"""

import json
import os
import sys
import time
from collections import Counter

import config
import ask                                # reuse retrieval + validation + execution
import value_linker                       # fix filter values ('Active' -> 'ACTIVE')

# Force UTF-8 output: model tokens (and our labels) can contain non-ASCII, and the
# default Windows console codepage (cp1252) raises UnicodeEncodeError on them mid
# stream. errors="replace" guarantees streaming never crashes on an odd character.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Opik logs a huge run-dict dump at WARNING when it can't extract token usage
# (e.g. a provider that doesn't return counts in streaming). Keep that noise out
# of the terminal — real errors still show at ERROR level.
import logging
logging.getLogger("opik").setLevel(logging.ERROR)

try:
    from langchain_ollama import ChatOllama
except ModuleNotFoundError:
    sys.exit("ERROR: pip install langchain-ollama")

from langchain_core.messages import SystemMessage, HumanMessage


# --- the model pool ----------------------------------------------------------
# reasoning=True models expose a separate thinking stream; the fast coder model
# has none, so we make it reason out loud in its answer via the prompt.
MODELS = [
    {"name": "qwen2.5-coder:7b", "provider": "ollama", "reasoning": False,
     "label": "qwen2.5-coder:7b (local, fast)"},
    {"name": "qwen3:8b",        "provider": "ollama", "reasoning": True,
     "label": "qwen3:8b (local, reasoning)"},
    {"name": "minimax-m3",      "provider": "openai", "reasoning": True,
     "label": "MiniMax-M3 (cloud, reasoning)"},
]

# the diversity matrix (all knobs — widen TEMPS/STYLES for a bigger spread)
STRATEGIES = ["shared", "per_model"]      # both prompt strategies, in parallel
TEMPS = [0.0]
STYLES = ["cot"]

# --- prompt strategies -------------------------------------------------------
SHARED_PREAMBLE = (
    "You are a MySQL expert. Reason briefly about which tables, columns and joins "
    "are needed, then write ONE MySQL SELECT query.")

PER_MODEL_PREAMBLE = {
    "qwen3:8b": "You are a meticulous data analyst. Think step by step about the "
                "schema, the joins, and any filters, then write the MySQL query.",
    "qwen2.5-coder:7b": "You are a precise MySQL code generator. State your plan in "
                        "one or two short lines, then output the query.",
    "minimax-m3": "You are an expert data engineer. Carefully consider join keys and "
                  "filter values against the schema, then produce correct MySQL.",
}

_RULES = (
    "Rules:\n"
    "- Use ONLY the tables, columns and join keys provided; never invent names.\n"
    "- MySQL dialect. A read-only SELECT only (no INSERT/UPDATE/DELETE/DDL).\n"
    "- Match string values exactly as shown in sample values.\n"
    "- Add LIMIT 100 unless the query is an aggregate.\n")


def build_prompt(question, context, strategy, model_name, history=None):
    """Return (system_text, user_text) as TWO separate messages.

    system_text = the role + rules — this is the actual 'system prompt'. It's the
    swappable piece: identical for every model (strategy='shared') or curated per
    model (strategy='per_model'). user_text = the focused schema context + any
    conversation history + the question. Splitting them means the system prompt
    shows up as a real system-role message in Opik, and the same-vs-per-model
    experiment becomes a clean one-variable swap."""
    preamble = (SHARED_PREAMBLE if strategy == "shared"
                else PER_MODEL_PREAMBLE.get(model_name, SHARED_PREAMBLE))
    system_text = f"{preamble}\n\n{_RULES}"
    convo = ""
    if history:
        # last few turns so follow-ups ("now group that by month") have context
        lines = [f"Q: {h['q']}\nSQL: {h.get('sql', '')}" for h in history[-3:]]
        convo = ("Earlier in this conversation (context for follow-up questions):\n"
                 + "\n".join(lines) + "\n\n")
    user_text = (f"{context}\n\n{convo}"
                 f"Question: {question}\n\n"
                 "First reason briefly, then give the final query in a ```sql code block.")
    return system_text, user_text


# --- model clients (cached so a model loads once) ----------------------------
_CLIENTS: dict = {}


def _client(model_cfg, temp):
    key = (model_cfg["name"], temp)
    if key in _CLIENTS:
        return _CLIENTS[key]
    if model_cfg["provider"] == "ollama":
        c = ChatOllama(model=model_cfg["name"], temperature=temp,
                       reasoning=True if model_cfg["reasoning"] else None)
    else:                                  # OpenAI-compatible (MiniMax)
        from langchain_openai import ChatOpenAI
        # stream_usage=True asks the OpenAI-compatible endpoint to include token
        # counts in the stream, so Opik can record them (and stops dumping a giant
        # "failed to extract usage" warning to the terminal).
        c = ChatOpenAI(model=model_cfg["name"], temperature=temp, stream_usage=True,
                       base_url=os.environ.get("MINIMAX_BASE_URL",
                                               "https://api.minimax.io/v1"),
                       api_key=os.environ.get("MINIMAX_API_KEY"))
    _CLIENTS[key] = c
    return c


# --- stream ONE candidate live to the terminal -------------------------------

def _opik_reachable():
    """Fast (<=1s) check that the self-hosted Opik server is actually up. Lets us
    skip tracing entirely when the stack is stopped (e.g. to free RAM) instead of
    paying connection-timeout overhead on every run."""
    import socket
    from urllib.parse import urlparse
    p = urlparse(os.environ.get("OPIK_URL_OVERRIDE", "http://localhost:5173/api"))
    host = p.hostname or "localhost"
    port = p.port or (443 if p.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _tracer():
    """One Opik tracer for the whole run (or None if Opik is off/unavailable)."""
    if os.environ.get("OPIK_TRACING", "1") == "0" or not _opik_reachable():
        return None
    try:
        import opik_ollama
        return opik_ollama.make_tracer(os.environ.get("OPIK_PROJECT", "text2sql-v2"))
    except Exception:
        return None


class _ThinkSplitter:
    """Routes a streamed CONTENT string into (think | answer) pieces by detecting
    <think>...</think> tags — the format MiniMax-M3 uses (its reasoning is inline
    in content, not in a separate reasoning_content field). Handles tags split
    across stream chunks by holding back any trailing partial-tag bytes. Models
    that don't use tags (qwen3, coder) just stream straight through as 'answer'."""

    def __init__(self):
        self.buf = ""
        self.in_think = False

    def _partial_tail(self, tag):
        for n in range(min(len(tag) - 1, len(self.buf)), 0, -1):
            if self.buf.endswith(tag[:n]):
                return n
        return 0

    def feed(self, text):
        self.buf += text
        out = []
        while True:
            if not self.in_think:
                i = self.buf.find("<think>")
                if i == -1:
                    keep = self._partial_tail("<think>")
                    if len(self.buf) > keep:
                        out.append(("answer", self.buf[:len(self.buf) - keep]))
                        self.buf = self.buf[len(self.buf) - keep:]
                    break
                if i > 0:
                    out.append(("answer", self.buf[:i]))
                self.buf = self.buf[i + len("<think>"):]
                self.in_think = True
            else:
                i = self.buf.find("</think>")
                if i == -1:
                    keep = self._partial_tail("</think>")
                    if len(self.buf) > keep:
                        out.append(("think", self.buf[:len(self.buf) - keep]))
                        self.buf = self.buf[len(self.buf) - keep:]
                    break
                if i > 0:
                    out.append(("think", self.buf[:i]))
                self.buf = self.buf[i + len("</think>"):]
                self.in_think = False
        return out

    def flush(self):
        if not self.buf:
            return []
        piece = ("think" if self.in_think else "answer", self.buf)
        self.buf = ""
        return [piece]


def stream_one(client, messages, label, tracer=None, metadata=None, show_thinking=True):
    """Stream a model's thinking + answer to the terminal as it happens. Returns
    (thinking_text, answer_text, usage, seconds). Two reasoning formats are unified:
    Ollama models put thinking in additional_kwargs['reasoning_content']; MiniMax
    inlines it as <think>...</think> in content (parsed by _ThinkSplitter). When
    show_thinking is False, reasoning is still captured but not printed (the
    /think off mode). Each candidate is logged to Opik as its own trace."""
    print("\n" + "=" * 74)
    print(f"  {label}")
    print("=" * 74)
    cfg = None
    if tracer is not None:
        cfg = {"callbacks": [tracer], "run_name": label, "metadata": metadata or {}}
    think_parts, ans_parts = [], []
    usage = {}
    last = [None]

    def emit(channel, text):
        if not text:
            return
        # always capture (for the Opik trace + the returned reasoning)...
        (think_parts if channel == "think" else ans_parts).append(text)
        # ...but only print thinking when asked to
        if channel == "think" and not show_thinking:
            return
        if channel != last[0]:
            sys.stdout.write(("\n\n" if last[0] is not None else "")
                             + ("  [thinking] " if channel == "think" else "  [output] "))
            last[0] = channel
        sys.stdout.write(text)
        sys.stdout.flush()

    t0 = time.time()
    splitter = _ThinkSplitter()
    for ch in client.stream(messages, config=cfg):
        rc = (ch.additional_kwargs or {}).get("reasoning_content")
        if rc:
            emit("think", rc)
        if ch.content:
            for channel, piece in splitter.feed(ch.content):
                emit(channel, piece)
        um = getattr(ch, "usage_metadata", None)
        if um:
            usage = um                      # final chunk carries the token counts
    for channel, piece in splitter.flush():
        emit(channel, piece)
    secs = round(time.time() - t0, 1)
    print()
    if tracer is not None:
        try:
            tracer.flush()
        except Exception:
            pass
    return "".join(think_parts), "".join(ans_parts), usage, secs


# --- result signature for voting ---------------------------------------------

def _result_key(rows):
    norm = sorted([tuple(str(v) for v in r) for r in rows])
    return hash(repr(norm))


# --- reusable engine pieces (shared by run() and the interactive CLI) ---------

DEFAULT_MODEL = "qwen2.5-coder:7b"             # Fix #4: fast coder first (was qwen3:8b)


def _minimax_ok():
    """True only if the cloud key is present AND clean (ASCII, no spaces). A
    malformed key would crash the HTTP layer, so we skip the cloud model instead."""
    k = os.environ.get("MINIMAX_API_KEY", "")
    return bool(k) and all(ord(c) < 128 and not c.isspace() for c in k)


def model_by_name(name):
    for m in MODELS:
        if m["name"] == name:
            return m
    return None


def prepare(question):
    """Retrieve tables + assemble the focused context (reuses the v1 pipeline).
    Returns (table_names, context, schema); table_names is None if out of scope."""
    schema = json.loads(config.SCHEMA_JSON.read_text(encoding="utf-8"))
    tables, _ = ask.retrieve(question)
    if not tables:
        return None, None, schema
    names = [t for t, _ in tables]
    added, _ = ask.expand_candidates(names)
    names += added
    joins, _ = ask.find_joins(names)
    context = ask.build_context([(t, {}) for t in names], joins, schema)
    return names, context, schema


def make_candidate(question, context, schema, model_cfg, strategy, temp,
                   tracer=None, history=None, show_thinking=True):
    """Run ONE model under ONE prompt strategy: stream its reasoning live, then
    validate + EXPLAIN + execute the SQL. Returns the candidate dict."""
    system_text, user_text = build_prompt(question, context, strategy,
                                          model_cfg["name"], history)
    messages = [SystemMessage(content=system_text), HumanMessage(content=user_text)]
    label = f"{model_cfg['label']}   |   prompt={strategy}   |   temp={temp}"
    meta = {"question": question, "model": model_cfg["name"],
            "strategy": strategy, "temperature": temp}
    think, ans, usage, secs = stream_one(_client(model_cfg, temp), messages, label,
                                         tracer=tracer, metadata=meta,
                                         show_thinking=show_thinking)
    sql = ask._extract_sql(ans)
    ok, err = ask.validate(sql, schema)
    vlinks = []
    if ok:                                  # value-linking: 'Active' -> 'ACTIVE'
        sql, vlinks = value_linker.link_values(sql, schema)
    executed, cols, rows, rkey = False, [], [], None
    if ok and ask._HAVE_MYSQL:
        exp_ok, exp_err = ask.explain(sql)
        if exp_ok is False:
            ok, err = False, f"MySQL rejected it: {exp_err}"
    # Fix #5: cost-gate — refuse execution if EXPLAIN estimates too many rows.
    if ok and ask._HAVE_MYSQL:
        cost_ok, est_rows, cost_reason = ask.explain_cost(sql)
        if not cost_ok:
            ok, err = False, cost_reason
    if ok and ask._HAVE_MYSQL:
        try:
            cols, rows = ask.execute(sql)
            executed, rkey = True, _result_key(rows)
        except Exception as exc:
            err = str(exc)
    return {"model": model_cfg["label"], "strategy": strategy, "sql": sql,
            "valid": ok, "error": err, "executed": executed, "value_links": vlinks,
            "cols": cols, "rows": rows, "rkey": rkey, "thinking": think,
            "tokens_in": usage.get("input_tokens"),
            "tokens_out": usage.get("output_tokens"), "seconds": secs}


def single(question, history=None, model_name=None, strategy="shared", tracer=None,
           show_thinking=True):
    """One-model streamed answer for the interactive CLI (the default turn).
    Returns (candidate, answer_text, table_names) — candidate is None if out of scope."""
    names, context, schema = prepare(question)
    if names is None:
        return None, None, None
    m = model_by_name(model_name or DEFAULT_MODEL) or MODELS[0]
    cand = make_candidate(question, context, schema, m, strategy,
                          TEMPS[0], tracer, history, show_thinking)
    answer = (ask.compose_answer(question, cand["cols"], cand["rows"])
              if cand["executed"] else None)
    return cand, answer, names


# --- orchestrate the FULL matrix (the /compare path) -------------------------

def run(question, history=None, show_thinking=True):
    names, context, schema = prepare(question)
    if names is None:
        print("\n** Out of scope — no table scored above the floor. **")
        return
    print(f"\nQUESTION: {question}")
    print(f"Tables in scope: {', '.join(names)}")

    tracer = _tracer()
    candidates = []
    for m in MODELS:
        if m["provider"] == "openai" and not _minimax_ok():
            print(f"\n(skipping {m['label']} — no usable MINIMAX_API_KEY)")
            continue
        for strat in STRATEGIES:
            for temp in TEMPS:
                try:
                    candidates.append(make_candidate(
                        question, context, schema, m, strat, temp, tracer,
                        history, show_thinking))
                except Exception as exc:
                    print(f"  [error running this candidate: {exc}]")
    _summary(question, candidates)


def _summary(question, candidates):
    if not candidates:
        print("\n** No candidates produced. **")
        return

    # per-candidate line: result + tokens + time
    print("\n\n" + "#" * 74)
    print("  CANDIDATES  (model | prompt | result | tokens | time)")
    print("#" * 74)
    for c in candidates:
        status = "valid" if c["valid"] else "INVALID"
        rc = len(c["rows"]) if c["executed"] else "-"
        tin = c.get("tokens_in") or "?"
        tout = c.get("tokens_out") or "?"
        print(f"\n* {c['model']}  [{c['strategy']}]  ->  {status}, rows={rc}"
              f"   ({tin} in / {tout} out tok, {c.get('seconds')}s)")
        if not c["valid"] and c["error"]:
            print(f"    reason: {c['error']}")

    # collapse identical SQL (whitespace-normalised) into distinct queries
    groups = {}
    for c in candidates:
        key = " ".join((c["sql"] or "").split())
        groups.setdefault(key, []).append(c)
    print("\n" + "-" * 74)
    print(f"  DISTINCT QUERIES: {len(groups)} unique out of {len(candidates)} candidates")
    print("-" * 74)
    for i, (key, cs) in enumerate(groups.items(), 1):
        who = ", ".join(f"{c['model'].split(' ')[0]}[{c['strategy']}]" for c in cs)
        print(f"\n[{i}]  ({len(cs)}x)  {who}")
        print(f"     {(cs[0]['sql'] or '(no sql)').strip()}")

    # consensus / agreement on the RESULT the queries actually returned
    executed = [c for c in candidates if c["executed"]]
    if not executed:
        print("\n** No candidate executed successfully. **")
        return
    vote = Counter(c["rkey"] for c in executed)
    win_key, win_n = vote.most_common(1)[0]
    winner = next(c for c in executed if c["rkey"] == win_key)

    print("\n" + "=" * 74)
    if len(vote) == 1:
        print(f"  CONSENSUS: all {len(executed)} executed candidates agree on the result")
    else:
        print(f"  AGREEMENT: {win_n}/{len(executed)} agree on the winning result "
              f"— {len(vote)} different results seen (models DISAGREE)")
    print("=" * 74)
    print(f"  WINNER: {winner['model']}  [{winner['strategy']}]")
    print(f"  {winner['sql']}")
    answer = ask.compose_answer(question, winner["cols"], winner["rows"])
    print(f"\n  ANSWER: {answer}\n")


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit('Usage: python src/multi_sql.py "your question here"')
    run(" ".join(args))


if __name__ == "__main__":
    main()
