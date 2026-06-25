"""
run_eval.py  —  grade the pipeline against the gold set and print the scorecard.

For each gold question it calls ask.answer(question, sql_model=MODEL) (the trace
function), then scores it by tier:

  CAPABILITY (expected="answer"): run the reference_sql for the TRUE rows, run the
     pipeline, compare result sets (order-insensitive, column-name-insensitive,
     floats rounded, tolerant of an extra column the model may add).
  out_of_scope (expected="reject"): correct iff the pipeline REFUSED (not in_scope).
  ambiguous   (expected="clarify"): correct iff the pipeline SIGNALLED uncertainty
     (low_confidence) or refused — i.e. it didn't confidently guess.
  adversarial (empty table): observational only — reported, never pass/fail.

Captures per-stage signals (right tables? valid SQL? executed?), tokens, and time
so the scorecard shows WHERE failures happen, not just how many.

Run:
  python eval/run_eval.py                       # default model in config
  python eval/run_eval.py qwen2.5-coder:7b      # pick a model (for the bake-off)
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import config
import graph as engine          # v2: the LangGraph port (drop-in twin of ask.answer)
import pymysql

GOLD = Path(__file__).resolve().parent / "gold.jsonl"


# --- result-set comparison --------------------------------------------------

def _norm_val(v):
    try:
        return round(float(v), 2)
    except (TypeError, ValueError):
        return v if v is None else str(v)


def _norm_rows(rows):
    normed = [tuple(_norm_val(v) for v in r) for r in rows]
    return sorted(normed, key=lambda t: [str(x) for x in t])


def rows_match(expected, generated):
    """True if `generated` answers as well as `expected`:
    exact set match, OR value-containment with equal row count (tolerates the
    model adding an extra column). Order / column-names / rounding ignored."""
    e, g = _norm_rows(expected), _norm_rows(generated)
    if e == g:
        return True
    if len(e) != len(g):
        return False
    used = [False] * len(g)
    for er in e:
        ec = Counter(er)
        hit = False
        for i, gr in enumerate(g):
            if not used[i] and all(Counter(gr)[k] >= n for k, n in ec.items()):
                used[i] = True
                hit = True
                break
        if not hit:
            return False
    return True


# --- grade one question -----------------------------------------------------

def grade_item(g, cur, model):
    r = {"id": g["id"], "category": g["category"], "expected": g["expected"],
         "retrieval_hit": None, "valid": None, "executed": None, "match": None,
         "attempts": None, "tokens": 0, "seconds": 0.0, "fail_stage": "",
         "correct": False}

    wall = time.time()
    tr = engine.answer(g["question"], sql_model=model, compose=False)
    r["seconds"] = round(time.time() - wall, 2)
    r["tokens"] = sum((e.get("in") or 0) + (e.get("out") or 0) for e in tr["tokens"])

    # ---- robustness tiers (no reference rows) ----
    if g["expected"] == "reject":
        r["correct"] = (tr["in_scope"] is False)
        r["fail_stage"] = "" if r["correct"] else "should_have_rejected"
        return r
    if g["expected"] == "clarify":
        # correct iff the pipeline didn't confidently guess: it either paused to
        # ask (M3 interrupt), flagged low retrieval confidence, or refused.
        r["correct"] = (bool(tr.get("needs_clarification")) or bool(tr["low_confidence"])
                        or (tr["in_scope"] is False))
        r["fail_stage"] = "" if r["correct"] else "answered_confidently"
        return r

    # ---- capability / adversarial (have reference SQL) ----
    r["attempts"] = len(tr["attempts"])
    r["valid"] = tr["valid"]
    r["executed"] = tr["executed"]
    got = set(tr["tables"])
    r["retrieval_hit"] = set(g.get("tables", [])).issubset(got) if g.get("tables") else None

    expected_rows = []
    if g.get("reference_sql"):
        cur.execute(g["reference_sql"])
        expected_rows = [list(x) for x in cur.fetchall()]
    if tr["executed"]:
        r["match"] = rows_match(expected_rows, tr["rows"])

    if g["category"] == "adversarial":
        r["correct"] = None                         # observational
        return r

    r["correct"] = bool(r["match"])
    if not r["correct"]:                            # attribute to FIRST broken stage
        if r["retrieval_hit"] is False:
            r["fail_stage"] = "retrieval"
        elif not r["valid"]:
            r["fail_stage"] = "sql_gen"
        elif not r["executed"]:
            r["fail_stage"] = "execution"
        else:
            r["fail_stage"] = "wrong_result"
    return r


# --- run + scorecard --------------------------------------------------------

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else config.OLLAMA_SQL_MODEL
    items = [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]
    conn = pymysql.connect(host=config.MYSQL_HOST, port=config.MYSQL_PORT,
                           user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
                           database=config.MYSQL_DB)
    cur = conn.cursor()

    print(f"\nGRADING with model = {model}   ({len(items)} questions)\n" + "=" * 70)
    results = []
    for g in items:
        res = grade_item(g, cur, model)
        results.append(res)
        mark = {True: "OK  ", False: "FAIL", None: "obs "}[res["correct"]]
        if res["expected"] == "answer":
            extra = (f"hit={res['retrieval_hit']} valid={res['valid']} "
                     f"exec={res['executed']} match={res['match']} att={res['attempts']}")
        else:
            extra = f"{res['expected']}_ok={res['correct']}"
        print(f"  [{mark}] {res['id']:<4} {res['category']:<12} "
              f"{res['seconds']:>5.1f}s {res['tokens']:>5}tok  {extra}")
    conn.close()

    ans = [r for r in results if r["expected"] == "answer" and r["category"] != "adversarial"]
    rej = [r for r in results if r["expected"] == "reject"]
    clar = [r for r in results if r["expected"] == "clarify"]
    adv = [r for r in results if r["category"] == "adversarial"]

    n = len(ans)
    correct = sum(1 for r in ans if r["correct"])
    valid = sum(1 for r in ans if r["valid"])
    hit = sum(1 for r in ans if r["retrieval_hit"])
    rej_ok = sum(1 for r in rej if r["correct"])
    clar_ok = sum(1 for r in clar if r["correct"])
    avg = lambda xs: round(sum(xs) / len(xs), 2) if xs else 0
    fails = Counter(r["fail_stage"] for r in ans if not r["correct"])

    cat_acc = {}
    for c in ["simple", "aggregate", "grouping", "join", "multi_join"]:
        cc = [r for r in ans if r["category"] == c]
        if cc:
            cat_acc[c] = f"{sum(1 for r in cc if r['correct'])}/{len(cc)}"

    print("\n" + "=" * 70)
    print(f"SCORECARD — {model}")
    print("=" * 70)
    print("  -- CAPABILITY tier --")
    print(f"  EXECUTION ACCURACY (rows match):  {correct}/{n} = {round(100*correct/n,1)}%")
    print(f"  valid SQL produced:               {valid}/{n} = {round(100*valid/n,1)}%")
    print(f"  retrieval hit (right tables):     {hit}/{n} = {round(100*hit/n,1)}%")
    print(f"  accuracy by category:             "
          + "  ".join(f"{k} {v}" for k, v in cat_acc.items()))
    print(f"  failures by stage:                "
          + (", ".join(f"{k}={v}" for k, v in fails.items()) if fails else "none"))
    print("  -- ROBUSTNESS tier --")
    print(f"  out-of-scope correctly rejected:  {rej_ok}/{len(rej)}")
    print(f"  ambiguous flagged-uncertain:      {clar_ok}/{len(clar)}")
    for r in adv:
        print(f"  adversarial {r['id']} (empty table):     "
              f"executed={r['executed']} match={r['match']}  (observational)")
    print("  -- EFFICIENCY (capability tier) --")
    print(f"  avg attempts / question:          {avg([r['attempts'] for r in ans])}")
    print(f"  avg tokens / question:            {avg([r['tokens'] for r in ans])}")
    print(f"  avg seconds / question:           {avg([r['seconds'] for r in ans])}")

    safe = model.replace("/", "_").replace(":", "_")
    out = Path(__file__).resolve().parent / f"results_{safe}.json"
    out.write_text(json.dumps({"model": model, "summary": {
        "accuracy": round(correct / n, 4), "valid_rate": round(valid / n, 4),
        "retrieval_rate": round(hit / n, 4), "reject_ok": rej_ok, "n_reject": len(rej),
        "ambiguous_ok": clar_ok, "n_ambiguous": len(clar), "n_answerable": n,
        "avg_attempts": avg([r['attempts'] for r in ans]),
        "avg_tokens": avg([r['tokens'] for r in ans]),
        "avg_seconds": avg([r['seconds'] for r in ans]),
        "fails_by_stage": dict(fails), "category_accuracy": cat_acc,
    }, "results": results}, indent=2), encoding="utf-8")
    print(f"\n  saved -> {out.name}")


if __name__ == "__main__":
    main()
