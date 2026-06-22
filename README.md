# Text-to-SQL RAG over a SQL dump

Ask a database a question in **plain English** → get the **correct SQL** → get the
**answer**. Built to scale to hundreds of tables, where the hard part is **schema
linking**: handing the LLM only the 3–6 relevant tables and the exact join keys
instead of the whole database.

> **Bring your own database.** This repo is the *engine* — point it at any MySQL
> dump. It ships with no data and no benchmark questions; you supply both (see below).

```
"<your question in plain English>"
   → embed question        (Ollama nomic-embed-text)
   → Pinecone              which tables, by MEANING
   → Neo4j graph-expand    add entity/bridge tables Pinecone missed
   → Neo4j join discovery  how the tables connect (declared + inferred FKs)
   → focused context       only those tables: columns, types, descriptions, samples, joins
   → qwen2.5-coder         writes the SQL
   → sqlglot validate      alias / column / join-type checks (offline)
   → MySQL EXPLAIN + run   execute the validated SELECT
   → llama3.1              compose the English answer
```

## The two-brain design

| Question | Tool | Holds |
|---|---|---|
| *Which* tables is this about? | **Pinecone** (vector DB) | the **meaning** of each table/column |
| *How* do they connect? | **Neo4j** (graph DB) | the **relationships** (FK graph) |

Neither alone is enough — Pinecone finds relevant tables but can't join them; Neo4j
joins but can't read English. Together they turn a vague question into a small,
join-correct table set, which a small local code model can reliably turn into SQL.

## Run it

**Prerequisites (all free):**
- **Ollama** running with `llama3.1:8b`, `qwen2.5-coder:7b`, `nomic-embed-text`
- **Pinecone** account → `PINECONE_API_KEY` in `.env`
- **Neo4j Aura** (free) → `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` in `.env`
- **Docker** → local MySQL container

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in PINECONE_API_KEY + NEO4J_*

# 1) Put YOUR dump in data/raw/ and point config.py at it
#    (edit SCHEMA_SQL / DATA_SQL / MYSQL_DB in src/config.py, or use env vars)

# 2) Build the knowledge layer (once)
python src/stage1_build_schema.py      # dump -> schema.json
python src/validate_schema.py          # structural health check
python src/stage2_describe.py          # Ollama fills descriptions (resumable)
python src/stage2a_build_graph.py      # schema -> graph.json (+ FK inference)
python src/stage4a_build_semantic.py   # schema -> semantic.json (keywords/aliases)
python src/stage5a_pinecone_upsert.py  # embed + upsert to Pinecone
python src/stage5b_neo4j_load.py       # load graph into Neo4j

# 3) MySQL in Docker, load YOUR dump
docker run --name textsql-mysql -e MYSQL_ROOT_PASSWORD=textsqlpass -p 3306:3306 -d mysql:8
docker exec -i textsql-mysql mysql -uroot -ptextsqlpass < data/raw/your_dump.sql

python src/healthcheck.py              # PASS/FAIL per component

# 4) ASK
python src/ask.py "<your question in plain English>"
```

A good public sample DB to try is **Sakila** (the standard MySQL example database).

## Benchmarking — bring your own gold set

The eval harness grades the pipeline against questions **you** author for **your**
database. Create `eval/gold.jsonl`, one JSON object per line:

```json
{"id":"q1","question":"how many orders did each customer place","category":"grouping","tables":["orders","customers"],"expected":"answer","reference_sql":"SELECT c.name, COUNT(*) FROM orders o JOIN customers c ON o.customer_id=c.id GROUP BY c.name"}
```

- `reference_sql` is the **ideal, verified** query — its result rows are the answer key.
- `expected`: `"answer"` (graded by comparing result rows), `"reject"` (should be
  refused as out-of-scope), or `"clarify"` (vague — should signal uncertainty).

Then:

```bash
python eval/run_eval.py                  # grade with the default model
python eval/run_eval.py qwen2.5-coder:7b # or pick a model (for a model bake-off)
```

It prints a scorecard: **execution accuracy** (result rows match the answer key),
valid-SQL rate, retrieval hit, per-category accuracy, **failures by stage**, plus
tokens and latency — and saves `eval/results_<model>.json`.

## File map

```
src/
  config.py                 all paths, models, connection settings (edit per DB)
  sql_parser.py             dump-parsing logic
  stage1_build_schema.py    dump        -> schema.json   (the TRUTH)
  validate_schema.py        structural health check
  stage2_describe.py        Ollama fills descriptions (guardrailed, incremental)
  stage2a_build_graph.py    schema.json -> graph.json    (+ FK inference, tagged)
  stage4a_build_semantic.py schema.json -> semantic.json (keywords/aliases, recall)
  stage5a_pinecone_upsert.py embed + upsert (meaning)
  stage5b_neo4j_load.py     load graph (structure)
  ask.py                    the ASK phase: question -> SQL -> answer
  healthcheck.py            PASS/FAIL per component
eval/
  run_eval.py               grade the pipeline against your eval/gold.jsonl
```

## Status
- **v1 (this branch):** hand-built pipeline, working end-to-end — simple, aggregate,
  and multi-table-join questions; rejects out-of-scope; SELECT-only; validated SQL.
- **v2 (`v2-langgraph` branch):** the same pipeline rebuilt on LangGraph for
  production-grade orchestration (retries, checkpointing, execution traces) — in progress.
- Local 8B models rely on the validation layer to catch multi-join mistakes.
