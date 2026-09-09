"""
Central configuration for the Text-to-SQL pipeline.

Every path and tunable knob lives here so that:
  - swapping in a different SQL dump means changing ONE line, not hunting
    through every script (this is what makes the pipeline "generic");
  - the same constants are shared by every stage (parser, describer, indexer)
    instead of being re-declared and drifting out of sync.
"""

import os
from pathlib import Path

# Load secrets from a gitignored .env (PINECONE_API_KEY, NEO4J_PASSWORD, ...).
# Wrapped in try/except so Stage 1 — which needs no secrets and no extra
# packages — still runs on a bare Python install without python-dotenv.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ModuleNotFoundError:
    pass

# ---------------------------------------------------------------------------
# Folder layout
# ---------------------------------------------------------------------------
# BASE_DIR = the project root (ai_project2/). We compute it relative to THIS
# file so the scripts work no matter what directory you run them from.
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"        # input: the downloaded SQL dump
OUTPUT_DIR = BASE_DIR / "data" / "output"  # output: generated artifacts

# ---------------------------------------------------------------------------
# Input dump files  (Stage 1 reads these)
# ---------------------------------------------------------------------------
# A dump can arrive in TWO shapes, and we support both with the same two names:
#
#   1. SPLIT dump (e.g. Sakila): structure and content in separate files.
#        SCHEMA_SQL = RAW_DIR / "sakila-schema.sql"
#        DATA_SQL   = RAW_DIR / "sakila-data.sql"
#
#   2. COMBINED dump (e.g. a normal `mysqldump` output): CREATE TABLE and
#      INSERT INTO live in ONE file. Point BOTH names at that same file —
#      Stage 1 detects the duplicate and reads it only once.
#        SCHEMA_SQL = DATA_SQL = RAW_DIR / "your_dump.sql"
#
# To test a different database, change the path(s) here and re-run Stage 1.
SCHEMA_SQL = RAW_DIR / "your_dump.sql"
DATA_SQL = RAW_DIR / "your_dump.sql"

# ---------------------------------------------------------------------------
# Output file  (Stage 1 writes this; Stages 2-4 read/update it)
# ---------------------------------------------------------------------------
# This single file is the source of truth for the whole system.
SCHEMA_JSON = OUTPUT_DIR / "schema.json"

# Derived artifacts. graph.json is built by Stage 2a; semantic.json by Stage 4a.
# Centralized here so no script hardcodes a path (the "one place to edit" rule).
GRAPH_JSON = OUTPUT_DIR / "graph.json"
SEMANTIC_JSON = OUTPUT_DIR / "semantic.json"

# Hand-authored metric catalog (Tier 0 semantic layer). NOT generated — you edit
# this by hand to teach the compiler your business metrics. See src/metrics_layer.py.
# (Unrelated to SEMANTIC_JSON above, which is retrieval embeddings.)
METRICS_YML = BASE_DIR / "metrics.yml"

# Tables to EXCLUDE from the Pinecone retrieval index (Stage 4a). These are
# migration/bookkeeping tables (Liquibase) — no business question is ever about
# them, so indexing them only adds retrieval noise. They stay in schema.json and
# graph.json (the truth); we just don't embed them for semantic search.
SKIP_TABLES = {"databasechangelog", "databasechangeloglock"}

# ---------------------------------------------------------------------------
# Stage 2 — descriptions via local Ollama
# ---------------------------------------------------------------------------
# DESC model writes the plain-English descriptions; EMBED model turns text into
# the 768-number vectors. They are DIFFERENT models with different jobs.
OLLAMA_DESC_MODEL = "llama3.1:8b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_SQL_MODEL = "qwen2.5-coder:7b"   # ASK phase: code-tuned model writes the SQL
EMBED_DIM = 768                # nomic-embed-text output size; MUST match the index

# ---------------------------------------------------------------------------
# Stage 5a — Pinecone (cloud vector DB). Secrets come from the environment,
# never hardcoded, so the API key never lands in git. See .env.example.
# ---------------------------------------------------------------------------
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "text2sql-rag")
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
PINECONE_METRIC = "cosine"     # compare meaning by direction, ignoring length

# ---------------------------------------------------------------------------
# Stage 5b — Neo4j (graph DB). Password from the environment, never hardcoded.
# ---------------------------------------------------------------------------
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

# ---------------------------------------------------------------------------
# Phase D — live MySQL (the ASK phase executes validated SELECTs here). Defaults
# match the local Docker container: docker run --name textsql-mysql
#   -e MYSQL_ROOT_PASSWORD=textsqlpass -p 3306:3306 -d mysql:8
# ---------------------------------------------------------------------------
MYSQL_HOST = os.environ.get("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "textsqlpass")
MYSQL_DB = os.environ.get("MYSQL_DB", "mydb")

# Answer composition model (turns result rows into an English sentence)
OLLAMA_ANSWER_MODEL = "llama3.1:8b"

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
# How many example rows to pull per table. 3 is enough to hint the LLM at what
# a column actually stores ('active' vs 1); more would bloat the JSON.
SAMPLE_ROWS = 3

# Cap on how long a single sample value can be before we truncate it. Some
# columns (e.g. film.description) hold paragraphs; we only need a hint.
MAX_VALUE_LEN = 60

# ---------------------------------------------------------------------------
# M5 semantic cache tunables
# ---------------------------------------------------------------------------
# The cache stores validated SQL (not answer text) and RE-EXECUTES on every hit
# so data is always fresh. TTL controls how long we trust the cached *SQL*
# before regenerating from scratch. See docs/v2-hardening-plan.md Fix #1.
CACHE_THRESHOLD = 0.96          # cosine sim to count as the "same" question
CACHE_HISTORY_TURNS = 2         # prior turns folded into the cache key
CACHE_MAX_ENTRIES = 500         # FIFO cap; cache_lookup is a linear scan

# TTLs (seconds) — how long we trust the cached SQL, not the data (always fresh).
CACHE_TTL_VOLATILE = 600        # 10 min  — churny tables (logs, transactions, etc.)
CACHE_TTL_DEFAULT  = 3600       # 1 hour  — aggregates over non-stable tables
CACHE_TTL_STABLE   = 86400      # 24 hour — pure lookups on catalog/config tables

# Substring match on table name → volatile (short SQL-trust window).
CACHE_VOLATILE_PATTERNS = {
    "log", "transaction", "execution", "session", "event", "wallet",
    "checkpoint", "cache", "audit", "message", "seq", "state", "lock",
    "grant", "invoice", "redemption", "conversion", "balance",
}

# Override: catalog/config tables are stable even if a volatile pattern matched.
# These are illustrative names — edit them to match your own schema's lookup tables.
CACHE_STABLE_TABLES = {
    "plans", "plan_configs", "products", "product_categories",
    "config_settings", "pricing_tiers", "countries", "currencies",
    "roles", "permissions", "feature_flags",
}

# ---------------------------------------------------------------------------
# Fix #5 — EXPLAIN cost-gate
# ---------------------------------------------------------------------------
# Estimated rows examined beyond this threshold → refuse execution.
# Heuristic guard (EXPLAIN estimates lie with stale stats); complemented by
# MAX_EXECUTION_TIME runtime cap as defense-in-depth.
EXPLAIN_MAX_ROWS = 1_000_000
QUERY_TIMEOUT_MS = 10_000                   # runtime cap for SELECT queries

