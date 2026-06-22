"""
healthcheck.py  —  verify every component of the pipeline is present and connected.

Run:  python src/healthcheck.py

Checks: artifact files, Python deps, Ollama models, Pinecone index + vectors,
Neo4j nodes/edges, MySQL container + tables. Prints a PASS/FAIL line per component
so you can see at a glance what is up before asking a question (or before a demo).
"""

import json
import sys

import config

OK, BAD = "[ OK ]", "[FAIL]"


def line(ok, label, detail=""):
    print(f"{OK if ok else BAD} {label:28} {detail}")
    return ok


def main():
    results = []

    # 1. artifact files
    for name, path, key in [
        ("schema.json", config.SCHEMA_JSON, "tables"),
        ("graph.json", config.GRAPH_JSON, "nodes"),
        ("semantic.json", config.SEMANTIC_JSON, None),
    ]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            n = len(data[key]) if key else len(data)
            results.append(line(True, name, f"{n} {key or 'docs'}"))
        except Exception as exc:
            results.append(line(False, name, str(exc)))

    # 2. python deps
    for mod in ["ollama", "pinecone", "neo4j", "sqlglot", "pymysql", "dotenv"]:
        try:
            __import__(mod)
            results.append(line(True, f"dep: {mod}"))
        except Exception as exc:
            results.append(line(False, f"dep: {mod}", str(exc)))

    # 3. Ollama models
    try:
        import ollama
        have = {m.get("model", m.get("name", "")) for m in ollama.list()["models"]}
        for want in [config.OLLAMA_DESC_MODEL, config.OLLAMA_EMBED_MODEL,
                     config.OLLAMA_SQL_MODEL]:
            present = any(h == want or h.startswith(want.split(":")[0]) for h in have)
            results.append(line(present, f"ollama: {want}"))
    except Exception as exc:
        results.append(line(False, "ollama server", str(exc)))

    # 4. Pinecone
    try:
        from pinecone import Pinecone
        idx = Pinecone(api_key=config.PINECONE_API_KEY).Index(config.PINECONE_INDEX)
        stats = idx.describe_index_stats()
        cnt = stats.get("total_vector_count", "?")
        dim = stats.get("dimension", "?")
        results.append(line(dim == config.EMBED_DIM,
                            "pinecone index", f"{cnt} vectors, dim {dim}"))
    except Exception as exc:
        results.append(line(False, "pinecone", str(exc)))

    # 5. Neo4j
    try:
        from neo4j import GraphDatabase
        drv = GraphDatabase.driver(config.NEO4J_URI,
                                   auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
        with drv.session() as s:
            n = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            e = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        drv.close()
        results.append(line(n > 0, "neo4j aura", f"{n} nodes, {e} edges"))
    except Exception as exc:
        results.append(line(False, "neo4j aura", str(exc).split(chr(10))[0]))

    # 6. MySQL
    try:
        import pymysql
        conn = pymysql.connect(host=config.MYSQL_HOST, port=config.MYSQL_PORT,
                               user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
                               database=config.MYSQL_DB, connect_timeout=5)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM information_schema.tables "
                        "WHERE table_schema=%s", (config.MYSQL_DB,))
            t = cur.fetchone()[0]
        conn.close()
        results.append(line(t > 0, "mysql (docker)", f"{t} tables in {config.MYSQL_DB}"))
    except Exception as exc:
        results.append(line(False, "mysql (docker)", str(exc).split(chr(10))[0]))

    passed = sum(1 for r in results if r)
    print("-" * 60)
    print(f"{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
