"""
stage5b_neo4j_load.py  —  Stage 5b: graph.json -> Neo4j (the graph DB)

WHAT
    Load the mapping-only graph.json into Neo4j as a real property graph so we can
    WALK relationships fast (join paths, bridge tables, backtracking).

DERIVING the typed graph from the minimal file
    graph.json stores only { id, properties:{name} } nodes and { source, target }
    edges. We re-derive the rest at load time from the id PREFIX:
        id 't...'      -> :Table label
        id 'c...'      -> :Column label
        edge t -> c    -> [:HAS_COLUMN]   (ownership)
        edge c -> c    -> [:REFERENCES]   (foreign key)
    So Neo4j gets proper labels and typed relationships, while the file on disk
    stays minimal (store once, derive the rest).

WIPE-THEN-LOAD (why a plain MERGE isn't enough)
    Surrogate ids (t1/c1) shift when the schema changes, so an old MERGE would
    leave orphan nodes behind. We DETACH DELETE everything first => the graph in
    Neo4j always matches graph.json exactly. Truly re-runnable.

CONCEPTS
    * MERGE      = Neo4j's upsert (create if absent, match if present).
    * constraint = "every id is unique" — speeds matching, blocks duplicates.

Run (needs Neo4j running + NEO4J_PASSWORD in .env):
    python src/stage5b_neo4j_load.py
"""

import json
import sys

import config

try:
    from neo4j import GraphDatabase
except ModuleNotFoundError:
    sys.exit("ERROR: the 'neo4j' package is not installed. "
             "Run: pip install -r requirements.txt")


def _label(node_id):
    """Node kind from the id prefix: 't' -> Table, 'c' -> Column."""
    return "Table" if node_id.startswith("t") else "Column"


def load():
    if not config.NEO4J_PASSWORD:
        sys.exit("ERROR: NEO4J_PASSWORD is not set. Copy .env.example to .env "
                 "and fill it in.")

    graph = json.loads(config.GRAPH_JSON.read_text(encoding="utf-8"))
    nodes, edges = graph["nodes"], graph["edges"]
    name_of = {n["id"]: n["properties"]["name"] for n in nodes}

    driver = GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))

    with driver.session() as session:
        # 0. wipe — clean slate so the graph matches graph.json exactly
        print("Wiping existing graph (DETACH DELETE) ...")
        session.run("MATCH (n) DETACH DELETE n")

        # 1. constraints — every Table/Column id is unique
        session.run("CREATE CONSTRAINT IF NOT EXISTS "
                    "FOR (t:Table) REQUIRE t.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS "
                    "FOR (c:Column) REQUIRE c.id IS UNIQUE")

        # 2. nodes — MERGE by id, set the name (label derived from prefix)
        tables = [{"id": n["id"], "name": name_of[n["id"]]}
                  for n in nodes if _label(n["id"]) == "Table"]
        columns = [{"id": n["id"], "name": name_of[n["id"]]}
                   for n in nodes if _label(n["id"]) == "Column"]
        session.run("UNWIND $rows AS r MERGE (t:Table {id:r.id}) SET t.name=r.name",
                    rows=tables)
        session.run("UNWIND $rows AS r MERGE (c:Column {id:r.id}) SET c.name=r.name",
                    rows=columns)

        # 3. edges — kind derived from the SOURCE prefix. REFERENCES edges also
        #    carry fk_type (declared / inferred_rule / inferred_naming) as a
        #    property, so a later query can join using ONLY declared FKs if it
        #    wants to be strict, or include the inferred ones for full coverage.
        own = [{"s": e["source"], "t": e["target"]}
               for e in edges if e["source"].startswith("t")]   # HAS_COLUMN
        fks = [{"s": e["source"], "t": e["target"],
                "fk": e.get("fk_type", "declared")}
               for e in edges if e["source"].startswith("c")]   # REFERENCES
        session.run(
            "UNWIND $rows AS r MATCH (a {id:r.s}),(b {id:r.t}) "
            "MERGE (a)-[:HAS_COLUMN]->(b)", rows=own)
        session.run(
            "UNWIND $rows AS r MATCH (a {id:r.s}),(b {id:r.t}) "
            "MERGE (a)-[rel:REFERENCES]->(b) SET rel.fk_type = r.fk", rows=fks)

        counts = session.run(
            "MATCH (t:Table) WITH count(t) AS nt "
            "MATCH (c:Column) WITH nt, count(c) AS nc "
            "RETURN nt, nc").single()

    driver.close()
    print(f"Loaded into Neo4j: {counts['nt']} Table + {counts['nc']} Column nodes, "
          f"{len(own)} HAS_COLUMN + {len(fks)} REFERENCES edges.")


def main():
    load()


if __name__ == "__main__":
    sys.exit(main())
