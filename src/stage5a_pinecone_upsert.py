"""
stage5a_pinecone_upsert.py  —  Stage 5a: embed semantic.json -> upsert to Pinecone

WHAT
    For every table document in semantic.json:
      1. embed its "embedding_text" locally with Ollama nomic-embed-text -> 768 numbers
      2. upsert {id, vector, metadata{table, description}} into a Pinecone index.

CONCEPTS (quick)
    * vector DB = stores number-lists and is fast at "given a new vector, find the
      closest stored ones" — exactly "find tables by meaning".
    * upsert    = update-if-exists-else-insert => safe to re-run (no duplicates).
    * dimension = how many numbers per vector (768). The index dimension MUST equal
      the model's output size or upsert fails.
    * metric    = cosine: compares meaning by DIRECTION, ignoring text length.
    * metadata  = small info stored beside the vector (table name + description) so a
      search hit tells us WHICH table it is.

Run (needs PINECONE_API_KEY in .env, Ollama running):
    python src/stage5a_pinecone_upsert.py
    python src/stage5a_pinecone_upsert.py --query "orders by customer"
"""

import json
import sys

import config

try:
    import ollama
    from pinecone import Pinecone, ServerlessSpec
except ModuleNotFoundError as exc:
    sys.exit(f"ERROR: missing package ({exc.name}). Run: pip install -r requirements.txt")


def _embed(text):
    """Local Ollama embedding -> list of 768 floats."""
    return ollama.embeddings(model=config.OLLAMA_EMBED_MODEL, prompt=text)["embedding"]


def _connect_index():
    """Return a Pinecone index handle, creating the index if it doesn't exist."""
    if not config.PINECONE_API_KEY:
        sys.exit("ERROR: PINECONE_API_KEY is not set. Copy .env.example to .env "
                 "and fill it in.")
    pc = Pinecone(api_key=config.PINECONE_API_KEY)

    # --- Index bootstrap: create with the RIGHT shape if missing -------------
    existing = [ix["name"] for ix in pc.list_indexes()]
    if config.PINECONE_INDEX not in existing:
        print(f"Creating index '{config.PINECONE_INDEX}' "
              f"(dim={config.EMBED_DIM}, metric={config.PINECONE_METRIC}) ...")
        pc.create_index(
            name=config.PINECONE_INDEX,
            dimension=config.EMBED_DIM,
            metric=config.PINECONE_METRIC,
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD,
                                region=config.PINECONE_REGION),
        )
    return pc.Index(config.PINECONE_INDEX)


def upsert(batch_size=50):
    docs = json.loads(config.SEMANTIC_JSON.read_text(encoding="utf-8"))
    index = _connect_index()

    vectors = []
    for i, doc in enumerate(docs, 1):
        vec = _embed(doc["embedding_text"])
        if len(vec) != config.EMBED_DIM:
            sys.exit(f"ERROR: model returned {len(vec)} numbers but the index "
                     f"expects {config.EMBED_DIM}. Check OLLAMA_EMBED_MODEL/EMBED_DIM.")
        # metadata lets a search hit map back to its source. A COLUMN hit also
        # carries its column name; both carry the table so retrieval -> table.
        meta = {"kind": doc["kind"], "table": doc["table"],
                "description": doc["description"]}
        if doc["kind"] == "column":
            meta["column"] = doc["column"]
        vectors.append({"id": doc["id"], "values": vec, "metadata": meta})
        label = doc["table"] if doc["kind"] == "table" \
            else f"{doc['table']}.{doc['column']}"
        print(f"  embedded [{i}/{len(docs)}] ({doc['kind']}) {label}")

        if len(vectors) >= batch_size:
            index.upsert(vectors=vectors)
            vectors = []
    if vectors:
        index.upsert(vectors=vectors)

    n_tab = sum(1 for d in docs if d["kind"] == "table")
    n_col = sum(1 for d in docs if d["kind"] == "column")
    print(f"\nUpserted {n_tab} table + {n_col} column vectors into "
          f"'{config.PINECONE_INDEX}'.")
    return index


def query(index, text, top_k=5):
    """Smoke test: embed a question, print the nearest tables by meaning."""
    vec = _embed(text)
    res = index.query(vector=vec, top_k=top_k, include_metadata=True)
    print(f"\nTop {top_k} matches for: {text!r}")
    for m in res["matches"]:
        md = m["metadata"]
        label = md["table"] if md.get("kind") == "table" \
            else f"{md['table']}.{md.get('column', '?')}"
        print(f"  {m['score']:.3f}  [{md.get('kind','?')}] {label}")


def main():
    # allow:  --query "..."  to also run a search after upserting
    q = None
    if "--query" in sys.argv:
        q = sys.argv[sys.argv.index("--query") + 1]
    index = upsert()
    if q:
        query(index, q)


if __name__ == "__main__":
    sys.exit(main())
