"""Verify retrieval quality and 4-series cross-model compatibility.

The corpus was embedded once with voyage-4-large. This script queries that
single index with three different query models and reports rank agreement,
proving 4-series embeddings share a vector space.
"""

import os
import pathlib
import time

from dotenv import load_dotenv
from pymongo import MongoClient

import voyageai

ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

DB_NAME = "mongodb_genai_devday_travel_agent"
VS_COLLECTION_NAME = "airbnb_listings_embeddings"
VS_INDEX_NAME = "vector_index"

QUERY_MODELS = ["voyage-4-large", "voyage-4", "voyage-4-lite"]
QUERIES = [
    "beachfront apartment in Barcelona for 4 people",
    "quiet place with character near the old town, good for writing",
    "modern high-rise studio in Hong Kong with skyline views",
    "family friendly home in Sydney with a kitchen and free parking",
]

client = MongoClient(os.environ["ATLAS_URI"], appname="devrel-workshop-travel-agent")
vs_collection = client[DB_NAME][VS_COLLECTION_NAME]
vo = voyageai.Client()

print(f"corpus: {vs_collection.count_documents({})} listings, "
      f"embedded once with voyage-4-large\n")


def search(query: str, model: str, limit: int = 5):
    t0 = time.time()
    qv = vo.embed([query], model=model, input_type="query").embeddings[0]
    embed_ms = (time.time() - t0) * 1000
    pipeline = [
        {
            "$vectorSearch": {
                "index": VS_INDEX_NAME,
                "path": "embedding",
                "queryVector": qv,
                "numCandidates": 150,
                "limit": limit,
            }
        },
        {
            "$project": {
                "_id": 0,
                "name": 1,
                "market": "$address.market",
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(vs_collection.aggregate(pipeline)), embed_ms


for query in QUERIES:
    print("=" * 78)
    print(f"QUERY: {query}")
    print("=" * 78)
    baseline = None
    for model in QUERY_MODELS:
        hits, embed_ms = search(query, model)
        names = [h.get("name") for h in hits]
        if baseline is None:
            baseline = names
            overlap = "baseline"
        else:
            common = len(set(names) & set(baseline))
            overlap = f"{common}/5 overlap vs large"
        print(f"\n  {model:16s} embed={embed_ms:6.0f}ms   {overlap}")
        for i, h in enumerate(hits, 1):
            print(f"    {i}. [{h['score']:.4f}] {str(h.get('name'))[:52]:54s} "
                  f"({h.get('market')})")
    print()
