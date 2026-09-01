"""Seed Voyage AI embeddings for the Airbnb travel-agent lab.

Reads the 5,555 listings from `sample_airbnb.listingsAndReviews` on the target
cluster, strips any pre-existing (stale) `embedding` field, composes a
retrieval-friendly `text` blob per listing, embeds it with `voyage-4-large`
(`input_type="document"`), and writes the result into NEW collections in the
`mongodb_genai_devday_travel_agent` database.

Shared sample data is never mutated: we read from `sample_airbnb` and write
only to our own database.

Embeddings are checkpointed to `.cache/embeddings.jsonl` after every batch, so
an interrupted run resumes instead of re-paying for ~2.8M tokens.

Usage:
    python scripts/seed_airbnb_voyage.py               # embed + seed
    python scripts/seed_airbnb_voyage.py --limit 50    # smoke test
    python scripts/seed_airbnb_voyage.py --skip-embed  # re-seed from checkpoint
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from decimal import Decimal
from typing import Any, Dict, Iterable, List

from bson import Decimal128
from dotenv import load_dotenv
from pymongo import MongoClient
from tenacity import retry, stop_after_attempt, wait_random_exponential

import voyageai

ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

# ----- Configuration -----
SRC_DB = "sample_airbnb"
SRC_COLLECTION = "listingsAndReviews"

DB_NAME = "mongodb_genai_devday_travel_agent"
FULL_COLLECTION_NAME = "airbnb_listings"
VS_COLLECTION_NAME = "airbnb_listings_embeddings"

# Ingest uses the quality model; queries use voyage-4 at runtime (4-series
# embeddings are mutually compatible, so the corpus is embedded only once).
DOC_MODEL = "voyage-4-large"
DIMS = 1024

# voyage-4-large allows <=1000 inputs and <=120K tokens per request.
# ~500 tokens/listing * 100 = ~50K tokens per request: comfortably under.
BATCH_SIZE = 100
MAX_TEXT_CHARS = 4000

CHECKPOINT = ROOT / ".cache" / "embeddings.jsonl"

# Fields dropped from the embeddings collection to keep it lean.
DROP_FROM_VS = ("reviews", "embedding")


def _clean(value: Any) -> Any:
    """Convert BSON types that don't round-trip through JSON."""
    if isinstance(value, Decimal128):
        return float(value.to_decimal())
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def compose_text(doc: Dict[str, Any]) -> str:
    """Build the string that gets embedded and returned by the retrieval tool."""
    address = doc.get("address") or {}
    market = address.get("market") or ""
    country = address.get("country") or ""
    location = ", ".join(p for p in (market, country) if p)

    price = doc.get("price")
    if isinstance(price, Decimal128):
        price = float(price.to_decimal())

    header_bits = [
        f"{doc.get('name') or 'Unnamed listing'}",
        f"{doc.get('property_type') or ''} / {doc.get('room_type') or ''}".strip(" /"),
    ]
    if location:
        header_bits.append(f"in {location}")
    header = " — ".join(b for b in header_bits if b)

    facts = []
    if doc.get("accommodates") is not None:
        facts.append(f"accommodates {doc['accommodates']}")
    for field, label in (
        ("bedrooms", "bedrooms"),
        ("beds", "beds"),
        ("bathrooms", "bathrooms"),
    ):
        val = doc.get(field)
        if isinstance(val, Decimal128):
            val = float(val.to_decimal())
        if val is not None:
            facts.append(f"{val:g} {label}")
    if price is not None:
        facts.append(f"${price:g} per night")

    scores = doc.get("review_scores") or {}
    rating = scores.get("review_scores_rating")
    if rating is not None:
        facts.append(f"guest rating {rating}/100")

    prose = " ".join(
        str(doc.get(f) or "").strip()
        for f in ("summary", "space", "neighborhood_overview", "transit")
    ).strip()

    amenities = doc.get("amenities") or []
    amenities_str = ", ".join(amenities[:25])

    parts = [header]
    if facts:
        parts.append(", ".join(facts) + ".")
    if prose:
        parts.append(prose)
    if amenities_str:
        parts.append(f"Amenities: {amenities_str}.")

    text = " ".join(p for p in parts if p)
    text = " ".join(text.split())  # collapse whitespace
    return text[:MAX_TEXT_CHARS]


@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(6))
def embed_batch(vo: voyageai.Client, texts: List[str]) -> List[List[float]]:
    return vo.embed(texts, model=DOC_MODEL, input_type="document").embeddings


def load_checkpoint() -> Dict[str, List[float]]:
    if not CHECKPOINT.exists():
        return {}
    done: Dict[str, List[float]] = {}
    with CHECKPOINT.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn final line
            done[rec["_id"]] = rec["embedding"]
    return done


def chunks(seq: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only process N listings")
    ap.add_argument("--skip-embed", action="store_true", help="seed from checkpoint only")
    args = ap.parse_args()

    client = MongoClient(os.environ["ATLAS_URI"], appname="devrel-workshop-travel-agent")
    src = client[SRC_DB][SRC_COLLECTION]

    # Drop the stale/partial embedding outright; we plant a fresh one.
    cursor = src.find({}, {"embedding": 0})
    if args.limit:
        cursor = cursor.limit(args.limit)

    print(f"Reading listings from {SRC_DB}.{SRC_COLLECTION} ...")
    docs: List[Dict[str, Any]] = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        doc["text"] = compose_text(doc)
        docs.append(doc)
    print(f"  loaded {len(docs)} listings")

    lens = sorted(len(d["text"]) for d in docs)
    print(
        f"  text chars: min={lens[0]} median={lens[len(lens) // 2]} max={lens[-1]}"
    )

    done = load_checkpoint()
    if done:
        print(f"  checkpoint: {len(done)} embeddings already computed")

    if not args.skip_embed:
        pending = [d for d in docs if d["_id"] not in done]
        print(f"\nEmbedding {len(pending)} listings with {DOC_MODEL} ...")
        vo = voyageai.Client()
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

        total_tokens = 0
        started = time.time()
        with CHECKPOINT.open("a") as fh:
            for i, batch in enumerate(chunks(pending, BATCH_SIZE), start=1):
                texts = [d["text"] for d in batch]
                embeddings = embed_batch(vo, texts)
                for d, emb in zip(batch, embeddings):
                    if len(emb) != DIMS:
                        raise SystemExit(f"unexpected dims {len(emb)} for {d['_id']}")
                    done[d["_id"]] = emb
                    fh.write(json.dumps({"_id": d["_id"], "embedding": emb}) + "\n")
                fh.flush()
                total_tokens += sum(len(t) // 4 for t in texts)
                n_done = min(i * BATCH_SIZE, len(pending))
                print(
                    f"  batch {i:3d}  {n_done}/{len(pending)}  "
                    f"~{total_tokens:,} tok  {time.time() - started:.0f}s",
                    flush=True,
                )
        print(f"  embedding complete in {time.time() - started:.0f}s")

    missing = [d["_id"] for d in docs if d["_id"] not in done]
    if missing:
        raise SystemExit(f"{len(missing)} listings still lack embeddings; re-run")

    # ----- Write to our own database (shared sample data untouched) -----
    full_collection = client[DB_NAME][FULL_COLLECTION_NAME]
    vs_collection = client[DB_NAME][VS_COLLECTION_NAME]

    print(f"\nSeeding {DB_NAME}.{FULL_COLLECTION_NAME} ...")
    full_collection.delete_many({})
    full_docs = [_clean({k: v for k, v in d.items() if k != "embedding"}) for d in docs]
    for batch in chunks(full_docs, 500):
        full_collection.insert_many(batch)
    print(f"  {full_collection.count_documents({})} documents")

    print(f"Seeding {DB_NAME}.{VS_COLLECTION_NAME} ...")
    vs_collection.delete_many({})
    vs_docs = []
    for d in docs:
        rec = _clean({k: v for k, v in d.items() if k not in DROP_FROM_VS})
        rec["embedding"] = done[d["_id"]]
        vs_docs.append(rec)
    for batch in chunks(vs_docs, 200):
        vs_collection.insert_many(batch)
    print(f"  {vs_collection.count_documents({})} documents")

    sample = vs_collection.find_one({}, {"name": 1, "text": 1, "embedding": 1})
    print(f"\nsample: {sample.get('name')!r}")
    print(f"  dims={len(sample['embedding'])}")
    print(f"  text={sample['text'][:180]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
