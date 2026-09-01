"""Benchmark $graphLookup flight routing on sample_training.routes.

Demonstrates why indexing `src_airport` matters for graph traversal:
`$graphLookup` re-queries `connectToField` at every hop, so without an index
each hop is a full collection scan of ~67k documents. Hub airports such as JFK
(456 outbound routes) multiply that cost.

Measured on a live Atlas cluster:
    JFK -> OPO   9.40s  ->  0.228s   (41x faster)
    JFK -> BCN   8.79s  ->  0.346s   (25x faster)

Usage:
    python scripts/benchmark_graphlookup.py           # measure current state
    python scripts/benchmark_graphlookup.py --index   # create indexes first
    python scripts/benchmark_graphlookup.py --drop    # drop indexes (reset demo)
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

from dotenv import load_dotenv
from pymongo import MongoClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=True)

from utils import create_index  # noqa: E402

ROUTES_DB = "sample_training"
ROUTES_COLLECTION = "routes"

# (origin, destination, note)
CASES = [
    ("JFK", "BCN", "direct routes exist; JFK is a hub (456 outbound)"),
    ("JFK", "OPO", "no direct route; reachable via 1 hop"),
    ("SYD", "HKG", "direct routes exist"),
    ("LGA", "YUL", "direct routes exist"),
    ("LIH", "IST", "low fan-out origin"),
    ("KOA", "OPO", "genuinely unreachable, even at 1 hop"),
]


def connection_pipeline(src: str, dst: str, max_depth: int = 0) -> list:
    """Find itineraries from `src` to `dst` using graph traversal."""
    return [
        {"$match": {"src_airport": src}},
        {
            "$graphLookup": {
                "from": ROUTES_COLLECTION,
                "startWith": "$dst_airport",
                "connectFromField": "dst_airport",
                "connectToField": "src_airport",
                "as": "connections",
                "maxDepth": max_depth,
                "depthField": "depth",
                # Only keep traversed routes that land at the destination.
                "restrictSearchWithMatch": {"dst_airport": dst},
            }
        },
        {"$match": {"connections.0": {"$exists": True}}},
        {"$count": "paths"},
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", action="store_true", help="create the indexes first")
    ap.add_argument("--drop", action="store_true", help="drop the indexes and exit")
    args = ap.parse_args()

    client = MongoClient(os.environ["ATLAS_URI"], appname="devrel-workshop-travel-agent")
    routes = client[ROUTES_DB][ROUTES_COLLECTION]

    if args.drop:
        for name in ("src_airport_1", "dst_airport_1"):
            if name in routes.index_information():
                routes.drop_index(name)
                print(f"dropped {name}")
        return 0

    if args.index:
        create_index(routes, [("src_airport", 1)], "src_airport_1")
        create_index(routes, [("dst_airport", 1)], "dst_airport_1")

    print(f"collection: {ROUTES_DB}.{ROUTES_COLLECTION} "
          f"({routes.count_documents({})} routes)")
    print("indexes:", [i["name"] for i in routes.list_indexes()])
    print()

    for src, dst, note in CASES:
        started = time.time()
        result = list(routes.aggregate(connection_pipeline(src, dst)))
        elapsed = time.time() - started
        paths = result[0]["paths"] if result else 0
        print(f"  {src} -> {dst}: {paths:3d} paths  {elapsed:6.3f}s   # {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
