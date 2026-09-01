"""Final verification of the seeded travel-agent data."""

import os
import pathlib

from dotenv import load_dotenv
from pymongo import MongoClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

client = MongoClient(os.environ["ATLAS_URI"], appname="devrel-workshop-travel-agent")
db = client["mongodb_genai_devday_travel_agent"]

print("DB:", db.name)
for name in sorted(db.list_collection_names()):
    print(f"  {name}: {db[name].count_documents({})} docs")

vs = db["airbnb_listings_embeddings"]
print("\nsearch indexes:")
for i in vs.list_search_indexes():
    print(f"  {i['name']} / {i['type']} / {i['status']}")

print("\nintegrity:")
print("  docs with 1024-dim embedding:", vs.count_documents({"embedding.1023": {"$exists": True}}))
print("  docs with 1025th element (should be 0):", vs.count_documents({"embedding.1024": {"$exists": True}}))
print("  docs with non-empty text:", vs.count_documents({"text": {"$exists": True, "$ne": ""}}))
print("  docs with stale embedding field leaked:", vs.count_documents({"text": {"$exists": False}}))

src = client["sample_airbnb"]["listingsAndReviews"]
print("\nshared sample data untouched:")
print("  sample_airbnb.listingsAndReviews:", src.count_documents({}), "docs")
print("  still has original embedding field:", src.count_documents({"embedding": {"$exists": True}}), "docs")

markets = sorted({d["_id"] for d in vs.aggregate([{"$group": {"_id": "$address.market"}}]) if d["_id"]})
print("\nmarkets:", markets)
