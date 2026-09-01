"""Create the 1024-dim vector search index for the travel-agent lab."""

import os
import pathlib
import sys

from dotenv import load_dotenv
from pymongo import MongoClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=True)

from utils import check_index_ready, create_search_index  # noqa: E402

DB_NAME = "mongodb_genai_devday_travel_agent"
VS_COLLECTION_NAME = "airbnb_listings_embeddings"
VS_INDEX_NAME = "vector_index"

client = MongoClient(os.environ["ATLAS_URI"], appname="devrel-workshop-travel-agent")
vs_collection = client[DB_NAME][VS_COLLECTION_NAME]

model = {
    "name": VS_INDEX_NAME,
    "type": "vectorSearch",
    "definition": {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 1024,
                "similarity": "cosine",
            }
        ]
    },
}

create_search_index(vs_collection, VS_INDEX_NAME, model)
check_index_ready(vs_collection, VS_INDEX_NAME)
print("done")
