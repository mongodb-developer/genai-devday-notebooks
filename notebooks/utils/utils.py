from pymongo.errors import OperationFailure
from pymongo.collection import Collection
import requests
from typing import Dict
import time
import os

SLEEP_TIMER = 5
SERVERLESS_URL = os.getenv("SERVERLESS_URL")
CODESPACE_NAME = os.getenv("CODESPACE_NAME")


def create_index(collection: Collection, index_name: str, model: Dict) -> None:
    """
    Create a search index

    Args:
        collection (Collection): Collection to create search index against
        index_name (str): Index name
        model (Dict): Index definition
    """
    try:
        print(f"Creating the {index_name} index")
        collection.create_search_index(model=model)
    except OperationFailure:
        print(f"{index_name} index already exists, recreating...")
        try:
            print(f"Dropping {index_name} index")
            collection.drop_search_index(name=index_name)

            # Poll for index deletion to complete
            while True:
                indexes = list(collection.list_search_indexes())
                index_exists = any(idx.get("name") == index_name for idx in indexes)
                if not index_exists:
                    print(f"{index_name} index deletion complete")
                    break
                print(f"Waiting for {index_name} index deletion to complete...")
                time.sleep(SLEEP_TIMER)

            print(f"Creating new {index_name} index")
            collection.create_search_index(model=model)
            print(f"Successfully recreated the {index_name} index")
        except Exception as e:
            raise Exception(f"Error during index recreation: {str(e)}")


def check_index_ready(collection: Collection, index_name: str) -> None:
    """
    Poll for index status until it's ready

    Args:
        collection (Collection): Collection to check index status against
        index_name (str): Name of the index to check
    """
    while True:
        indexes = list(collection.list_search_indexes())
        matching_indexes = [idx for idx in indexes if idx.get("name") == index_name]

        if not matching_indexes:
            print(f"{index_name} index not found")
            time.sleep(SLEEP_TIMER)
            continue

        index = matching_indexes[0]
        status = index["status"]
        if status == "READY":
            print(f"{index_name} index status: READY")
            print(f"{index_name} index definition: {index['latestDefinition']}")
            break

        print(f"{index_name} index status: {status}")
        time.sleep(SLEEP_TIMER)


def track_progress(task: str, workshop_id: str) -> None:
    """
    Track progress of a task

    Args:
        task (str): Task name
        workshop (str): Workshop name
    """
    print(f"Tracking progress for task {task}")
    payload = {"task": task, "workshop_id": workshop_id, "sandbox_id": CODESPACE_NAME}
    requests.post(url=SERVERLESS_URL, json={"task": "track_progress", "data": payload})
