from pymongo.errors import OperationFailure
from pymongo.collection import Collection
from langchain_aws import ChatBedrock
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
from typing import Dict, List
import time
import os

SLEEP_TIMER = 5
PROXY_ENDPOINT = "https://vtqjvgchmwcjwsrela2oyhlegu0hwqnw.lambda-url.us-west-2.on.aws/"
SANDBOX_NAME = os.getenv("CODESPACE_NAME") or os.getenv("_SANDBOX_ID")


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
    payload = {"task": task, "workshop_id": workshop_id, "sandbox_id": SANDBOX_NAME}
    requests.post(url=PROXY_ENDPOINT, json={"task": "track_progress", "data": payload})


def set_env(providers: List[str], passkey: str) -> None:
    """
    Set environment variables in sandbox

    Args:
        providers (List[str]): List of provider names
        passkey (str): Passkey to get token
    """
    for provider in providers:
        payload = {"provider": provider, "passkey": passkey}
        response = requests.post(url=PROXY_ENDPOINT, json={"task": "get_token", "data": payload})
        status_code = response.status_code
        if status_code == 200:
            result = response.json().get("token")
            for key in result:
                os.environ[key] = result[key]
                print(f"Successfully set {key} environment variable.")
        elif status_code == 401:
            raise Exception(f"{response.json()['error']} Follow steps in the lab documentation to obtain your own credentials and set them as environment variables.")
        else:
            raise Exception(f"{response.json()['error']}")


def get_llm(provider: str):
    if provider == "aws":
        return ChatBedrock(
            model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            model_kwargs=dict(temperature=0),
            region_name="us-west-2",
        )
    elif provider == "google":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0,
        )
    elif provider == "microsoft":
        return AzureChatOpenAI(
            azure_endpoint="https://gai-326.openai.azure.com/",
            azure_deployment="gpt-4o",
            api_version="2023-06-01-preview",
            temperature=0,
        )
    else:
        raise Exception("Unsupported provider. provider can be one of 'aws', 'google', 'microsoft'.")
