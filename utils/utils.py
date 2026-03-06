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


def create_search_index(collection: Collection, index_name: str, model: Dict) -> None:
    """
    Create a search index, dropping and recreating if it already exists.

    Args:
        collection (Collection): Collection to create search index against
        index_name (str): Index name
        model (Dict): Index definition with 'name', 'type', and 'definition' keys
    """
    # Check if index already exists
    indexes = list(collection.list_search_indexes(name=index_name))

    if len(indexes) > 0:
        print(f"{index_name} index exists, dropping...")
        collection.drop_search_index(name=index_name)

        # Wait for deletion to complete
        while True:
            indexes = list(collection.list_search_indexes(name=index_name))
            if len(indexes) == 0:
                print(f"{index_name} index dropped")
                break
            print(f"Waiting for {index_name} index to be dropped...")
            time.sleep(SLEEP_TIMER)

    print(f"Creating {index_name} index...")
    collection.create_search_index(model=model)
    print(f"Successfully created {index_name} index")


def check_index_ready(collection: Collection, index_name: str) -> None:
    """
    Poll for search index status until it's ready

    Args:
        collection (Collection): Collection to check index status against
        index_name (str): Name of the index to check
    """
    while True:
        indexes = list(collection.list_search_indexes(name=index_name))

        if not indexes:
            print(f"{index_name} index not found, waiting...")
            time.sleep(SLEEP_TIMER)
            continue

        status = indexes[0].get("status", "UNKNOWN")
        if status == "READY":
            print(f"{index_name} index is READY")
            return

        print(f"{index_name} index status: {status}")
        time.sleep(SLEEP_TIMER)


def create_index(collection: Collection, keys: List, index_name: str, **kwargs) -> None:
    """
    Create a regular index, dropping and recreating if it already exists.

    Args:
        collection (Collection): Collection to create index against
        keys (List): List of keys for the index
        index_name (str): Index name
        **kwargs: Additional arguments for index creation (e.g., unique, expireAfterSeconds)
    """
    # Check if index already exists
    if index_name in collection.index_information():
        print(f"{index_name} index exists, dropping...")
        collection.drop_index(index_name)
        print(f"{index_name} index dropped")

    print(f"Creating {index_name} index...")
    collection.create_index(keys, name=index_name, **kwargs)
    print(f"Successfully created {index_name} index")


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
            model_id="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
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
            azure_deployment="gpt-4.1",
            api_version="2024-12-01-preview",
            temperature=0,
        )
    else:
        raise Exception("Unsupported provider. provider can be one of 'aws', 'google', 'microsoft'.")
