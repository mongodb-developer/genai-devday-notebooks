from langchain_aws import ChatBedrock
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import os

MAX_TOKENS = 4096

def get_llm(provider: str):
    """
    Get the appropriate LLM instance based on the provider.

    Args:
        provider (str): Name of the provider. One of "aws", "google", "microsoft"
    """
    if provider == "aws":
        return ChatBedrock(
            model_id="global.anthropic.claude-sonnet-4-6",
            model_kwargs=dict(temperature=0),
            region_name="us-west-2",
            max_tokens=MAX_TOKENS,
        )
    elif provider == "google":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0,
            max_tokens=MAX_TOKENS,
        )
    elif provider == "microsoft":
        return AzureChatOpenAI(
            azure_endpoint="https://gai-326.openai.azure.com/",
            azure_deployment="gpt-5.1",
            api_version="2024-12-01-preview",
            temperature=0,
            max_tokens=MAX_TOKENS,
        )
    else:
        raise Exception("Unsupported provider. provider can be one of 'aws', 'google', 'microsoft'.")
