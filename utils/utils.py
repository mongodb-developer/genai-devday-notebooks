from langchain_aws import ChatBedrock
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import os

SLEEP_TIMER = 5
PROXY_ENDPOINT = "https://vtqjvgchmwcjwsrela2oyhlegu0hwqnw.lambda-url.us-west-2.on.aws/"

def get_llm():

    provider = os.environ.get("AILABS_LLM_PROVIDER", "aws")
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
