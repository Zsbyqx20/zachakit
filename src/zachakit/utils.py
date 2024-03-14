import json
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletionMessageParam
from rich import print
from tiktoken import Encoding


def load_azure_client() -> OpenAI:
    api_key = os.getenv("AZURE_OPENAI_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    return AzureOpenAI(api_key=api_key, api_version="2023-05-15", azure_endpoint=azure_endpoint)


def load_local_client() -> OpenAI:
    try:
        load_dotenv()
    except:
        print("[red1 b]The environment file `.env` is not found. Check `python-dotenv` for more details.[/]")

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("BASE_URL")
    except:
        print("[red1 b]The env variables OPENAI_API_KEY and BASE_URL are invalid. Please check.[/]")

    return OpenAI(api_key=api_key, base_url=base_url)


def rjl(file: Path):
    with open(file, "r") as f:
        dt = [json.loads(line) for line in f]
    return dt


def wjl(file: Path, dt: list):
    with open(file, "w") as f:
        for d in dt:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return


def estimate_tokens(message_param: List[ChatCompletionMessageParam], enc: Encoding):
    tokens = 0
    extra_tokens = {"system": 4, "user": 7, "assistant": 1}
    for mp in message_param:
        tokens += len(enc.encode(mp["content"])) + extra_tokens[mp["role"]]
    return tokens
