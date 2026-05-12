import os
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from openai import OpenAI

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


def _build_neo4j_driver():
    neo4j_uri = os.environ.get("NEO4J_URI")
    if not neo4j_uri:
        return None

    # A local single-instance Neo4j server does not need routing.
    # Using bolt:// avoids repeated routing-table retries when the demo
    # connects to 127.0.0.1 / localhost.
    if neo4j_uri.startswith("neo4j://127.0.0.1") or neo4j_uri.startswith("neo4j://localhost") or neo4j_uri.startswith("neo4j://[::1]"):
        neo4j_uri = "bolt://" + neo4j_uri[len("neo4j://"):]

    return GraphDatabase.driver(
        neo4j_uri,
        auth=(os.environ.get("NEO4J_USERNAME"), os.environ.get("NEO4J_PASSWORD")),
        notifications_min_severity="OFF",
    )


def _build_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


neo4j_driver = _build_neo4j_driver()
open_ai_client = _build_openai_client()


def ensure_neo4j_driver():
    if neo4j_driver is None:
        raise RuntimeError(
            "Neo4j driver is not configured. Set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD."
        )
    return neo4j_driver


def ensure_openai_client():
    if open_ai_client is None:
        raise RuntimeError("OpenAI client is not configured. Set OPENAI_API_KEY.")
    return open_ai_client


def chunk_text(text, chunk_size, overlap, split_on_whitespace_only=True):
    chunks = []
    index = 0

    while index < len(text):
        if split_on_whitespace_only:
            prev_whitespace = 0
            left_index = index - overlap
            while left_index >= 0:
                if text[left_index] == " ":
                    prev_whitespace = left_index
                    break
                left_index -= 1
            next_whitespace = text.find(" ", index + chunk_size)
            if next_whitespace == -1:
                next_whitespace = len(text)
            chunk = text[prev_whitespace:next_whitespace].strip()
            chunks.append(chunk)
            index = next_whitespace + 1
        else:
            start = max(0, index - overlap + 1)
            end = min(index + chunk_size + overlap, len(text))
            chunk = text[start:end].strip()
            chunks.append(chunk)
            index += chunk_size

    return chunks


def num_tokens_from_string(string: str, model: str = "gpt-5-nano") -> int:
    if tiktoken is None:
        return max(1, len(string.split()))
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(string))


def embed(texts, model="text-embedding-3-small"):
    client = ensure_openai_client()
    response = client.embeddings.create(
        input=texts,
        model=model,
    )
    return [item.embedding for item in response.data]


def chat(messages, model="gpt-5-nano", config=None):
    client = ensure_openai_client()
    if config is None:
        config = {}
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **config,
    )
    return response.choices[0].message.content


def tool_choice(messages, model="gpt-5-nano", tools=None, config=None):
    client = ensure_openai_client()
    if tools is None:
        tools = []
    if config is None:
        config = {}

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools or None,
        tool_choice="required" if tools else None,
        **config,
    )

    choice = response.choices[0]
    return {
        "tool_calls": choice.message.tool_calls or [],
        "content": choice.message.content,
        "finish_reason": choice.finish_reason,
    }


def project_root() -> Path:
    return Path(__file__).resolve().parent


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")
