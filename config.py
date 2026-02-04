from __future__ import annotations

import os
from typing import Any, Dict

os.environ.setdefault("MEM0_TELEMETRY", "False")


def normalize_openai_base_url(ollama_base_url: str, openai_base_url: str | None) -> str:
    if openai_base_url:
        return openai_base_url.rstrip("/")
    base = ollama_base_url.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OPENAI_BASE_URL = normalize_openai_base_url(OLLAMA_BASE_URL, os.getenv("OPENAI_BASE_URL"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "huihui_ai/qwen3-next-abliterated:80b-a3b-instruct-q4_K_M")
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1")
EMBEDDING_DIMS = int(os.getenv("EMBEDDING_DIMS", "768"))
USER_ID = os.getenv("MEM0_USER_ID", "local_user")
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "mem0demo123")
MEM0_GRAPH_PROMPT = os.getenv("MEM0_GRAPH_PROMPT", "").strip()


def build_llm_config() -> Dict[str, Any]:
    return {
        "provider": "openai",
        "config": {
            "model": OLLAMA_MODEL,
            "openai_base_url": OPENAI_BASE_URL,
            "api_key": OPENAI_API_KEY,
        },
    }


def build_graph_store(llm_config: Dict[str, Any]) -> Dict[str, Any]:
    graph_store: Dict[str, Any] = {
        "provider": "neo4j",
        "config": {
            "url": NEO4J_URL,
            "username": NEO4J_USERNAME,
            "password": NEO4J_PASSWORD,
        },
        "llm": llm_config,
    }
    if MEM0_GRAPH_PROMPT:
        graph_store["custom_prompt"] = MEM0_GRAPH_PROMPT
    return graph_store


def build_mem0_config() -> Dict[str, Any]:
    llm_config = build_llm_config()
    graph_store = build_graph_store(llm_config)
    return {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "personal_memory",
                "host": "localhost",
                "port": 6333,
                "embedding_model_dims": EMBEDDING_DIMS,
                "on_disk": True,
            },
        },
        "graph_store": graph_store,
        "llm": llm_config,
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": HF_EMBED_MODEL,
                "model_kwargs": {
                    "device": "cpu",
                    "trust_remote_code": True,
                },
            },
        },
    }
