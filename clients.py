from __future__ import annotations

from mem0 import Memory
from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_BASE_URL, build_mem0_config
from mem0_patch import patch_mem0_graph


def create_memory() -> Memory:
    patch_mem0_graph()
    return Memory.from_config(build_mem0_config())


def create_openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
