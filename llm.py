from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from config import OLLAMA_MODEL
from mem0_models import Mem0Response, MemoryItem
from utils import format_relation
from pydantic import BaseModel, Field, ValidationError


class AssistantResponse(BaseModel):
    answer: str = Field(
        description=(
            "A neutral paraphrase of the user's latest message that keeps the same meaning "
            "without adding new information or opinions."
        )
    )


def _memory_items(relevant_memories: Mem0Response | Dict[str, Any] | None) -> Tuple[List[Any], List[Any]]:
    if relevant_memories is None:
        return [], []
    if isinstance(relevant_memories, Mem0Response):
        return relevant_memories.results, relevant_memories.relations
    return relevant_memories.get("results", []), relevant_memories.get("relations", [])


def _memory_text(item: Any) -> str:
    if isinstance(item, MemoryItem):
        return item.memory or ""
    if isinstance(item, dict):
        return str(item.get("memory") or "")
    return str(item)


def build_context(relevant_memories: Mem0Response | Dict[str, Any] | None) -> str:
    context_parts: List[str] = []
    results, relations = _memory_items(relevant_memories)
    if results:
        facts = "\n".join([f"- {_memory_text(m)}" for m in results if _memory_text(m)])
        if facts:
            context_parts.append(f"Facts:\n{facts}")
    if relations:
        rel_lines = "\n".join([f"- {format_relation(r)}" for r in relations])
        context_parts.append(f"Relations:\n{rel_lines}")
    return "\n\n".join(context_parts)


def build_system_prompt(structured: bool) -> str:
    prompt = (
        "You are a neutral rephrasing assistant. "
        "Your only task is to restate the user's latest message in different words. "
        "Keep the same meaning and keep it concise. "
        "Do not add opinions, advice, analysis, assumptions, emotional framing, or new facts. "
        "Do not answer the user's question; only paraphrase what the user said."
    )
    if structured:
        prompt += " Return only valid JSON that matches the provided schema."
    return prompt


def extract_output_text(response: Any) -> str:
    if response is None:
        return ""
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    if isinstance(response, dict):
        output = response.get("output")
    else:
        output = getattr(response, "output", None)
    if not output:
        return ""

    chunks: List[str] = []
    for item in output:
        if isinstance(item, dict):
            content = item.get("content") or item.get("text") or []
        else:
            content = getattr(item, "content", None) or getattr(item, "text", None) or []

        if isinstance(content, str):
            chunks.append(content)
            continue
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") in {"output_text", "text", "message"} and part.get("text"):
                        chunks.append(part["text"])
                else:
                    text = getattr(part, "text", None)
                    if text:
                        chunks.append(text)
    return "\n".join(chunks).strip()


def _structured_text_format() -> Dict[str, Any]:
    schema = AssistantResponse.model_json_schema()
    return {
        "format": {
            "type": "json_schema",
            "json_schema": {
                "name": "assistant_response",
                "strict": True,
                "schema": schema,
            },
        }
    }


def _parse_structured_text(raw_text: str) -> str | None:
    if not raw_text:
        return None
    try:
        parsed = AssistantResponse.model_validate_json(raw_text)
        return parsed.answer.strip()
    except ValidationError:
        pass

    def extract_answer(text: str) -> str | None:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(data, dict) and "answer" in data:
            answer = data.get("answer")
            if answer is None:
                return None
            return str(answer).strip()
        if isinstance(data, dict):
            for key in ("message", "response", "text", "final", "output"):
                if key in data and data[key]:
                    return str(data[key]).strip()
            string_values = [v for v in data.values() if isinstance(v, str) and v.strip()]
            if string_values:
                return max(string_values, key=len).strip()
        return None

    answer = extract_answer(raw_text)
    if answer:
        return answer

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        answer = extract_answer(fenced.group(1))
        if answer:
            return answer

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end > start:
        answer = extract_answer(raw_text[start : end + 1])
        if answer:
            return answer

    return None


def _sanitize_paraphrase(candidate: str | None, user_input: str) -> str:
    text = (candidate or "").strip()
    if not text:
        return user_input.strip()
    return text


def get_ai_response(
    client: Any,
    user_input: str,
    relevant_memories: Mem0Response | Dict[str, Any] | None,
    model: str = OLLAMA_MODEL,
) -> str:
    _ = relevant_memories
    system_prompt = build_system_prompt(structured=True)

    final_prompt = f"""
User message:
{user_input}

Paraphrase:
"""

    try:
        response = client.responses.create(
            model=model,
            instructions=system_prompt,
            input=final_prompt,
            text=_structured_text_format(),
        )
        raw_text = extract_output_text(response)
        parsed = _parse_structured_text(raw_text)
        if parsed:
            return _sanitize_paraphrase(parsed, user_input)
    except Exception:
        pass

    system_prompt = build_system_prompt(structured=False)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt},
        ],
    )
    return _sanitize_paraphrase(response.choices[0].message.content, user_input)
