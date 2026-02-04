from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class MemoryItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: Optional[str] = None
    memory: Optional[str] = None
    user_id: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    created_at: Optional[str] = None
    score: Optional[float] = None
    event: Optional[str] = None
    previous_memory: Optional[str] = None


class Mem0Response(BaseModel):
    model_config = ConfigDict(extra="ignore")

    results: List[MemoryItem] = Field(default_factory=list)
    relations: List[Any] = Field(default_factory=list)


def parse_mem0_response(raw: Any) -> Mem0Response:
    if raw is None:
        return Mem0Response()
    if isinstance(raw, Mem0Response):
        return raw
    if isinstance(raw, list):
        items = []
        for item in raw:
            try:
                items.append(MemoryItem.model_validate(item))
            except ValidationError:
                continue
        return Mem0Response(results=items)
    if isinstance(raw, dict):
        payload = dict(raw)
        if "results" not in payload and "result" in payload:
            payload["results"] = payload["result"]
        if "results" in payload and not isinstance(payload["results"], list):
            payload["results"] = [payload["results"]]
        relations = payload.get("relations", [])
        if isinstance(relations, dict):
            payload["relations"] = [relations]
        elif relations is None:
            payload["relations"] = []
        elif not isinstance(relations, list):
            payload["relations"] = [relations]
        try:
            return Mem0Response.model_validate(payload)
        except ValidationError:
            results = payload.get("results", [])
            items = []
            for item in results if isinstance(results, list) else [results]:
                try:
                    items.append(MemoryItem.model_validate(item))
                except ValidationError:
                    continue
            safe_relations = payload.get("relations", [])
            if isinstance(safe_relations, dict):
                safe_relations = [safe_relations]
            elif safe_relations is None:
                safe_relations = []
            elif not isinstance(safe_relations, list):
                safe_relations = [safe_relations]
            return Mem0Response(results=items, relations=safe_relations)
    return Mem0Response()
