from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List


def time_short() -> str:
    return datetime.now().strftime("%H:%M")


def time_full() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_score(score: Any) -> str:
    try:
        return f"{float(score):.2f}"
    except Exception:
        return "-"


def format_relation(relation: Any) -> str:
    if relation is None:
        return "-"
    if isinstance(relation, str):
        return relation
    if isinstance(relation, (list, tuple)):
        if len(relation) >= 3:
            return f"{relation[0]} -[{relation[1]}]-> {relation[2]}"
        return " ".join([str(item) for item in relation])
    if isinstance(relation, dict):
        raw = relation.get("raw")
        if raw:
            return str(raw)
        source = relation.get("source") or relation.get("from") or relation.get("start") or "?"
        relationship = (
            relation.get("relationship") or relation.get("relation") or relation.get("type") or "?"
        )
        target = relation.get("target") or relation.get("destination") or relation.get("to") or relation.get(
            "end"
        ) or "?"
        return f"{source} -[{relationship}]-> {target}"
    return str(relation)


def normalize_graph_relations(relations: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []

    def add_relation(item: Any, event: str | None = None) -> None:
        if item is None:
            return
        if isinstance(item, dict):
            if "added_entities" in item or "deleted_entities" in item:
                added_entities = item.get("added_entities")
                deleted_entities = item.get("deleted_entities")
                if added_entities is not None:
                    add_relation(added_entities, "ADD")
                if deleted_entities is not None:
                    add_relation(deleted_entities, "DELETE")
                return
            if {"source", "relationship", "target"} & set(item.keys()) or "destination" in item or "raw" in item:
                rel = dict(item)
                if event:
                    rel["event"] = rel.get("event", event)
                normalized.append(rel)
                return
            rel = {"raw": str(item)}
            if event:
                rel["event"] = event
            normalized.append(rel)
            return
        if isinstance(item, (list, tuple)):
            if len(item) == 0:
                return
            if all(not isinstance(x, (list, tuple, dict)) for x in item) and len(item) >= 3:
                rel = {"source": item[0], "relationship": item[1], "target": item[2]}
                if event:
                    rel["event"] = event
                normalized.append(rel)
                return
            for sub in item:
                add_relation(sub, event)
            return
        if isinstance(item, str):
            rel = {"raw": item}
            if event:
                rel["event"] = event
            normalized.append(rel)
            return
        rel = {"raw": str(item)}
        if event:
            rel["event"] = event
        normalized.append(rel)

    if isinstance(relations, dict):
        added = relations.get("added_entities")
        deleted = relations.get("deleted_entities")
        if added is not None:
            add_relation(added, "ADD")
        if deleted is not None:
            add_relation(deleted, "DELETE")
        if not normalized and {"source", "relationship", "target", "destination", "raw"} & set(relations.keys()):
            add_relation(relations)
        return normalized

    add_relation(relations)
    return normalized
