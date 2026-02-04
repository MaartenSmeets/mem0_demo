from __future__ import annotations

import os
import re


def patch_mem0_graph() -> None:
    """Patch Mem0 graph extraction behavior at runtime (no file edits outside repo)."""
    import mem0.memory.graph_memory as graph_memory

    if getattr(graph_memory, "_patched_by_chat", False):
        return
    graph_memory._patched_by_chat = True

    logger = graph_memory.logger

    def _retrieve_nodes_from_data(self, data, filters):
        _tools = [graph_memory.EXTRACT_ENTITIES_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [graph_memory.EXTRACT_ENTITIES_STRUCT_TOOL]

        tool_name = _tools[0]["function"]["name"] if _tools else None
        tool_choice = "auto"
        if self.llm_provider in ["openai", "azure_openai"] and tool_name:
            tool_choice = {"type": "function", "function": {"name": tool_name}}

        search_results = self.llm.generate_response(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a smart assistant who understands entities and their types in a given text. "
                        f"If user message contains self reference such as 'I', 'me', 'my' etc. then use {filters['user_id']} as the source entity. "
                        "Extract all the entities from the text. ***DO NOT*** answer the question itself if the given text is a question."
                    ),
                },
                {"role": "user", "content": data},
            ],
            tools=_tools,
            tool_choice=tool_choice,
        )

        entity_type_map = {}
        try:
            for tool_call in search_results.get("tool_calls", []):
                if tool_call.get("name") != "extract_entities":
                    continue
                entities = tool_call.get("arguments", {}).get("entities")
                if not entities:
                    continue
                for item in entities:
                    entity_type_map[item["entity"]] = item["entity_type"]
        except Exception as exc:
            logger.exception(
                "Error in search tool: %s, llm_provider=%s, search_results=%s",
                exc,
                self.llm_provider,
                search_results,
            )

        if not entity_type_map:
            lowered = data.lower()
            if os.getenv("MEM0_GRAPH_DEBUG") == "1":
                print(f"[graph] tool_calls={search_results.get('tool_calls')} input={data!r}")
            if re.search(r"\b(i|me|my|mine|myself|i'm|im)\b", lowered):
                entity_type_map[filters["user_id"]] = "person"
            if not entity_type_map:
                entity_type_map[filters["user_id"]] = "person"

        entity_type_map = {
            k.lower().replace(" ", "_"): v.lower().replace(" ", "_")
            for k, v in entity_type_map.items()
        }
        if os.getenv("MEM0_GRAPH_DEBUG") == "1":
            print(f"[graph] extracted_entities={entity_type_map} input={data!r}")
        logger.debug("Entity type map: %s\n search_results=%s", entity_type_map, search_results)
        return entity_type_map

    def _establish_nodes_relations_from_data(self, data, filters, entity_type_map):
        user_identity = f"{filters['user_id']}"

        if self.config.graph_store.custom_prompt:
            system_content = graph_memory.EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_identity)
            system_content = system_content.replace(
                "CUSTOM_PROMPT", f"4. {self.config.graph_store.custom_prompt}"
            )
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": data},
            ]
        else:
            system_content = graph_memory.EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_identity)
            messages = [
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": f"List of entities: {list(entity_type_map.keys())}. \n\nText: {data}",
                },
            ]

        _tools = [graph_memory.RELATIONS_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [graph_memory.RELATIONS_STRUCT_TOOL]

        tool_name = _tools[0]["function"]["name"] if _tools else None
        tool_choice = "auto"
        if self.llm_provider in ["openai", "azure_openai"] and tool_name:
            tool_choice = {"type": "function", "function": {"name": tool_name}}

        extracted_entities = self.llm.generate_response(
            messages=messages,
            tools=_tools,
            tool_choice=tool_choice,
        )

        entities = []
        for tool_call in extracted_entities.get("tool_calls", []):
            if tool_call.get("name") not in {"establish_relations", "establish_relationships"}:
                continue
            entities = tool_call.get("arguments", {}).get("entities", [])
            break

        entities = self._remove_spaces_from_entities(entities)
        logger.debug("Extracted entities: %s", entities)
        return entities

    def _get_delete_entities_from_search_output(self, search_output, data, filters):
        search_output_string = graph_memory.format_entities(search_output)
        user_identity = f"{filters['user_id']}"

        system_prompt, user_prompt = graph_memory.get_delete_messages(
            search_output_string, data, user_identity
        )

        _tools = [graph_memory.DELETE_MEMORY_TOOL_GRAPH]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [graph_memory.DELETE_MEMORY_STRUCT_TOOL_GRAPH]

        tool_name = _tools[0]["function"]["name"] if _tools else None
        tool_choice = "auto"
        if self.llm_provider in ["openai", "azure_openai"] and tool_name:
            tool_choice = {"type": "function", "function": {"name": tool_name}}

        memory_updates = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=_tools,
            tool_choice=tool_choice,
        )

        to_be_deleted = []
        for item in memory_updates.get("tool_calls", []):
            if item.get("name") == "delete_graph_memory":
                to_be_deleted.append(item.get("arguments"))
        to_be_deleted = self._remove_spaces_from_entities(to_be_deleted)
        logger.debug("Deleted relationships: %s", to_be_deleted)
        return to_be_deleted

    graph_memory.MemoryGraph._retrieve_nodes_from_data = _retrieve_nodes_from_data
    graph_memory.MemoryGraph._establish_nodes_relations_from_data = _establish_nodes_relations_from_data
    graph_memory.MemoryGraph._get_delete_entities_from_search_output = _get_delete_entities_from_search_output
