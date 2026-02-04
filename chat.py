import asyncio
import os
import re
from datetime import datetime
from typing import Any, Dict, List

from openai import OpenAI
from mem0 import Memory
from nicegui import ui

# 1. SETUP: Configure Privacy-First Memory
# We use Qdrant for storage, an OpenAI-compatible API (Ollama) for reasoning, and local CPU for embeddings.
os.environ["MEM0_TELEMETRY"] = "False"


def _patch_mem0_graph() -> None:
    """Patch Mem0 graph extraction behavior at runtime (no file edits outside chat.py)."""
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
            if "favorite" in lowered and "pizza" in lowered:
                entity_type_map.setdefault("favorite_pizza", "food")
            if not entity_type_map:
                entity_type_map[filters["user_id"]] = "person"

        entity_type_map = {k.lower().replace(" ", "_"): v.lower().replace(" ", "_") for k, v in entity_type_map.items()}
        if os.getenv("MEM0_GRAPH_DEBUG") == "1":
            print(f"[graph] extracted_entities={entity_type_map} input={data!r}")
        logger.debug("Entity type map: %s\n search_results=%s", entity_type_map, search_results)
        return entity_type_map

    def _establish_nodes_relations_from_data(self, data, filters, entity_type_map):
        # Use raw user_id to keep node naming consistent
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
                {"role": "user", "content": f"List of entities: {list(entity_type_map.keys())}. \n\nText: {data}"},
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

        # Use raw user_id to keep node naming consistent
        user_identity = f"{filters['user_id']}"

        system_prompt, user_prompt = graph_memory.get_delete_messages(search_output_string, data, user_identity)

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

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", f"{OLLAMA_BASE_URL.rstrip('/')}/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "huihui_ai/qwen3-next-abliterated:80b-a3b-instruct-q4_K_M")
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1")
EMBEDDING_DIMS = int(os.getenv("EMBEDDING_DIMS", "768"))
USER_ID = os.getenv("MEM0_USER_ID", "local_user")
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "mem0demo123")
MEM0_GRAPH_PROMPT = os.getenv("MEM0_GRAPH_PROMPT", "").strip()

llm_config: Dict[str, Any] = {
    "provider": "openai",
    "config": {
        "model": OLLAMA_MODEL,
        "openai_base_url": OPENAI_BASE_URL,
        "api_key": OPENAI_API_KEY,
    },
}

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

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "personal_memory",
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": EMBEDDING_DIMS,  # Matches the embedder dimensions
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
                "device": "cpu",  # Forces embedding to run on CPU
                "trust_remote_code": True,  # Required by some HF models like nomic-embed-text-v1
            },
        },
    },
}

# Initialize Memory
print("Loading Embedding Models (this may take a moment first time)...")
_patch_mem0_graph()
mem = Memory.from_config(config)
openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

interaction_log: List[Dict[str, Any]] = []


def _time_short() -> str:
    return datetime.now().strftime("%H:%M")


def _time_full() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _format_score(score: Any) -> str:
    try:
        return f"{float(score):.2f}"
    except Exception:
        return "-"


def _format_relation(relation: Any) -> str:
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
        relationship = relation.get("relationship") or relation.get("relation") or relation.get("type") or "?"
        target = relation.get("target") or relation.get("destination") or relation.get("to") or relation.get("end") or "?"
        return f"{source} -[{relationship}]-> {target}"
    return str(relation)


def _normalize_graph_relations(relations: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []

    def add_relation(item: Any, event: str | None = None) -> None:
        if item is None:
            return
        if isinstance(item, dict):
            if {"source", "relationship", "target"} & set(item.keys()) or "destination" in item or "raw" in item:
                rel = dict(item)
                if event:
                    rel["event"] = rel.get("event", event)
                normalized.append(rel)
                return
            # dict but not a relation -> add as raw for visibility
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


chat_log: List[Dict[str, Any]] = [
    {
        "role": "assistant",
        "text": "Hello! I am ready to chat. I will remember what you tell me.",
        "stamp": _time_short(),
        "sent": False,
    }
]


@ui.refreshable
def render_chat() -> None:
    for entry in chat_log:
        ui.chat_message(
            entry["text"],
            name="You" if entry.get("sent") else "Assistant" if entry["role"] == "assistant" else entry["role"],
            stamp=entry.get("stamp", _time_short()),
            sent=entry.get("sent", False),
        )


@ui.refreshable
def render_mem0_activity() -> None:
    if not interaction_log:
        ui.label("No memory activity yet.").classes("text-sm text-slate-500")
        return

    for item in reversed(interaction_log[-50:]):
        retrieved = item.get("retrieved", [])
        actions = item.get("actions", [])
        graph_retrieved = _normalize_graph_relations(item.get("graph_retrieved"))
        graph_actions = _normalize_graph_relations(item.get("graph_actions"))
        with ui.card().props("flat bordered").classes("w-full p-4"):
            with ui.row().classes("w-full items-start justify-between gap-4"):
                with ui.column().classes("gap-1"):
                    ui.label(item["query"]).classes("text-sm font-semibold text-slate-900")
                    ui.label(item["ts"]).classes("text-xs text-slate-500")
                with ui.row().classes("items-center gap-2"):
                    ui.badge(f"{len(retrieved)} retrieved", color="blue-2", text_color="blue-10")
                    ui.badge(f"{len(actions)} actions", color="amber-2", text_color="amber-10")
                    ui.badge(
                        f"{len(graph_retrieved)} graph retrieved", color="indigo-2", text_color="indigo-10"
                    )
                    ui.badge(f"{len(graph_actions)} graph actions", color="purple-2", text_color="purple-10")

            ui.element("div").classes("h-px w-full bg-slate-200 my-3")

            with ui.row().classes("w-full gap-4 flex-wrap"):
                with ui.column().classes("w-full md:w-[48%] gap-2"):
                    ui.label("Retrieved").classes("text-xs uppercase tracking-wide text-slate-500")
                    if not retrieved:
                        ui.label("No relevant memories.").classes("text-xs text-slate-500")
                    for mem_item in retrieved:
                        with ui.row().classes("items-start gap-2"):
                            ui.badge(_format_score(mem_item.get("score")), color="blue-2", text_color="blue-10")
                            ui.label(mem_item.get("memory", "")).classes("text-sm text-slate-800")

                with ui.column().classes("w-full md:w-[48%] gap-2"):
                    ui.label("Actions").classes("text-xs uppercase tracking-wide text-slate-500")
                    if not actions:
                        ui.label("No memory changes.").classes("text-xs text-slate-500")
                    for action in actions:
                        event = action.get("event", "ADD")
                        colors = {
                            "ADD": ("green-3", "green-10"),
                            "UPDATE": ("amber-3", "amber-10"),
                            "DELETE": ("red-3", "red-10"),
                            "NONE": ("grey-4", "grey-9"),
                        }.get(event, ("grey-4", "grey-9"))
                        with ui.row().classes("items-start gap-2"):
                            ui.badge(event, color=colors[0], text_color=colors[1])
                            ui.label(action.get("memory", "")).classes("text-sm text-slate-800")
                        previous = action.get("previous_memory")
                        if event == "UPDATE" and previous:
                            ui.label(f"Previous: {previous}").classes("text-xs text-slate-500")

            ui.element("div").classes("h-px w-full bg-slate-200 my-3")

            with ui.row().classes("w-full gap-4 flex-wrap"):
                with ui.column().classes("w-full md:w-[48%] gap-2"):
                    ui.label("Graph Retrieved").classes("text-xs uppercase tracking-wide text-slate-500")
                    if not graph_retrieved:
                        ui.label("No graph relations.").classes("text-xs text-slate-500")
                    for relation in graph_retrieved:
                        with ui.row().classes("items-start gap-2"):
                            ui.badge("REL", color="indigo-2", text_color="indigo-10")
                            ui.label(_format_relation(relation)).classes("text-sm text-slate-800")

                with ui.column().classes("w-full md:w-[48%] gap-2"):
                    ui.label("Graph Actions").classes("text-xs uppercase tracking-wide text-slate-500")
                    if not graph_actions:
                        ui.label("No graph updates.").classes("text-xs text-slate-500")
                    for relation in graph_actions:
                        event = relation.get("event", "REL") if isinstance(relation, dict) else "REL"
                        event_colors = {
                            "ADD": ("green-3", "green-10"),
                            "DELETE": ("red-3", "red-10"),
                            "REL": ("purple-2", "purple-10"),
                        }.get(event, ("purple-2", "purple-10"))
                        with ui.row().classes("items-start gap-2"):
                            ui.badge(event, color=event_colors[0], text_color=event_colors[1])
                            ui.label(_format_relation(relation)).classes("text-sm text-slate-800")


def get_ai_response(user_input: str, relevant_memories: Dict[str, Any]) -> str:
    """
    Generates a response using Ollama, injecting the retrieved memories as context.
    """
    context_parts: List[str] = []
    if relevant_memories and relevant_memories.get("results"):
        facts = "\n".join([f"- {m['memory']}" for m in relevant_memories["results"]])
        context_parts.append(f"Facts:\n{facts}")
    if relevant_memories and relevant_memories.get("relations"):
        relations = "\n".join([f"- {_format_relation(r)}" for r in relevant_memories["relations"]])
        context_parts.append(f"Relations:\n{relations}")
    context_str = "\n\n".join(context_parts)

    system_prompt = (
        "You are a helpful AI assistant with long-term memory. "
        "Use the provided [MEMORY] facts and relations to personalize your responses. "
        "If the memory doesn't help with the specific question, just answer naturally."
    )

    final_prompt = f"""
[MEMORY]
{context_str}
[/MEMORY]

User: {user_input}
Assistant:
"""

    response = openai_client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt},
        ],
    )

    return response.choices[0].message.content or ""


def chat_loop() -> None:
    print("\n\n=== SYSTEM READY: TYPE 'exit' TO QUIT ===")
    print("AI: Hello! I am ready to chat. I will remember what you tell me.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        previous_memories = mem.search(user_input, user_id=USER_ID)
        response = get_ai_response(user_input, previous_memories)
        print(f"AI: {response}")

        mem.add(user_input, user_id=USER_ID)


def run_ui() -> None:
    ui.add_head_html(
        """
<style>
  :root { --app-header-height: 72px; }
  html, body, #q-app { height: 100%; margin: 0; }
  body { overflow: hidden; }
</style>
"""
    )

    with ui.header(elevated=True, fixed=False).classes("bg-slate-900 text-white").style(
        "height: var(--app-header-height)"
    ):
        with ui.row().classes("w-full items-center justify-between px-6 py-4"):
            with ui.column().classes("gap-1"):
                ui.label("Mem0 Command Center").classes("text-xl font-semibold tracking-tight")
                ui.label("Chat with durable memory and transparent retrievals").classes(
                    "text-xs text-slate-300"
                )
            with ui.row().classes("items-center gap-2"):
                ui.badge(f"LLM: {OLLAMA_MODEL}", color="blue-4", text_color="blue-11")
                ui.badge("Qdrant: localhost", color="teal-4", text_color="teal-11")
                ui.badge(f"Neo4j: {NEO4J_URL}", color="indigo-4", text_color="indigo-11")
                ui.badge(f"OpenAI API: {OPENAI_BASE_URL}", color="cyan-4", text_color="cyan-11")
                ui.badge("Embedder: CPU", color="orange-4", text_color="orange-11")

    with ui.column().classes("w-full bg-slate-50 px-6 py-6 gap-4").style(
        "height: calc(100vh - var(--app-header-height))"
    ):
        with ui.column().classes("w-full h-full min-h-0 gap-4"):
            with ui.tabs().classes("w-full") as tabs:
                chat_tab = ui.tab("Chat")
                memory_tab = ui.tab("Mem0 Activity")

            with ui.tab_panels(tabs, value=chat_tab).classes("w-full flex-1 min-h-0"):
                with ui.tab_panel(chat_tab).classes("h-full min-h-0"):
                    with ui.card().props("flat bordered").classes("w-full h-full min-h-0"):
                        with ui.column().classes("w-full h-full min-h-0 p-6 gap-3"):
                            ui.label("Conversation").classes("text-lg font-semibold text-slate-900")
                            ui.label(
                                "Your messages stay on the chat tab. Memory details live separately."
                            ).classes("text-xs text-slate-500")
                            chat_scroll_area = ui.scroll_area().classes("w-full flex-1 min-h-0 h-full")
                            with chat_scroll_area:
                                render_chat()

                            input_box = ui.textarea(
                                label="Message",
                                placeholder="Ask a question or share a preference...",
                            ).props("outlined autogrow").classes("w-full")

                            with ui.row().classes("w-full items-center justify-between"):
                                with ui.row().classes("items-center gap-2"):
                                    spinner = ui.spinner("dots", size="sm", color="blue").classes("text-slate-500")
                                    spinner.visible = False
                                    status_label = ui.label("Ready").classes("text-xs text-slate-500")
                                send_button = ui.button("Send", color="dark", on_click=None).classes(
                                    "text-white"
                                )

                            async def handle_send(_: Any = None) -> None:
                                text = (input_box.value or "").strip()
                                if not text:
                                    return

                                input_box.value = ""
                                status_label.set_text("Retrieving memories...")
                                mem0_status_label.set_text("Status: Retrieving memories")
                                spinner.visible = True
                                send_button.disable()
                                input_box.disable()

                                chat_log.append(
                                    {"role": "user", "text": text, "stamp": _time_short(), "sent": True}
                                )
                                render_chat.refresh()
                                chat_scroll_area.scroll_to(percent=1.0)

                                try:
                                    search_result = await asyncio.to_thread(mem.search, text, user_id=USER_ID)
                                    status_label.set_text("Querying model...")
                                    mem0_status_label.set_text("Status: Querying model")
                                    response = await asyncio.to_thread(get_ai_response, text, search_result)
                                    chat_log.append(
                                        {"role": "assistant", "text": response, "stamp": _time_short(), "sent": False}
                                    )
                                    render_chat.refresh()
                                    chat_scroll_area.scroll_to(percent=1.0)

                                    status_label.set_text("Updating memory...")
                                    mem0_status_label.set_text("Status: Updating memory")
                                    add_result = await asyncio.to_thread(mem.add, text, user_id=USER_ID)

                                    interaction_log.append(
                                        {
                                            "ts": _time_full(),
                                            "query": text,
                                            "retrieved": (search_result or {}).get("results", []),
                                            "actions": (add_result or {}).get("results", []),
                                            "graph_retrieved": (search_result or {}).get("relations", []),
                                            "graph_actions": (add_result or {}).get("relations", []),
                                        }
                                    )

                                    render_mem0_activity.refresh()
                                    status_label.set_text("Done")
                                    mem0_status_label.set_text("Status: Done")
                                except Exception as exc:
                                    chat_log.append(
                                        {
                                            "role": "System",
                                            "text": f"System error: {exc}",
                                            "stamp": _time_short(),
                                            "sent": False,
                                        }
                                    )
                                    render_chat.refresh()
                                    chat_scroll_area.scroll_to(percent=1.0)
                                    status_label.set_text("Error (see message)")
                                    mem0_status_label.set_text("Status: Error (see message)")
                                    ui.notify(f"Request failed: {exc}", type="negative")
                                finally:
                                    spinner.visible = False
                                    send_button.enable()
                                    input_box.enable()

                            send_button.on("click", handle_send)
                            input_box.on(
                                "keydown",
                                handle_send,
                                js_handler=(
                                    "(event) => {"
                                    "  if (event.key === 'Enter' && !event.shiftKey) {"
                                    "    event.preventDefault();"
                                    "    emit('send', {shiftKey: event.shiftKey});"
                                    "    return false;"
                                    "  }"
                                    "}"
                                ),
                            )


                with ui.tab_panel(memory_tab).classes("h-full min-h-0"):
                    with ui.column().classes("w-full h-full min-h-0 gap-3"):
                        with ui.row().classes("w-full items-center justify-between"):
                            mem0_status_label = ui.label("Status: Idle").classes("text-xs text-slate-500")
                            clear_button = ui.button("Clear memories", color="red", on_click=None).props("outline")

                        async def handle_clear(_: Any = None) -> None:
                            mem0_status_label.set_text("Status: Clearing memories")
                            try:
                                await asyncio.to_thread(mem.delete_all, user_id=USER_ID)
                                interaction_log.clear()
                                render_mem0_activity.refresh()
                                mem0_status_label.set_text("Status: Cleared")
                                ui.notify("Memories cleared", type="positive")
                            except Exception as exc:
                                mem0_status_label.set_text("Status: Clear failed")
                                ui.notify(f"Clear failed: {exc}", type="negative")

                        clear_button.on("click", handle_clear)

                        with ui.scroll_area().classes("w-full flex-1 min-h-0"):
                            render_mem0_activity()

    ui.run(title="Mem0 Chat", reload=False)


if __name__ == "__main__":
    run_ui()
