from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from nicegui import ui

from clients import create_memory, create_openai_client
from config import NEO4J_URL, OLLAMA_MODEL, OPENAI_BASE_URL, USER_ID
from llm import get_ai_response
from mem0_models import parse_mem0_response
from utils import (
    format_relation,
    format_score,
    normalize_graph_relations,
    time_full,
    time_short,
)

chat_log: List[Dict[str, Any]] = [
    {
        "role": "assistant",
        "text": "Hello! I am ready to chat. I will remember what you tell me.",
        "stamp": time_short(),
        "sent": False,
    }
]
interaction_log: List[Dict[str, Any]] = []


@ui.refreshable
def render_chat() -> None:
    for entry in chat_log:
        ui.chat_message(
            entry["text"],
            name="You" if entry.get("sent") else "Assistant" if entry["role"] == "assistant" else entry["role"],
            stamp=entry.get("stamp", time_short()),
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
        graph_retrieved = normalize_graph_relations(item.get("graph_retrieved"))
        graph_actions = normalize_graph_relations(item.get("graph_actions"))
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
                            ui.badge(format_score(mem_item.get("score")), color="blue-2", text_color="blue-10")
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
                            ui.label(format_relation(relation)).classes("text-sm text-slate-800")

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
                            ui.label(format_relation(relation)).classes("text-sm text-slate-800")


def run_ui() -> None:
    mem0_status_label: Any | None = None

    print("Loading Embedding Models (this may take a moment first time)...")
    mem = create_memory()
    openai_client = create_openai_client()

    ui.add_head_html(
        """
<style>
  :root { --app-header-height: 104px; }
  html, body, #q-app { height: 100%; margin: 0; }
  body { overflow: hidden; }
</style>
"""
    )

    with ui.header(elevated=True, fixed=False).classes("bg-slate-900 text-white").style(
        "height: var(--app-header-height)"
    ):
        with ui.row().classes("w-full items-start justify-between px-6 py-3 gap-4 flex-wrap"):
            with ui.column().classes("gap-1"):
                ui.label("Mem0 Command Center").classes("text-xl font-semibold tracking-tight text-white")
                ui.label("Chat with durable memory and transparent retrievals").classes(
                    "text-xs text-slate-200"
                )

            def info_chip(label: str, value: str, accent: str) -> None:
                with ui.element("div").classes(
                    "flex flex-col gap-0.5 rounded-md border border-slate-700/70 "
                    "bg-slate-950/40 px-2 py-1"
                ):
                    ui.label(label).classes("text-[10px] uppercase tracking-wide text-slate-400")
                    ui.label(value).classes(f"text-xs font-medium {accent} truncate max-w-[240px]")

            with ui.row().classes("items-start gap-2 flex-wrap justify-end w-full md:w-auto"):
                info_chip("LLM", OLLAMA_MODEL, "text-blue-200")
                info_chip("Qdrant", "localhost", "text-teal-200")
                info_chip("Neo4j", NEO4J_URL, "text-indigo-200")
                info_chip("OpenAI API", OPENAI_BASE_URL, "text-cyan-200")
                info_chip("Embedder", "CPU", "text-amber-200")

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

                            def append_chat(role: str, text: str, sent: bool) -> None:
                                chat_log.append(
                                    {"role": role, "text": text, "stamp": time_short(), "sent": sent}
                                )
                                render_chat.refresh()
                                chat_scroll_area.scroll_to(percent=1.0)

                            def set_mem0_status(message: str) -> None:
                                if mem0_status_label is not None:
                                    mem0_status_label.set_text(f"Status: {message}")

                            def set_status(message: str) -> None:
                                status_label.set_text(message)
                                set_mem0_status(message)

                            async def handle_send(_: Any = None) -> None:
                                text = (input_box.value or "").strip()
                                if not text:
                                    return

                                input_box.value = ""
                                set_status("Retrieving memories...")
                                spinner.visible = True
                                send_button.disable()
                                input_box.disable()

                                append_chat("user", text, True)

                                try:
                                    raw_search = await asyncio.to_thread(mem.search, text, user_id=USER_ID)
                                    search_result = parse_mem0_response(raw_search)
                                    set_status("Querying model...")
                                    response = await asyncio.to_thread(
                                        get_ai_response, openai_client, text, search_result
                                    )
                                    append_chat("assistant", response, False)

                                    set_status("Updating memory...")
                                    raw_add = await asyncio.to_thread(mem.add, text, user_id=USER_ID)
                                    add_result = parse_mem0_response(raw_add)

                                    interaction_log.append(
                                        {
                                            "ts": time_full(),
                                            "query": text,
                                            "retrieved": [item.model_dump() for item in search_result.results],
                                            "actions": [item.model_dump() for item in add_result.results],
                                            "graph_retrieved": list(search_result.relations),
                                            "graph_actions": list(add_result.relations),
                                        }
                                    )

                                    render_mem0_activity.refresh()
                                    set_status("Done")
                                except Exception as exc:
                                    append_chat("System", f"System error: {exc}", False)
                                    set_status("Error (see message)")
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
                            clear_button = ui.button("Clear memories", color="red", on_click=None).props(
                                "outline"
                            )

                        async def handle_clear(_: Any = None) -> None:
                            set_mem0_status("Clearing memories")
                            try:
                                await asyncio.to_thread(mem.delete_all, user_id=USER_ID)
                                interaction_log.clear()
                                render_mem0_activity.refresh()
                                set_mem0_status("Cleared")
                                ui.notify("Memories cleared", type="positive")
                            except Exception as exc:
                                set_mem0_status("Clear failed")
                                ui.notify(f"Clear failed: {exc}", type="negative")

                        clear_button.on("click", handle_clear)

                        with ui.scroll_area().classes("w-full flex-1 min-h-0"):
                            render_mem0_activity()

    ui.run(title="Mem0 Chat", reload=False)
