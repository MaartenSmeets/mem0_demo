import asyncio
import os
from datetime import datetime
from typing import Any, Dict, List

from ollama import Client
from mem0 import Memory
from nicegui import ui

# 1. SETUP: Configure Privacy-First Memory
# We use Qdrant for storage, Ollama for reasoning, and local CPU for embeddings.
os.environ["MEM0_TELEMETRY"] = "False"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral-small3.2")
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1")
EMBEDDING_DIMS = int(os.getenv("EMBEDDING_DIMS", "768"))
USER_ID = os.getenv("MEM0_USER_ID", "local_user")

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
    "llm": {
        "provider": "ollama",
        "config": {
            "model": OLLAMA_MODEL,
            "ollama_base_url": OLLAMA_BASE_URL,
        },
    },
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
mem = Memory.from_config(config)
ollama_client = Client(host=OLLAMA_BASE_URL)

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
        with ui.card().props("flat bordered").classes("w-full p-4"):
            with ui.row().classes("w-full items-start justify-between gap-4"):
                with ui.column().classes("gap-1"):
                    ui.label(item["query"]).classes("text-sm font-semibold text-slate-900")
                    ui.label(item["ts"]).classes("text-xs text-slate-500")
                with ui.row().classes("items-center gap-2"):
                    ui.badge(f"{len(retrieved)} retrieved", color="blue-2", text_color="blue-10")
                    ui.badge(f"{len(actions)} actions", color="amber-2", text_color="amber-10")

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


def get_ai_response(user_input: str, relevant_memories: Dict[str, Any]) -> str:
    """
    Generates a response using Ollama, injecting the retrieved memories as context.
    """
    context_str = ""
    if relevant_memories and "results" in relevant_memories:
        context_str = "\n".join([f"- {m['memory']}" for m in relevant_memories["results"]])

    system_prompt = (
        "You are a helpful AI assistant with long-term memory. "
        "Use the provided [MEMORY] section to personalize your responses. "
        "If the memory doesn't help with the specific question, just answer naturally."
    )

    final_prompt = f"""
[MEMORY]
{context_str}
[/MEMORY]

User: {user_input}
Assistant:
"""

    response = ollama_client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt},
        ],
    )

    return response["message"]["content"]


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
