from __future__ import annotations

from clients import create_memory, create_openai_client
from config import USER_ID
from llm import get_ai_response
from mem0_models import parse_mem0_response


def chat_loop() -> None:
    print("\n\n=== SYSTEM READY: TYPE 'exit' TO QUIT ===")
    print("AI: Hello! I am ready to chat. I will remember what you tell me.\n")

    mem = create_memory()
    client = create_openai_client()

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        previous_raw = mem.search(user_input, user_id=USER_ID)
        previous_memories = parse_mem0_response(previous_raw)
        response = get_ai_response(client, user_input, previous_memories)
        print(f"AI: {response}")

        mem.add(user_input, user_id=USER_ID)
