# Mem0 NiceGUI Chat

A local‑first chat UI that uses **Mem0** for long‑term memory, **Qdrant** for vector storage, **Hugging Face embeddings on CPU**, and **Ollama** for generation. The UI is built with **NiceGUI** and provides a separate Mem0 Activity tab that groups retrieved memories and actions by request.

## What This Does
- Chat with a local model via Ollama.
- Store and retrieve memories with Mem0 + Qdrant.
- Use a local CPU embedding model (SentenceTransformers).
- Inspect Mem0’s retrievals and add/update/delete actions per request.
- Clear memories for the current user from the UI.

## Quickstart

### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Torch note**: If `pip install torch` fails, install the CPU wheel from the official PyTorch instructions for your OS.

### 2) Start Qdrant
```bash
docker compose up -d
```

### 3) Start Ollama
```bash
ollama serve
ollama pull mistral-small3.2
```

### 4) Run the app
```bash
python chat.py
```

Open the URL printed by NiceGUI (usually `http://localhost:8080`).

## Usage
- **Enter** sends a message.
- **Shift+Enter** inserts a newline.
- Switch to **Mem0 Activity** to see grouped retrievals and memory actions per request.
- Click **Clear memories** to delete memories for the current user.

## Configuration
Set these env vars as needed:

- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (default: `mistral-small3.2`)
- `HF_EMBED_MODEL` (default: `nomic-ai/nomic-embed-text-v1`)
- `EMBEDDING_DIMS` (default: `768`) — must match the embedding model
- `MEM0_USER_ID` (default: `local_user`)
- `HF_TOKEN` (optional) — avoids Hugging Face rate limits

Example:
```bash
export OLLAMA_MODEL=llama3.1
export HF_EMBED_MODEL=sentence-transformers/all-mpnet-base-v2
export EMBEDDING_DIMS=768
python chat.py
```

## Code Overview (chat.py)
- **Config block**: Sets Mem0 vector store (Qdrant), LLM (Ollama), and embedder (Hugging Face on CPU).
- **Mem0 + Ollama clients**: `Memory.from_config(...)` and `ollama.Client(...)`.
- **Chat flow**:
  1. `mem.search` to retrieve relevant memories
  2. `get_ai_response` to generate with Ollama (injects memory context)
  3. `mem.add` to store updated facts
- **UI**:
  - Chat tab with streaming status indicator and scrollable message history.
  - Mem0 Activity tab that groups retrievals and actions per request.
  - Clear memories button (calls `mem.delete_all` for the current user ID).

## Troubleshooting
- **No response**: Ensure Ollama server is running and the model is pulled.
- **Empty memory activity**: Confirm Qdrant is running (`docker compose ps`).
- **Embedding errors**: Ensure the embedding model and `EMBEDDING_DIMS` match.
