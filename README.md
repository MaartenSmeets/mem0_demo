# Mem0 NiceGUI Chat

A local‑first chat UI that uses **Mem0** for long‑term memory, **Qdrant** for vector storage, **Neo4j** for graph memory, **Hugging Face embeddings on CPU**, and the **OpenAI Python client** pointed at **Ollama’s OpenAI‑compatible API** for generation. The UI is built with **NiceGUI** and provides a separate Mem0 Activity tab that groups retrieved memories, actions, and graph relations by request.

## What This Does
- Chat with a local model via Ollama.
- Store and retrieve memories with Mem0 + Qdrant.
- Capture and query relationships with Mem0 Graph + Neo4j.
- Use a local CPU embedding model (SentenceTransformers).
- Inspect Mem0’s retrievals and add/update/delete actions per request.
- Inspect graph relations retrieved and written per request.
- Clear memories for the current user from the UI.

## Quickstart

### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Torch note**: If `pip install torch` fails, install the CPU wheel from the official PyTorch instructions for your OS.

### 2) Start Qdrant + Neo4j
```bash
docker compose up -d
```
Neo4j defaults to `neo4j / mem0demo123` unless you override `NEO4J_AUTH` in `docker-compose.yaml`.

### 3) Start Ollama
First download and install it (https://ollama.com/download)
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
- `OPENAI_BASE_URL` (default: `${OLLAMA_BASE_URL}/v1`) — OpenAI‑compatible API endpoint
- `OPENAI_API_KEY` (default: `ollama`) — required by the OpenAI client, any string works for local Ollama
- `OLLAMA_MODEL` (default: `mistral-small3.2`)
- `HF_EMBED_MODEL` (default: `nomic-ai/nomic-embed-text-v1`)
- `EMBEDDING_DIMS` (default: `768`) — must match the embedding model
- `MEM0_USER_ID` (default: `local_user`)
- `NEO4J_URL` (default: `bolt://localhost:7687`)
- `NEO4J_USERNAME` (default: `neo4j`)
- `NEO4J_PASSWORD` (default: `mem0demo123`)
- `NEO4J_DATABASE` (default: `neo4j`)
- `MEM0_GRAPH_PROMPT` (optional) — guide which relations are extracted
- `HF_TOKEN` (optional) — avoids Hugging Face rate limits

Example:
```bash
export OLLAMA_MODEL=llama3.1
export HF_EMBED_MODEL=sentence-transformers/all-mpnet-base-v2
export EMBEDDING_DIMS=768
python chat.py
```

## Code Overview (chat.py)
- **Config block**: Sets Mem0 vector store (Qdrant), graph store (Neo4j), LLM (OpenAI client → Ollama API), and embedder (Hugging Face on CPU).
- **Mem0 + OpenAI clients**: `Memory.from_config(...)` and `openai.OpenAI(...)`.
- **Chat flow**:
  1. `mem.search` to retrieve relevant memories
  2. `get_ai_response` to generate with Ollama (injects memory context)
  3. `mem.add` to store updated facts
- **UI**:
  - Chat tab with streaming status indicator and scrollable message history.
  - Mem0 Activity tab that groups retrievals, actions, and graph relations per request.
  - Clear memories button (calls `mem.delete_all` for the current user ID).

## Troubleshooting
- **No response**: Ensure Ollama server is running and the model is pulled.
- **Empty memory activity**: Confirm Qdrant and Neo4j are running (`docker compose ps`).
- **Embedding errors**: Ensure the embedding model and `EMBEDDING_DIMS` match.
