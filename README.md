# Mem0 NiceGUI Chat

A local-first chat UI that uses **Mem0** for long-term memory, **Qdrant** for vector storage, **Neo4j** for graph memory, **Hugging Face embeddings on CPU**, and the **OpenAI Python client** pointed at **Ollama’s OpenAI-compatible API** for generation. The UI is built with **NiceGUI** and provides a separate Mem0 Activity tab that groups retrieved memories, actions, and graph relations by request.

## What This Does
- Chat with a local model via Ollama using the Responses API when available; assistant replies are neutral paraphrases of the latest user message.
- Store and retrieve memories with Mem0 + Qdrant.
- Capture and query relationships with Mem0 Graph + Neo4j.
- Use a local CPU embedding model (SentenceTransformers).
- Inspect Mem0 retrievals, add/update/delete actions, and graph relations per request.
- Clear memories for the current user from the UI.

## Mem0 Behavior In This App
- Retrieval: each user message calls `mem.search`. The vector store uses embeddings of the full query to run a similarity search in Qdrant.
- Graph retrieval: Mem0 Graph uses the LLM to extract entities from the query, embeds each entity string, and performs a Neo4j vector similarity search (`n.embedding` cosine). Results are filtered by the graph threshold (default `0.7`) and then BM25 re-ranked against the query tokens before returning the top relations.
- Update: after the assistant response is rendered, the app calls `mem.add` with the user message only. Mem0 extracts facts and relations with the LLM, upserts memory vectors into Qdrant, embeds entity names for graph node merge, and writes relations to Neo4j.
- Note: the assistant response is not added to memory unless you change the code.

## Quickstart

### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Torch note: If `pip install torch` fails, install the CPU wheel from the official PyTorch instructions for your OS.

### 2) Start Qdrant + Neo4j
```bash
docker compose up -d
```
Neo4j defaults to `neo4j / mem0demo123` unless you override `NEO4J_AUTH` in `docker-compose.yaml`.

### 3) Start Ollama
First download and install it (https://ollama.com/download)
```bash
ollama serve
```
Pull a model and set `OLLAMA_MODEL` accordingly. Example:
```bash
ollama pull llama3.1
export OLLAMA_MODEL=llama3.1
```

### 4) Run the app
```bash
python chat.py
```

Open the URL printed by NiceGUI (usually `http://localhost:8080`).

## Usage
- Enter sends a message.
- Shift+Enter inserts a newline.
- Switch to Mem0 Activity to see grouped retrievals, memory actions, and graph updates per request.
- Click Clear memories to delete memories for the current user.

## Configuration
Set these env vars as needed:

- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OPENAI_BASE_URL` (default: `${OLLAMA_BASE_URL}/v1`) — OpenAI-compatible API endpoint
- `OPENAI_API_KEY` (default: `ollama`) — required by the OpenAI client; any string works for local Ollama
- `OLLAMA_MODEL` (default: `huihui_ai/qwen3-next-abliterated:80b-a3b-instruct-q4_K_M`)
- `HF_EMBED_MODEL` (default: `nomic-ai/nomic-embed-text-v1`)
- `EMBEDDING_DIMS` (default: `768`) — must match the embedding model
- `MEM0_USER_ID` (default: `local_user`)
- `NEO4J_URL` (default: `bolt://localhost:7687`)
- `NEO4J_USERNAME` (default: `neo4j`)
- `NEO4J_PASSWORD` (default: `mem0demo123`)
- `MEM0_GRAPH_PROMPT` (optional) — guide which relations are extracted
- `HF_TOKEN` (optional) — avoids Hugging Face rate limits

Example:
```bash
export OLLAMA_MODEL=llama3.1
export HF_EMBED_MODEL=sentence-transformers/all-mpnet-base-v2
export EMBEDDING_DIMS=768
python chat.py
```

## Structured Outputs
The app asks the model for a structured response using a JSON schema via the Responses API. The result is parsed and only the final answer is displayed. If the Responses API is unavailable or the output cannot be parsed, the app falls back to a standard chat completion response.

## Code Overview
- `chat.py`: UI entrypoint.
- `ui_app.py`: NiceGUI layout, chat handlers, and Mem0 activity UI.
- `llm.py`: Responses API calls, structured output parsing, and neutral paraphrasing prompts.
- `clients.py`: OpenAI + Mem0 client construction.
- `config.py`: Env settings and Mem0 configuration.
- `mem0_patch.py`: Runtime patch for Mem0 graph extraction behavior.
- `mem0_models.py`: Pydantic models for Mem0 responses.
- `utils.py`: Formatting and normalization helpers.
- `cli.py`: Optional CLI loop for terminal chat (import `chat_loop`).

## Troubleshooting
- No response: Ensure Ollama is running and the model is pulled.
- Empty memory activity: Confirm Qdrant and Neo4j are running (`docker compose ps`).
- Embedding errors: Ensure the embedding model and `EMBEDDING_DIMS` match.
- Responses API not available: Update Ollama or set `OPENAI_BASE_URL` to an OpenAI-compatible server that supports `/v1/responses`.
