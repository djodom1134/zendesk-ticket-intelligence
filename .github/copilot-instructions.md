# Zendesk Ticket Intelligence (ZTI) - AI Coding Instructions

## Project Overview

ZTI is a **local-first ticket intelligence system** built on NVIDIA's txt2kg playbook. It ingests Zendesk tickets, creates embeddings, clusters them into "problem families", and provides visualization + Tier-0 chatbot capabilities.

**Architecture**: Zendesk MCP → Ingest → Normalize → Summarize → Embed → Cluster → Label → API → UI

## Key Technologies

- **Graph DB**: ArangoDB (tickets, clusters, knowledge graph)
- **Vector DB**: Qdrant (embeddings) - chosen for ARM compatibility over Pinecone
- **LLM**: Ollama with GPU acceleration (summary: `gpt-oss:120b`, embed: `qwen3-embedding:8b`)
- **Backend**: FastAPI (`services/api/main.py`)
- **ML**: UMAP + HDBSCAN for clustering, sentence-transformers fallback

## Service Architecture

```
services/
├── ingest/        # Zendesk MCP client - SSE transport, JSON-RPC
├── normalize/     # Raw → TicketDocument (shared/schemas/ticket.py)
├── embed_cluster/ # Summarize → Embed (4096-dim) → UMAP → HDBSCAN cluster
├── pipeline/      # Orchestrates: run_pipeline.py (batch), incremental.py (delta)
├── api/           # FastAPI serving clusters, search, Tier-0 chat
└── chat/          # Tier-0 prototype with RAG
```

## Critical Patterns

### Schema-First Design
All ticket data flows through `shared/schemas/ticket.py:TicketDocument`. Always use Pydantic models:
```python
from shared.schemas.ticket import TicketDocument, TicketStatus
```

### Ollama Integration
Embeddings and LLM calls use `httpx` async with Ollama's REST API:
```python
async with httpx.AsyncClient(timeout=120.0) as client:
    response = await client.post(f"{OLLAMA_URL}/api/embed", json={"model": EMBED_MODEL, "input": text})
```

### Configuration via Environment
All service configs read from environment variables with sensible defaults. See `.env.example` for required vars.

### Lazy Model Initialization
Heavy ML models (UMAP, HDBSCAN) use lazy init pattern - see `clusterer.py:_init_models()`.

## Development Commands

```bash
# Start infrastructure
docker compose -f docker/docker-compose.yml up -d arangodb qdrant ollama

# Run full pipeline (on GPU machine)
docker compose exec pipeline python -m services.pipeline.run_pipeline -i /data/raw/tickets.json

# Ingest from Zendesk MCP
docker compose exec pipeline python -m services.ingest.cli fetch --days 90

# Run tests
pytest tests/ -v --cov=services

# Lint/format
ruff check --fix . && ruff format .
```

## Known Issues & Gotchas

1. **GPU Fallback**: Ollama runs CPU unless `OLLAMA_LLM_LIBRARY=cuda_v13` and GPU reservations in compose
2. **Path Typos**: txt2kg submodule lives at `dgx-spark-playbooks/nvidia/txt2kg` - verify path exists
3. **Healthcheck**: Ollama healthcheck uses `ollama list` not `curl` (curl missing in image)
4. **Embedding Dimensions**: `qwen3-embedding:8b` produces 4096-dim vectors, not 384

## File Conventions

- **Imports**: Use `from services.X import` and `from shared.schemas import` for cross-service
- **Logging**: Use `structlog` consistently: `logger = structlog.get_logger()`
- **Async**: API endpoints are async, pipeline batch ops are sync
- **Data Files**: Pipeline outputs to `/data/zti/` (normalized.json, summaries.json, embeddings.json, clusters.json)

## Testing Strategy

- Unit tests in `tests/` - run with `pytest`
- Integration tests require running infrastructure
- Pre-commit hooks enforce ruff linting
