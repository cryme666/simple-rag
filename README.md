# Simple RAG

Retrieval-Augmented Generation (RAG) system with:

- **Backend:** FastAPI + PostgreSQL/pgvector + Alembic
- **LLM:** Groq (`llama-3.3-70b-versatile`)
- **Embeddings:** local `sentence-transformers/all-mpnet-base-v2` (768 dim)
- **Retrieval:** hybrid search (pgvector + BM25 + RRF)
- **Frontend:** React + Tailwind + OpenAPI-generated client

## Architecture

```
User -> React UI (:3000)
         -> OpenAPI client -> FastAPI (:8000)
                                   -> SentenceTransformer (local)
                                   -> Groq API
                                   -> PostgreSQL + pgvector
```

## Quick Start (Docker)

### 1. Configure environment

```bash
cp .env.example .env
```

Set at minimum:

```env
GROQ_API_KEY=your_groq_api_key
```

### 2. Start the stack

```bash
docker compose up --build
```

Services:

| Service  | URL |
|----------|-----|
| Frontend | http://localhost:3000 |
| Backend  | http://localhost:8000 |
| API docs | http://localhost:8000/docs |

Postgres runs inside Docker network only (not exposed on host by default).

### 3. Use the app

1. Open http://localhost:3000
2. Ingest a PDF or URL in the sidebar
3. Ask questions in the chat panel
4. Sources appear under assistant answers when retrieval finds relevant chunks

## Local Development (without Docker)

### Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt

cp .env.example .env
# Set GROQ_API_KEY and DATABASE_URL (e.g. localhost:5433 if using external postgres)

docker compose up -d postgres   # or your own Postgres with pgvector
alembic upgrade head
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
cp .env.example .env
npm install
npm run generate:api   # requires backend running on :8000
npm run dev
```

Frontend dev server: http://localhost:5173

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/chat` | RAG chat |
| POST | `/ingest/file` | Ingest PDF |
| POST | `/ingest/url` | Ingest URL |
| DELETE | `/ingest/clear` | Clear knowledge base |
| POST | `/debug/search` | Debug vector search (tune `distance_threshold`) |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key | required |
| `GROQ_MODEL` | Groq chat model | `llama-3.3-70b-versatile` |
| `EMBEDDING_MODEL` | SentenceTransformer model | `sentence-transformers/all-mpnet-base-v2` |
| `DATABASE_URL` | Async PostgreSQL URL | see `.env.example` |

## Migrations

Schema is managed by Alembic only (no `create_all` on startup).

```bash
alembic upgrade head
```

After changing embedding model/dimension, clear and re-ingest:

```bash
curl -X DELETE http://localhost:8000/ingest/clear
```

## Notes

- **Embedding dimension:** 768 (`all-mpnet-base-v2`). Old 3072-dim data is incompatible.
- **`distance_threshold`:** default `0.5`. Use `/debug/search` to calibrate for your corpus.
