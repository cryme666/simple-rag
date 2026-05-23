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

## RAG Techniques

This section describes the retrieval and generation techniques implemented in this project, how they are wired together, and where they live in the codebase.

### Pipeline overview

The system follows a classic **indexing → retrieval → generation** RAG flow:

1. **Indexing:** documents are parsed, split into chunks, embedded, and stored in PostgreSQL with pgvector.
2. **Retrieval:** each chat message is optionally rewritten, embedded, and matched against the knowledge base using **hybrid search** (dense vectors + sparse BM25), fused with **Reciprocal Rank Fusion (RRF)**.
3. **Generation:** retrieved chunks are injected into a grounded system prompt; Groq generates the final answer using conversation history from the client.

```
Ingest (PDF/URL)
  -> chunk_text()
  -> generate_embeddings_batch()
  -> insert_chunks() + bm25_index.mark_dirty()

Chat (/chat)
  -> split_and_clarify_query() [optional]
  -> generate_embedding()
  -> hybrid_search_rrf()  [vector + BM25 + RRF]
  -> retry / context fallback [optional]
  -> chat_completion() with retrieved context
```

### 1. Document ingestion

Two ingestion paths populate the same `document_chunks` table.

#### PDF ingestion (`POST /ingest/file`)

- **Parser:** `pypdf` extracts text page by page (`app/services/pdf_parser.py`).
- **Chunking:** each page is chunked independently (`app/routers/ingest.py`).
- **Metadata:** page number is stored per chunk (`{"page": N}`).
- **Source identity:** filename; `source_type = "file"`.
- **Overwrite:** if `overwrite=true`, existing chunks for that source are deleted before re-insert.

#### URL ingestion (`POST /ingest/url`)

- **Fetcher:** `httpx` downloads HTML with redirect following (`app/services/web_scraper.py`).
- **Cleanup:** BeautifulSoup removes boilerplate tags (`script`, `style`, `nav`, `footer`, `header`, `aside`, `form`, `noscript`).
- **Text extraction:** body text is normalized to plain text.
- **Chunking:** the full scraped document is chunked as one stream.
- **Metadata:** title and URL are stored per chunk.
- **Source identity:** URL string; `source_type = "url"`.

After every write, the in-memory BM25 index is marked dirty and rebuilt on the next search.

### 2. Sentence-aware chunking with overlap

**File:** `app/utils/chunker.py` → `chunk_text()`

This is a **fixed-size, sentence-boundary chunking** strategy:

- Whitespace is collapsed before splitting.
- Text is split into sentences using punctuation boundaries (`(?<=[.!?])\s+`).
- Sentences are packed until the chunk reaches `chunk_size` characters (default **1000**).
- When a chunk is flushed, **overlap** is created by walking backward through sentences until roughly `chunk_overlap` characters (default **200**) are included in the next chunk.

Why this helps RAG:

- Sentence boundaries reduce mid-sentence cuts that hurt both embeddings and keyword search.
- Overlap preserves context across chunk borders, so answers that span two adjacent chunks are still retrievable.

### 3. Dense retrieval with local embeddings

**File:** `app/services/embedding.py`

- **Model:** `sentence-transformers/all-mpnet-base-v2` (768 dimensions).
- **Inference:** runs locally on CPU inside the backend container.
- **Normalization:** embeddings are L2-normalized (`normalize_embeddings=True`).
- **Batching:** ingest uses batch encoding (batch size 32); chat uses single-query encoding.

Because vectors are normalized, pgvector distance with the `<=>` operator behaves like **cosine distance** for ranking.

**Storage:** `app/models.py` defines `DocumentChunk.embedding` as `Vector(768)`; Alembic migration enables the `vector` extension.

**Search:** `app/services/vector_store.py` → `similarity_candidates()` runs:

```sql
ORDER BY embedding <=> query_embedding
LIMIT top_k
```

There is no approximate nearest-neighbor index (HNSW/IVFFlat) yet, so vector search is an exact top-K sort over stored rows.

### 4. Sparse retrieval with BM25

**File:** `app/services/bm25_index.py`

BM25 complements dense search by matching exact terms, names, and rare keywords that embeddings may miss.

- **Library:** `rank_bm25.BM25Okapi`.
- **Index:** built in memory from all chunks in the database.
- **Invalidation:** any ingest/delete/clear marks the index dirty; it is rebuilt lazily on the next query.
- **Tokenization:**
  - Regex keeps Latin, Cyrillic, and Ukrainian letters/numbers.
  - Tokens are lowercased.
  - English NLTK stopwords are removed.
  - Tokens with length ≤ 2 are dropped.
- **Quality gate:** if the best BM25 score is ≤ 0, BM25 returns no results (avoids noisy matches on empty or weak queries).

NLTK stopwords are prepared at startup (`app/services/nltk_setup.py`) and used **only for BM25**, not for embeddings or vector search.

### 5. Hybrid retrieval with Reciprocal Rank Fusion (RRF)

**File:** `app/services/hybrid_retriever.py` → `hybrid_search_rrf()`

This is the core retrieval technique used by `/chat`.

For each query, two ranked lists are produced:

| Branch | Input | Filtering | Default top-K |
|--------|-------|-----------|---------------|
| **Vector** | query embedding | keep chunks with `distance < distance_threshold` | 5 |
| **BM25** | original query text | keep only if best score > 0 | 5 |

Then **RRF** merges the two lists:

```
RRF_score(chunk) = Σ 1 / (k + rank)
```

with `k = 60` (constant `RRF_K`).

Properties of this fusion:

- Scores from vector distance and BM25 are **not directly comparable**, so rank-based fusion is used instead of weighted averaging.
- Duplicate chunks appearing in both lists get contributions from both ranks.
- Final ordering is by fused score; ties prefer chunks that appeared in the vector branch.
- The top `top_k` fused chunks are returned.

Why hybrid search helps:

- **Dense retrieval** captures semantic similarity and paraphrases.
- **BM25** captures lexical overlap and exact terminology.
- **RRF** combines both without needing manual score calibration between modalities.

### 6. Distance thresholding (vector branch only)

**Setting:** `distance_threshold` (default **0.5**)

After vector search returns top-K candidates, chunks are kept only if:

```
cosine_distance < distance_threshold
```

This reduces irrelevant semantic neighbors. BM25 has its own score gate instead of a distance threshold.

If vector filtering removes everything but BM25 still finds strong keyword matches, those chunks can still appear in the fused result set.

Use `POST /debug/search` to inspect raw candidate distances and tune the threshold for your corpus.

### 7. Query transformation before retrieval

**File:** `app/services/query_transform.py`  
**Orchestration:** `app/routers/chat.py` → `_handle_normal_chat()`

The system uses the LLM **before retrieval** to improve search queries. This is not HyDE (no hypothetical document generation) and not multi-query retrieval over every split question.

#### 7.1 Split + clarify (primary transform)

When enabled and the user message is long enough (`query_transform_min_query_len`, default 8):

1. `split_and_clarify_query()` asks Groq to split compound input into standalone questions.
2. Each question is clarified into a retrieval-friendly form.
3. **Only the first clarified question** is used as `retrieval_message` for embedding.

The LLM transform runs at `temperature=0` and must return strict JSON.

#### 7.2 Retry with the original query

If hybrid search returns **zero chunks** and the transformed query differs from the original user message:

- the system re-embeds the **original** user message
- runs hybrid search again

This protects against over-aggressive rewriting.

#### 7.3 Context-aware fallback rewrite

If retrieval is still empty and conversation history exists:

- `transform_query_with_context()` rewrites follow-up messages (e.g. “What about it?”) into standalone search queries
- it uses up to the last 5 previous **user** messages for disambiguation
- the rewritten query is embedded and hybrid search runs once more

#### Important asymmetry

- **Vector branch** uses the transformed / rewritten / retried embedding.
- **BM25 branch** always searches with the **original** `user_message`.

This keeps keyword matching anchored to the user's actual wording while allowing semantic search to benefit from clearer rewritten queries.

### 8. Context assembly and grounded generation

**File:** `app/services/llm.py` → `chat_completion()`

Retrieved chunk texts are joined with:

```
chunk_1

---

chunk_2
```

If nothing is retrieved, the model receives:

```
No relevant context found in the knowledge base.
```

#### Prompting strategy

The system prompt (`RAG_SYSTEM_PROMPT`) instructs the model to:

- answer **only** from the provided `<context>`
- admit when the context is insufficient
- avoid inventing facts outside retrieved material

Message order sent to Groq:

1. system prompt with injected context
2. full client-provided conversation history
3. current user message

Generation defaults:

- model: `llama-3.3-70b-versatile`
- temperature: `0.7`
- max tokens: `2048`

#### Source attribution

Sources returned to the client are deduplicated by `(source, source_type)` from retrieved chunks. They are **not** passed as structured citations inside the LLM prompt; the model sees raw chunk text only.

### 9. Conversation history model

There is **no server-side chat memory**. The frontend sends `conversation_history` on every `/chat` request.

History usage by stage:

| Stage | Uses history? |
|-------|---------------|
| Split + clarify transform | No |
| BM25 search | No |
| Context fallback rewrite | Yes (previous user messages only) |
| Answer generation | Yes (full user/assistant history) |

### 10. Configuration reference for RAG behavior

These settings in `app/config.py` control retrieval quality:

| Variable | Default | Purpose |
|----------|---------|---------|
| `chunk_size` | `1000` | Max characters per chunk |
| `chunk_overlap` | `200` | Sentence overlap between chunks |
| `top_k` | `5` | Final number of retrieved chunks |
| `distance_threshold` | `0.5` | Vector distance cutoff |
| `bm25_top_k` | `5` | BM25 candidate pool size |
| `query_transform_enabled` | `true` | Master switch for query rewriting |
| `query_transform_split_clarify_enabled` | `true` | Enable split + clarify |
| `query_transform_fallback_with_context_enabled` | `true` | Enable conversational fallback rewrite |
| `query_transform_min_query_len` | `8` | Skip transform for very short queries |
| `query_transform_fallback_max_prev_user_messages` | `5` | History window for fallback rewrite |

Hardcoded constants:

- `EMBEDDING_DIMENSION = 768` in `app/models.py`
- `RRF_K = 60` in `app/services/hybrid_retriever.py`

### 11. Techniques intentionally not implemented

To avoid confusion, these common RAG extensions are **not** present in this codebase:

- HyDE (hypothetical document embeddings)
- Multi-query retrieval over all split questions
- Cross-encoder or LLM reranking after RRF
- pgvector HNSW / IVFFlat indexes
- Metadata-filtered retrieval (by source, type, or date)
- Server-side session memory

The current design prioritizes a clear, inspectable baseline: **sentence chunking + local embeddings + BM25 hybrid fusion + LLM query rewriting + grounded generation**.

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
| `CHUNK_SIZE` | Max characters per chunk | `1000` |
| `CHUNK_OVERLAP` | Sentence overlap between chunks | `200` |
| `TOP_K` | Retrieved chunks after fusion | `5` |
| `DISTANCE_THRESHOLD` | Vector distance cutoff | `0.5` |
| `BM25_TOP_K` | BM25 candidate pool size | `5` |
| `QUERY_TRANSFORM_ENABLED` | Enable query rewriting | `true` |

See [RAG Techniques](#rag-techniques) for how these settings affect retrieval and generation.

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
