# Spring AI RAG POC

A Retrieval-Augmented Generation proof of concept: upload documents, embed them
into a vector database, and (from Layer 3 onward) ask Claude questions grounded
in their contents.

**Build status:** All three layers complete and verified end-to-end, with 42
passing tests and a web UI. **No API key required** — the default chat provider
is Ollama, running locally and free. Anthropic and OpenAI are one profile flag away.

## Documentation

| Doc | Contents |
|---|---|
| **[docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md)** | **Start here.** First principles — what a vector is, what RAG is, why each piece exists, and one question traced end to end |
| [docs/LAYER-1-INGESTION.md](docs/LAYER-1-INGESTION.md) | Full walkthrough of the ingestion pipeline — diagrams, stage-by-stage data flow, code analysis, chunk-size tradeoffs, and the traps |
| [docs/LAYER-2-RETRIEVAL.md](docs/LAYER-2-RETRIEVAL.md) | Semantic search — query flow, measured similarity scores, threshold calibration, and why vector search always returns something |
| [docs/LAYER-3-CHAT.md](docs/LAYER-3-CHAT.md) | Grounded chat — the three providers, how the advisor rewrites your prompt, citations, and why model size decides whether grounding holds |

---

## Architecture

| Concern | Choice | Why |
|---|---|---|
| Framework | Spring Boot 4.1.2 + Spring AI 2.0.2 | Spring AI 2.x requires Boot 4.x / Framework 7 |
| Generation | **Ollama `llama3.2:3b`** (default) · Anthropic · OpenAI | One profile flag switches provider; retrieval is identical in all three |
| Embeddings | `all-MiniLM-L6-v2` via ONNX, in-process | **Anthropic has no embeddings endpoint.** This runs locally: no API key, no cost, works offline. 384 dimensions. |
| Vector store | PGVector (Postgres 17 + HNSW) | Real vector DB, inspectable with plain SQL |
| Parsing | Apache Tika | PDF, DOCX, PPTX, HTML + ~20 more formats |

**The whole stack runs locally and free by default** — local embeddings, local
vector DB, local LLM. No API key anywhere.

### Switching provider

```bash
./mvnw spring-boot:run                                    # ollama (default, free)
./mvnw spring-boot:run -Dspring-boot.run.profiles=anthropic
./mvnw spring-boot:run -Dspring-boot.run.profiles=openai
```

`GET /api/info` reports which provider actually won the wiring. The UI shows it
as a badge in the top-right.

> ⚠️ Keep `spring.ai.model.embedding: transformers`. The OpenAI and Ollama
> starters ship their own embedding models; without the pin one could supply a
> different dimension than the 384 the pgvector table was built for.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| JDK 21 | Installed at `/usr/lib/jvm/java-21-openjdk-amd64` |
| Docker + Compose | For PGVector |
| Ollama | For the default chat provider. `ollama serve`, then the model auto-pulls (~2GB) |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | **Only** if you switch off the Ollama default |

### ⚠️ JAVA_HOME gotcha

`JAVA_HOME` on this machine points at **JDK 17**, but the project targets 21.
Maven follows `JAVA_HOME`, not your PATH, and fails with
`release version 21 not supported`. Fix it permanently:

```bash
echo 'export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64' >> ~/.bashrc
source ~/.bashrc
```

Every command below assumes this is done. If you skip it, prefix each Maven
command with `JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64`.

---

## Running it

### 1. Start PGVector

```bash
docker compose up -d
```

Wait for health (~6s):

```bash
docker compose ps
```

### 2. Start the application

```bash
./mvnw spring-boot:run
```

Ready when you see `Started RagApplication` (~7 seconds).

**Open <http://localhost:8080> for the web UI** — drag-and-drop multi-file
upload, a document library with chunk counts, and a chat panel that shows the
retrieved sources and similarity score behind every answer. The REST endpoints
below remain available for scripting.

> Layers 1–2 need **no credentials**. For Layer 3, put your key in
> `config/application.yml` (gitignored, read automatically — see
> `config/application.yml.example`). Prefer that over `export`, which only
> applies to the shell it was typed in and will not reach an app started
> elsewhere.

### 3. Ingest a document

```bash
curl -X POST -F "file=@sample-docs/company-handbook.txt" \
  http://localhost:8080/api/documents
```

```json
{"source":"company-handbook.txt","chunks":4}
```

### 4. Verify what was stored

```bash
docker exec rag-pgvector psql -U rag -d ragdb -c \
  "SELECT metadata->>'source' AS source, length(content) AS chars,
          vector_dims(embedding) AS dims FROM vector_store;"
```

### 5. Stop

```bash
# Ctrl+C the app, then:
docker compose down          # keeps data
docker compose down -v       # also wipes the database
```

---

## Experimenting with chunk size

Chunk size is the highest-leverage knob in RAG. Override it without editing files:

```bash
./mvnw spring-boot:run -Dspring-boot.run.arguments="--rag.chunking.chunk-size=300"
```

Clear the table between runs so you compare like with like:

```bash
docker exec rag-pgvector psql -U rag -d ragdb -c "TRUNCATE vector_store;"
```

The sample handbook is ~450 tokens, so at the default `chunk-size: 800` it stays
a **single** chunk. Drop to 120 and it splits into 4.

---

## The ONNX model

Two files in `models/all-MiniLM-L6-v2/` are gitignored and **must exist** for
the app to start offline:

| File | Size | Source |
|---|---|---|
| `model.onnx` | 87 MB | Git LFS — must be downloaded once |
| `tokenizer.json` | 695 KB | Extractable straight from the jar |

**Both** are fetched from GitHub on every startup at default settings, into a
*randomly named* temp directory — so each restart re-downloads them and fails
if the network blips. `application.yml` pins both to local copies.

The `model.onnx` inside `spring-ai-transformers-2.0.2.jar` is a **133-byte Git
LFS pointer**, not the model. `tokenizer.json` in the jar *is* real, but Spring
AI's default URI still points at `raw.githubusercontent.com`, so it downloads
anyway.

To regenerate on a fresh machine:

```bash
# tokenizer.json — straight out of the jar, no network
unzip -o -j ~/.m2/repository/org/springframework/ai/spring-ai-transformers/2.0.2/spring-ai-transformers-2.0.2.jar \
  onnx/all-MiniLM-L6-v2/tokenizer.json -d models/all-MiniLM-L6-v2/
```

For `model.onnx`: comment out `spring.ai.embedding.transformer.onnx.model-uri`,
run `./mvnw spring-boot:run` once with network access, then
`cp /tmp/spring-ai-model-cache*/*/model.onnx models/all-MiniLM-L6-v2/` and
uncomment the property.

Verify integrity — it should match the LFS pointer's `oid`:

```bash
sha256sum models/all-MiniLM-L6-v2/model.onnx
# e3dde332c13808c718680e7bf74a574e7e5d06f55bd6e1527e51509dcb8206f3
```

---

## Config reference

All in `src/main/resources/application.yml`.

| Property | Value | Notes |
|---|---|---|
| `spring.ai.anthropic.chat.model` | `claude-opus-4-8` | In Spring AI 2.x the key is `chat.model`, **not** `chat.options.model`. The 1.x form fails *silently* and leaves you on the `claude-haiku-4-5` default. |
| `spring.ai.vectorstore.pgvector.dimensions` | `384` | Must match the embedding model. PGVector defaults to 1536 (OpenAI-shaped); a mismatch fails at **insert** time, not startup, so it looks like an ingestion bug. |
| `spring.ai.vectorstore.pgvector.initialize-schema` | `true` | Auto-creates the table and extensions. Fine for a POC; use Flyway in production. |
| `rag.chunking.chunk-size` | `800` | Tokens per chunk. |
| `rag.retrieval.top-k` | `4` | Chunks fed to Claude. Also a **cost** lever — prompt size scales with it. |
| `rag.retrieval.similarity-threshold` | `0.1` | Calibrated for MiniLM in Layer 2. **Not portable** to other embedding models. |
| `rag.retrieval.max-full-document-chunks` | `40` | Caps the FULL_DOCUMENT prompt. Exceeding it sets `truncated: true` — never silent. |
| `rag.intent.overview-examples` | 12 examples | Example questions wanting a whole document. **Edit for your domain** — they are compared by meaning, not matched literally. |
| `rag.intent.specific-examples` | 10 examples | Example questions wanting one fact. |
| `rag.intent.min-overview-similarity` | `0.35` | Floor below which a question resembles neither list and defaults to the safe `TOP_K` path. |

Changing the embedding model means changing `dimensions` **and** re-ingesting —
vectors from different models are not comparable.

---

## Tests

```bash
./mvnw test
```

42 tests, no API key needed. `RagPipelineIntegrationTest` runs against the real
PGVector instance (so `docker compose up -d` first) and asserts the Layer 2
findings directly — semantic match scores above 0.3, off-topic below 0.1, the
threshold filtering noise, and idempotent re-ingestion.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `release version 21 not supported` | `JAVA_HOME` points at JDK 17. See above. |
| `Connection timed out` fetching `model.onnx` **or** `tokenizer.json` | A local model file is missing; see *The ONNX model*. Both must be pinned. |
| `AccessDeniedException` under `~/.m2/repository` | Root-owned dirs from a past `sudo mvn`. Fix: `sudo chown -R "$USER:$USER" ~/.m2/repository` |
| `Could not resolve placeholder 'ANTHROPIC_API_KEY'` | Export the key (only required for Layer 3). |
| Search returns irrelevant results | Expected at `threshold=0.0` — vector search always returns its `topK`. Pass `threshold=0.1`. |
| Search returns nothing at all | Threshold too high. MiniLM scores are compressed; 0.75 rejects everything. Try 0.1. |
| `/api/chat` returns 502 `anthropic_api_error` | Read the `hint` field. Usually a missing/invalid key — or switch to the free Ollama default. |
| `/api/chat` hangs ~30–60s on Ollama | Normal on CPU-only inference. The UI shows a typing indicator throughout. |
| Ollama connection refused | `ollama serve` isn't running. Start it, then restart the app. |
| Model answers questions the docs don't cover | Model too small to honor the grounding prompt. `llama3.2:1b` fails this; `3b` passes. See [docs/LAYER-3-CHAT.md](docs/LAYER-3-CHAT.md) §9. |
| A question routes to the wrong strategy | Check `strategyReason` in the response — it shows both scores. Add an example to `rag.intent.*-examples` in `application.yml`; no recompile needed. |
| Asking for an overview gives a partial answer | Name the document ("what's in the **company handbook**"). Unnamed overview requests fall back to top-k rather than guessing which document you meant. |
| Answers cite the wrong document | Retrieval searched everything. Set a `source` scope — naming the document in the question does nothing. |
| `/api/chat` answers with `grounded: false` | Nothing cleared the threshold. Either the docs don't cover it, or the threshold is too high. |
| Model says "not in the context" when it clearly is | `llama3.2:3b` literalism (e.g. question says "pay", document says "stipend"). Retrieval is fine — check `citations`. Larger model or a hosted provider fixes it. |

---

## Project layout

```
compose.yaml                        PGVector service
models/all-MiniLM-L6-v2/            ONNX embedding model (gitignored, 87MB)
sample-docs/company-handbook.txt    Test document
src/main/java/com/example/rag/
  RagApplication.java
  ingest/IngestionService.java      Tika -> TokenTextSplitter -> VectorStore
  ingest/IngestionController.java   POST /api/documents
  search/SearchService.java         VectorStore.similaritySearch
  search/SearchController.java      GET /api/search
  search/ScoredChunk.java           result record (rank, score, excerpt)
  ingest/DocumentLibraryService.java per-source chunk counts (JDBC)
  chat/ChatService.java             ChatClient + QuestionAnswerAdvisor
  chat/ChatController.java          POST /api/chat
  chat/ChatPayloads.java            request/answer/citation records
  chat/RetrievalPlanner.java        picks TOP_K vs FULL_DOCUMENT per question
  chat/IntentClassifier.java        semantic intent via embeddings (no keywords)
  chat/IntentProperties.java        intent examples, bound from application.yml
  chat/InfoController.java          GET /api/info
  chat/ChatExceptionHandler.java    maps Anthropic errors to 502 + hint
src/main/resources/application.yml
src/main/resources/static/        web UI (no build step, no CDN)
  index.html
  styles.css
  app.js
```

## Endpoints

| Method | Path | Status |
|---|---|---|
| `POST` | `/api/documents` | Layer 1 — multipart upload. Field `files` (repeatable) or `file` (single) |
| `GET` | `/api/documents` | List ingested sources with chunk counts |
| `DELETE` | `/api/documents/{source}` | Remove one document and all its chunks |
| `GET` | `/api/search` | Layer 2 — semantic search. Params: `q` (required), `topK` (4), `threshold` (0.0), `source` (scope to one document) |
| `POST` | `/api/chat` | Layer 3 — grounded Q&A. Body: `{"question", "source"?, "topK"?, "threshold"?}`. Retrieval strategy is chosen per question |
| `GET` | `/api/info` | Active chat provider, model, embedding dimensions |

### Searching

```bash
curl -G --data-urlencode "q=how much is on-call pay" \
     --data-urlencode "threshold=0.1" \
     http://localhost:8080/api/search | jq
```

Start with `threshold=0.1` for this embedding model. Thresholds are **not**
portable between models — see [docs/LAYER-2-RETRIEVAL.md](docs/LAYER-2-RETRIEVAL.md) §9.

### Chatting

Requires `ANTHROPIC_API_KEY`.

```bash
curl -sX POST http://localhost:8080/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"how much is on-call pay?"}' | jq
```

Every answer carries `citations` (the chunks that grounded it) and `grounded`.
A `grounded: false` answer means nothing cleared the threshold — distrust it.

