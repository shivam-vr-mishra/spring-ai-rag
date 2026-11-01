# How It All Works

A first-principles explanation of this project: what a vector is, what RAG is,
why each piece exists, and how they fit together.

Every number here is measured from this running system — not a generic
example. Where something is a simplification, it says so.

**Reading order:** this document explains the *concepts*. The three layer
documents explain the *implementation*:
[Layer 1 — Ingestion](LAYER-1-INGESTION.md) ·
[Layer 2 — Retrieval](LAYER-2-RETRIEVAL.md) ·
[Layer 3 — Chat](LAYER-3-CHAT.md)

---

## Table of contents

1. [The problem RAG solves](#1-the-problem-rag-solves)
2. [What an embedding actually is](#2-what-an-embedding-actually-is)
3. [Measuring similarity](#3-measuring-similarity)
4. [What a vector database does](#4-what-a-vector-database-does)
5. [Why chunking exists](#5-why-chunking-exists)
6. [The two phases](#6-the-two-phases)
7. [One question, end to end](#7-one-question-end-to-end)
8. [Concepts mapped to code](#8-concepts-mapped-to-code)
9. [The failure modes](#9-the-failure-modes)
10. [Why these technology choices](#10-why-these-technology-choices)
11. [Glossary](#11-glossary)

---

## 1. The problem RAG solves

An LLM knows what was in its training data. It does not know your company
handbook, last week's incident report, or your customer records.

There are four ways to fix that. Three are usually wrong:

| Approach | How | Why it usually fails |
|---|---|---|
| **Ask anyway** | Just ask the model | It invents an answer. Confident and wrong. |
| **Fine-tune** | Retrain the model on your documents | Expensive, slow, needs redoing on every change, and teaches *style* far better than *facts* |
| **Paste everything** | Put all documents in the prompt | Hits the context limit, costs a fortune per call, and accuracy *drops* as irrelevant text crowds the prompt |
| **RAG** ✅ | Find the few relevant passages, put only those in the prompt | Cheap, instant to update, and the model can cite what it used |

**RAG = Retrieval-Augmented Generation.** Three words, in order:

```
  RETRIEVAL    find the passages relevant to this question
  AUGMENTED    add them to the prompt
  GENERATION   let the model answer using them
```

The crucial insight, and the one most people miss:

> **The intelligence is in the retrieval, not the generation.**

If retrieval hands the model the wrong passages, no model — however capable —
can produce a right answer. Most "the AI hallucinated" bugs are retrieval
bugs.

### The hard part

"Find the relevant passages" sounds like search, and search sounds solved.
It isn't, because keyword search fails on the normal way people ask questions.

A user asks: **"how much do I get paid for being on call?"**

The handbook says: *"Engineers on the primary on-call rotation **receive a
stipend** of 800 dollars per week."*

The word "paid" **never appears**. `SELECT ... WHERE content LIKE '%paid%'`
returns nothing. The information is right there, and keyword search cannot see
it.

What's needed is search by *meaning*. That is what embeddings provide.

---

## 2. What an embedding actually is

An **embedding** is a list of numbers that represents a piece of text's
meaning. In this project, every chunk of text becomes **384 numbers**:

```
"Engineers on the primary on-call rotation receive a stipend of 800 dollars..."
                              │
                              ▼
                   ┌──────────────────────┐
                   │  all-MiniLM-L6-v2    │   a small neural network
                   │  (ONNX, in-process)  │   87 MB, runs locally
                   └──────────────────────┘
                              │
                              ▼
  [-0.16015553, -0.12794536,  0.12585087, -0.042256076,  0.08593752,
   -0.17796439, -0.055528745, 0.1418549,  -0.13686329,   0.04582468, ... ]
                        ↑
            384 floats — actual values from this database
```

Storage: **1,540 bytes** per vector — 384 × 4-byte float32 = 1,536, plus a
4-byte header (`pg_column_size`, measured).

### What do the 384 numbers mean?

Individually: nothing human-readable. Nobody can tell you what dimension 203
represents. They are not "topic" or "sentiment" slots.

Collectively: a **position**. The model was trained so that text with similar
meaning lands at nearby positions, and unrelated text lands far apart.

Think of it as a map. On a real map, two numbers (latitude, longitude) place a
city, and cities near each other have similar coordinates. An embedding does
the same with 384 numbers instead of 2, placing *meanings* instead of cities.

```
        A 384-dimensional space, flattened to 2D for illustration

                 ●  "deployment policy"
                  ╲
                   ╲  small angle  =  similar meaning
                    ╲
                 ●  "when can we ship code?"


                                    ●  "robot battery life"
                                       far away = unrelated
```

> **Simplification:** you cannot actually draw 384 dimensions. The 2D picture
> conveys *near/far*, which is the part that matters. Human intuition about
> distance mostly survives the jump to high dimensions; intuition about
> *volume* and *angles* does not.

### Why 384?

It's the output size of `all-MiniLM-L6-v2`, the model this project uses.
Different models produce different sizes:

| Model | Dimensions | Runs |
|---|---|---|
| **all-MiniLM-L6-v2** (ours) | **384** | locally, in-process, free |
| nomic-embed-text | 768 | locally, via Ollama |
| text-embedding-3-small | 1536 | OpenAI API, paid |

More dimensions can capture finer distinctions, but cost more storage and
compute. 384 is plenty for a POC.

> **Critical:** vectors from different models are **not comparable**. They are
> different coordinate systems. Swapping the embedding model means re-embedding
> every document — and the `vector_store` column is typed `vector(384)`, so a
> mismatch is a hard error.

### Anthropic has no embeddings API

Worth stating plainly, because it shapes this project's whole architecture:
Claude generates text but does not produce embeddings. So RAG with Claude
always needs a **second** provider for the embedding half. This project uses a
local ONNX model, which is why ingestion and search need no API key at all.

---

## 3. Measuring similarity

Once text is a position in space, "find relevant passages" becomes "find
nearby positions". Nearness is measured by the **angle** between two vectors —
**cosine similarity**.

```
  cosine similarity = 1.0   identical direction  (same meaning)
                    = 0.0   perpendicular        (unrelated)
                    = -1.0  opposite direction
```

Angle is used rather than straight-line distance because it ignores magnitude.
A one-sentence summary and a three-paragraph passage about the same topic
point the same *way*, even though one vector is "longer".

### Distance vs score

PostgreSQL's `pgvector` extension provides the `<=>` operator, which returns
cosine **distance**. This project converts it to a **score**:

```
  score = 1 − cosine_distance
```

Measured directly — chunk #3 of the handbook compared against all 8 chunks:

```
 chunk | cosine_distance | score
-------+-----------------+--------
 3     |          0.0000 | 1.0000   ← itself: zero distance, perfect score
 1     |          0.5528 | 0.4472
 0     |          0.6916 | 0.3084
 2     |          0.8577 | 0.1423
```

So a `similarityThreshold` of 0.1 means "cosine distance below 0.9".

### Real similarities between real chunks

Every pair of the 8 stored chunks, top 6 by similarity:

```
    chunk A         chunk B          similarity
 ------------------------------------------------
 security-pol#1  security-pol#0        0.663   ← adjacent chunks, same doc
 security-pol#0  company-hand#0        0.502   ← both are policy preambles
 security-pol#1  security-pol#2        0.500   ← adjacent chunks, same doc
 company-hand#2  company-hand#0        0.458
 company-hand#1  company-hand#3        0.447
 security-pol#0  company-hand#2        0.402
```

Note what the model learned without being told: **adjacent chunks of the same
document are the most similar**, and the two documents' opening sections
resemble each other (both are corporate-policy preambles) more than they
resemble their own later sections. Nobody labelled any of this.

### Scores are relative, not absolute

This is the single most common RAG configuration mistake. A measured spread
from this system:

```
  0.0        0.1        0.3         0.5              1.0
   │──────────│──────────│───────────│────────────────│
   │  noise   │   weak   │   good    │     strong     │
              ▲                       ▲
        pizza: 0.027            on-call: 0.53
```

A threshold of **0.75** — a figure that appears in plenty of OpenAI-based
tutorials — would reject **every result in this project**, including the 0.53
bullseye. It would look like "my RAG is broken". The cause would be a constant
copied from a different embedding model.

Calibrate on your own model and corpus, always.

---

## 4. What a vector database does

Storing 384 numbers is easy — any database can do that. The hard part is:

> Given a query vector, find the nearest of **N** stored vectors, fast.

Compared one at a time, that's O(n): fine for 8 chunks, hopeless for 8 million.

### The index

This project uses **HNSW** — *Hierarchical Navigable Small World* — a layered
graph where upper layers are sparse for coarse jumps and lower layers dense for
refinement:

```
  Layer 2    ●─────────────────────●              few nodes, long hops
             │                     │
  Layer 1    ●────────●────────●───●              medium density
             │        │        │   │
  Layer 0    ●──●──●──●──●──●──●──●──●──●──●      every vector

             ▲
             └─ search enters at the top, greedily hops toward the query
                vector, then descends and refines
```

| | Exact scan | HNSW |
|---|---|---|
| Complexity | O(n) | ~O(log n) |
| Recall | 100% | ~95–99% |
| 8 rows | instant | instant |
| 8M rows | seconds | milliseconds |

**HNSW is approximate** — it trades a little recall for a lot of speed. At this
scale you'll never see a miss; at millions of rows, a relevant chunk will
occasionally not surface. That is the deliberate bargain.

Confirmed in the live query plan:

```
 Limit
   ->  Index Scan using spring_ai_vector_index on vector_store
         Order By: (embedding <=> $1)
```

`Index Scan`, not `Seq Scan` — the graph is doing the work.

### The actual table

```sql
                     Table "public.vector_store"
  Column   |    Type     | Nullable |      Default
-----------+-------------+----------+--------------------
 id        | uuid        | not null | uuid_generate_v4()
 content   | text        |          |   ← the original text
 metadata  | json        |          |   ← source filename, chunk index
 embedding | vector(384) |          |   ← the 384 numbers
Indexes:
    "vector_store_pkey" PRIMARY KEY, btree (id)
    "spring_ai_vector_index" hnsw (embedding vector_cosine_ops)
```

**The original text is stored alongside the vector.** This matters: embeddings
are one-way. You cannot reconstruct text from 384 numbers. The vector is only
used to *find* the row; the `content` column is what gets sent to the LLM.

---

## 5. Why chunking exists

Documents are not embedded whole. They are split into **chunks** first.

Two reasons:

**1. Precision.** One vector for a 50-page manual is an average of everything
in it — a blur that matches every query weakly and none strongly.

**2. Prompt budget.** Retrieved text goes into the prompt. Retrieving an entire
manual to answer one question wastes the context window and money.

### The tradeoff

```
   too small (50 tokens)          balanced (500-800)         too large (4000)
   ───────────────────            ──────────────────         ────────────────
   "The response time             "SEV-1 means complete      [entire handbook]
    target is 5 minutes"           loss of service. The
                                   response time target      Retrieves on any
   Retrieves precisely, but        is 5 minutes and it       query, but 95% of
   WHAT has a 5 minute             pages the VP."            what reaches the
   target? Context is lost.                                  LLM is irrelevant
                                   Self-contained.           filler.
```

Measured on the sample handbook (~450 tokens):

| `chunk-size` | Chunks produced |
|---|---|
| 800 (default) | 1 — document is smaller than the chunk size |
| 120 | 4 |

Chunk size is the highest-leverage knob in RAG. It is externalised to config in
this project precisely so you can sweep it:

```bash
./mvnw spring-boot:run -Dspring-boot.run.arguments="--rag.chunking.chunk-size=300"
```

---

## 6. The two phases

RAG has two completely separate phases. Conflating them is a common source of
confusion.

```
╔══════════════════════════════════════════════════════════════════════╗
║  PHASE 1 — INGESTION          offline · once per document · no LLM   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║   document.pdf                                                       ║
║        │                                                             ║
║        ▼  EXTRACT        Apache Tika strips formatting → plain text  ║
║   "ACME Robotics…"                                                   ║
║        │                                                             ║
║        ▼  CHUNK          TokenTextSplitter → N passages              ║
║   ["chunk 0", "chunk 1", …]                                          ║
║        │                                                             ║
║        ▼  EMBED          each chunk → float[384]                     ║
║   [[-0.16, …], [0.07, …], …]                                         ║
║        │                                                             ║
║        ▼  STORE          INSERT text + metadata + vector             ║
║   PostgreSQL / pgvector                                              ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║  PHASE 2 — QUERY              online · once per question · uses LLM  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║   "how much is on-call pay?"                                         ║
║        │                                                             ║
║        ▼  EMBED          SAME model as ingestion → float[384]        ║
║   [0.04, -0.11, …]                                                   ║
║        │                                                             ║
║        ▼  SEARCH         nearest neighbours, drop below threshold    ║
║   [chunk#3 (0.61), chunk#1 (0.20), …]                                ║
║        │                                                             ║
║        ▼  AUGMENT        splice chunk TEXT into the prompt           ║
║   system + context + question                                        ║
║        │                                                             ║
║        ▼  GENERATE       the only step that uses an LLM              ║
║   "Primary on-call receives 800 dollars per week…"                   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

**The same embedding model must be used in both phases.** This is the hinge the
whole system turns on. Query and documents must live in the same coordinate
system or the comparison is meaningless — and it fails silently, producing
plausible-looking nonsense rather than an error.

Note also: **ingestion never touches an LLM.** Embedding and generation are
different kinds of model. That is why Layers 1 and 2 of this project run with
no API key at all.

---

## 7. One question, end to end

Following `"how much is on-call pay?"` through the live system, with real
measured values.

**Step 1 — the question is embedded.** ~5 ms, in-process, no network.

```
"how much is on-call pay?"  →  [0.0412, -0.1134, 0.0871, …]   (384 floats)
```

**Step 2 — PostgreSQL finds the nearest chunks.**

```sql
SELECT content, metadata, 1 - (embedding <=> $queryVector) AS score
FROM vector_store
ORDER BY embedding <=> $queryVector
LIMIT 4;
```

Result:

```
 rank  score   source                 chunk
 ─────────────────────────────────────────────────────────────
   1   0.6091  company-handbook.txt   #3   ← ON-CALL COMPENSATION
   2   0.2014  company-handbook.txt   #1
   3   0.1451  company-handbook.txt   #0
   4   0.1342  security-policy.txt    #3
```

The top hit scores **3× the runner-up**. Note it was found despite the word
"paid"/"pay" not matching "stipend" — meaning, not keywords.

**Step 3 — anything below the threshold is dropped.** At 0.1, all four survive.

**Step 4 — the chunks are spliced into the prompt.** The model receives:

```
┌─ SYSTEM ────────────────────────────────────────────────────────────┐
│ You answer questions strictly from the provided context documents.  │
│ - Use only information present in the context…                      │
│ - If the context does not contain the answer, say so plainly…       │
└─────────────────────────────────────────────────────────────────────┘
┌─ USER ──────────────────────────────────────────────────────────────┐
│ how much is on-call pay?                                            │
│                                                                     │
│ Context information is below…                                       │
│ ---------------------                                               │
│ SECTION 4: ON-CALL COMPENSATION                                     │
│ Engineers on the primary on-call rotation receive a stipend of 800  │
│ dollars per week. Secondary on-call receives 300 dollars per week.  │
│ ---------------------                                               │
│                                                                     │
│ Given the context information and no prior knowledge, answer the    │
│ query.                                                              │
└─────────────────────────────────────────────────────────────────────┘
```

Measured: **483 input tokens**.

**Step 5 — the model answers.** 84 output tokens, 567 total.

**Step 6 — the response carries its own evidence.**

```json
{
  "grounded": true,
  "answer": "…Engineers on the primary on-call rotation receive a stipend of
             800 dollars per week. Secondary on-call receives 300 dollars…",
  "citations": [
    { "rank": 1, "score": 0.6091, "source": "company-handbook.txt",
      "chunkIndex": 3, "excerpt": "…SECTION 4: ON-CALL COMPENSATION…" }
  ],
  "usage": { "promptTokens": 483, "completionTokens": 84, "totalTokens": 567 }
}
```

The citations are the point. They let you check whether a wrong answer came
from bad retrieval or bad generation — a distinction you cannot make from the
answer text alone.

---

## 8. Concepts mapped to code

| Concept | Spring AI type | This project |
|---|---|---|
| Read a file into text | `DocumentReader` | `TikaDocumentReader` |
| Split into chunks | `DocumentTransformer` | `TokenTextSplitter` |
| Text → vector | `EmbeddingModel` | `TransformersEmbeddingModel` (ONNX, 384d) |
| Store / search vectors | `VectorStore` | `PgVectorStore` (HNSW, cosine) |
| One passage + metadata | `Document` | has `getText()`, `getMetadata()`, `getScore()` |
| Search parameters | `SearchRequest` | `topK`, `similarityThreshold` |
| Talk to an LLM | `ChatModel` / `ChatClient` | Ollama · Anthropic · OpenAI |
| Retrieve-then-ask | `Advisor` | `QuestionAnswerAdvisor` |

Spring AI's abstraction is the reason this is swappable. `DocumentReader`,
`DocumentTransformer` and `DocumentWriter` are just
`Supplier`, `Function` and `Consumer` — so the whole ingestion pipeline
collapses to one line:

```java
vectorStore.accept(splitter.apply(reader.get()));
```

### Where each file lives

```
ingest/IngestionService.java      Phase 1: extract → chunk → embed → store
ingest/IngestionController.java   POST /api/documents  (multi-file upload)
ingest/DocumentLibraryService.java per-source chunk counts
search/SearchService.java         Phase 2 steps 1–3: embed → search → score
search/SearchController.java      GET  /api/search     (no LLM — inspect retrieval)
chat/ChatService.java             Phase 2 steps 4–6: augment → generate → cite
chat/ChatController.java          POST /api/chat
chat/InfoController.java          GET  /api/info       (which provider is live)
```

`/api/search` exists specifically so you can look at retrieval **without** the
LLM. That separation is the most useful debugging tool in the project.

---

## 9. The failure modes

Four ways RAG goes wrong, all observed in this project.

### 1. Vector search always returns something

Measured, with no threshold, against the 2-document / 8-chunk corpus:

```
Q: "what is the best topping for a pizza?"
  1. score=0.0208  chunk#3
  2. score=0.0137  chunk#1
```

There is no pizza in the corpus. (Exact scores shift as the corpus changes —
these were 0.027 / 0.006 when only one document was loaded. The *band* is what
matters: an order of magnitude below any real match.) Nearest-neighbour search returns the
*nearest* neighbours, never the *relevant* ones — nothing in the mechanism
knows the difference between "closest" and "close enough".

**Guard:** `similarityThreshold`. At 0.1 the pizza query returns zero chunks.

### 2. A threshold cannot control what the model says

With zero chunks retrieved, the same prompt, two different local models:

| Model | Answer to the pizza question |
|---|---|
| `llama3.2:1b` | ❌ **"Melted mozzarella cheese."** |
| `llama3.2:3b` | ✅ "The text does not mention pizzas or toppings at all." |

Retrieval behaved **identically** in both runs. The 1b model simply ignored the
instruction and answered from its training data.

**Guard:** an explicit "say so plainly if the context lacks the answer"
instruction — *and* a model capable of following it. Two guards, not one:

```
   threshold      → controls WHAT the model sees
   system prompt  → controls what the model DOES with it
```

A team that tests RAG safety by checking retrieval alone will ship this bug.

### 3. Thresholds are not portable

Copying `0.75` from an OpenAI tutorial rejects every result here. Scores depend
on the embedding model. Always calibrate.

### 4. Embeddings are bad at exact tokens

Vector search matches *meaning*. A query for the error code `SEV-2`, an order
number, or a person's exact name matches semantically similar text rather than
the literal string. Production systems usually run vector **and** keyword
(BM25) search and fuse the rankings — "hybrid search".

Related, measured: a single trailing `?` moved a score by 0.004.

```
"can I plug a USB stick into my work laptop?"   → 0.3507
"can I plug a USB stick into my work laptop"    → 0.3549
```

Harmless here, but don't treat a threshold as precise to three decimals.

---

## 10. Why these technology choices

| Choice | Why |
|---|---|
| **Spring AI** | Provider-neutral abstractions — swapping the LLM is a config change, not a rewrite |
| **Local ONNX embeddings** | No API key, no cost, works offline. Anthropic has no embeddings endpoint, so a second provider is required regardless |
| **PGVector** | A real vector index, but it's just Postgres — inspectable with plain SQL, and you probably already run one |
| **Ollama default** | The whole stack runs free and local. Switch to a hosted provider with one flag |
| **Apache Tika** | One reader for PDF, DOCX, PPTX, HTML and ~20 more formats |

### The thing worth internalising

Retrieval, embeddings and storage are **identical** across all three chat
providers. Only the final generation step changes:

```bash
./mvnw spring-boot:run                                    # ollama  (free, local)
./mvnw spring-boot:run -Dspring-boot.run.profiles=anthropic
./mvnw spring-boot:run -Dspring-boot.run.profiles=openai
```

That is RAG's real claim: the retrieved context does the work, and the model
mostly phrases it. A better model writes a nicer sentence; it cannot rescue a
passage that retrieval failed to find.

---

## 11. Glossary

| Term | Meaning |
|---|---|
| **Embedding / vector** | A list of numbers representing meaning. Here, 384 floats. |
| **Dimension** | How many numbers in the vector. Ours: 384. |
| **Cosine similarity** | Angle between two vectors. 1.0 = same meaning, 0.0 = unrelated. |
| **Cosine distance** | `1 − similarity`. What pgvector's `<=>` returns. |
| **Chunk** | A passage a document was split into; the unit of retrieval. |
| **topK** | Maximum number of chunks to retrieve. |
| **Similarity threshold** | Minimum score to keep a chunk. Bounds *quality*; `topK` bounds *quantity*. |
| **Vector store** | Database that indexes vectors for nearest-neighbour search. |
| **HNSW** | The approximate-nearest-neighbour index. ~O(log n) instead of O(n). |
| **Grounded** | The answer was built from retrieved text. `grounded: false` = distrust it. |
| **Citation** | Which chunk produced the answer, with its score. |
| **Advisor** | Spring AI hook that intercepts a chat call. `QuestionAnswerAdvisor` does retrieve-then-augment. |
| **Hallucination** | Model states something unsupported. In RAG, usually caused by bad retrieval. |
| **Hybrid search** | Vector + keyword search combined, to fix the exact-token weakness. |
| **Reranking** | Second pass that re-scores retrieved candidates more accurately. |

---

## Where to go next

| Document | Covers |
|---|---|
| [Layer 1 — Ingestion](LAYER-1-INGESTION.md) | Extract, chunk, embed, store — and the Git LFS trap |
| [Layer 2 — Retrieval](LAYER-2-RETRIEVAL.md) | Similarity scores, threshold calibration, pgvector internals |
| [Layer 3 — Chat](LAYER-3-CHAT.md) | The three providers, prompt augmentation, citations, grounding |
| [README](../README.md) | Setup, running it, configuration reference, troubleshooting |

The most instructive thing you can do next is open `/api/search` and run your
own questions against your own documents. Watch the scores. Find the query
that *should* match and doesn't — that is where RAG actually gets interesting.
