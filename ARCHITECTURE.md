# GANP-Chat — RAG Architecture

## Overview

GANP-Chat is a legal Q&A system grounded exclusively in Cameroonian law. It retrieves
relevant legal articles from a pre-built index and generates structured answers using
DeepSeek. No hallucination is possible by design: the LLM is forbidden from introducing
any legal claim not present in the retrieved context.

---

## Retrieval Pipeline

Every query goes through three independent rankers whose results are fused before the LLM
ever sees a single token.

```
User query
    │
    ├─── 1. Dense retrieval      (multilingual-e5-base + FAISS)
    ├─── 2. BM25 keyword search  (rank_bm25)
    └─── 3. HyDE dense retrieval (DeepSeek → hypothetical article → FAISS)
                │
                ▼
         RRF Fusion (k=60)
                │
                ▼
         Top-5 chunks + metadata
                │
                ▼
         DeepSeek generation
                │
                ▼
         Structured answer + sources
```

---

## Components

### 1. Embedding Model — `intfloat/multilingual-e5-base`

**Why:** The previous model (`all-MiniLM-L6-v2`) was trained predominantly on English.
Cameroonian law documents are in French. `multilingual-e5-base` was trained on 100+
languages including French, producing semantically accurate representations of French
legal vocabulary.

**How it works:**
- Documents are encoded with the prefix `passage: <text>` at index-build time.
- Queries are encoded with the prefix `query: <text>` at retrieval time.
- This asymmetric prefixing is required by the e5 architecture and significantly improves
  retrieval quality over symmetric encoding.
- Vectors are L2-normalized; the FAISS index uses inner product (= cosine similarity on
  normalized vectors).
- Dimension: 768 (vs 384 for MiniLM).

**Memory footprint:** ~550 MB RAM on Railway Hobby.

---

### 2. BM25 — `rank_bm25` (BM25Okapi)

**Why:** Dense models can miss exact legal terms — article numbers (`Article 34`), law
codes (`LOI N°92/007`), specific legal phrases. BM25 is a proven sparse retrieval method
that scores documents by term frequency weighted by inverse document frequency. It excels
at exact-match retrieval.

**How it works:**
- At startup, all chunks are tokenized (lowercased, split on word boundaries).
- `BM25Okapi` builds a frequency-weighted index over all 9,000+ chunks.
- At query time, query tokens are scored against the index; the top-N chunks by BM25
  score form the BM25 ranking list.
- No GPU required; pure Python; negligible memory (~20 MB).

---

### 3. HyDE — Hypothetical Document Embeddings

**Why:** Short or vague queries ("mon patron ne me paie pas") contain almost no legal
vocabulary, so both dense and BM25 retrieval struggle. HyDE generates a synthetic legal
article that *would* answer the question, then searches for real articles similar to that
hypothetical one. This bridges the vocabulary gap between user language and legal language.

**How it works:**
1. DeepSeek receives a short prompt: *"Write a 2-3 sentence hypothetical legal article
   that directly answers this question."*
2. The hypothetical article is encoded with the `passage:` prefix using the same e5 model.
3. This embedding is used to search FAISS, returning a third ranked list.
4. If DeepSeek fails (network error, timeout), HyDE is silently skipped and the system
   falls back to Dense + BM25 only.

**Latency:** One additional DeepSeek API call (~300–600 ms). Acceptable for legal Q&A
where answer quality matters more than sub-second response.

---

### 4. RRF — Reciprocal Rank Fusion

**Why:** Each ranker has different strengths. Rather than choosing one winner, RRF
combines all three rankings into a single score without requiring any tuning of per-ranker
weights.

**Formula:**

```
RRF_score(document) = Σ  1 / (k + rank(document, ranker))
```

where `k = 60` (standard constant that reduces sensitivity to top-rank outliers).

**How it works:**
- Each ranker returns a ranked list of chunk indices.
- RRF assigns a score to every chunk that appeared in any list.
- Chunks appearing in multiple lists (i.e., agreed upon by multiple rankers) receive
  higher combined scores.
- The top-5 chunks by RRF score are sent to the LLM.

---

### 5. DeepSeek Generation

**Model:** `deepseek-chat`, temperature 0.3, max 800 tokens.

The system prompt enforces an 8-step legal analysis protocol:
1. Context verification (is the retrieved context sufficient?)
2. Question augmentation (what is the user really asking?)
3. Decomposition (break into doctrinal / procedural / practical sub-issues)
4. Context-grounded analysis (cite only retrieved text)
5. Synthesis (coherent integrated answer)
6. Structured output (headers, bullets, [X] citations)
7. Style constraints (formal, neutral, plain language)
8. Quality control (no hallucination, no external law)

The LLM is explicitly forbidden from:
- Introducing legal rules not in the retrieved context
- Citing articles not referenced by the retriever
- Speculating on intent or policy

---

## What the System Can Answer

| Query type | Example | Capability |
|---|---|---|
| Direct article lookup | "Que dit l'article 34 du Code du Travail ?" | ✅ Exact citation |
| Rights questions | "Quels sont mes droits en cas de licenciement ?" | ✅ Full analysis |
| Penalty questions | "Quelle est la peine pour vol selon le Code Pénal ?" | ✅ With article ref |
| Cross-law questions | "Comment le code du travail et l'OHADA diffèrent sur les contrats ?" | ✅ Multi-source synthesis |
| Vague/colloquial | "Mon boss ne me paie pas" | ✅ HyDE bridges vocabulary gap |
| English queries | "What are the rules for company registration?" | ✅ Multilingual model |
| Out-of-scope | "What does French law say about X?" | ⚠️ System states: not in context |
| Hallucination attempt | "Make up an article about X" | ❌ Refused by system prompt |

---

## Deployment (Railway)

| Resource | Requirement | Railway Hobby |
|---|---|---|
| RAM | ~850 MB | 8 GB ✅ |
| CPU | 1 vCPU | Shared ✅ |
| GPU | None | N/A ✅ |
| Storage | ~50 MB (index files) | Persistent volume ✅ |
| External APIs | DeepSeek API key | `DEEPSEEK_API_KEY` env var |

---

## Rebuilding the Index

The FAISS index must be rebuilt whenever:
- New law documents are added to the corpus
- The embedding model changes

```bash
cd "RAG_BUILDING"
python build_index.py
```

This re-encodes all PDFs using `multilingual-e5-base` with `passage:` prefix and writes:
- `index_file.index` — FAISS binary index (768-dim, IndexFlatIP)
- `index_file.meta.json` — article + law metadata per chunk
- `index_file.meta.chunks.json` — raw chunk texts
