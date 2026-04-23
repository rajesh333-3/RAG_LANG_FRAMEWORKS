# RAG Patterns — Learning Journal

A progressive study of RAG (Retrieval-Augmented Generation) patterns, built incrementally.

---

## Setup Notes

- **Embeddings:** OpenAI `/embeddings` endpoint is geo-blocked for this org key (US-only restriction). Workaround: `all-MiniLM-L6-v2` via HuggingFace runs locally, no API call needed.
- **LLM:** `gpt-4o-mini` via OpenAI chat completions — works fine (not geo-blocked).
- **Tracing:** LangSmith — set `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` in `.env`.

---

## P1 — Simple RAG

**File:** `patterns/p1_simple_rag.py`
**Embedding:** `all-MiniLM-L6-v2` (local, 384-dim)
**LLM:** `gpt-4o-mini`
**Vector store:** FAISS (saved to `index/p1/`, skips re-embed on subsequent runs)

### Pipeline

```
PDF
 │
 ▼
LOAD ──► list of Document objects (one per page)
         .page_content = text  |  .metadata = {source, page, ...}
 │
 ▼
SPLIT ──► RecursiveCharacterTextSplitter(chunk_size=512, overlap=50)
          breaks long pages into overlapping chunks (~100 words each)
 │
 ▼
EMBED + INDEX ──► each chunk → 384-dim vector via all-MiniLM-L6-v2
                  FAISS stores vectors for cosine nearest-neighbour search
                  index saved to disk — reloaded on next run (no re-embed)
 │
 ▼
RETRIEVE ──► question → retriever → top 4 most similar chunks by cosine sim
 │
 ▼
PROMPT ──► chunks joined with \n\n, inserted into prompt template as {context}
 │
 ▼
GENERATE ──► gpt-4o-mini reads prompt, returns grounded answer
 │
 ▼
ANSWER
```

### Chain Construction

```python
chain = (
    {
        "context":  retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

The `{}` dict is a **fan-out** — the question goes into both branches in parallel:

```
question
   │
   ├──► retriever ──► [Doc1..Doc4] ──► lambda ──► joined text  →  {context}
   │                                                                    │
   └──► RunnablePassthrough() ──► question unchanged             →  {question}
                                                                        │
                                                                  prompt template
                                                                        │
                                                                   gpt-4o-mini
                                                                        │
                                                                  StrOutputParser
                                                                        │
                                                                    answer
```

**Key concepts:**
- The `|` operator composes LangChain Runnables (like Unix pipe)
- A plain `lambda` piped with `|` is auto-wrapped into `RunnableLambda`
- `RunnablePassthrough()` is a no-op Runnable — required because every value in the dict must be a Runnable; a raw string would break it
- `retriever` is a Runnable wrapper around FAISS — accepts a question string, internally calls `similarity_search`, returns Documents

---

## P2 — *(coming next)*
