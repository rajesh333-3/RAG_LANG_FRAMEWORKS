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

## P2 — Memory RAG

**File:** `patterns/p2_memory_rag.py`
**Embedding:** `all-MiniLM-L6-v2` (local, reuses P1 index)
**LLM:** `gpt-4o-mini`
**New concepts:** `MessagesPlaceholder`, `HumanMessage`, `AIMessage`, condenser chain

### Problem with P1 in multi-turn conversations

Follow-up questions contain pronouns with no retrievable context:

```
Turn 1: "What method does this paper propose?"    ← retrievable ✓
Turn 2: "How does it compare to chunking?"        ← "it" = ? ✗
Turn 3: "Who funded that research?"               ← "that" = ? ✗
```

### Fix — Condenser chain rewrites before retrieval

```
"Who funded that research?" + chat_history
        │
        ▼
   CONDENSER (LLM)
        │
        ▼
"Who provided the funding for the research discussed in the paper?"
        │
        ▼
   RETRIEVER → chunks → LLM → answer
```

### Pipeline

```
user question
     │
     ▼
CONDENSER  ← MessagesPlaceholder injects chat_history here
  prompt:  system: rewrite as standalone
           {chat_history}          ← all prior HumanMessage/AIMessage turns
           human: {question}
     │
     ▼
standalone question
     │
     ▼
RETRIEVER  (top-4 chunks)
     │
     ▼
RAG PROMPT  (context + standalone question)
     │
     ▼
LLM → answer
     │
     ▼
append HumanMessage(question) + AIMessage(answer) to chat_history
```

### Key concepts

- `MessagesPlaceholder(variable_name="chat_history")` — a slot in the prompt that expands to the full list of message objects at runtime. The variable name must match the key passed in `.invoke()`
- `HumanMessage` / `AIMessage` — typed message objects LangChain uses to represent conversation turns. The condenser LLM sees them as a real conversation, not just a string
- Condenser chain input: `{"chat_history": [...], "question": "..."}` → outputs a plain string (the rewritten question)
- Turn 1 skips the condenser entirely (`if chat_history` guard) — no history means nothing to resolve, and it saves one LLM call at the start of every fresh conversation
- For turns 2+, only the last 10 messages (`history[-10:]` = last 5 turns) are passed — prevents the prompt from growing unbounded in long conversations
- Memory is just a plain Python list — you manage it manually, appending after each turn
- In production: use Redis with a TTL so history is scoped per session and auto-expires

### Condenser chain construction

```python
condenser_prompt = ChatPromptTemplate.from_messages([
    ("system", "rewrite follow-up as standalone..."),
    MessagesPlaceholder(variable_name="chat_history"),  # ← expands to all prior turns
    ("human", "{question}"),
])

condenser_chain = condenser_prompt | llm | StrOutputParser()
```

---

## P3 — Branched RAG

**File:** `patterns/p3_branched_rag.py`
**Embedding:** `all-MiniLM-L6-v2` (local, reuses P1 index)
**LLM:** `gpt-4o-mini`
**New concepts:** `JsonOutputParser`, decompose chain, synthesise chain

### Problem with P1/P2 for complex questions

A single retrieval call fetches 4 chunks — likely all from one section, missing others:

```
"What is Landmark Embedding, how does it work, and what experiments validated it?"
        │
        ▼
retriever → 4 chunks from intro section only  ✗  (misses methods + experiments)
```

### Fix — Decompose → retrieve per sub-Q → synthesise

```
complex question
     │
     ▼
DECOMPOSE → ["What is it?", "How does it work?", "What experiments?"]
     │
     ├──► sub-q 1 → RETRIEVE → RAG CHAIN → sub-answer 1
     ├──► sub-q 2 → RETRIEVE → RAG CHAIN → sub-answer 2
     └──► sub-q 3 → RETRIEVE → RAG CHAIN → sub-answer 3
     │
     ▼
SYNTHESISE → final coherent answer
```

### Key concepts

- `JsonOutputParser` — instead of returning a string like `StrOutputParser`, it parses the LLM output as JSON and returns a native Python object (here: a list of strings). The LLM must be instructed to output valid JSON
- Decompose chain: `{question}` → `["sub-q1", "sub-q2", ...]` (Python list)
- Each sub-question runs through the same RAG chain as P1, independently — different chunks retrieved per sub-question
- Synthesise chain sees: original question + all `Q: ...\nA: ...` pairs → one final answer
- Total chunks = N sub-questions × 4 — broader document coverage than a single retrieval

### Chain construction

```python
# Returns a Python list, not a string
decompose_chain = (
    ChatPromptTemplate.from_template("...Return a JSON array...")
    | llm
    | JsonOutputParser()   # ← str → list
)

# Runs once per sub-question (same as P1)
rag_chain = (
    {"context": retriever | join_lambda, "question": RunnablePassthrough()}
    | rag_prompt | llm | StrOutputParser()
)

# Combines all sub-answers
synthesise_chain = (
    ChatPromptTemplate.from_template("...{original}...{sub_answers}...")
    | llm
    | StrOutputParser()
)
```

---

## P4 — *(coming next)*
