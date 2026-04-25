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

## P4 — HyDE + Semantic Chunking

**File:** `patterns/p4_hyde.py`
**Embedding:** `all-MiniLM-L6-v2` (local, own index at `index/p4/`)
**LLM:** `gpt-4o-mini`
**New concepts:** `SemanticChunker`, `embed_query()`, `similarity_search_by_vector()`

### Two problems solved together

**Problem 1 — Fixed chunking (P1) cuts mid-idea:**
`RecursiveCharacterTextSplitter` splits at 512 chars regardless of meaning. A sentence spanning a topic boundary gets cut wherever the counter hits 512.

**Problem 2 — Query/document language mismatch:**
User asks: `"what keeps him going?"` → conversational, short
Document says: `"he relies on his will and prayers…"` → formal prose
These embed to different vectors → poor retrieval on vague queries.

### Fix 1 — SemanticChunker

```
P1: 319 chunks, avg 440 chars  (fixed-size, blind to meaning)
P4: 153 chunks, avg 909 chars  (topic-aware boundaries)
```

`SemanticChunker` embeds each sentence, computes cosine similarity between adjacent sentences, and inserts a boundary wherever similarity drops sharply (topic shift). Fewer, larger, more coherent chunks.

`breakpoint_threshold_type="percentile"` — boundary inserted at the sharpest similarity drops (bottom 70th percentile of all drops).

### Fix 2 — HyDE (Hypothetical Document Embeddings)

```
Standard:  question  ──► embed ──► search ──► chunks
HyDE:      question  ──► LLM generates hypothesis paragraph
                         ──► embed hypothesis ──► search ──► chunks
```

The hypothesis is written in document-like language → its embedding is much closer to real document chunks → better recall on vague queries.

### Key methods

- `embed_query(text)` — embeds a string, returns `list[float]` (the raw vector)
- `similarity_search_by_vector(vector, k)` — searches FAISS with a pre-computed vector instead of a string. Used because we embed the hypothesis ourselves before searching.
- `temperature=0.2` on hyde_chain — slightly creative so it generates natural prose, not a robotic answer

### Why rag_chain differs from P1–P3

In P1–P3, the chain fan-out `{"context": retriever | lambda, "question": RunnablePassthrough()}` handles retrieval internally. In P4, retrieval is done externally (either standard or HyDE path), so the chain is simplified — it just takes a pre-joined context string + question directly.

---

## P5 — Adaptive RAG

**File:** `patterns/p5_adaptive_rag.py`
**Embedding:** `all-MiniLM-L6-v2` (local, reuses P1 index)
**LLM:** `gpt-4o-mini`
**New concepts:** `PydanticOutputParser`, `Enum` routing, `BaseModel` schema

### Problem

Not every question needs the same strategy — or any retrieval at all:

```
"Hi!"                   → no FAISS needed       (P1 wastes a search call)
"What is 2+2?"          → no FAISS needed
"Who is Santiago?"      → simple lookup
"Dreams + days + shark" → needs branched strategy
"Give me a recipe"      → should be refused
```

### Fix — Classify first, then route

```
question
     │
     ▼
ROUTER → RouteDecision(route=Route.SIMPLE, reason="single fact lookup")
     │
     ├── no_retrieval → LLM direct        (0 FAISS calls)
     ├── simple       → P1 chain          (1 FAISS call)
     ├── branched     → P3 chain          (N FAISS calls)
     └── refuse       → static message   (0 LLM calls)
```

### PydanticOutputParser vs JsonOutputParser

| | JsonOutputParser (P3) | PydanticOutputParser (P5) |
|---|---|---|
| Returns | raw `dict` | typed Python object |
| Validation | none | Pydantic enforces types |
| Enum constraint | no | yes — wrong value → error |
| Access | `result["route"]` | `result.route` |

### Schema definition

```python
class Route(str, Enum):
    NO_RETRIEVAL = "no_retrieval"
    SIMPLE       = "simple"
    BRANCHED     = "branched"
    REFUSE       = "refuse"

class RouteDecision(BaseModel):
    route:  Route = Field(description="Which route to take")
    reason: str   = Field(description="One sentence justification")
```

### Key concept — `get_format_instructions()`

```python
router_parser = PydanticOutputParser(pydantic_object=RouteDecision)
router_parser.get_format_instructions()  # → JSON schema string
```

This generates a JSON schema description that gets injected into the prompt so the LLM knows exactly what structure to return. The parser then validates the response against the schema — if `route` is not one of the 4 enum values, Pydantic raises a `ValidationError`.

### Demo results

```
"Hello!"             → no_retrieval  (0 chunks)
"How many days..."   → simple        (4 chunks)
"Dreams + Manolin…"  → branched      (12 chunks)
"Recipe for marlin"  → refuse        (0 chunks, 0 LLM calls)
```

---

## P6 — *(coming next)*
