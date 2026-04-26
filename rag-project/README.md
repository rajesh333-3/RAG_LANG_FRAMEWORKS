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

## langgraph_101.py — LangGraph anatomy (read before P6)

**File:** `patterns/langgraph_101.py`

A runnable tutorial covering every LangGraph primitive used in P6–P8.

| Thing | What it is | Key rule |
|---|---|---|
| `State` | TypedDict shared by all nodes | init ALL fields in `invoke()` |
| `Node` | `fn(state) → dict` | return ONLY changed fields |
| `Edge` | `add_edge(A, B)` | A always goes to B |
| Cond. edge | `fn(state) → str → node` | this is how loops work |
| `END` | special constant | graph stops here |
| `compile()` | validates + returns app | `app.invoke()` = `chain.invoke()` |

The loop pattern (impossible with chains):
```python
wf.add_edge("rewrite", "retrieve")          # goes BACKWARDS
wf.add_conditional_edges("grade", fn, {
    "retry": "rewrite"                       # fn can return "retry"
})
```

---

## P6 — CRAG (Corrective RAG)

**File:** `patterns/p6_crag.py`
**Embedding:** `all-MiniLM-L6-v2` (local, reuses P1 index)
**LLM:** `gpt-4o-mini`
**New concepts:** `StateGraph`, `add_conditional_edges`, corrective loop

### Problem with P1

P1 is a linear chain: retrieve → prompt → generate. If retrieved chunks are irrelevant, P1 still generates — likely hallucinated. There's no way to check, retry, or refuse within a chain.

### Fix — Grade → loop → refuse

```
START → retrieve → grade ──► relevant      → generate → END
                    │
                    ├──► irrelevant (retries<2) → rewrite → retrieve  (LOOP)
                    │
                    └──► irrelevant (retries≥2) → refuse  → END
```

### State

```python
class CRAGState(TypedDict):
    question:  str    # may be rewritten by node_rewrite
    documents: list   # filled by node_retrieve
    grade:     str    # "relevant" | "irrelevant"
    answer:    str    # filled by generate or refuse
    retries:   int    # loop counter, max 2
```

### The loop — what makes this impossible as a chain

```python
wf.add_edge("rewrite", "retrieve")   # BACKWARDS arrow = loop
wf.add_conditional_edges("grade", route_after_grade, {
    "generate": "generate",
    "rewrite":  "rewrite",            # bad grade → go back
    "refuse":   "refuse",
})
```

### Demo results

```
"How many days without fish?" → relevant on first try → answer: "Eighty-seven days"  (0 retries)
"What does he think at night?" → irrelevant → rewrite x2 → relevant → answer found   (2 retries)
"Population of Cuba?"          → irrelevant → rewrite x2 → still off-topic → refuse  (2 retries)
```

### Key rule — nodes return only what they changed

```python
def node_retrieve(state: CRAGState) -> dict:
    docs = retriever.invoke(state["question"])
    return {"documents": docs}   # only this field — LangGraph merges the rest
```

---

## P7 — Self-RAG

**File:** `patterns/p7_self_rag.py`
**Embedding:** `all-MiniLM-L6-v2` (local, reuses P1 index)
**LLM:** `gpt-4o-mini`
**New concepts:** answer grading node, graph extension pattern, negative prompting, `Literal` type

### Problem with P6

CRAG grades documents but not the answer. The LLM can still hallucinate from relevant docs:
```
Docs:   "Santiago caught the marlin on day 3 at sea"
Answer: "Santiago caught the marlin and sold it for $500"  ← Gate 1 passes, Gate 2 catches this
```

### Fix — Add Gate 2 (answer grade)

```
P6: generate → END
P7: generate → grade_answer → END           (faithful)
                             → regen_strict → END   (hallucinated)
```

### Graph extension pattern

P7 doesn't rewrite P6 — it extends it. The pattern:
1. Add new fields to State (`ans_grade`, `violations`)
2. Write new node functions (`node_grade_answer`, `node_regen_strict`)
3. Rewire one edge (`generate→END` becomes `generate→grade_answer`)
4. Add new conditional edge from `grade_answer`

### Key concepts

- **Negative prompting** — `node_regen_strict` passes the violations list explicitly: `"Do NOT include: X, Y, Z"`. More reliable than just `"be faithful"` — the LLM knows exactly what to avoid
- **`violations: list[str]`** — the grader extracts specific unsupported claims, not just a bool. These feed back into the regen prompt
- **`Literal["faithful", "hallucinated"]`** — inline Pydantic type constraint. Same as `Enum` but declared directly in the field. Use `Enum` when the values are reused across models; `Literal` for one-off constraints
- **Two quality gates** — Gate 1 (doc grade) + Gate 2 (answer grade). Either can loop/refuse independently

### Real-world lesson from the run

The grader flagged `"Eighty-four days."` as hallucinated — not because it was wrong, but because it "lacks context." **Grader prompt engineering matters as much as the graph architecture.** A poorly tuned grader causes false positives → unnecessary regen → higher latency and cost. Always evaluate your grader's precision/recall separately.

### State additions

```python
ans_grade:  str    # "faithful" | "hallucinated"
violations: list   # ["claim 1 not in context", "claim 2 not in context"]
```

---

## P8 — Agentic RAG

**File:** `patterns/p8_agentic_rag.py`
**Embedding:** `all-MiniLM-L6-v2` (local, reuses P1 index)
**LLM:** `gpt-4o-mini`
**New concepts:** `@tool` decorator, `create_agent`, ReAct loop, `.stream()`

### The shift

| | P1–P7 | P8 |
|---|---|---|
| Recipe | You write it (fixed steps) | LLM writes it on the fly |
| Path | Always the same | Changes per question |
| Tools | One strategy hardcoded | LLM picks from a menu |

### ReAct loop (Reason + Act)

```
Question: "How many days without fish, and how many hours is that?"

[Action]       search_book("Santiago go without catching a fish")
[Observation]  "...eighty-seven days..."
[Action]       calculate("87 * 24")
[Observation]  2088
[Final Answer] Santiago went 87 days = 2,088 hours without catching a fish.
```

### `@tool` — docstring is the prompt

```python
@tool
def search_book(query: str) -> str:
    """Search The Old Man and the Sea for specific facts, quotes, events,
    or character details. Use for precise factual lookups from the book."""
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)
```

The LLM sees only the **name + docstring** to decide when to call the tool. The implementation is invisible. Treat docstrings with the same care as prompts.

### LangChain 1.x API change

```python
# Old (LangChain 0.x):
from langchain.agents import create_react_agent, AgentExecutor

# New (LangChain 1.x):
from langchain.agents import create_agent   # returns compiled StateGraph directly
agent = create_agent(model=llm, tools=tools)
agent.invoke({"messages": [{"role": "user", "content": question}]})
agent.stream(...)   # yields each tool call + LLM step live
```

No separate `AgentExecutor` — the compiled graph IS the executor, consistent with P6/P7.

### Key concepts

- **`@tool`** — decorator that registers a Python function as a tool. Docstring = routing instruction for the LLM
- **`create_agent`** — wires LLM + tools into a compiled ReAct graph
- **`.stream()`** — yields each step live: `[Action]`, `[Observation]`, `[Final Answer]`. Use instead of `.invoke()` when you want to watch the loop or stream to a UI
- **Two tool calls on one question** — the agent called `search_book` twice for the second question (days + dreams) without being told to. The LLM decided independently

---

## P9 — Multimodal RAG

**File:** `patterns/p9_multimodal_rag.py`
**PDF:** `llama2_tech_report.pdf` (a technical paper with architecture diagrams + performance charts)
**Embedding (text):** `all-MiniLM-L6-v2` (384-dim, same as P1)
**Embedding (image):** CLIP `openai/clip-vit-base-patch32` (512-dim, shared text+image space)
**Vision LLM:** `gpt-4o-mini` (supports vision, no need for full `gpt-4o`)

### Why P1–P8 fail on visual content

Standard RAG ingests text only. A technical paper has:
- Architecture diagrams (e.g., model overview on page 6)
- Results tables rendered as images (page 12)
- Performance charts (pages 13–14)

P1 would retrieve the text *around* a figure — "See Figure 3" — but not the figure itself. The LLM gets the caption, not the diagram. For any visual information (architecture, benchmark plots), the answer is incomplete or fabricated.

### Fix: two parallel retrieval gates

```
question
     │
     ├──► text_retriever (MiniLM, 384-dim)  → top-3 text chunks
     │
     ├──► retrieve_images (CLIP, 512-dim)   → top-2 matching images
     │
     └──► GPT-4o vision message
              │
              ├── text context + question   (type: "text")
              └── images as base64          (type: "image_url")
                   │
                   ▼
              answer — can describe diagrams AND cite text
```

### CLIP — why it enables text→image retrieval

CLIP (Contrastive Language-Image Pretraining) was trained on 400M (text, image) pairs. Its key property:

> Text and images land in the **same 512-dim vector space**.

"attention mechanism diagram" (text query) and an actual attention diagram image will have similar embeddings. That's what makes semantic image retrieval possible — you embed the query as text and compare against image embeddings using cosine similarity.

```python
# Both functions return 512-dim L2-normalised vectors in the same CLIP space
embed_image(b64)        # PIL image  → 512-dim vector
embed_text_clip(text)   # string     → 512-dim vector

# Retrieval = dot product (cosine sim of normalised vectors)
scores = [np.dot(q_vec, img["emb"]) for img in img_store]
```

**Important:** MiniLM (384-dim) and CLIP (512-dim) are completely separate vector spaces. Never mix them. Text retrieval uses MiniLM; image retrieval uses CLIP.

### Vision message format

GPT-4o and GPT-4o-mini accept multimodal content in a single message:

```python
content = [
    {
        "type": "text",
        "text": f"Context:\n{text_context}\n\nQuestion: {question}"
    },
    {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{img['b64']}"}
    },
    # ... more images
]
messages = [{"role": "user", "content": content}]
llm.invoke(messages)
```

The model reads the text context AND visually processes the images in a single LLM call.

### Image extraction with pymupdf (fitz)

`pymupdf` (imported as `fitz`) extracts embedded images directly from the PDF — no OCR, no system dependencies.

```python
import fitz
doc = fitz.open(pdf_path)
for page in doc:
    for img in page.get_images():
        xref = img[0]
        pix  = fitz.Pixmap(doc, xref)
        if pix.n > 4:                  # CMYK → convert to RGB
            pix = fitz.Pixmap(fitz.csRGB, pix)
        if pix.width < 200 or pix.height < 200:
            continue                   # skip decorative icons
```

Why pymupdf over `unstructured hi_res`:
- `unstructured`: needs detectron2 + tesseract + poppler (~2 GB, complex install)
- `pymupdf`: `pip install pymupdf`, pure Python, same image extraction result

PDF had 5 meaningful images (after filtering icons <200px) on pages 6, 7, 12, 13, 14.

### transformers v5 API change

In transformers v5, `get_image_features()` and `get_text_features()` return `BaseModelOutputWithPooling`, not a plain tensor. You must use `.pooler_output`:

```python
# Old (transformers <5):
vec = feats[0].numpy()

# New (transformers v5):
vec = feats.pooler_output[0].numpy()   # ← .pooler_output required
```

### Key concepts

- **CLIP shared vector space** — trained on (text, image) pairs → semantic similarity across modalities
- **Two separate indexes** — MiniLM for text (384-dim), CLIP cosine similarity for images (512-dim); never mix
- **pymupdf** — lightweight PDF image extraction, one pip install, no system deps
- **Vision message format** — text + base64 images in the same `content` list; `gpt-4o-mini` handles both
- **Pre-compute image embeddings** — embed all images once at startup, not per query (CLIP is slow)
- **Filter small images** — skip anything under ~200px to avoid decorative icons polluting the image index
