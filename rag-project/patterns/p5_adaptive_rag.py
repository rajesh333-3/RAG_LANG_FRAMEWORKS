import os
from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

DIVIDER = "=" * 60

def section(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)

# ═══════════════════════════════════════════════════════════════
# STEPS 1-3 — LOAD / SPLIT / EMBED (reusing P1 index)
# ═══════════════════════════════════════════════════════════════
section("STEPS 1-3 — LOAD, SPLIT, EMBED (reusing P1 index)")

pdf_path   = os.path.join(os.path.dirname(__file__), "../data/oldman_and_the_sea.pdf")
docs       = PyPDFLoader(pdf_path).load()
splitter   = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks     = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
index_path = os.path.join(os.path.dirname(__file__), "../index/p1")

if os.path.exists(index_path):
    print("  Index found on disk — loading (skipping re-embed)")
    vectordb = FAISS.load_local(index_path, embeddings,
                                allow_dangerous_deserialization=True)
else:
    print(f"  Embedding {len(chunks)} chunks ...")
    vectordb = FAISS.from_documents(chunks, embeddings)
    os.makedirs(index_path, exist_ok=True)
    vectordb.save_local(index_path)

llm       = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# ═══════════════════════════════════════════════════════════════
# STEP 4 — DEFINE THE ROUTE SCHEMA  ← NEW in P5
#
# JsonOutputParser (P3) returns a raw dict — no type safety.
# PydanticOutputParser returns a validated Python object.
#
# How it works:
#   1. You define a BaseModel with typed fields
#   2. parser.get_format_instructions() generates a JSON schema
#      description that gets injected into the prompt
#   3. The LLM must return JSON matching that schema exactly
#   4. The parser validates it — wrong types → raises an error
#
# Route is a str Enum — restricts the LLM to exactly 4 values.
# If the LLM returns anything else, Pydantic raises ValidationError.
# ═══════════════════════════════════════════════════════════════
section("STEP 4 — ROUTE SCHEMA (PydanticOutputParser)")

class Route(str, Enum):
    NO_RETRIEVAL = "no_retrieval"  # greetings, maths, general knowledge
    SIMPLE       = "simple"        # single factual lookup in the doc
    BRANCHED     = "branched"      # multi-part question needing multiple sections
    REFUSE       = "refuse"        # off-topic, harmful, unanswerable

class RouteDecision(BaseModel):
    route:  Route = Field(description="Which route to take")
    reason: str   = Field(description="One sentence justification")

router_parser = PydanticOutputParser(pydantic_object=RouteDecision)

print("  RouteDecision fields:")
print("    .route  → Route enum  (no_retrieval | simple | branched | refuse)")
print("    .reason → str         (one sentence justification)")
print()
print("  Format instructions injected into prompt:")
print("  " + router_parser.get_format_instructions()[:300].replace("\n", "\n  ") + "...")

# ═══════════════════════════════════════════════════════════════
# STEP 5 — ROUTER CHAIN  ← NEW in P5
#
# The router chain:
#   input : {query, format_instructions}
#   output: RouteDecision object  (not a string, not a dict)
#
# get_format_instructions() returns a string describing the JSON
# schema — injected into the prompt so the LLM knows exactly what
# structure to return. The parser then validates the response.
# ═══════════════════════════════════════════════════════════════
section("STEP 5 — ROUTER CHAIN")

router_prompt = ChatPromptTemplate.from_template(
    """Classify this query for a book Q&A system about "The Old Man and the Sea".

Routes:
  no_retrieval : greetings, maths, or facts the LLM already knows well
  simple       : lookup a single fact from the book
  branched     : multi-part question needing multiple sections of the book
  refuse       : harmful, off-topic, or impossible to answer from the book

Query: {query}

{format_instructions}"""
)

router_chain = (
    router_prompt
    | llm
    | router_parser   # ← returns RouteDecision, not a string
)

print("  router_chain output type: RouteDecision (Pydantic object)")
print("  access via: decision.route, decision.reason")

# ═══════════════════════════════════════════════════════════════
# STEP 6 — HANDLERS (reuse P1 and P3 chain patterns)
# ═══════════════════════════════════════════════════════════════
section("STEP 6 — HANDLERS")

# Shared RAG prompt
rag_prompt = ChatPromptTemplate.from_template("""
Answer ONLY using the context below.
If the answer is not there, say "I don't know."

Context:
{context}

Question: {question}
Answer:""")

# SIMPLE handler — same as P1
simple_chain = (
    {
        "context":  retriever | (lambda d: "\n\n".join(x.page_content for x in d)),
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# BRANCHED handler — same decompose+synthesise pattern as P3
decompose_chain = (
    ChatPromptTemplate.from_template(
        """Break this question into 2–4 simpler sub-questions answerable
from a single book section. Return a JSON array of strings only.
Question: {question}
Example: ["sub-question 1", "sub-question 2"]"""
    )
    | llm
    | JsonOutputParser()
)

synthesise_chain = (
    ChatPromptTemplate.from_template(
        """Answer the original question by combining the sub-answers.
Be concise. Note gaps where sub-answers say "I don't know."

Original: {original}
Sub-answers:
{sub_answers}

Final answer:"""
    )
    | llm
    | StrOutputParser()
)

print("  no_retrieval → LLM direct (no FAISS, cheapest)")
print("  simple       → P1 chain  (retrieve + answer)")
print("  branched     → P3 chain  (decompose + retrieve per sub-q + synthesise)")
print("  refuse       → static message (no LLM call)")

# ═══════════════════════════════════════════════════════════════
# STEP 7 — ADAPTIVE ROUTER FUNCTION
# ═══════════════════════════════════════════════════════════════
section("STEP 7 — ADAPTIVE ROUTER")

def rag_p5(question: str) -> dict:
    # Classify
    decision = router_chain.invoke({
        "query":               question,
        "format_instructions": router_parser.get_format_instructions(),
    })

    route  = decision.route
    reason = decision.reason

    # Route
    if route == Route.NO_RETRIEVAL:
        answer = llm.invoke(question).content
        return {"answer": answer, "contexts": [], "route": route, "reason": reason}

    elif route == Route.SIMPLE:
        answer = simple_chain.invoke(question)
        docs   = retriever.invoke(question)
        return {"answer": answer, "contexts": [d.page_content for d in docs],
                "route": route, "reason": reason}

    elif route == Route.BRANCHED:
        sub_qs = decompose_chain.invoke({"question": question})
        all_docs, sub_answers = [], []
        for sq in sub_qs:
            retrieved = retriever.invoke(sq)
            answer    = simple_chain.invoke(sq)
            all_docs.extend(retrieved)
            sub_answers.append(f"Q: {sq}\nA: {answer}")
        final = synthesise_chain.invoke({
            "original":    question,
            "sub_answers": "\n\n".join(sub_answers),
        })
        return {"answer": final, "contexts": [d.page_content for d in all_docs],
                "route": route, "reason": reason}

    else:  # REFUSE
        return {"answer": "I can't answer that from the available documents.",
                "contexts": [], "route": route, "reason": reason}

# ═══════════════════════════════════════════════════════════════
# STEP 8 — DEMO: show all 4 routes being triggered
# ═══════════════════════════════════════════════════════════════
section("STEP 8 — DEMO (all 4 routes)")

test_queries = [
    "Hello! How are you today?",                                      # → no_retrieval
    "How many days had the old man gone without catching a fish?",    # → simple
    "What does Santiago dream about, who is Manolin, and what attacks the fish on the way back?",  # → branched
    "Give me the recipe for the marlin Santiago caught",              # → refuse
]

for q in test_queries:
    print(f"\n  Query   : {q}")
    result = rag_p5(q)
    print(f"  Route   : {result['route']}   ← {result['reason']}")
    print(f"  Answer  : {result['answer'][:200]}")
    print(f"  Contexts: {len(result['contexts'])} chunks retrieved")

# ═══════════════════════════════════════════════════════════════
# STEP 9 — PIPELINE SUMMARY
# ═══════════════════════════════════════════════════════════════
section("STEP 9 — PIPELINE SUMMARY")

print(f"  LLM          : gpt-4o-mini")
print(f"  Embedding    : all-MiniLM-L6-v2 (local)")
print(f"""
  Flow:
    question
         │
         ▼
    ROUTER (PydanticOutputParser → RouteDecision object)
         │
         ├── no_retrieval → LLM direct answer      (0 FAISS calls)
         ├── simple       → P1 retrieve + answer   (1 FAISS call)
         ├── branched     → P3 decompose + N×FAISS (N FAISS calls)
         └── refuse       → static message         (0 LLM calls)

  PydanticOutputParser vs JsonOutputParser:
    JsonOutputParser  → raw dict  (no validation)
    PydanticOutputParser → typed Python object (validated, enum-constrained)
    RouteDecision.route is always one of 4 exact values — Pydantic enforces this
""")
