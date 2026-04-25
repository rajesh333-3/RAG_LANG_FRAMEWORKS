import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

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

retriever = vectordb.as_retriever(search_kwargs={"k": 4})
llm       = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ═══════════════════════════════════════════════════════════════
# STEP 4 — STATE
#
# Why P1 can't do what CRAG does:
#   P1 is a linear chain: retrieve → prompt → generate
#   If the retrieved chunks are irrelevant, P1 still generates
#   an answer — likely hallucinated or wrong.
#
#   CRAG adds a question between retrieval and generation:
#   "Are these chunks actually relevant to the question?"
#   If not → rewrite the query and try again (loop).
#   If still not after 2 tries → refuse honestly.
#
#   A chain can't loop. A graph can. That's the whole point.
#
# CRAGState carries:
#   question  — may be rewritten by node_rewrite
#   documents — filled by node_retrieve
#   grade     — "relevant" or "irrelevant" — set by node_grade
#   answer    — final answer — set by node_generate or node_refuse
#   retries   — counts how many rewrite+retry cycles have happened
# ═══════════════════════════════════════════════════════════════
section("STEP 4 — STATE (CRAGState)")

class CRAGState(TypedDict):
    question:  str
    documents: list
    grade:     str   # "relevant" | "irrelevant"
    answer:    str
    retries:   int

print("  CRAGState fields:")
print("    question  — may be rewritten mid-graph")
print("    documents — filled by retrieve node")
print("    grade     — set by grade node (relevant | irrelevant)")
print("    answer    — set by generate or refuse node")
print("    retries   — loop counter (max 2 before refuse)")

# ═══════════════════════════════════════════════════════════════
# STEP 5 — GRADER CHAIN (reuses PydanticOutputParser from P5)
# ═══════════════════════════════════════════════════════════════
section("STEP 5 — GRADER + REWRITER CHAINS")

class DocGrade(BaseModel):
    score:  Literal["relevant", "irrelevant"] = Field(description="Relevance score")
    reason: str                                = Field(description="One sentence justification")

grade_parser = PydanticOutputParser(pydantic_object=DocGrade)

grader_chain = (
    ChatPromptTemplate.from_template(
        """Strictly assess if these document chunks can answer the question.
Score "relevant" only if the chunks contain a direct answer.
Score "irrelevant" if the chunks are off-topic or too vague.

Question: {question}
Documents: {docs}

{fmt}"""
    )
    | llm
    | grade_parser
)

rewriter_chain = (
    ChatPromptTemplate.from_template(
        """Rewrite this query to be more specific for retrieving relevant
passages from a literary text. Keep it concise.

Original query: {question}
Rewritten query:"""
    )
    | llm
    | StrOutputParser()
)

print("  grader_chain  : (question + docs) → DocGrade(score, reason)")
print("  rewriter_chain: question → better query string")

# ═══════════════════════════════════════════════════════════════
# STEP 6 — NODES
#
# Each node: (state: CRAGState) → dict of ONLY changed fields
# ═══════════════════════════════════════════════════════════════
section("STEP 6 — NODES")

def node_retrieve(state: CRAGState) -> dict:
    # Reads:   question
    # Returns: documents
    docs = retriever.invoke(state["question"])
    print(f"    [retrieve] query='{state['question'][:60]}' → {len(docs)} chunks")
    return {"documents": docs}

def node_grade(state: CRAGState) -> dict:
    # Reads:   question, documents
    # Returns: grade
    docs_text = "\n\n".join(d.page_content for d in state["documents"])
    result    = grader_chain.invoke({
        "question": state["question"],
        "docs":     docs_text[:2000],   # cap to avoid huge prompts
        "fmt":      grade_parser.get_format_instructions(),
    })
    print(f"    [grade]    score={result.score}  reason={result.reason[:60]}")
    return {"grade": result.score}

def node_rewrite(state: CRAGState) -> dict:
    # Reads:   question, retries
    # Returns: question (rewritten), retries (incremented)
    new_q = rewriter_chain.invoke({"question": state["question"]})
    print(f"    [rewrite]  '{state['question'][:50]}' → '{new_q[:50]}'  (retry #{state['retries']+1})")
    return {"question": new_q, "retries": state["retries"] + 1}

def node_generate(state: CRAGState) -> dict:
    # Reads:   documents, question
    # Returns: answer
    ctx    = "\n\n".join(d.page_content for d in state["documents"])
    answer = llm.invoke(
        f"Answer ONLY from context. If not in context say 'I don't know.'\n"
        f"Context: {ctx}\nQuestion: {state['question']}\nAnswer:"
    ).content
    print(f"    [generate] answer='{answer[:80]}'")
    return {"answer": answer}

def node_refuse(state: CRAGState) -> dict:
    # Reads:   question (original, possibly rewritten)
    # Returns: answer
    msg = f"I couldn't find reliable information about this in the book after {state['retries']} retries."
    print(f"    [refuse]   {msg}")
    return {"answer": msg}

print("  node_retrieve : question → documents")
print("  node_grade    : question + documents → grade")
print("  node_rewrite  : question → rewritten question + retries+1")
print("  node_generate : documents + question → answer")
print("  node_refuse   : → honest refusal message")

# ═══════════════════════════════════════════════════════════════
# STEP 7 — CONDITIONAL EDGE (the decision diamond)
#
# This is the function that makes the loop possible.
# Returns a string key that maps to the next node.
#   "generate" → docs are good, move forward
#   "rewrite"  → docs are bad, go BACKWARDS to retrieve (loop!)
#   "refuse"   → too many retries, give up honestly
# ═══════════════════════════════════════════════════════════════
section("STEP 7 — CONDITIONAL EDGE (the loop logic)")

def route_after_grade(state: CRAGState) -> str:
    if state["grade"] == "relevant":
        return "generate"
    elif state["retries"] < 2:
        return "rewrite"   # ← LOOP BACK
    else:
        return "refuse"

print("  route_after_grade:")
print("    grade=relevant          → 'generate'  (move forward)")
print("    grade=irrelevant + retries<2 → 'rewrite'  (LOOP BACK)")
print("    grade=irrelevant + retries≥2 → 'refuse'   (give up honestly)")

# ═══════════════════════════════════════════════════════════════
# STEP 8 — BUILD AND COMPILE
# ═══════════════════════════════════════════════════════════════
section("STEP 8 — BUILD GRAPH")

wf = StateGraph(CRAGState)

wf.add_node("retrieve", node_retrieve)
wf.add_node("grade",    node_grade)
wf.add_node("rewrite",  node_rewrite)
wf.add_node("generate", node_generate)
wf.add_node("refuse",   node_refuse)

wf.set_entry_point("retrieve")
wf.add_edge("retrieve", "grade")           # retrieve → grade (always)
wf.add_conditional_edges("grade", route_after_grade, {
    "generate": "generate",                # relevant  → generate
    "rewrite":  "rewrite",                 # irrelevant, retries<2 → rewrite
    "refuse":   "refuse",                  # irrelevant, retries≥2 → refuse
})
wf.add_edge("rewrite",  "retrieve")        # THE LOOP — rewrite → back to retrieve
wf.add_edge("generate", END)
wf.add_edge("refuse",   END)

crag_app = wf.compile()

print("""
  Graph flow:

  START
    │
    ▼
  retrieve
    │
    ▼
  grade ──► [route_after_grade]
               │
               ├── relevant         → generate → END
               │
               ├── irrelevant       → rewrite
               │   retries < 2        │
               │                      └──► retrieve  (LOOP)
               │
               └── irrelevant       → refuse → END
                   retries ≥ 2
""")

# ═══════════════════════════════════════════════════════════════
# STEP 9 — DEMO
# Three cases:
#   1. Good retrieval  → straight through, no loop
#   2. Vague query     → loop triggers, rewrite improves it
#   3. Off-topic       → hits retry limit, refuses honestly
# ═══════════════════════════════════════════════════════════════
section("STEP 9 — DEMO")

test_cases = [
    ("Good retrieval",
     "How many days had Santiago gone without catching a fish?"),
    ("Vague query that may trigger rewrite",
     "What does the old man think about at night?"),
    ("Off-topic — should refuse",
     "What is the population of Cuba?"),
]

for label, question in test_cases:
    print(f"\n  [{label}]")
    print(f"  Q: {question}")
    result = crag_app.invoke({
        "question":  question,
        "documents": [],
        "grade":     "",
        "answer":    "",
        "retries":   0,
    })
    print(f"  A: {result['answer'][:200]}")
    print(f"  retries used: {result['retries']}")

# ═══════════════════════════════════════════════════════════════
# STEP 10 — PIPELINE SUMMARY
# ═══════════════════════════════════════════════════════════════
section("STEP 10 — PIPELINE SUMMARY")

print(f"  LLM          : gpt-4o-mini")
print(f"  Embedding    : all-MiniLM-L6-v2 (local)")
print(f"  New pattern  : StateGraph with corrective loop")
print(f"""
  Why P1 can't do this:
    P1 chain is linear — retrieve → prompt → generate
    No way to loop back, no way to grade, no way to rewrite
    Bad retrieval → hallucinated answer, silently

  What CRAG adds:
    Grade step after retrieval — "are these chunks actually useful?"
    Rewrite loop  — bad grade → rewrite query → retry (max 2x)
    Refuse node   — loop exhausted → honest "I don't know"
    Result: faithfulness jumps from ~0.42 (P1) to ~0.82 (P6)
""")

# ═══════════════════════════════════════════════════════════════
# REUSABLE FUNCTION
# ═══════════════════════════════════════════════════════════════

def rag_p6(question: str) -> dict:
    result = crag_app.invoke({
        "question":  question,
        "documents": [],
        "grade":     "",
        "answer":    "",
        "retries":   0,
    })
    return {
        "answer":   result["answer"],
        "contexts": [d.page_content for d in result["documents"]],
        "retries":  result["retries"],
    }
