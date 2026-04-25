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
    vectordb = FAISS.from_documents(chunks, embeddings)
    os.makedirs(index_path, exist_ok=True)
    vectordb.save_local(index_path)

retriever = vectordb.as_retriever(search_kwargs={"k": 4})
llm       = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ═══════════════════════════════════════════════════════════════
# STEP 4 — STATE
#
# EXTRA CONCEPT: Graph Extension Pattern
#
# P7 doesn't rewrite P6 from scratch — it extends it.
# The pattern is always the same:
#   1. Add new fields to State
#   2. Write new node functions
#   3. Rewire one edge (generate→END becomes generate→grade_answer)
#   4. Add the new nodes and edges
#
# SelfRAGState adds two fields to CRAGState:
#   ans_grade  — "faithful" or "hallucinated"
#   violations — list of specific unsupported claims found in the answer
#
# EXTRA CONCEPT: Why store violations as a list?
#
# When regenerating, we pass violations explicitly to the LLM:
#   "Do NOT include these unsupported claims: <list>"
#
# This is more effective than just saying "be faithful" because:
#   - "be faithful" is vague — LLM may repeat the same mistake
#   - "do not say X, Y, Z" is specific — LLM knows exactly what to avoid
#   - It's the same principle as few-shot negative examples
# ═══════════════════════════════════════════════════════════════
section("STEP 4 — STATE (SelfRAGState extends CRAGState)")

class SelfRAGState(TypedDict):
    # ── from CRAGState (P6) ──────────────────────────────────
    question:   str
    documents:  list
    grade:      str    # doc grade: "relevant" | "irrelevant"
    answer:     str
    retries:    int
    # ── new in P7 ────────────────────────────────────────────
    ans_grade:  str    # answer grade: "faithful" | "hallucinated"
    violations: list   # unsupported claims found in the answer

print("  New fields in P7:")
print("    ans_grade  — 'faithful' | 'hallucinated'")
print("    violations — list of specific unsupported claims")
print()
print("  Why violations as a list (not just a bool)?")
print("  → Passed to node_regen_strict as 'do NOT say X, Y, Z'")
print("  → Specific instructions beat vague 'be faithful' prompts")

# ═══════════════════════════════════════════════════════════════
# STEP 5 — DOC GRADER + REWRITER (same as P6)
# ═══════════════════════════════════════════════════════════════
section("STEP 5 — DOC GRADER + REWRITER (same as P6)")

class DocGrade(BaseModel):
    score:  Literal["relevant", "irrelevant"] = Field(description="Relevance score")
    reason: str = Field(description="One sentence justification")

doc_grade_parser = PydanticOutputParser(pydantic_object=DocGrade)

doc_grader_chain = (
    ChatPromptTemplate.from_template(
        """Strictly assess if these chunks can answer the question.
"relevant" only if chunks contain a direct answer.
Question: {question}
Documents: {docs}
{fmt}"""
    )
    | llm
    | doc_grade_parser
)

rewriter_chain = (
    ChatPromptTemplate.from_template(
        "Rewrite this query to be more specific for a literary text retrieval:\n{question}"
    )
    | llm
    | StrOutputParser()
)

print("  doc_grader_chain : (question + docs) → DocGrade")
print("  rewriter_chain   : question → better query string")

# ═══════════════════════════════════════════════════════════════
# STEP 6 — ANSWER GRADER  ← NEW in P7
#
# This is the second quality gate — checks the ANSWER, not the docs.
#
# Two-gate logic:
#   Gate 1 (P6): Are the retrieved docs relevant to the question?
#   Gate 2 (P7): Is the generated answer faithful to those docs?
#
# Gate 1 can pass but Gate 2 still catches hallucination.
# Example:
#   Docs: "Santiago caught the marlin on day 3 at sea"
#   Answer: "Santiago caught the marlin and sold it for $500"
#                                          ↑ not in context — Gate 2 catches this
#
# EXTRA CONCEPT: Literal type in Pydantic
# Literal["faithful", "hallucinated"] restricts the field to
# exactly those two strings — same as Enum but inline.
# ═══════════════════════════════════════════════════════════════
section("STEP 6 — ANSWER GRADER (second quality gate)")

class AnswerGrade(BaseModel):
    faithful:   bool      = Field(description="True if every claim is supported by context")
    violations: list[str] = Field(description="List of unsupported claims, empty if faithful")

ans_parser = PydanticOutputParser(pydantic_object=AnswerGrade)

answer_grader_chain = (
    ChatPromptTemplate.from_template(
        """Grade whether EVERY claim in this answer is directly supported by the context.
Be strict — if ANY detail was added beyond the context, mark as unfaithful and list it.

Context:
{context}

Answer: {answer}

{fmt}"""
    )
    | llm
    | ans_parser
)

print("  answer_grader_chain: (context + answer) → AnswerGrade")
print("  AnswerGrade.faithful   : bool")
print("  AnswerGrade.violations : list[str]  ← specific unsupported claims")
print()
print("  Two gates:")
print("    Gate 1 (doc grade)    — are these docs relevant to the question?")
print("    Gate 2 (answer grade) — is this answer grounded in the docs?")

# ═══════════════════════════════════════════════════════════════
# STEP 7 — NODES
# ═══════════════════════════════════════════════════════════════
section("STEP 7 — NODES")

def node_retrieve(state: SelfRAGState) -> dict:
    docs = retriever.invoke(state["question"])
    print(f"    [retrieve]     '{state['question'][:55]}' → {len(docs)} chunks")
    return {"documents": docs}

def node_grade_docs(state: SelfRAGState) -> dict:
    docs_text = "\n\n".join(d.page_content for d in state["documents"])
    result    = doc_grader_chain.invoke({
        "question": state["question"],
        "docs":     docs_text[:2000],
        "fmt":      doc_grade_parser.get_format_instructions(),
    })
    print(f"    [grade_docs]   score={result.score}")
    return {"grade": result.score}

def node_rewrite(state: SelfRAGState) -> dict:
    new_q = rewriter_chain.invoke({"question": state["question"]})
    print(f"    [rewrite]      retry #{state['retries']+1}: '{new_q[:55]}'")
    return {"question": new_q, "retries": state["retries"] + 1}

def node_generate(state: SelfRAGState) -> dict:
    ctx    = "\n\n".join(d.page_content for d in state["documents"])
    answer = llm.invoke(
        f"Answer ONLY from context. If not in context say 'I don't know.'\n"
        f"Context: {ctx}\nQuestion: {state['question']}\nAnswer:"
    ).content
    print(f"    [generate]     '{answer[:75]}'")
    return {"answer": answer}

def node_grade_answer(state: SelfRAGState) -> dict:
    # ← NEW in P7
    ctx    = "\n\n".join(d.page_content for d in state["documents"])
    result = answer_grader_chain.invoke({
        "context": ctx[:2000],
        "answer":  state["answer"],
        "fmt":     ans_parser.get_format_instructions(),
    })
    grade = "faithful" if result.faithful else "hallucinated"
    print(f"    [grade_answer] {grade}  violations={result.violations}")
    return {"ans_grade": grade, "violations": result.violations}

def node_regen_strict(state: SelfRAGState) -> dict:
    # ← NEW in P7
    # EXTRA CONCEPT: negative prompting
    # Explicitly listing violations is more reliable than "be faithful".
    # The LLM knows exactly what NOT to include.
    ctx   = "\n\n".join(d.page_content for d in state["documents"])
    viols = ", ".join(state.get("violations", [])) or "none"
    answer = llm.invoke(
        f"Answer ONLY from context below.\n"
        f"Do NOT include these unsupported claims: {viols}\n\n"
        f"Context: {ctx}\n"
        f"Question: {state['question']}\n"
        f"Answer:"
    ).content
    print(f"    [regen_strict] violations banned: {viols[:80]}")
    print(f"    [regen_strict] new answer: '{answer[:75]}'")
    return {"answer": answer}

def node_refuse(state: SelfRAGState) -> dict:
    msg = f"I couldn't find reliable information after {state['retries']} retries."
    print(f"    [refuse]       {msg}")
    return {"answer": msg}

# ═══════════════════════════════════════════════════════════════
# STEP 8 — CONDITIONAL EDGES
# ═══════════════════════════════════════════════════════════════
section("STEP 8 — CONDITIONAL EDGES")

def route_after_doc_grade(state: SelfRAGState) -> str:
    if state["grade"] == "relevant":   return "generate"
    elif state["retries"] < 2:         return "rewrite"
    else:                              return "refuse"

def route_after_ans_grade(state: SelfRAGState) -> str:
    # ← NEW in P7
    return "end" if state["ans_grade"] == "faithful" else "regen"

print("  route_after_doc_grade : relevant → generate | irrelevant → rewrite/refuse")
print("  route_after_ans_grade : faithful → END      | hallucinated → regen_strict")

# ═══════════════════════════════════════════════════════════════
# STEP 9 — BUILD GRAPH
#
# EXTRA CONCEPT: How P7 extends P6
#
# P6 had:   generate → END
# P7 has:   generate → grade_answer → (END or regen_strict → END)
#
# That's the only structural change. Everything else is identical.
# This is the graph extension pattern — add nodes, rewire one edge.
# ═══════════════════════════════════════════════════════════════
section("STEP 9 — BUILD GRAPH (extending P6)")

wf = StateGraph(SelfRAGState)

# ── nodes (same as P6 + two new ones) ───────────────────────
wf.add_node("retrieve",     node_retrieve)
wf.add_node("grade_docs",   node_grade_docs)
wf.add_node("rewrite",      node_rewrite)
wf.add_node("generate",     node_generate)
wf.add_node("grade_answer", node_grade_answer)   # NEW
wf.add_node("regen_strict", node_regen_strict)   # NEW
wf.add_node("refuse",       node_refuse)

# ── edges ────────────────────────────────────────────────────
wf.set_entry_point("retrieve")
wf.add_edge("retrieve", "grade_docs")
wf.add_conditional_edges("grade_docs", route_after_doc_grade, {
    "generate": "generate",
    "rewrite":  "rewrite",
    "refuse":   "refuse",
})
wf.add_edge("rewrite",  "retrieve")             # doc-grade loop (same as P6)
wf.add_edge("generate", "grade_answer")         # ← replaces generate→END from P6
wf.add_conditional_edges("grade_answer", route_after_ans_grade, {
    "end":  END,                                # faithful   → done
    "regen": "regen_strict",                    # hallucinated → fix it
})
wf.add_edge("regen_strict", END)               # after strict regen → done
wf.add_edge("refuse",       END)

self_rag_app = wf.compile()

print("""
  P6 graph:          P7 graph (one rewire + two new nodes):

  retrieve           retrieve
      │                  │
  grade_docs         grade_docs
      │                  │
  ┌───┴───┐          ┌───┴───┐
  │       │          │       │
  rewrite generate   rewrite generate
  │       │          │       │
  └─►retrieve  END   └─►retrieve  grade_answer
                                      │
                              ┌───────┴───────┐
                              │               │
                             END         regen_strict
                          (faithful)         │
                                            END
                                      (hallucinated)
""")

# ═══════════════════════════════════════════════════════════════
# STEP 10 — DEMO
# ═══════════════════════════════════════════════════════════════
section("STEP 10 — DEMO")

test_cases = [
    ("Factual — should pass both gates",
     "How many days had the old man gone without catching a fish?"),
    ("Interpretive — answer grade may catch over-reaching",
     "What motivates old man to keep fighting the fish from shark attacks?"),
    ("Off-topic — should refuse",
     "What year was Ernest Hemingway born?"),
]

for label, question in test_cases:
    print(f"\n  [{label}]")
    print(f"  Q: {question}")
    result = self_rag_app.invoke({
        "question":   question,
        "documents":  [],
        "grade":      "",
        "answer":     "",
        "retries":    0,
        "ans_grade":  "",
        "violations": [],
    })
    print(f"  A: {result['answer'][:220]}")
    print(f"  doc retries: {result['retries']}  |  ans_grade: {result['ans_grade']}")

# ═══════════════════════════════════════════════════════════════
# STEP 11 — SUMMARY
# ═══════════════════════════════════════════════════════════════
section("STEP 11 — PIPELINE SUMMARY")

print(f"  LLM        : gpt-4o-mini")
print(f"  Embedding  : all-MiniLM-L6-v2 (local)")
print(f"""
  P6 → P7 change (graph extension pattern):
    Before: generate → END
    After:  generate → grade_answer → END  (faithful)
                                    → regen_strict → END  (hallucinated)

  Two quality gates:
    Gate 1 — doc grade   : are retrieved chunks relevant to the question?
    Gate 2 — answer grade: is the generated answer grounded in those chunks?

  Extra concepts introduced:
    ① Graph extension pattern — add nodes, rewire one edge, done
    ② Negative prompting     — "do NOT say X, Y, Z" beats "be faithful"
    ③ violations list        — specific unsupported claims extracted by grader,
                               fed back to regen node to prevent repetition
    ④ Literal["a","b"]       — inline Pydantic type constraint (vs Enum)
""")

# ═══════════════════════════════════════════════════════════════
# REUSABLE FUNCTION
# ═══════════════════════════════════════════════════════════════

def rag_p7(question: str) -> dict:
    result = self_rag_app.invoke({
        "question":   question,
        "documents":  [],
        "grade":      "",
        "answer":     "",
        "retries":    0,
        "ans_grade":  "",
        "violations": [],
    })
    return {
        "answer":     result["answer"],
        "contexts":   [d.page_content for d in result["documents"]],
        "retries":    result["retries"],
        "ans_grade":  result["ans_grade"],
        "violations": result["violations"],
    }
