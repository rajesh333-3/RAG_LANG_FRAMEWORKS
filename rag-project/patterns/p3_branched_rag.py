import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

load_dotenv()

DIVIDER = "=" * 60

def section(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)

# ═══════════════════════════════════════════════════════════════
# STEPS 1-3 — LOAD / SPLIT / EMBED (same as P1, reusing index)
# ═══════════════════════════════════════════════════════════════
section("STEPS 1-3 — LOAD, SPLIT, EMBED (reusing P1 index)")

pdf_path   = os.path.join(os.path.dirname(__file__), "../data/oldman_and_the_sea.pdf")
docs       = PyPDFLoader(pdf_path).load()
splitter   = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks     = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
index_path = os.path.join(os.path.dirname(__file__), "../index/p1")  # reuse P1 index

if os.path.exists(index_path):
    print("  Index found on disk — loading (skipping re-embed)")
    vectordb = FAISS.load_local(index_path, embeddings,
                                allow_dangerous_deserialization=True)
else:
    print(f"  Embedding {len(chunks)} chunks via all-MiniLM-L6-v2 ...")
    vectordb = FAISS.from_documents(chunks, embeddings)
    os.makedirs(index_path, exist_ok=True)
    vectordb.save_local(index_path)
    print(f"  Index saved to {index_path}")

retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# ═══════════════════════════════════════════════════════════════
# STEP 4 — DECOMPOSE CHAIN  ← NEW in P3
#
# Problem with P1 and P2 for complex questions:
#   "What is Landmark Embedding, how does it work, and what
#    experiments were run to validate it?"
#
#   A single retrieval call fetches 4 chunks — likely all from
#   one section, missing the others entirely.
#
# Fix: decompose into sub-questions, retrieve independently
#   for each, then synthesise all answers into one final answer.
#
# JsonOutputParser — instead of returning a string like
#   StrOutputParser does, it parses the LLM output as JSON
#   and returns a native Python object (here: a list of strings).
#   The LLM must be instructed to output valid JSON.
# ═══════════════════════════════════════════════════════════════
section("STEP 4 — DECOMPOSE CHAIN (JsonOutputParser)")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

decompose_chain = (
    ChatPromptTemplate.from_template(
        """Break this question into 2–4 simpler sub-questions that can each
be answered by a single document section.
Return a JSON array of strings, nothing else.
Question: {question}
Example output: ["sub-question 1", "sub-question 2"]"""
    )
    | llm
    | JsonOutputParser()  # ← returns Python list, not a string
    # Compare with P1/P2 which use StrOutputParser() → plain string
)

print("  decompose_chain output type : Python list (via JsonOutputParser)")
print("  decompose_chain input       : {question}")
print("  decompose_chain output      : ['sub-q 1', 'sub-q 2', ...]")

# ═══════════════════════════════════════════════════════════════
# STEP 5 — RAG CHAIN (same as P1 — answers each sub-question)
# ═══════════════════════════════════════════════════════════════
section("STEP 5 — RAG CHAIN (same as P1, runs per sub-question)")

rag_prompt = ChatPromptTemplate.from_template("""
Answer ONLY using the context below.
If the answer is not there, say "I don't know."
Never guess or use knowledge from outside the context.

Context:
{context}

Question: {question}
Answer:""")

rag_chain = (
    {
        "context":  retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# ═══════════════════════════════════════════════════════════════
# STEP 6 — SYNTHESISE CHAIN  ← NEW in P3
#
# After all sub-questions are answered independently,
# the synthesiser sees:
#   - the original complex question
#   - all sub-Q + sub-A pairs
# and produces one coherent final answer.
# ═══════════════════════════════════════════════════════════════
section("STEP 6 — SYNTHESISE CHAIN")

synthesise_chain = (
    ChatPromptTemplate.from_template(
        """Answer the original question by combining the sub-answers below.
Be concise. Note any gaps where sub-answers say "I don't know."

Original question: {original}
Sub-answers:
{sub_answers}

Final answer:"""
    )
    | llm
    | StrOutputParser()
)

print("  synthesise_chain input  : {original, sub_answers}")
print("  synthesise_chain output : final combined answer string")

# ═══════════════════════════════════════════════════════════════
# STEP 7 — DEMO
#
# Flow per question:
#   complex question
#        │
#        ▼
#   DECOMPOSE  → ["sub-q1", "sub-q2", "sub-q3"]
#        │
#        ▼ (loop — each sub-question runs independently)
#   RETRIEVE + ANSWER  → "Q: sub-q1\nA: ..."
#   RETRIEVE + ANSWER  → "Q: sub-q2\nA: ..."
#   RETRIEVE + ANSWER  → "Q: sub-q3\nA: ..."
#        │
#        ▼
#   SYNTHESISE  → final answer
# ═══════════════════════════════════════════════════════════════
section("STEP 7 — BRANCHED RAG DEMO")

QUESTION = "What does Santiago dream about, how many days had he gone without a fish, and what did the shark do?"

print(f"\n  Complex question:\n  {QUESTION}\n")

# Step A: decompose
sub_qs = decompose_chain.invoke({"question": QUESTION})
print(f"  Decomposed into {len(sub_qs)} sub-questions:")
for i, sq in enumerate(sub_qs):
    print(f"    [{i+1}] {sq}")

# Step B: retrieve + answer each sub-question independently
all_docs, sub_answers = [], []
print()
for i, sq in enumerate(sub_qs):
    retrieved = retriever.invoke(sq)
    answer    = rag_chain.invoke(sq)
    all_docs.extend(retrieved)
    sub_answers.append(f"Q: {sq}\nA: {answer}")
    print(f"  [{i+1}] Sub-Q  : {sq}")
    print(f"       Sub-A  : {answer[:180]}")
    print()

# Step C: synthesise
final_answer = synthesise_chain.invoke({
    "original":    QUESTION,
    "sub_answers": "\n\n".join(sub_answers),
})

print(f"  Final synthesised answer:\n")
print(f"  {final_answer}")

# ═══════════════════════════════════════════════════════════════
# STEP 8 — PIPELINE SUMMARY
# ═══════════════════════════════════════════════════════════════
section("STEP 8 — PIPELINE SUMMARY")

unique_docs = {d.metadata.get("page"): d for d in all_docs}
print(f"  Sub-questions generated : {len(sub_qs)}")
print(f"  Total chunks retrieved  : {len(all_docs)} ({len(unique_docs)} unique pages)")
print(f"  Embedding model         : all-MiniLM-L6-v2 (local)")
print(f"  LLM                     : gpt-4o-mini")
print(f"""
  Flow:
    complex question
         │
         ▼
    DECOMPOSE (JsonOutputParser → Python list of sub-questions)
         │
         ├──► sub-q 1 → RETRIEVE → RAG CHAIN → sub-answer 1
         ├──► sub-q 2 → RETRIEVE → RAG CHAIN → sub-answer 2
         └──► sub-q N → RETRIEVE → RAG CHAIN → sub-answer N
         │
         ▼
    SYNTHESISE (original + all sub-answers → final answer)
""")

# ═══════════════════════════════════════════════════════════════
# REUSABLE FUNCTION (for benchmarking with RAGAS later)
# ═══════════════════════════════════════════════════════════════

def rag_p3(question: str) -> dict:
    sub_qs = decompose_chain.invoke({"question": question})

    all_docs, sub_answers = [], []
    for sq in sub_qs:
        retrieved = retriever.invoke(sq)
        answer    = rag_chain.invoke(sq)
        all_docs.extend(retrieved)
        sub_answers.append(f"Q: {sq}\nA: {answer}")

    final = synthesise_chain.invoke({
        "original":    question,
        "sub_answers": "\n\n".join(sub_answers),
    })
    return {
        "answer":         final,
        "contexts":       [d.page_content for d in all_docs],
        "sub_questions":  sub_qs,
    }
