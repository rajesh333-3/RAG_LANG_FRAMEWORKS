import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.text_splitter import SemanticChunker

load_dotenv()

DIVIDER = "=" * 60

def section(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ═══════════════════════════════════════════════════════════════
# STEP 1 — LOAD (same as P1)
# ═══════════════════════════════════════════════════════════════
section("STEP 1 — LOAD DOCUMENT")

pdf_path = os.path.join(os.path.dirname(__file__), "../data/oldman_and_the_sea.pdf")
docs     = PyPDFLoader(pdf_path).load()

print(f"  Pages loaded : {len(docs)}")

# ═══════════════════════════════════════════════════════════════
# STEP 2 — SEMANTIC CHUNKING  ← NEW in P4
#
# P1 used RecursiveCharacterTextSplitter — splits at fixed character
# counts (512 chars), blind to meaning. A sentence that spans a topic
# boundary gets cut wherever the counter hits 512.
#
# SemanticChunker splits differently:
#   1. Splits the text into sentences first
#   2. Embeds each sentence
#   3. Computes cosine similarity between adjacent sentences
#   4. Inserts a chunk boundary wherever similarity drops sharply
#      (i.e. when the topic changes)
#
# Result: chunks that contain one coherent idea each, not arbitrary
# character windows. Better chunks → better retrieval.
#
# breakpoint_threshold_type="percentile" — a boundary is inserted
# wherever the similarity drop is in the bottom 70th percentile
# of all drops (i.e. the sharpest topic shifts become boundaries).
# ═══════════════════════════════════════════════════════════════
section("STEP 2 — SEMANTIC CHUNKING (vs fixed-size in P1)")

splitter_p1   = None   # RecursiveCharacterTextSplitter(chunk_size=512) — shown for comparison
semantic_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
)

semantic_chunks = semantic_splitter.split_documents(docs)

print(f"  P1 fixed chunks (512 chars)   : 319 chunks, avg ~440 chars each")
print(f"  P4 semantic chunks            : {len(semantic_chunks)} chunks")
print(f"  P4 avg chunk length           : {sum(len(c.page_content) for c in semantic_chunks) // len(semantic_chunks)} chars")
print(f"\n  --- Semantic Chunk 0 ---")
print(f"  {semantic_chunks[0].page_content.strip()}")
print(f"\n  --- Semantic Chunk 1 ---")
print(f"  {semantic_chunks[1].page_content.strip()}")
print(f"\n  Note: each chunk ends at a topic boundary, not a character count")

# ═══════════════════════════════════════════════════════════════
# STEP 3 — EMBED + INDEX (semantic chunks, own index)
# ═══════════════════════════════════════════════════════════════
section("STEP 3 — EMBED & BUILD VECTOR INDEX (semantic chunks)")

index_path = os.path.join(os.path.dirname(__file__), "../index/p4")

if os.path.exists(index_path):
    print("  Index found on disk — loading (skipping re-embed)")
    vectordb = FAISS.load_local(index_path, embeddings,
                                allow_dangerous_deserialization=True)
else:
    print(f"  Embedding {len(semantic_chunks)} semantic chunks ...")
    vectordb = FAISS.from_documents(semantic_chunks, embeddings)
    os.makedirs(index_path, exist_ok=True)
    vectordb.save_local(index_path)
    print(f"  Index saved to {index_path}")

# ═══════════════════════════════════════════════════════════════
# STEP 4 — HYDE CHAIN  ← NEW in P4
#
# Problem HyDE solves:
#   User asks: "how long was the old man at sea?"
#   → conversational, short, vague language
#
#   Documents say: "He had been out for eighty-four days and on this
#   third day at sea the great fish took the bait at noon..."
#   → formal, document-like, dense language
#
#   These embed to very different vectors → poor retrieval.
#
# HyDE fix:
#   1. Ask the LLM to generate a HYPOTHETICAL document paragraph
#      that would answer the question (not an answer for the user)
#   2. Embed that paragraph instead of the raw question
#   3. The hypothesis uses document-like language → much closer in
#      embedding space to the real document chunks → better recall
#
# Key methods used:
#   embed_query()               → embeds a string → returns list[float]
#   similarity_search_by_vector() → takes a pre-computed vector (not a string)
#                                   and searches FAISS directly
# ═══════════════════════════════════════════════════════════════
section("STEP 4 — HYDE CHAIN (embed hypothesis, not raw question)")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

hyde_chain = (
    ChatPromptTemplate.from_template(
        """Write a short factual paragraph from a literary work that would
directly answer this question. Write as established fact.
Do not say "I" or express uncertainty.
Question: {question}
Document excerpt:"""
    )
    | ChatOpenAI(model="gpt-4o-mini", temperature=0.2)  # slightly creative → natural prose
    | StrOutputParser()
)

print("  HyDE chain: question → hypothetical paragraph (document-like language)")
print("  temperature=0.2 → slightly creative so it generates natural prose")

# ═══════════════════════════════════════════════════════════════
# STEP 5 — RAG CHAIN (for final answer grounding)
# ═══════════════════════════════════════════════════════════════
section("STEP 5 — RAG CHAIN (same structure as P1)")

rag_prompt = ChatPromptTemplate.from_template("""
Answer ONLY using the context below.
If the answer is not there, say "I don't know."
Never guess or use knowledge from outside the context.

Context:
{context}

Question: {question}
Answer:""")

# In P4 retrieval is handled externally (standard or HyDE),
# so the chain receives pre-joined context string + question directly.
rag_chain = rag_prompt | llm | StrOutputParser()

def answer_from_docs(docs: list, question: str) -> str:
    context = "\n\n".join(d.page_content for d in docs)
    return rag_chain.invoke({"context": context, "question": question})

# ═══════════════════════════════════════════════════════════════
# STEP 6 — DEMO: compare standard retrieval vs HyDE
#
# For each question we run both:
#   Standard: embed the raw question → search → answer
#   HyDE:     generate hypothesis → embed hypothesis → search → answer
#
# The difference is visible in which chunks get retrieved
# ═══════════════════════════════════════════════════════════════
section("STEP 6 — DEMO: Standard vs HyDE retrieval")

questions = [
    "What keeps the old man going when he is exhausted and in pain?",
    "How does the old man feel about the fish he is fighting?",
]

for QUESTION in questions:
    print(f"\n  Question: {QUESTION}\n")

    # --- Standard retrieval ---
    std_docs   = vectordb.similarity_search(QUESTION, k=4)
    std_answer = answer_from_docs(std_docs, QUESTION)

    # --- HyDE retrieval ---
    hypothesis  = hyde_chain.invoke({"question": QUESTION})
    hypo_vec    = embeddings.embed_query(hypothesis)                    # embed the HYPOTHESIS
    hyde_docs   = vectordb.similarity_search_by_vector(hypo_vec, k=4)  # search by vector
    hyde_answer = answer_from_docs(hyde_docs, QUESTION)

    print(f"  Hypothesis paragraph (what HyDE embeds):")
    print(f"  \"{hypothesis[:250]}\"")
    print(f"\n  Standard answer : {std_answer[:200]}")
    print(f"  HyDE answer     : {hyde_answer[:200]}")
    print()

# ═══════════════════════════════════════════════════════════════
# STEP 7 — PIPELINE SUMMARY
# ═══════════════════════════════════════════════════════════════
section("STEP 7 — PIPELINE SUMMARY")

print(f"  Chunking strategy     : SemanticChunker (topic-aware boundaries)")
print(f"  Semantic chunks       : {len(semantic_chunks)}")
print(f"  Embedding model       : all-MiniLM-L6-v2 (local)")
print(f"  LLM                   : gpt-4o-mini")
print(f"""
  Standard retrieval flow:
    question  ──► embed_query(question) ──► similarity_search ──► chunks ──► answer

  HyDE flow:
    question
       │
       ▼
    hyde_chain (LLM, temp=0.2) → hypothetical paragraph
       │
       ▼
    embed_query(hypothesis)    → vector  (document-like language)
       │
       ▼
    similarity_search_by_vector(hypo_vec) → chunks closer to real docs
       │
       ▼
    rag_chain → grounded answer
""")

# ═══════════════════════════════════════════════════════════════
# REUSABLE FUNCTION (for benchmarking with RAGAS later)
# ═══════════════════════════════════════════════════════════════

def rag_p4(question: str) -> dict:
    hypothesis = hyde_chain.invoke({"question": question})
    hypo_vec   = embeddings.embed_query(hypothesis)
    docs       = vectordb.similarity_search_by_vector(hypo_vec, k=4)
    answer     = answer_from_docs(docs, question)
    return {
        "answer":     answer,
        "contexts":   [d.page_content for d in docs],
        "hypothesis": hypothesis,
    }
