import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# REASON: OpenAI /embeddings endpoint is geo-blocked for this key (US-only restriction).
#         Chat completions (/chat/completions) works fine — only embeddings is affected.
#         Workaround: use a local HuggingFace model (runs on-device, no API call needed).
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DIVIDER = "=" * 60

def section(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)

# ═══════════════════════════════════════════════════════════════
# STEP 1 — LOAD
# PDF → list of Document objects, one per page
# Each Document has .page_content (str) and .metadata (dict)
# ═══════════════════════════════════════════════════════════════
section("STEP 1 — LOAD DOCUMENT")

pdf_path = os.path.join(os.path.dirname(__file__), "../data/llama2_tech_report.pdf")
docs = PyPDFLoader(pdf_path).load()

print(f"  Pages loaded     : {len(docs)}")
print(f"  First page chars : {len(docs[0].page_content)}")
print(f"  Metadata sample  : {docs[0].metadata}")
print(f"\n  --- First 300 chars of page 1 ---")
print(f"  {docs[0].page_content[:300].strip()}")

# ═══════════════════════════════════════════════════════════════
# STEP 2 — SPLIT
# Long pages → smaller overlapping chunks
# chunk_size=512 chars ≈ 100 words
# chunk_overlap=50  keeps context across chunk boundaries
# ═══════════════════════════════════════════════════════════════
section("STEP 2 — SPLIT INTO CHUNKS")

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks   = splitter.split_documents(docs)

print(f"  Pages  → Chunks  : {len(docs)} → {len(chunks)}")
print(f"  Avg chunk length : {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
print(f"\n  --- Chunk 0 ---")
print(f"  {chunks[0].page_content.strip()}")
print(f"\n  --- Chunk 1 (notice overlap with chunk 0) ---")
print(f"  {chunks[1].page_content.strip()}")

# ═══════════════════════════════════════════════════════════════
# STEP 3 — EMBED + INDEX
# Each chunk → 1536-dim vector via text-embedding-3-small
# FAISS stores vectors for fast cosine nearest-neighbour search
# Saved to disk so you don't re-embed on every run
# ═══════════════════════════════════════════════════════════════
section("STEP 3 — EMBED & BUILD VECTOR INDEX")

index_path = os.path.join(os.path.dirname(__file__), "../index/p1")
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # geo-blocked (see import above)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # 384-dim, runs locally

if os.path.exists(index_path):
    print("  Index found on disk — loading (skipping re-embed)")
    vectordb = FAISS.load_local(index_path, embeddings,
                                allow_dangerous_deserialization=True)
else:
    print(f"  Embedding {len(chunks)} chunks via all-MiniLM-L6-v2 (local) ...")
    vectordb = FAISS.from_documents(chunks, embeddings)
    os.makedirs(index_path, exist_ok=True)
    vectordb.save_local(index_path)
    print(f"  Index saved to {index_path}")

sample_vector = embeddings.embed_query("rajesh")
print(f"  Vector dimensions : {len(sample_vector)}")
print(f"  Sample vector[:5] : {[round(x, 4) for x in sample_vector[:5]]}")

# ═══════════════════════════════════════════════════════════════
# STEP 4 — RETRIEVE
# Given a question, find the 4 most similar chunks by cosine sim
# Shows exactly what context the LLM will see
# ═══════════════════════════════════════════════════════════════
section("STEP 4 — RETRIEVAL (semantic search)")

# QUESTION = "What safety measures were used in Llama 2?"
QUESTION = "Who are the authors of the document?"

retriever   = vectordb.as_retriever(search_kwargs={"k": 4})
raw_results = vectordb.similarity_search_with_score(QUESTION, k=4)

print(f"  Query : {QUESTION}")
print(f"\n  Top 4 chunks retrieved (lower score = more similar):\n")
for i, (doc, score) in enumerate(raw_results):
    print(f"  [{i+1}] score={score:.4f}  source={doc.metadata.get('source','?')}  page={doc.metadata.get('page','?')}")
    print(f"       {doc.page_content[:180].strip()} ...")
    print()

# ═══════════════════════════════════════════════════════════════
# STEP 5 — PROMPT CONSTRUCTION
# Shows the exact prompt sent to the LLM after filling in context
# ═══════════════════════════════════════════════════════════════
section("STEP 5 — PROMPT CONSTRUCTION")

prompt = ChatPromptTemplate.from_template("""
Answer ONLY using the context below.
If the answer is not there, say "I don't know."
Never guess or use knowledge from outside the context.

Context:
{context}

Question: {question}
Answer:""")

context_text = "\n\n".join(doc.page_content for doc, _ in raw_results)
filled_prompt = prompt.format_messages(context=context_text, question=QUESTION)

print("  Prompt sent to LLM:\n")
print(filled_prompt[0].content[:800].strip())
print("  ...")

# ═══════════════════════════════════════════════════════════════
# STEP 6 — GENERATE
# LLM reads the prompt and generates a grounded answer
# temperature=0 → deterministic, no hallucination risk
# ═══════════════════════════════════════════════════════════════
section("STEP 6 — GENERATE ANSWER (LLM)")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain = (
    {
        "context":  retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

answer = chain.invoke(QUESTION)

print(f"  Question : {QUESTION}")
print(f"\n  Answer   : {answer}")

# ═══════════════════════════════════════════════════════════════
# STEP 7 — FULL PIPELINE SUMMARY
# ═══════════════════════════════════════════════════════════════
section("STEP 7 — PIPELINE SUMMARY")

print(f"  PDF pages          : {len(docs)}")
print(f"  Chunks created     : {len(chunks)}")
print(f"  Embedding dims     : {len(sample_vector)}")
print(f"  Chunks retrieved   : 4")
print(f"  Model used         : gpt-4o-mini")
print(f"  Question           : {QUESTION}")
print(f"  Answer             : {answer[:120]} ...")
print(f"\n  Flow: PDF → Pages → Chunks → Vectors → FAISS → Retrieve → Prompt → LLM → Answer")
print()

# ═══════════════════════════════════════════════════════════════
# REUSABLE FUNCTION (for benchmarking with RAGAS later)
# ═══════════════════════════════════════════════════════════════

def rag_p1(question: str) -> dict:
    docs_used = retriever.invoke(question)
    answer    = chain.invoke(question)
    return {
        "answer":   answer,
        "contexts": [d.page_content for d in docs_used]
    }
q="what is the problem being addressed?"
print("\nQuestion\n")
print(q)
print("\nAnswer\n")
print(rag_p1(q))

# cmd-o/p:

# (venv) rajeshtvd@Rajeshs-MacBook-Air rag-project % python patterns/p1_simple_rag.py

# ============================================================
#   STEP 1 — LOAD DOCUMENT
# ============================================================
#   Pages loaded     : 14
#   First page chars : 5273
#   Metadata sample  : {'producer': 'pdfTeX-1.40.22', 'creator': 'LaTeX with hyperref', 'creationdate': '2024-08-18T13:38:22+01:00', 'author': '', 'title': '', 'subject': '', 'keywords': '', 'moddate': '2024-08-18T13:38:22+01:00', 'trapped': '/False', 'ptex.fullbanner': 'This is pdfTeX, Version 3.141592653-2.6-1.40.22 (TeX Live 2022/dev/Debian) kpathsea version 6.3.4/dev', 'source': '/Users/rajeshtvd/Documents/PROJECTS/ai_agentic_trails/RAG_types/rag-project/patterns/../data/llama2_tech_report.pdf', 'total_pages': 14, 'page': 0, 'page_label': '3268'}

#   --- First 300 chars of page 1 ---
#   Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (V olume 1: Long Papers) , pages 3268–3281
# August 11-16, 2024 ©2024 Association for Computational Linguistics
# Landmark Embedding: A Chunking-Free Embedding Method For
# Retrieval Augmented Long-Context Large Langua

# ============================================================
#   STEP 2 — SPLIT INTO CHUNKS
# ============================================================
#   Pages  → Chunks  : 14 → 123
#   Avg chunk length : 460 chars

#   --- Chunk 0 ---
#   Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (V olume 1: Long Papers) , pages 3268–3281
# August 11-16, 2024 ©2024 Association for Computational Linguistics
# Landmark Embedding: A Chunking-Free Embedding Method For
# Retrieval Augmented Long-Context Large Language Models
# Kun Luo1,2 Zheng Liu2††Shitao Xiao2 Tong Zhou1 Yubo Chen1 Jun Zhao1 Kang Liu1,2†
# 1Institute of Automation, Chinese Academy of Sciences
# 2Beijing Academy of Artificial Intelligence

#   --- Chunk 1 (notice overlap with chunk 0) ---
#   2Beijing Academy of Artificial Intelligence
# {luokun695, zhengliu1026}@gmail.com kliu@nlpr.ia.ac.cn
# Abstract
# Retrieval augmentation is a promising approach
# to handle long-context language modeling.
# However, the existing retrieval methods usu-
# ally work with the chunked context, which is
# prone to inferior quality of semantic represen-
# tation and incomplete retrieval of useful in-
# formation. In this work, we propose a new
# method for the retrieval augmentation of long-

# ============================================================
#   STEP 3 — EMBED & BUILD VECTOR INDEX
# ============================================================
# Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
# Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 8060.10it/s]
#   Index found on disk — loading (skipping re-embed)
#   Vector dimensions : 384
#   Sample vector[:5] : [0.0116, 0.0251, -0.0367, 0.0593, -0.0071]

# ============================================================
#   STEP 4 — RETRIEVAL (semantic search)
# ============================================================
#   Query : What safety measures were used in Llama 2?

#   Top 4 chunks retrieved (lower score = more similar):

#   [1] score=1.0713  source=/Users/rajeshtvd/Documents/PROJECTS/ai_agentic_trails/RAG_types/rag-project/patterns/../data/llama2_tech_report.pdf  page=8
#        as those associated with LLaMA-2. In particular,
# open-source LLMs may involve the incorporation
# of private or contentious data during the training
# phase. The usage of synthetic dat ...

#   [2] score=1.2845  source=/Users/rajeshtvd/Documents/PROJECTS/ai_agentic_trails/RAG_types/rag-project/patterns/../data/llama2_tech_report.pdf  page=8
#        unlimited length, multi-sourced scenarios. Deeper
# exploration into more efficient methods for curat-
# ing high-quality synthetic data can also be pursued
# in the future.
# 7 Ethical co ...

#   [3] score=1.3344  source=/Users/rajeshtvd/Documents/PROJECTS/ai_agentic_trails/RAG_types/rag-project/patterns/../data/llama2_tech_report.pdf  page=0
#        fortunately, the existing LLMs are usually con-
# strained by a limited size of context window, e.g.,
# 2K for LLaMA-1 (Touvron et al., 2023a) and 4K
# for LLaMA-2 (Touvron et al., 2023b ...

#   [4] score=1.3899  source=/Users/rajeshtvd/Documents/PROJECTS/ai_agentic_trails/RAG_types/rag-project/patterns/../data/llama2_tech_report.pdf  page=5
#        uation samples are longer than 4K, which is far
# beyond the context length of LLaMA-2. However,
# many of them are shorter than 16K, especially for
# Qasper, MultifieldQA, 2WikiMQA, and ...


# ============================================================
#   STEP 5 — PROMPT CONSTRUCTION
# ============================================================
#   Prompt sent to LLM:

# Answer ONLY using the context below.
# If the answer is not there, say "I don't know."
# Never guess or use knowledge from outside the context.

# Context:
# as those associated with LLaMA-2. In particular,
# open-source LLMs may involve the incorporation
# of private or contentious data during the training
# phase. The usage of synthetic data may also lead
# to potential bias during retrieval process.
# 8 Ackonwledgements
# This work was supported by the Strategic Priority
# Research Program of Chinese Academy of Sci-
# ences (No. XDA27020203) and National Science
# and Technology Major Project (2023ZD0121504).
# References
# 2023. Localllama. ntk-aware scaled rope allows llama

# unlimited length, multi-sourced scenarios. Deeper
# exploration into more efficient methods for curat-
# ing high-quality synthetic data can als
#   ...

# ============================================================
#   STEP 6 — GENERATE ANSWER (LLM)
# ============================================================
#   Question : What safety measures were used in Llama 2?

#   Answer   : I don't know.

# ============================================================
#   STEP 7 — PIPELINE SUMMARY
# ============================================================
#   PDF pages          : 14
#   Chunks created     : 123
#   Embedding dims     : 384
#   Chunks retrieved   : 4
#   Model used         : gpt-4o-mini
#   Question           : What safety measures were used in Llama 2?
#   Answer             : I don't know. ...

# ============================================================
#   STEP 7 — PIPELINE SUMMARY trail 2
# ============================================================
#   PDF pages          : 14
#   Chunks created     : 123
#   Embedding dims     : 384
#   Chunks retrieved   : 4
#   Model used         : gpt-4o-mini
#   Question           : Who are the authors of the document?
#   Answer             : The authors of the document are Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Ch ...


#   Flow: PDF → Pages → Chunks → Vectors → FAISS → Retrieve → Prompt → LLM → Answer

