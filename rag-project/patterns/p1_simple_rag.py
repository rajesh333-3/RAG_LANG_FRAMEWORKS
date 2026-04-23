from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ═══ PHASE 1 — INDEXING (run once, reuse) ════════════════════

# Load PDF → list of Document objects (one per page)
docs     = PyPDFLoader("data/your_document.pdf").load()

# Split into overlapping chunks
# chunk_size=512 chars ≈ 100 words   chunk_overlap=50 preserves boundary context
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks   = splitter.split_documents(docs)
print(f"Document split into {len(chunks)} chunks")

# Embed every chunk and build FAISS index
# Each chunk → 1536-dim vector (text-embedding-3-small)
# FAISS stores them for fast cosine nearest-neighbour lookup
vectordb = FAISS.from_documents(chunks, OpenAIEmbeddings())
vectordb.save_local("index/p1")   # save so you don't re-embed every run

# ═══ PHASE 2 — QUERYING (run per user question) ═════════════

vectordb  = FAISS.load_local("index/p1", OpenAIEmbeddings(),
                              allow_dangerous_deserialization=True)

# .as_retriever() wraps FAISS as a LangChain Runnable
# k=4 → return 4 most similar chunks per query
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_template("""
Answer ONLY using the context below.
If the answer is not there, say "I don't know."
Never guess or use knowledge from outside the context.

Context:
{context}

Question: {question}
Answer:""")

# LCEL chain — read this left to right:
# 1. {"context": retriever, "question": RunnablePassthrough()}
#    → retriever runs on the question, returns 4 docs
#    → RunnablePassthrough passes the question through unchanged
#    → result: {"context": "chunk1\n\nchunk2...", "question": "..."}
# 2. | prompt  → fills {context} and {question} into the template
# 3. | ChatOpenAI → sends to GPT, returns AIMessage
# 4. | StrOutputParser → extracts .content → plain string
chain = (
    {
        "context":  retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        "question": RunnablePassthrough()
    }
    | prompt
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | StrOutputParser()
)

# Wrapper — returns both answer and raw docs (RAGAS needs the docs)
def rag_p1(question: str) -> dict:
    docs_used = retriever.invoke(question)
    answer    = chain.invoke(question)
    return {"answer": answer,
            "contexts": [d.page_content for d in docs_used]}

# Run benchmark after this:
# from eval.benchmark import run_benchmark
# run_benchmark(rag_p1, "p1_simple_rag")



# you now own:
# PyPDFLoader
# TextSplitter
# FAISS
# Retriever
# LCEL chain
# ChatPromptTemplate
# ChatOpenAI
# StrOutputParser