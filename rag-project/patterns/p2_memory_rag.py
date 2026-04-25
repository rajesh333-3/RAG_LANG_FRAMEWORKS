import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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
# STEP 4 — CONDENSER CHAIN  ← NEW in P2
#
# Problem with P1 in multi-turn conversations:
#   Turn 1: "What method does the paper propose?"  ← retrievable ✓
#   Turn 2: "How does it compare to chunking?"     ← retrievable ✓
#   Turn 3: "What are its limitations?"            ← retrievable ✓
#   Turn 4: "Who funded that?"                     ← "that" has no context ✗
#
# Fix: before retrieval, rewrite the follow-up into a standalone
# question using the chat history.
#   "Who funded that?" + history → "Who funded the Landmark Embedding research?"
#
# MessagesPlaceholder — injects the full chat_history list
# (HumanMessage/AIMessage objects) into the prompt at that position.
# ═══════════════════════════════════════════════════════════════
section("STEP 4 — CONDENSER CHAIN (rewrite follow-ups)")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

condenser_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given the conversation history and a follow-up question, "
     "rewrite the follow-up as a fully self-contained standalone question. "
     "If it is already standalone, return it unchanged. "
     "Output ONLY the rewritten question, nothing else."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

# condenser_chain: {chat_history, question} → standalone question string
condenser_chain = condenser_prompt | llm | StrOutputParser()

print("  Condenser prompt template:")
print("    system: rewrite follow-up as standalone question")
print("    MessagesPlaceholder(chat_history)  ← injects conversation turns here")
print("    human: {question}")

# ═══════════════════════════════════════════════════════════════
# STEP 5 — RAG CHAIN (same structure as P1)
# ═══════════════════════════════════════════════════════════════
section("STEP 5 — RAG CHAIN (same as P1)")

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
# STEP 6 — MULTI-TURN CONVERSATION LOOP
#
# chat_history accumulates HumanMessage/AIMessage objects.
# Each turn:
#   1. condenser rewrites the follow-up into a standalone question
#   2. standalone question → retriever → top-k chunks → LLM → answer
#   3. original question + answer appended to history
#
# HumanMessage — wraps what the user said
# AIMessage    — wraps what the assistant replied
# ═══════════════════════════════════════════════════════════════
section("STEP 6 — MULTI-TURN CONVERSATION DEMO")

chat_history = []

conversation = [
    "What is old man and what is his story?",
    "Who is the boy and what is his relationship with him?",   # "him" needs rewriting
    "How many days had he gone without catching a fish?",           # "he" needs rewriting
    "What does he dream about at night?",                      # "he" needs rewriting
]

def chat(question: str) -> str:
    # Step A: rewrite follow-up if needed
    # Skip condenser on turn 1 — no history means nothing to resolve,
    # and the question is standalone by definition. Saves one LLM call
    # at the start of every fresh conversation.
    if chat_history:
        standalone = condenser_chain.invoke({
            "history":  chat_history[-10:],  # last 5 turns (10 messages) — prevents unbounded growth
            "question": question,
        })
    else:
        standalone = question  # turn 1: always standalone, skip the rewriter

    print(f"\n  Original   : {question}")
    if standalone != question:
        print(f"  Rewritten  : {standalone}")
    else:
        print(f"  Rewritten  : (unchanged — already standalone)")

    # Step B: retrieve + generate using standalone question
    answer = rag_chain.invoke(standalone)

    # Step C: update memory
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))

    print(f"  Answer     : {answer[:200]}")
    return answer

for q in conversation:
    chat(q)
    print()

# ═══════════════════════════════════════════════════════════════
# STEP 7 — INSPECT MEMORY
# Shows the raw HumanMessage/AIMessage objects in chat_history
# ═══════════════════════════════════════════════════════════════
section("STEP 7 — CHAT HISTORY (HumanMessage / AIMessage objects)")

for msg in chat_history:
    role = "Human" if isinstance(msg, HumanMessage) else "AI"
    print(f"\n  [{role}]")
    print(f"  {msg.content[:200]}")

# ═══════════════════════════════════════════════════════════════
# STEP 8 — PIPELINE SUMMARY
# ═══════════════════════════════════════════════════════════════
section("STEP 8 — PIPELINE SUMMARY")

print(f"  Turns in conversation  : {len(conversation)}")
print(f"  Messages in history    : {len(chat_history)}")
print(f"  Embedding model        : all-MiniLM-L6-v2 (local)")
print(f"  LLM                    : gpt-4o-mini")
print(f"""
  Flow:
    user question
         │
         ▼
    CONDENSER  (chat_history + follow-up → standalone question)
         │
         ▼
    RETRIEVER  (standalone → top-4 chunks)
         │
         ▼
    RAG PROMPT (context + standalone question)
         │
         ▼
    LLM → answer
         │
         ▼
    append HumanMessage + AIMessage to chat_history
""")

# ═══════════════════════════════════════════════════════════════
# REUSABLE FUNCTION (for benchmarking with RAGAS later)
# ═══════════════════════════════════════════════════════════════

def rag_p2(question: str, history: list = None) -> dict:
    history = history or []
    # Same guard — skip rewriter on turn 1 (empty history)
    if history:
        standalone = condenser_chain.invoke({
            "history":  history[-10:],
            "question": question,
        })
    else:
        standalone = question
    docs_used = retriever.invoke(standalone)
    answer    = rag_chain.invoke(standalone)
    return {
        "answer":     answer,
        "contexts":   [d.page_content for d in docs_used],
        "standalone": standalone,
    }
