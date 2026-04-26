import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
# NOTE: In LangChain 1.x, AgentExecutor is removed.
# create_agent (langchain.agents) replaces the old create_react_agent + AgentExecutor.
# It returns a compiled StateGraph — same .invoke()/.stream() interface as P6/P7 graphs.

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
# STEP 4 — THE SHIFT: from fixed recipes to LLM-as-orchestrator
#
# P1–P7: YOU decide the recipe.
#   P1: always retrieve → generate
#   P5: classify → route to fixed handler
#   P6: retrieve → grade → rewrite → retrieve (fixed loop)
#
# P8: the LLM decides the recipe on the fly.
#   "I need fact A first. Now B depends on A. Now calculate C from B."
#   You give it tools. It picks which ones, in what order, how many times.
#
# This is the ReAct loop (Reason + Act):
#   Thought:        "I need to find how many days the old man was at sea"
#   Action:         search_book
#   Action Input:   "how many days Santiago at sea"
#   Observation:    "eighty-four days without fish..."
#   Thought:        "Now I can calculate hours: 84 * 24"
#   Action:         calculate
#   Action Input:   84 * 24
#   Observation:    2016
#   Final Answer:   "Santiago went 84 days without fish, which is 2016 hours."
#
# The loop repeats until the LLM emits "Final Answer:" — then AgentExecutor stops.
# ═══════════════════════════════════════════════════════════════
section("STEP 4 — THE SHIFT (fixed recipe → LLM orchestrator)")

print("  P1–P7: you write the recipe  →  retrieve → grade → generate")
print("  P8:    LLM writes the recipe →  Thought → Action → Observation → repeat")
print()
print("  ReAct loop:")
print("    Thought    : LLM reasons about what to do next")
print("    Action     : LLM picks a tool by name")
print("    Action Input: LLM provides the tool's input string")
print("    Observation: tool result fed back to LLM")
print("    ... repeats until LLM emits 'Final Answer:'")

# ═══════════════════════════════════════════════════════════════
# STEP 5 — @tool DECORATOR  ← NEW in P8
#
# @tool turns a plain Python function into a LangChain Tool.
# The function's DOCSTRING is what the LLM reads to decide
# when to call this tool. The routing quality depends entirely
# on the docstring — treat it like a prompt.
#
# Bad docstring:  "search stuff"
# Good docstring: "Search the book for specific facts about characters,
#                  events, or descriptions. Use for precise factual lookups."
#
# The LLM sees: tool name + docstring. It never sees the implementation.
# ═══════════════════════════════════════════════════════════════
section("STEP 5 — @tool DECORATOR")

@tool
def search_book(query: str) -> str:
    """Search The Old Man and the Sea for specific facts, quotes, events,
    or character details. Use for precise factual lookups from the book.
    Input: a specific search query string."""
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)

@tool
def summarise_topic(topic: str) -> str:
    """Retrieve and summarise ALL available information about a broad topic
    from The Old Man and the Sea. Use when the question asks for an overview
    or summary rather than a specific fact.
    Input: the topic to summarise (e.g. 'Santiago's character', 'the marlin')."""
    docs = retriever.invoke(topic)
    ctx  = "\n\n".join(d.page_content for d in docs)
    return llm.invoke(
        f"Summarise all content about '{topic}' from the text below. Be concise.\n\n{ctx}"
    ).content

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.
    Use for any arithmetic — converting days to hours, calculating percentages, totals.
    Input: a valid Python math expression e.g. '84 * 24' or '(84 / 7)'."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"

tools = [search_book, summarise_topic, calculate]

print("  Tools registered:")
for t in tools:
    print(f"    @tool {t.name}")
    print(f"           {t.description[:90]}")
print()
print("  Rule: the LLM reads ONLY the name + docstring to decide when to call each tool")
print("  Rule: docstrings are prompts — write them with the same care")

# ═══════════════════════════════════════════════════════════════
# STEP 6 — REACT PROMPT + AGENT
#
# hub.pull("hwchase17/react") fetches the canonical ReAct prompt
# from LangChain Hub. It contains the structured template that
# teaches the LLM to emit Thought / Action / Action Input / Observation.
#
# create_react_agent wires the LLM + tools + prompt together.
# It does NOT run anything yet — it just builds the agent object.
#
# EXTRA CONCEPT: why hub.pull instead of writing the prompt yourself?
# The ReAct prompt has been carefully engineered and tested across
# many models. Writing your own is error-prone — small formatting
# differences break the Thought/Action parsing. Use the canonical one.
# ═══════════════════════════════════════════════════════════════
section("STEP 6 — CREATE REACT AGENT")

# In LangChain 1.x, create_react_agent lives in langgraph.prebuilt.
# It returns a compiled StateGraph — no separate AgentExecutor needed.
# The ReAct prompt is built in; you don't need hub.pull anymore.
#
# EXTRA CONCEPT: why the API changed
# LangChain 0.x: create_react_agent (langchain.agents) + AgentExecutor
# LangChain 1.x: create_react_agent (langgraph.prebuilt) returns a graph
#   The graph IS the executor — compile() + invoke() in one object.
#   This aligns agents with the same StateGraph pattern from P6/P7.

agent = create_agent(
    model=llm,
    tools=tools,
)

print("  create_react_agent(model, tools) → compiled StateGraph")
print("  No hub.pull, no AgentExecutor — the graph drives the ReAct loop")
print("  Same .invoke() interface as every chain and graph in this series")

# ═══════════════════════════════════════════════════════════════
# STEP 7 — AGENTEXECUTOR  ← NEW in P8
#
# AgentExecutor drives the Thought→Action→Observation loop.
# It repeatedly:
#   1. Asks the agent (LLM) what to do next
#   2. Parses the Action + Action Input from the response
#   3. Calls the chosen tool with the input
#   4. Appends the Observation to the context
#   5. Repeats until the LLM emits "Final Answer:"
#
# Key parameters:
#   verbose=True        — prints every Thought/Action/Observation live
#   max_iterations=8    — hard stop to prevent infinite loops
#   handle_parsing_errors=True — if the LLM returns malformed output,
#                                recover instead of crashing
# ═══════════════════════════════════════════════════════════════
section("STEP 7 — INVOKING THE AGENT")

# The agent graph is invoked with a messages list — same pattern as
# LangChain chat models. The graph internally runs the ReAct loop:
#   1. LLM decides which tool to call
#   2. Tool is called, result appended as ToolMessage
#   3. LLM reads the result and decides next step
#   4. Repeats until LLM emits a final AIMessage with no tool calls
#
# To watch the loop: iterate over agent.stream() instead of .invoke()
# Each chunk shows one step — tool call or LLM thought.

print("  agent.invoke({'messages': [...]}) — drives the full ReAct loop")
print("  agent.stream(...)                — yields each step live (tool calls + LLM)")
print()
print("  Key difference from chain.invoke():")
print("    chain.invoke()  → fixed steps, always the same path")
print("    agent.invoke()  → LLM decides steps, path changes per question")

# ═══════════════════════════════════════════════════════════════
# STEP 8 — DEMO
#
# Multi-step questions that require multiple tools in sequence.
# Watch verbose output to see the LLM's reasoning live.
# ═══════════════════════════════════════════════════════════════
section("STEP 8 — DEMO (multi-step questions, watch the ReAct loop)")

questions = [
    # Needs: search_book (days) + calculate (days → hours)
    "How many days did Santiago go without catching a fish, and how many hours is that?",

    # Needs: search_book (days at sea) + search_book (dreams) — two tool calls
    "How long was the old man at sea on his final trip, and what does he dream about?",
]

for q in questions:
    print(f"\n{'─'*60}")
    print(f"  Question: {q}")
    print(f"{'─'*60}\n")

    # stream() yields each step so we can watch the ReAct loop live
    final_answer = ""
    for step in agent.stream(
        {"messages": [{"role": "user", "content": q}]},
        stream_mode="values",
    ):
        msg = step["messages"][-1]
        # ToolMessage = tool was called; AIMessage = LLM thought or final answer
        role = type(msg).__name__
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"    [Action]       {tc['name']}({tc['args']})")
        elif hasattr(msg, "content") and msg.content and role != "ToolMessage":
            if not getattr(msg, "tool_calls", None):
                final_answer = msg.content
                print(f"    [Final Answer] {final_answer[:200]}")
        elif role == "ToolMessage":
            print(f"    [Observation]  {str(msg.content)[:120]}")

    print(f"\n  Answer: {final_answer[:300]}")

# ═══════════════════════════════════════════════════════════════
# STEP 9 — PIPELINE SUMMARY
# ═══════════════════════════════════════════════════════════════
section("STEP 9 — PIPELINE SUMMARY")

print(f"  LLM        : gpt-4o-mini")
print(f"  Embedding  : all-MiniLM-L6-v2 (local)")
print(f"""
  Architecture shift across the series:
    P1–P3  : fixed linear chain
    P4     : smarter retrieval (HyDE)
    P5     : classify → route to fixed handler
    P6–P7  : loop with quality gates (LangGraph)
    P8     : LLM IS the orchestrator — writes the recipe on the fly

  @tool docstring is a prompt:
    Bad:  "search stuff"
    Good: "Search The Old Man and the Sea for specific facts, quotes,
           events, or character details. Use for precise factual lookups."

  AgentExecutor parameters that matter in production:
    max_iterations      — prevent runaway loops (set to 6–10)
    handle_parsing_errors — LLMs sometimes emit malformed output; recover gracefully
    verbose             — essential for debugging; turn off in production

  Why this matters beyond RAG:
    @tool + AgentExecutor is exactly how Ford SFG and Content Optimizer work.
    RAG is just one tool. Add web search, a calculator, a database query,
    an email sender — the LLM orchestrates all of them to answer one question.
""")

# ═══════════════════════════════════════════════════════════════
# REUSABLE FUNCTION
# ═══════════════════════════════════════════════════════════════

def rag_p8(question: str) -> dict:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    final  = result["messages"][-1].content
    return {
        "answer":   final,
        "contexts": [],   # agent doesn't expose retrieved docs directly
    }
