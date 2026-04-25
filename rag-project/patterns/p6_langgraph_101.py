"""
langgraph_101.py — anatomy of every LangGraph you'll build
============================================================
Read this before P6. Every graph in P6–P8 is just this pattern
applied to a more sophisticated problem.

Core mental model:
  Nodes  = boxes      (Python functions that do work)
  Edges  = arrows     (permanent, always-run connections)
  Cond.  = diamonds   (routing functions that read State)
  State  = whiteboard (shared dict every node reads & writes)
"""

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_openai import ChatOpenAI

load_dotenv()

DIVIDER = "=" * 60

def section(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)

# ═══════════════════════════════════════════════════════════════
# CONCEPT 1 — STATE
#
# A TypedDict shared across ALL nodes in the graph.
# Initialised once when you call app.invoke({...}).
# Every node reads fields it needs and returns ONLY what changed.
# LangGraph merges the returned dict back into State for you.
#
# Think of it as a sticky-note board the whole team writes on.
# ═══════════════════════════════════════════════════════════════
section("CONCEPT 1 — STATE (shared dict)")

class MyState(TypedDict):
    question:    str    # set at the start, may be rewritten by nodes
    documents:   list   # filled by retrieve node
    answer:      str    # filled by generate node
    retry_count: int    # incremented by rewrite node
    grade:       str    # "good" or "bad" — set by grade node

print("  MyState fields:")
print("    question    : str   — the current query (may be rewritten)")
print("    documents   : list  — retrieved chunks")
print("    answer      : str   — final answer")
print("    retry_count : int   — how many rewrites have happened")
print()
print("  Rule: every field must be initialised when you call app.invoke()")
print("  Rule: nodes return ONLY the fields they changed, not the full state")

# ═══════════════════════════════════════════════════════════════
# CONCEPT 2 — NODES
#
# Plain Python functions. Signature: (state: MyState) → dict
# Read whatever fields you need. Return only what you changed.
# LangGraph merges the returned dict into State automatically.
# ═══════════════════════════════════════════════════════════════
section("CONCEPT 2 — NODES (Python functions)")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def node_retrieve(state: MyState) -> dict:
    # Reads:   state["question"]
    # Returns: only "documents" — that's all this node changed
    print(f"    [node_retrieve] searching for: {state['question']}")
    docs = [{"page_content": f"Fake doc about: {state['question']}"}]
    return {"documents": docs}

def node_generate(state: MyState) -> dict:
    # Reads:   state["documents"], state["question"]
    # Returns: only "answer"
    ctx = "\n".join(d["page_content"] for d in state["documents"])
    ans = llm.invoke(f"Context: {ctx}\nQuestion: {state['question']}\nAnswer briefly:")
    print(f"    [node_generate] generated answer")
    return {"answer": ans.content}

def node_grade(state: MyState) -> dict:
    # Reads:   state["answer"]
    # Returns: "grade" (new field) and increments retry_count
    grade = "good" if len(state["answer"]) > 20 else "bad"
    retries = state.get("retry_count", 0) + 1
    print(f"    [node_grade] grade={grade}, retries={retries}")
    return {"grade": grade, "retry_count": retries}

print("  node_retrieve : question → documents")
print("  node_generate : documents + question → answer")
print("  node_grade    : answer → grade + retry_count")

# ═══════════════════════════════════════════════════════════════
# CONCEPT 3 — CONDITIONAL EDGE
#
# A Python function that reads State and returns a STRING.
# That string is looked up in the edge map to find the next node.
# This is how branching AND looping work — same mechanism.
#
# "retry" → maps to "retrieve" → goes BACKWARDS = loop
# "done"  → maps to END        → graph terminates
# ═══════════════════════════════════════════════════════════════
section("CONCEPT 3 — CONDITIONAL EDGE (decision diamond)")

def route_after_grade(state: MyState) -> str:
    if state["grade"] == "good":
        print("    [route] grade=good → done")
        return "done"
    elif state["retry_count"] < 2:
        print("    [route] grade=bad, retries<2 → retry (LOOP BACK)")
        return "retry"
    else:
        print("    [route] grade=bad, retries>=2 → give_up")
        return "give_up"

print("  route_after_grade returns: 'done' | 'retry' | 'give_up'")
print("  These strings map to node names in add_conditional_edges()")
print("  'retry' → 'retrieve' creates the LOOP")

# ═══════════════════════════════════════════════════════════════
# CONCEPT 4 — BUILD THE GRAPH
#
# wf = StateGraph(MyState)     — create graph with this state type
# add_node(name, fn)           — register a node function
# set_entry_point(name)        — which node runs first
# add_edge(A, B)               — permanent arrow: A always → B
# add_conditional_edges(       — decision diamond after node A:
#     A, fn, {key: node})        fn(state) returns key → next node
# compile()                    — validate + return app object
# ═══════════════════════════════════════════════════════════════
section("CONCEPT 4 — BUILD AND COMPILE")

wf = StateGraph(MyState)

wf.add_node("retrieve", node_retrieve)
wf.add_node("generate", node_generate)
wf.add_node("grade",    node_grade)

wf.set_entry_point("retrieve")           # START → retrieve
wf.add_edge("retrieve", "generate")      # retrieve → generate (always)
wf.add_edge("generate", "grade")         # generate → grade    (always)

wf.add_conditional_edges("grade", route_after_grade, {
    "done":    END,          # good answer  → finish
    "retry":   "retrieve",   # bad answer   → LOOP BACK to retrieve
    "give_up": END,          # too many     → finish anyway
})

app = wf.compile()  # validates graph, returns .invoke()-able object

print("  Graph compiled. Flow:")
print("""
  START
    │
    ▼
  retrieve ──► generate ──► grade ──► [route_after_grade]
    ▲                                       │
    │                                 done → END
    └──────── retry ◄────────────────┘
                                      give_up → END
""")

# ═══════════════════════════════════════════════════════════════
# CONCEPT 5 — INVOKE
#
# app.invoke() takes the initial State dict.
# ALL fields must be present — LangGraph won't fill defaults.
# Returns the final State after the graph reaches END.
# ═══════════════════════════════════════════════════════════════
section("CONCEPT 5 — INVOKE AND TRACE")

print("  Running graph with: 'What is the Gulf Stream?'\n")

result = app.invoke({
    "question":    "What is the Gulf Stream?",
    "documents":   [],   # empty — retrieve node will fill this
    "answer":      "",   # empty — generate node will fill this
    "retry_count": 0,    # always start at 0
    "grade":       "",   # empty — grade node will fill this
})

print(f"\n  Final State:")
print(f"    question    : {result['question']}")
print(f"    answer      : {result['answer'][:100]}")
print(f"    retry_count : {result['retry_count']}")
print(f"    docs count  : {len(result['documents'])}")

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
section("SUMMARY — what to carry into P6")

print("""
  Thing          What it is                     Key rule
  ─────────────────────────────────────────────────────────────
  State          TypedDict shared by all nodes  init ALL fields in invoke()
  Node           fn(state) → dict               return ONLY changed fields
  Edge           add_edge(A, B)                 A always goes to B
  Cond. edge     fn(state) → str → node name   this is how loops work
  END            special constant               graph stops here
  compile()      validates + returns app        app.invoke() = chain.invoke()

  The loop pattern (impossible with chains):
    add_edge("rewrite", "retrieve")             ← goes BACKWARDS
    add_conditional_edges("grade", fn, {
        "retry": "rewrite"                      ← fn can return "retry"
    })
""")
