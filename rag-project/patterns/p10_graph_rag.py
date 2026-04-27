import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
# NOTE: langchain 1.x — Neo4jGraph + GraphCypherQAChain now live in langchain-neo4j.
# langchain_community.graphs.Neo4jGraph and langchain.chains are both removed/deprecated.

load_dotenv()

DIVIDER = "=" * 60

def section(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)

# ═══════════════════════════════════════════════════════════════
# STEP 1 — WHY GRAPH RAG
#
# Vector search answers "find me chunks that SOUND LIKE this question."
# It cannot answer relational questions, for example:
#
#   "Which birds are found in both high-altitude and coastal zones?"
#   "What species share the same habitat as the Gray Wagtail?"
#   "Which endangered birds breed in the Western Ghats?"
#
# Why vector search fails here:
#   Every chunk is an isolated island of text.
#   There is no stored link between "Gray Wagtail" and "riverine habitat"
#   and "Western Ghats" — those connections exist as scattered sentences
#   across dozens of pages.
#
# Graph RAG fixes this at INGEST time:
#   LLMGraphTransformer reads each chunk → extracts entities + relationships
#   → stores them as nodes and edges in Neo4j.
#
#   At query time: natural language → Cypher query → graph traversal
#   → structured answer.
#
# The key mental model:
#   Vector RAG : chunks ↔ chunks  (similarity in embedding space)
#   Graph RAG  : entities → relationships → entities  (structured traversal)
#
#   "Which bird shares habitat with the Malabar Trogon?"
#   → MATCH (b:Bird)-[:FOUND_IN]->(h:Habitat)<-[:FOUND_IN]-(b2:Bird)
#     WHERE b.name = 'Malabar Trogon' RETURN b2.name
#   → Graph returns exact matches. No hallucination about habitat names.
# ═══════════════════════════════════════════════════════════════
section("STEP 1 — WHY GRAPH RAG")

PDF_PATH = os.path.join(os.path.dirname(__file__), "../data/oldman_and_the_sea.pdf")
# Using the Old Man story — its named entities (people, fish, places, relationships)
# are clear and easy to visualise as a graph. Bird atlas works too but has 500+ pages.

print("  Vector RAG: every chunk is an island, retrieval by text similarity")
print("  Graph RAG:  entities + relationships stored in Neo4j at ingest time")
print("              query time = graph traversal, not similarity search")
print()
print("  Questions vector search CANNOT answer well:")
print("    'What did Santiago and the marlin have in common?'  → needs RELATIONSHIP")
print("    'Who cared for Santiago?'                          → needs ENTITY LINK")
print("    'What threatened the catch?'                       → needs ENTITY LINK")

# ═══════════════════════════════════════════════════════════════
# STEP 2 — LOAD AND SPLIT (same as P1, fewer chunks for demo)
#
# We use a small sample (first 20 pages) so the LLMGraphTransformer
# call — which hits GPT-4o once per chunk — stays fast and cheap.
# In production you'd process all pages; here we illustrate the pattern.
# ═══════════════════════════════════════════════════════════════
section("STEP 2 — LOAD AND SPLIT")

docs     = PyPDFLoader(PDF_PATH).load()
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks   = splitter.split_documents(docs)

# Cap to first 30 chunks so graph extraction stays fast during this demo.
# Remove this slice for full-document ingestion.
DEMO_CHUNKS = 30
sample_chunks = chunks[:DEMO_CHUNKS]

print(f"  Total chunks  : {len(chunks)}")
print(f"  Demo sample   : {DEMO_CHUNKS} chunks (first ~15 pages)")
print(f"  Why sample?   : LLMGraphTransformer calls GPT-4o once per chunk")
print(f"                  30 chunks ≈ 30 API calls. Full doc = full cost.")

# ═══════════════════════════════════════════════════════════════
# STEP 3 — NEO4J SETUP
#
# Neo4j is the graph database. It stores nodes (entities) and
# directed edges (relationships) with properties on both.
#
# To start Neo4j locally:
#
#   Docker (recommended):
#     docker run -d \
#       --name neo4j \
#       -p 7474:7474 -p 7687:7687 \
#       -e NEO4J_AUTH=neo4j/password \
#       neo4j:latest
#   Then open http://localhost:7474 to see the graph visually.
#
#   Homebrew (Mac):
#     brew install neo4j
#     brew services start neo4j
#
# Connection uses the Bolt protocol (port 7687).
# HTTP browser UI is on port 7474.
#
# Add to .env:
#   NEO4J_URI=bolt://localhost:7687
#   NEO4J_USERNAME=neo4j
#   NEO4J_PASSWORD=password
# ═══════════════════════════════════════════════════════════════
section("STEP 3 — NEO4J CONNECTION")

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

print(f"  Connecting to Neo4j at {NEO4J_URI} ...")
try:
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
    )
    # Clear existing data so re-runs start fresh
    graph.query("MATCH (n) DETACH DELETE n")
    print("  Connected — existing graph cleared")
    NEO4J_AVAILABLE = True
except Exception as e:
    print(f"  Neo4j not available: {e}")
    print("  → Running in DRY RUN mode: graph extraction only, no DB writes")
    print("  → Start Neo4j with Docker to enable full graph querying:")
    print("    docker run -d -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest")
    graph = None
    NEO4J_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════
# STEP 4 — LLMGraphTransformer  ← CORE OF P10
#
# This is the ingest-time step that separates Graph RAG from all others.
#
# LLMGraphTransformer sends each chunk to GPT-4o with a structured
# extraction prompt. GPT-4o reads the text and returns:
#   - Nodes: entities with a type label (Person, Fish, Place, ...)
#   - Relationships: directed edges between those nodes
#
# allowed_nodes / allowed_relationships constrain the schema.
# Why constrain?
#   Unconstrained → GPT-4o creates inconsistent labels ("CHARACTER",
#   "PERSON", "MAN", "HUMAN" for the same concept).
#   Constrained  → consistent schema → reliable Cypher queries.
#
# Example extraction from the chunk:
#   "The old man had gone eighty-four days now without taking a fish.
#    The boy's name was Manolin."
#
#   Nodes:      (Santiago:Person), (Manolin:Person), (Marlin:Fish)
#   Edges:      (Manolin)-[:COMPANION_OF]->(Santiago)
#               (Santiago)-[:PURSUES]->(Marlin)
#
# This is what makes relational questions answerable:
#   "Who is Santiago's companion?" → traverse COMPANION_OF → Manolin
# ═══════════════════════════════════════════════════════════════
section("STEP 4 — LLMGraphTransformer (chunks → entities + relationships)")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Person", "Fish", "Animal", "Place", "Boat", "Concept"],
    allowed_relationships=[
        "COMPANION_OF", "PURSUES", "CATCHES", "ATTACKS",
        "LOCATED_IN", "OWNS", "ADMIRES", "DREAMS_OF",
    ],
    # ENTITY NORMALIZATION — critical for reliable Cypher matching.
    # Without this, the same person becomes "Old Man", "The Old Man",
    # "He", "Santiago" in different chunks. Cypher does exact id matching,
    # so it would miss 3 of those 4 variants every time.
    # prompt_override tells the LLM to always use canonical proper names.
    prompt=None,   # use default; entity normalization via node_properties below
    node_properties=["description"],   # store raw mention so we can debug
)

print("  LLMGraphTransformer — extraction schema:")
print("  Nodes   : Person | Fish | Animal | Place | Boat | Concept")
print("  Edges   : COMPANION_OF | PURSUES | CATCHES | ATTACKS |")
print("            LOCATED_IN   | OWNS    | ADMIRES | DREAMS_OF")
print()
print(f"  Extracting graph from {DEMO_CHUNKS} chunks (GPT-4o-mini per chunk) ...")

graph_docs = transformer.convert_to_graph_documents(sample_chunks)

# Count what was extracted
all_nodes = {n.id for gd in graph_docs for n in gd.nodes}
all_rels  = [(r.source.id, r.type, r.target.id) for gd in graph_docs for r in gd.relationships]

print(f"  Extracted {len(all_nodes)} unique nodes, {len(all_rels)} relationships")
print()
print("  Sample nodes:")
for n in list(all_nodes)[:12]:
    print(f"    · {n}")
print()
print("  Sample relationships:")
for src, rel, tgt in all_rels[:10]:
    print(f"    ({src}) -[{rel}]→ ({tgt})")

# ═══════════════════════════════════════════════════════════════
# STEP 5 — WRITE GRAPH TO NEO4J
#
# graph.add_graph_documents():
#   - Creates nodes with their type as the Neo4j label
#   - Creates directed edges with the relationship type as the label
#   - include_source=True also stores the raw chunk text on the node
#     so you can trace which passage produced each entity
#
# After this call you can open http://localhost:7474 and run:
#   MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50
# to visualise the full extracted graph.
# ═══════════════════════════════════════════════════════════════
section("STEP 5 — WRITE GRAPH TO NEO4J")

if NEO4J_AVAILABLE:
    graph.add_graph_documents(graph_docs, include_source=True)
    # Refresh schema so GraphCypherQAChain knows what labels/relationships exist
    graph.refresh_schema()
    print("  Graph written to Neo4j")
    print("  View it: http://localhost:7474")
    print("  Cypher:  MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50")
    print()
    print("  Graph schema:")
    print(graph.schema)
else:
    print("  Skipped — Neo4j not available (dry run)")
    print("  Graph docs ready in memory — connect Neo4j to persist and query")

# ═══════════════════════════════════════════════════════════════
# STEP 6 — GraphCypherQAChain  ← QUERY TIME
#
# Two-step chain:
#   Step 1 — Cypher generation:
#     LLM reads the graph schema + natural language question
#     → generates a Cypher query
#   Step 2 — Answer generation:
#     Cypher runs on Neo4j → returns structured rows
#     LLM reads rows + original question → natural language answer
#
# verbose=True prints the generated Cypher live — essential for
# understanding what the LLM actually queries.
#
# allow_dangerous_requests=True:
#   Required because LangChain doesn't validate the Cypher before
#   running it. In production, add query whitelisting or read-only
#   Neo4j credentials so write queries can't be injected.
#
# Fallback to P7 (Self-RAG):
#   If Cypher generation fails or returns empty results, we fall
#   back to vector search. This makes the system robust — graph
#   traversal for relational questions, vector for semantic ones.
# ═══════════════════════════════════════════════════════════════
section("STEP 6 — GraphCypherQAChain (NL → Cypher → answer)")

if NEO4J_AVAILABLE:
    from langchain_core.prompts import PromptTemplate

    # FUZZY CYPHER PROMPT — fixes the entity ID mismatch problem.
    #
    # Default GraphCypherQAChain generates exact id matches:
    #   MATCH (p:Person {id: 'Santiago'}) ...
    # This fails when the same entity was stored as "Old Man", "He", etc.
    #
    # Fix: instruct the LLM to always use case-insensitive CONTAINS matching:
    #   WHERE toLower(p.id) CONTAINS 'santiago'
    # This catches "Santiago", "The Santiago", "santiago" — all variants.
    CYPHER_GENERATION_TEMPLATE = """Task: Generate a Cypher statement to query a Neo4j graph database.

Instructions:
- Use only node labels and relationship types from the schema.
- NEVER use exact property matching like {{id: 'Name'}}.
- ALWAYS use case-insensitive fuzzy matching: WHERE toLower(n.id) CONTAINS 'name'
- Use CONTAINS not = for all string comparisons.
- Return only the Cypher query, no explanation.

Schema:
{schema}

Question: {question}
Cypher query:"""

    cypher_prompt = PromptTemplate(
        input_variables=["schema", "question"],
        template=CYPHER_GENERATION_TEMPLATE,
    )

    cypher_chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        allow_dangerous_requests=True,
        cypher_prompt=cypher_prompt,
    )
    print("  GraphCypherQAChain ready (fuzzy CONTAINS matching)")
    print("  Flow: question → LLM generates Cypher → Neo4j runs it → LLM answers")
else:
    cypher_chain = None
    print("  Skipped — Neo4j not available")

print("""
  Two-LLM architecture:
    LLM 1 : question + schema → Cypher query   (structured retrieval)
    Neo4j  : runs Cypher → structured rows
    LLM 2 : rows + question   → natural language answer
""")

# ═══════════════════════════════════════════════════════════════
# STEP 7 — FULL GRAPH RAG FUNCTION WITH VECTOR FALLBACK
# ═══════════════════════════════════════════════════════════════
section("STEP 7 — GRAPH RAG PIPELINE (with Self-RAG fallback)")

def rag_p10(question: str) -> dict:
    """
    Try graph traversal first (relational questions).
    If Cypher fails or returns nothing, fall back to vector RAG.

    Graph is better for:  "Who does X?", "What connects A to B?"
    Vector is better for: "What is the mood of...", open-ended summaries
    """
    if NEO4J_AVAILABLE and cypher_chain:
        try:
            result = cypher_chain.invoke({"query": question})
            answer = result.get("result", "")
            # Fallback triggers when:
            #   - answer is empty
            #   - Cypher returned no rows ("I don't know" is the LLM's response to [])
            #   - answer is clearly a non-answer
            bad_answer = not answer or "don't know" in answer.lower() or "no information" in answer.lower()
            if not bad_answer:
                return {"answer": answer, "source": "graph", "contexts": []}
            print(f"  [graph] Empty result — falling back to vector")
        except Exception as e:
            print(f"  [graph] Cypher failed: {e} — falling back to vector")

    # Fallback: import and use P7 Self-RAG
    from p7_self_rag import rag_p7
    result = rag_p7(question)
    return {**result, "source": "vector_fallback"}

print("""
  rag_p10(question):
    1. Generate Cypher from NL question + graph schema
    2. Run Cypher on Neo4j → structured rows
    3. If rows are non-empty → LLM generates answer from rows
    4. If Cypher fails / empty → fall back to rag_p7() (Self-RAG)

  Source field in return dict tells you which path was taken:
    "graph"          → answered by Neo4j traversal
    "vector_fallback" → answered by P7 Self-RAG
""")

# ═══════════════════════════════════════════════════════════════
# STEP 8 — DEMO
# ═══════════════════════════════════════════════════════════════
section("STEP 8 — DEMO")

questions = [
    "Who is Santiago's companion?",
    "What does Santiago pursue in the sea?",
    "What attacked the catch?",
    "What does the old man dream about?",
]

if NEO4J_AVAILABLE:
    for q in questions:
        print(f"\n  Q: {q}")
        r = rag_p10(q)
        print(f"  Source : {r['source']}")
        print(f"  A      : {r['answer'][:300]}")
else:
    print("  Neo4j not running — showing dry run extraction results instead")
    print()
    print("  Extracted graph (from memory, not queried):")
    for src, rel, tgt in all_rels:
        print(f"    ({src}) -[{rel}]→ ({tgt})")
    print()
    print("  To run the full demo:")
    print("    docker run -d -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest")
    print("    # add NEO4J_URI/USERNAME/PASSWORD to .env, then re-run this script")

# ═══════════════════════════════════════════════════════════════
# STEP 9 — PIPELINE SUMMARY
# ═══════════════════════════════════════════════════════════════
section("STEP 9 — PIPELINE SUMMARY")

print(f"  LLM          : gpt-4o-mini")
print(f"  Graph DB     : Neo4j (Bolt protocol, port 7687)")
print(f"  Chunks used  : {DEMO_CHUNKS} of {len(chunks)} total")
print(f"""
  RAG pattern comparison — what each can answer:

    P1  (simple vector) : "What happened when Santiago caught the marlin?"
    P6  (CRAG)          : same, but retries if retrieval quality is low
    P7  (Self-RAG)      : same, but also checks answer faithfulness
    P10 (Graph RAG)     : "Who is connected to Santiago?" — follows edges

  Key concepts introduced in P10:
    ① LLMGraphTransformer — GPT-4o reads text → outputs nodes + edges
    ② Neo4j               — graph database storing entities + relationships
    ③ Cypher              — graph query language (like SQL for graphs)
    ④ GraphCypherQAChain  — NL → Cypher → Neo4j → NL answer (two LLM calls)
    ⑤ Graph schema        — constrain node/edge types at ingest → reliable queries
    ⑥ Fallback pattern    — graph first, vector if Cypher fails

  Why schema matters:
    Unconstrained → GPT-4o labels same entity as "Person", "Man", "Character"
                    → Cypher can't match them reliably
    Constrained   → consistent labels → reliable traversal

  Vector vs Graph at a glance:
    Vector : finds chunks SIMILAR to the question (semantic proximity)
    Graph  : follows EXPLICIT RELATIONSHIPS between entities (structural)
    Best systems use both — graph for relational, vector for semantic
""")

q="did the old man die? what was the moral of the story?"
r=rag_p10(q)
print(f"\n  Source : {r['source']}")
print(f"  A      : {r['answer']}")
# print(rag_p10(question))
#  → {"answer": str, "contexts": list[str], "source": "graph" | "vector_fallback"}
