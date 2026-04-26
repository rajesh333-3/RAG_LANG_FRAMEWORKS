import os
import base64
import io
import numpy as np
import torch
import fitz  # pymupdf — lighter than unstructured, no system deps
from dotenv import load_dotenv
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

DIVIDER = "=" * 60

def section(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)

# ═══════════════════════════════════════════════════════════════
# STEP 1 — WHY MULTIMODAL RAG
#
# P1–P8 are blind to images. A technical paper like this one has:
#   - Architecture diagrams (page 6, 7)
#   - Results tables as images (page 12)
#   - Performance charts (page 13, 14)
#
# Standard RAG ingests the text around the figure but not the figure.
# "See Figure 3" in the text → chunk retrieved → LLM sees the caption,
# not the diagram. For visual information (architecture, charts) this fails.
#
# Fix:
#   1. Extract images from the PDF with pymupdf (fitz)
#   2. Embed images with CLIP (same vector space as text)
#   3. At query time: retrieve top text chunks AND top matching images
#   4. Pass both to GPT-4o vision — it reads diagrams and text together
# ═══════════════════════════════════════════════════════════════
section("STEP 1 — WHY MULTIMODAL RAG")

# PDF_PATH = os.path.join(os.path.dirname(__file__), "../data/llama2_tech_report.pdf")
PDF_PATH = os.path.join(os.path.dirname(__file__), "../data/Kerala-Bird-Atlas-Final-Compressed.pdf")

print("  Standard RAG: ingests text only → blind to diagrams, charts, tables")
print("  Multimodal RAG:")
print("    1. Extract images with pymupdf")
print("    2. Embed images with CLIP (text + image → same vector space)")
print("    3. Retrieve top text chunks + top image matches")
print("    4. GPT-4o vision reads both → answers about figures")

# ═══════════════════════════════════════════════════════════════
# STEP 2 — EXTRACT IMAGES WITH PYMUPDF
#
# fitz.open() opens the PDF. page.get_images() returns a list of
# image references. fitz.Pixmap(doc, xref) renders the image.
# We skip tiny images (< 100x100) — they're decorative icons.
# We store each image as base64 for later sending to GPT-4o.
#
# Why pymupdf over unstructured?
#   unstructured hi_res: needs detectron2 + tesseract + poppler (~2GB)
#   pymupdf: one pip install, pure Python, same image extraction result
# ═══════════════════════════════════════════════════════════════
section("STEP 2 — EXTRACT IMAGES (pymupdf / fitz)")

def extract_images_from_pdf(pdf_path: str, min_size: int = 100) -> list:
    """Extract images from PDF, skip decorative icons smaller than min_size px."""
    doc    = fitz.open(pdf_path)
    images = []
    for page_num, page in enumerate(doc):
        for img in page.get_images():
            xref = img[0]
            pix  = fitz.Pixmap(doc, xref)
            if pix.n > 4:                          # CMYK → RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)
            if pix.width < min_size or pix.height < min_size:
                continue                            # skip icons
            buf = io.BytesIO()
            Image.frombytes("RGB", [pix.width, pix.height], pix.samples).save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            images.append({
                "b64":    b64,
                "page":   page_num + 1,
                "width":  pix.width,
                "height": pix.height,
            })
    return images

img_store = extract_images_from_pdf(PDF_PATH, min_size=200)

print(f"  Images extracted (>200px): {len(img_store)}")
for img in img_store:
    print(f"    Page {img['page']}: {img['width']}x{img['height']}px")

# ═══════════════════════════════════════════════════════════════
# STEP 3 — TEXT INDEX (same as P1)
# ═══════════════════════════════════════════════════════════════
section("STEP 3 — TEXT INDEX (same as P1)")

text_docs  = PyPDFLoader(PDF_PATH).load()
splitter   = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks     = splitter.split_documents(text_docs)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Index path is keyed to the PDF filename — swap the PDF, get a separate index.
# e.g. Kerala-Bird-Atlas.pdf → index/p9_Kerala-Bird-Atlas/
pdf_stem   = os.path.splitext(os.path.basename(PDF_PATH))[0]
index_path = os.path.join(os.path.dirname(__file__), f"../index/p9_{pdf_stem}")

if os.path.exists(index_path):
    print("  Text index found — loading")
    vectordb = FAISS.load_local(index_path, embeddings,
                                allow_dangerous_deserialization=True)
else:
    print(f"  Embedding {len(chunks)} text chunks ...")
    vectordb = FAISS.from_documents(chunks, embeddings)
    os.makedirs(index_path, exist_ok=True)
    vectordb.save_local(index_path)
    print(f"  Text index saved")

text_retriever = vectordb.as_retriever(search_kwargs={"k": 3})
print(f"  Text chunks: {len(chunks)}")

# ═══════════════════════════════════════════════════════════════
# STEP 4 — CLIP EMBEDDINGS  ← NEW in P9
#
# CLIP (Contrastive Language-Image Pretraining) by OpenAI:
#   - Trained on 400M (text, image) pairs
#   - Embeds text and images into the SAME 512-dim vector space
#   - "diagram of attention mechanism" (text) and an actual
#     attention diagram (image) will have similar embeddings
#   - This is what makes text→image retrieval possible
#
# Two embedding functions:
#   embed_image()     : PIL image → 512-dim vector
#   embed_text_clip() : string    → 512-dim vector (CLIP space, not MiniLM)
#
# Important: text_retriever uses MiniLM (384-dim).
#            image retrieval uses CLIP (512-dim).
#            These are SEPARATE vector spaces — don't mix them.
# ═══════════════════════════════════════════════════════════════
section("STEP 4 — CLIP EMBEDDINGS (image + text → same vector space)")

print("  Loading CLIP model (openai/clip-vit-base-patch32) ...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
print("  CLIP loaded")

def embed_image(b64: str) -> np.ndarray:
    """Embed a base64 image into CLIP's 512-dim vector space.

    NOTE (transformers v5): get_image_features() returns BaseModelOutputWithPooling,
    not a plain tensor. Use .pooler_output[0] to get the 512-dim vector.
    """
    img    = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    inputs = clip_proc(images=img, return_tensors="pt")
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
    vec = feats.pooler_output[0].numpy()   # v5: .pooler_output[0] not feats[0]
    return vec / np.linalg.norm(vec)       # normalise for cosine similarity

def embed_text_clip(text: str) -> np.ndarray:
    """Embed a text string into CLIP's 512-dim vector space.

    NOTE (transformers v5): get_text_features() returns BaseModelOutputWithPooling,
    not a plain tensor. Use .pooler_output[0] to get the 512-dim vector.
    """
    inputs = clip_proc(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        feats = clip_model.get_text_features(**inputs)
    vec = feats.pooler_output[0].numpy()   # v5: .pooler_output[0] not feats[0]
    return vec / np.linalg.norm(vec)

print("  Pre-computing CLIP embeddings for all extracted images ...")
for img in img_store:
    img["emb"] = embed_image(img["b64"])
print(f"  Done — {len(img_store)} image embeddings ready")
print()
print("  CLIP space: 512-dim, shared for text + images")
print("  MiniLM space: 384-dim, text only (used by text_retriever)")
print("  These are SEPARATE spaces — never mix them")

# ═══════════════════════════════════════════════════════════════
# STEP 5 — IMAGE RETRIEVAL FUNCTION
#
# cosine similarity = dot product of normalised vectors
# We pre-normalised in embed_image/embed_text_clip, so it's just dot()
# ═══════════════════════════════════════════════════════════════
section("STEP 5 — IMAGE RETRIEVAL (cosine similarity in CLIP space)")

def retrieve_images(query: str, top_k: int = 2) -> list:
    """Find top-k images most semantically similar to the query."""
    q_vec = embed_text_clip(query)
    scored = sorted(
        img_store,
        key=lambda img: np.dot(q_vec, img["emb"]),   # dot of normalised = cosine
        reverse=True
    )
    return scored[:top_k]

print("  retrieve_images(query, k=2):")
print("    embed query with CLIP text encoder")
print("    cosine similarity against all image embeddings")
print("    return top-k most similar images")

# ═══════════════════════════════════════════════════════════════
# STEP 6 — GPT-4o VISION MESSAGE FORMAT  ← NEW in P9
#
# GPT-4o accepts a multimodal message: text + images in the same
# content list. Images are passed as base64 data URLs.
#
# Message format:
#   {"role": "user", "content": [
#       {"type": "text",      "text": "...context + question..."},
#       {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
#   ]}
#
# The LLM reads the text context AND visually processes the images.
# "See Figure 3" in the context + the actual Figure 3 as an image →
# the model can describe, compare, and reason about the diagram.
# ═══════════════════════════════════════════════════════════════
section("STEP 6 — GPT-4o VISION MESSAGE FORMAT")

llm_vision = ChatOpenAI(model="gpt-4o-mini")  # gpt-4o-mini supports vision too

def build_vision_message(question: str, text_docs: list, images: list) -> list:
    text_context = "\n\n".join(d.page_content for d in text_docs)
    content = [
        {
            "type": "text",
            "text": (
                f"Answer the question using the text context and images provided.\n"
                f"If an image is relevant, describe what you see in it.\n\n"
                f"Text context:\n{text_context}\n\n"
                f"Question: {question}"
            )
        }
    ]
    for img in images:
        content.append({
            "type":      "image_url",
            "image_url": {"url": f"data:image/png;base64,{img['b64']}"}
        })
    return [{"role": "user", "content": content}]

print("  Vision message: [{type: text, text: context+question},")
print("                   {type: image_url, image_url: {url: data:image/png;base64,...}},")
print("                   ...]")
print("  gpt-4o-mini supports vision — no need for full gpt-4o here")

# ═══════════════════════════════════════════════════════════════
# STEP 7 — FULL MULTIMODAL RAG FUNCTION
# ═══════════════════════════════════════════════════════════════
section("STEP 7 — MULTIMODAL RAG PIPELINE")

def rag_p9(question: str, image_k: int = 2) -> dict:
    # Gate 1: text retrieval (MiniLM space)
    retrieved_text = text_retriever.invoke(question)

    # Gate 2: image retrieval (CLIP space)
    retrieved_imgs = retrieve_images(question, top_k=image_k)

    # Build multimodal message and invoke
    messages = build_vision_message(question, retrieved_text, retrieved_imgs)
    answer   = llm_vision.invoke(messages).content

    return {
        "answer":       answer,
        "contexts":     [d.page_content for d in retrieved_text],
        "image_pages":  [img["page"] for img in retrieved_imgs],
    }

print("""
  Flow:
    question
         │
         ├──► text_retriever (MiniLM)  → top-3 text chunks
         │
         ├──► retrieve_images (CLIP)   → top-2 matching images
         │
         └──► GPT-4o vision message
                  │
                  ├── text context + question
                  └── images as base64
                       │
                       ▼
                  answer (can describe diagrams + cite text)
""")

# ═══════════════════════════════════════════════════════════════
# STEP 8 — DEMO
# ═══════════════════════════════════════════════════════════════
section("STEP 8 — DEMO")

questions = [
    # "What architecture or model diagram is shown in this paper?",
    # "What do the results or performance charts show?",
    # "Describe the key components shown in the figures.",
    "describe the bird gray wagtail",
    "describe the bird Indian yellow tit and what is its distribution?",
    "what is the difference between the distribution of Indian yellow tit and gray wagtail?",
    "what is the fastest bird in the document?",
    "most found bird as pr the document?",
    "which bird is found in higher elevations?"

]

for q in questions:
    print(f"\n  Q: {q}")
    result = rag_p9(q)
    print(f"  Images from pages: {result['image_pages']}")
    print(f"  A: {result['answer'][:300]}")
    print()

# ═══════════════════════════════════════════════════════════════
# STEP 9 — SUMMARY
# ═══════════════════════════════════════════════════════════════
section("STEP 9 — PIPELINE SUMMARY")

print(f"  PDF           : {os.path.basename(PDF_PATH)}")
print(f"  Images found  : {len(img_store)} (after filtering icons <200px)")
print(f"  Text model    : all-MiniLM-L6-v2 (384-dim)")
print(f"  Image model   : CLIP vit-base-patch32 (512-dim)")
print(f"  Vision LLM    : gpt-4o-mini")
print(f"""
  Two separate vector spaces:
    MiniLM (384-dim) — text chunks ↔ text query
    CLIP   (512-dim) — images      ↔ text query (same space!)

  Key insight: CLIP was trained on (text, image) pairs so
  "attention mechanism diagram" (text) and an actual diagram image
  land near each other in CLIP space — that's what enables retrieval.

  Why pymupdf over unstructured:
    unstructured hi_res → detectron2 + tesseract (~2GB, complex install)
    pymupdf (fitz)      → pip install pymupdf, pure Python, same result
""")
qq="what are the different plots you found? and what are the shapes and colors of the plotted lined in graphs"
res=rag_p9(qq, image_k=2) 
print(res)
# → {"answer": str, "contexts": list[str], "image_pages": list[int]}

