from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List
import re

EMBEDDINGS_FILE = "data/toe_embeddings.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    top_k: int = 5


def load_index():
    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [d["text"] for d in data]
    sources = [d["source"] for d in data]
    embeddings = np.array([d["embedding"] for d in data], dtype="float32")
    return texts, sources, embeddings


print("Loading index and model...")
TEXTS, SOURCES, EMBEDDINGS = load_index()
MODEL = SentenceTransformer(MODEL_NAME)
print(f"Loaded {len(TEXTS)} chunks.")


def search(query: str, top_k: int = 5):
    q_emb = MODEL.encode([query])[0]
    norms = np.linalg.norm(EMBEDDINGS, axis=1) * np.linalg.norm(q_emb)
    sims = EMBEDDINGS @ q_emb / norms
    idxs = np.argsort(-sims)[:top_k]
    results = []
    for i in idxs:
        results.append({
            "score": float(sims[i]),
            "text": TEXTS[i],
            "source": SOURCES[i],
        })
    return results


# ------------------------------------------------------------
# NEW ToE‑Optimized Summarizer (markdown-aware, structure-aware)
# ------------------------------------------------------------
def simple_summarize(text: str, max_sections: int = 4) -> str:
    # Remove markdown symbols
    cleaned = re.sub(r"[#>*`]+", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Split on semantic boundaries: headings, bullets, colons, list markers
    parts = re.split(r"(?: - |\•|\–|: )", cleaned)
    parts = [p.strip() for p in parts if len(p.strip()) > 40]

    if not parts:
        return cleaned

    # Rank parts by length (proxy for importance)
    ranked = sorted(parts, key=lambda p: -len(p))
    selected = ranked[:max_sections]

    # Compress each selected part to its first sentence-like unit
    compressed = []
    for p in selected:
        s = re.split(r"[.!?]", p)[0]
        s = s.strip()
        if len(s) > 20:
            compressed.append(s)

    # Final assembly
    summary = ". ".join(compressed)
    if not summary.endswith("."):
        summary += "."

    return summary


@app.post("/search")
def search_endpoint(req: SearchRequest):
    results = search(req.query, req.top_k)
    return {"results": results}


@app.post("/chat")
def chat_endpoint(req:ChatRequest):
    user_messages = [m for m in req.messages if m.role == "user"]
    if not user_messages:
        return {"answer": "No user message provided.", "contexts": []}

    query = user_messages[-1].content

    results = search(query, req.top_k)
    combined_text = "\n\n".join([r["text"] for r in results])

    # NEW summarizer
    summary = simple_summarize(combined_text)

    citations = "\n".join(
        [f"[{i+1}] {r['source']} (score {r['score']:.3f})" for i, r in enumerate(results)]
    )

    answer = (
        f"{summary}\n\n"
        f"---\n"
        f"Sources:\n{citations}"
    )

    return {
        "answer": answer,
        "contexts": results
    }
