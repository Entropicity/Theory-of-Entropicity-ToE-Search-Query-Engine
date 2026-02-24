from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List
import re
import os
import requests

# -----------------------------
# CONFIG
# -----------------------------
EMBEDDINGS_FILE = "data/toe_embeddings.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3"   # or "mistral", "phi3", etc.

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


# -----------------------------
# LOAD INDEX
# -----------------------------
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


# -----------------------------
# SEARCH (RAG RETRIEVAL)
# -----------------------------
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


# -----------------------------
# GENERATIVE LAYER (OLLAMA)
# -----------------------------
def generate_answer_with_llm(context: str, query: str) -> str:
    """
    Produces a coherent answer using retrieved context.
    Uses a local Ollama model (free, offline).
    """
    prompt = f"""
You are an expert explainer of the Theory of Entropicity (ToE).
Use ONLY the context below to answer the question.
If the context does not clearly support an answer, say you are not sure.

Context:
{context}

Question:
{query}

Write a clear, coherent answer in 2–4 sentences.
Do not mention the context or sources. Just answer directly.
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt}
    )

    data = response.json()
    return data.get("response", "").strip()


# -----------------------------
# FALLBACK SUMMARIZER
# -----------------------------
def simple_summarize(text: str, query: str, max_sections: int = 4) -> str:
    cleaned = re.sub(r"[#>*`]+", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    parts = re.split(r"[.!?]", cleaned)
    parts = [p.strip() for p in parts if len(p.strip()) > 30]

    if not parts:
        return cleaned

    q_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", query.lower()))
    def_keywords = {"is", "refers", "defined", "framework", "concept", "theory", "describes"}

    def score(part):
        p_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", part.lower()))
        overlap = len(q_words & p_words)
        def_bonus = 1 if any(k in part.lower() for k in def_keywords) else 0
        return overlap + def_bonus

    ranked = sorted(parts, key=score, reverse=True)
    selected = ranked[:max_sections]

    summary = ". ".join(selected).strip()
    if not summary.endswith("."):
        summary += "."

    return summary


# -----------------------------
# ENDPOINTS
# -----------------------------
@app.post("/search")
def search_endpoint(req: SearchRequest):
    results = search(req.query, req.top_k)
    return {"results": results}


@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    user_messages = [m for m in req.messages if m.role == "user"]
    if not user_messages:
        return {"answer": "No user message provided.", "contexts": []}

    query = user_messages[-1].content

    # 1. Retrieve chunks
    results = search(query, req.top_k)

    # 2. Build context for LLM
    context_text = ""
    for i, r in enumerate(results):
        context_text += f"[{i+1}] Source: {r['source']}\n{r['text']}\n\n"

    # 3. Generate answer using Ollama
    try:
        answer_text = generate_answer_with_llm(context_text, query)
    except Exception as e:
        combined_text = "\n\n".join([r["text"] for r in results])
        answer_text = simple_summarize(combined_text, query)

    # 4. Build citations
    citations = "\n".join(
        [f"[{i+1}] {r['source']} (score {r['score']:.3f})" for i, r in enumerate(results)]
    )

    answer = (
        f"{answer_text}\n\n"
        f"---\n"
        f"Sources:\n{citations}"
    )

    return {
        "answer": answer,
        "contexts": results
    }
