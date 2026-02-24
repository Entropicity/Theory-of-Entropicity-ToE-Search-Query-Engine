from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List # ⭐ ADD THIS

EMBEDDINGS_FILE = "data/toe_embeddings.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (for local dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class ChatMessage(BaseModel):
    role: str   # "user" or "assistant"
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

@app.post("/search")
def search_endpoint(req: SearchRequest):
    results = search(req.query, req.top_k)
    return {"results": results}

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    # get last user message
    user_messages = [m for m in req.messages if m.role == "user"]
    if not user_messages:
        return {"answer": "No user message provided.", "contexts": []}

    query = user_messages[-1].content

    # run semantic search
    results = search(query, req.top_k)

    # build a structured answer with intro + citations
    intro = (
        "Here is a response grounded in the most relevant ToE passages "
        "I found for your question:\n"
    )

    answer_sections = []
    for i, r in enumerate(results, start=1):
        section = (
            f"[{i}] Source: {r['source']}\n"
            f"Score: {r['score']:.3f}\n\n"
            f"{r['text']}"
        )
        answer_sections.append(section)

    answer_body = "\n\n---\n\n".join(answer_sections)
    answer = intro + "\n\n" + answer_body

    return {
        "answer": answer,
        "contexts": results
    }
