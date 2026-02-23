import json
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

EMBEDDINGS_FILE = "data/toe_embeddings.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

app = FastAPI()

class SearchRequest(BaseModel):
    query: str
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
