import json
import re
import numpy as np
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# --- Configuration ---
EMBEDDINGS_FILE = "data/toe_embeddings.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

app = FastAPI(title="Vector Search API")

# --- CORS FIX ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],
)

# --- Models ---
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    top_k: int = 5

# --- Data Loading ---
def load_index():
    try:
        with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        texts = [d["text"] for d in data]
        sources = [d["source"] for d in data]
        # Convert list of lists to a proper numpy matrix
        embeddings = np.array([d["embedding"] for d in data], dtype="float32")
        return texts, sources, embeddings
    except FileNotFoundError:
        print(f"Error: {EMBEDDINGS_FILE} not found.")
        return [], [], np.array([])
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], [], np.array([])

print("Loading index and model...")
TEXTS, SOURCES, EMBEDDINGS = load_index()
MODEL = SentenceTransformer(MODEL_NAME)
print(f"Loaded {len(TEXTS)} chunks.")

# --- Logic Functions ---
def search_logic(query: str, top_k: int = 5):
    if EMBEDDINGS.size == 0:
        return []
        
    q_emb = MODEL.encode([query])[0]
    
    # Cosine Similarity: (A · B) / (||A|| * ||B||)
    norms = np.linalg.norm(EMBEDDINGS, axis=1) * np.linalg.norm(q_emb)
    sims = EMBEDDINGS @ q_emb / (norms + 1e-9) # Added epsilon to avoid div by zero
    
    idxs = np.argsort(-sims)[:top_k]
    
    results = []
    for i in idxs:
        results.append({
            "score": float(sims[i]),
            "text": TEXTS[i],
            "source": SOURCES[i],
        })
    return results

def extract_definition(text: str, query: str) -> str:
    cleaned = re.sub(r"[#>*`]+", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    sentences = re.split(r"[.!?]", cleaned)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return ""

    q_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", query.lower()))
    patterns = [r"\bis\b", r"\brefers to\b", r"\bdefined as\b", r"\bdescribes\b"]

    candidates = []
    for s in sentences:
        if len(q_words & set(s.lower().split())) == 0:
            continue
        if any(re.search(p, s.lower()) for p in patterns):
            candidates.append(s)

    if candidates:
        return max(candidates, key=len) + "."
    return ""

# --- API Endpoints ---

@app.post("/search")
async def perform_search(request: SearchRequest):
    results = search_logic(request.query, request.top_k)
    # Add a definition highlight if possible
    definition = ""
    if results:
        definition = extract_definition(results[0]["text"], request.query)
        
    return {
        "query": request.query,
        "results": results,
        "extracted_definition": definition
    }

@app.get("/health")
async def health_check():
    return {"status": "online", "index_size": len(TEXTS)}
