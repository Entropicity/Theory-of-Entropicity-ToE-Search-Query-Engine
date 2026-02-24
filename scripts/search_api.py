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
# DEFINITION EXTRACTOR (primary)
# ------------------------------------------------------------
def extract_definition(text: str, query: str) -> str:
    cleaned = re.sub(r"[#>*`]+", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    sentences = re.split(r"[.!?]", cleaned)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return ""

    q_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", query.lower()))

    patterns = [
        r"\bis\b",
        r"\brefers to\b",
        r"\bdefined as\b",
        r"\bdescribes\b",
        r"\bframework\b",
        r"\btheory\b",
        r"\bconcept\b",
    ]

    candidates = []
    for s in sentences:
        if len(q_words & set(s.lower().split())) == 0:
            continue
        if any(re.search(p, s.lower()) for p in patterns):
            candidates.append(s)

    if candidates:
        best = max(candidates, key=len)
        return best + "."

    return ""


# ------------------------------------------------------------
# FALLBACK QUERY‑AWARE SUMMARIZER
# ------------------------------------------------------------
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

    results = search(query, req.top_k)

    # Prefer the definition file if present
    definition_hit = None
    for r in results:
        if "definition.md" in r["source"]:
            definition_hit = r
            break

    if definition_hit is not None:
        # Use only the definition text
        summary = definition_hit["text"].strip()
        used_results = [definition_hit]
    else:
        # Fall back to combined text + summarizer
        combined_text = "\n\n".join([r["text"] for r in results])
        summary = simple_summarize(combined_text, query)
        used_results = results

    citations = "\n".join(
        [f"[{i+1}] {r['source']} (score {r['score']:.3f})" for i, r in enumerate(used_results)]
    )

    answer = (
        f"{summary}\n\n"
        f"---\n"
        f"Sources:\n{citations}"
    )

    return {
        "answer": answer,
        "contexts": used_results
    }
