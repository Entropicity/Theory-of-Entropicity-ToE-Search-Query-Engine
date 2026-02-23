import json
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDINGS_FILE = "data/toe_embeddings.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_index():
    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [d["text"] for d in data]
    sources = [d["source"] for d in data]
    embeddings = np.array([d["embedding"] for d in data], dtype="float32")
    return texts, sources, embeddings

def build_model():
    return SentenceTransformer(MODEL_NAME)

def search(query, texts, sources, embeddings, model, top_k=5):
    q_emb = model.encode([query])[0]
    # cosine similarity
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb)
    sims = embeddings @ q_emb / norms
    idxs = np.argsort(-sims)[:top_k]
    results = []
    for i in idxs:
        results.append({
            "score": float(sims[i]),
            "text": texts[i],
            "source": sources[i],
        })
    return results

def main():
    print("Loading index...")
    texts, sources, embeddings = load_index()
    model = build_model()
    print(f"Loaded {len(texts)} chunks.")

    while True:
        query = input("\nEnter query (or 'q' to quit): ").strip()
        if query.lower() in {"q", "quit", "exit"}:
            break
        results = search(query, texts, sources, embeddings, model)
        print("\nTop results:\n")
        for r in results:
            print(f"[{r['score']:.3f}] {r['source']}")
            print(r["text"])
            print("-" * 80)

if __name__ == "__main__":
    main()
