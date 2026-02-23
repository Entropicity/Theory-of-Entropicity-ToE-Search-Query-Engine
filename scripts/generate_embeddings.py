import os
import glob
import json
from sentence_transformers import SentenceTransformer

# Paths
CORPUS_DIR = "data/toe_corpus"
OUTPUT_FILE = "data/toe_embeddings.json"

# Model name
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_model():
    return SentenceTransformer(MODEL_NAME)

def load_documents(corpus_dir):
    docs = []
    for path in glob.glob(os.path.join(corpus_dir, "**", "*.*"), recursive=True):
        if any(path.endswith(ext) for ext in [".md", ".txt", ".html"]):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                docs.append({"path": path, "text": text})
    return docs

def chunk_text(text, max_chars=800):
    chunks = []
    current = []
    length = 0
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if length + len(para) > max_chars and current:
            chunks.append("\n\n".join(current))
            current = [para]
            length = len(para)
        else:
            current.append(para)
            length += len(para)
    if current:
        chunks.append("\n\n".join(current))
    return chunks

def main():
    model = get_model()
    docs = load_documents(CORPUS_DIR)
    all_chunks = []

    for doc in docs:
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            all_chunks.append({
                "source": doc["path"],
                "text": chunk
            })

    texts = [c["text"] for c in all_chunks]
    print(f"Embedding {len(texts)} chunks...")

    embeddings = model.encode(texts, show_progress_bar=True)

    data = []
    for chunk, emb in zip(all_chunks, embeddings):
        data.append({
            "source": chunk["source"],
            "text": chunk["text"],
            "embedding": emb.tolist()
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)

    print(f"Saved {len(data)} chunks with embeddings to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
