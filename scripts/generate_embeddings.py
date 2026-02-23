import os
import glob
import json
from sentence_transformers import SentenceTransformer

# 1. Configure paths
CORPUS_DIR = "data/toe_corpus"
OUTPUT_FILE = "data/toe_embeddings.json"

# 2. Load embedding model only when needed
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_model():
    return SentenceTransformer(MODEL_NAME)

def load_documents(corpus_dir):
    ...
    return docs

def chunk_text(text, max_chars=800):
    ...
    return chunks

def main():
    model = get_model()
    docs = load_documents(CORPUS_DIR)
    ...
    # rest of your code

if __name__ == "__main__":
    main()
