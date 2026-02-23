# ← your script goes HERE

import os
import glob
import json
from sentence_transformers import SentenceTransformer

# 1. Configure paths
CORPUS_DIR = "data/toe_corpus"
OUTPUT_FILE = "data/toe_embeddings.json"

# 2. Load embedding model (free, small, good)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

