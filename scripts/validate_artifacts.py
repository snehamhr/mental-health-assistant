from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
chunks_path = BASE_DIR / "data" / "combined_chunks.pkl"
embeddings_path = BASE_DIR / "data" / "embeddings.npy"

if not chunks_path.is_file() or not embeddings_path.is_file():
    raise SystemExit("Build or copy combined_chunks.pkl and embeddings.npy first.")

with chunks_path.open("rb") as file:
    chunks = pickle.load(file)
embeddings = np.load(embeddings_path, allow_pickle=False)

print(f"Chunks: {len(chunks):,}")
print(f"Embeddings shape: {embeddings.shape}")
if len(chunks) != embeddings.shape[0]:
    raise SystemExit("ERROR: chunk and embedding counts do not match.")
print("Artifacts look valid.")
