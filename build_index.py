"""Step 2: turn every paper into an SBERT vector -> embeddings.npy"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_csv("papers.csv")
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = (df["title"] + ". " + df["abstract"]).tolist()
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,  # so dot product = cosine similarity
)

np.save("embeddings.npy", embeddings)
print("Saved embeddings:", embeddings.shape)
