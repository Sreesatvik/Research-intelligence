import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import adjusted_rand_score, silhouette_score

from clustering.kmeans import kmeans
from clustering.naming import cluster_keywords

truth = json.load(open("data/eval/topics.json", encoding="utf-8"))
paths = sorted(Path("data/corpus_arxiv").glob("*.txt"))
true_labels = [truth[p.stem] for p in paths]
texts = [p.read_text(encoding="utf-8") for p in paths]
X = SentenceTransformer("all-MiniLM-L6-v2").encode(texts, normalize_embeddings=True)

print(" k | tightness | silhouette | ARI vs my topics")
best_by_k = {}
for k in range(2, 13):
    runs = [kmeans(X, k, seed=s) for s in range(10)]
    labels, centroids, score = max(runs, key=lambda r: r[2])
    best_by_k[k] = labels
    sil = silhouette_score(X, labels, metric="cosine")
    ari = adjusted_rand_score(true_labels, labels)
    print(f"{k:2d} |   {score:6.2f}  |   {sil:5.3f}    | {ari:.2f}")

k = 7
labels = best_by_k[k]
print(f"\nCluster names for k={k} (most distinctive words):")
for j, words in enumerate(cluster_keywords(texts, labels, k)):
    size = int((labels == j).sum())
    print(f"  cluster {j} ({size:2d} papers): {', '.join(words)}")
