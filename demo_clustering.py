import json
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import adjusted_rand_score

from clustering.kmeans import kmeans

truth = json.load(open("data/eval/topics.json", encoding="utf-8"))
paths = sorted(Path("data/corpus_arxiv").glob("*.txt"))
ids = [p.stem for p in paths]
texts = [p.read_text(encoding="utf-8") for p in paths]
assert set(ids) == set(truth), set(ids) ^ set(truth)  # every paper must have a label
true_labels = [truth[i] for i in ids]

X = SentenceTransformer("all-MiniLM-L6-v2").encode(texts, normalize_embeddings=True)
print("one vector per paper:", X.shape)

K = 7
runs = []
for seed in range(10):
    labels, centroids, score = kmeans(X, K, seed=seed)
    ari = adjusted_rand_score(true_labels, labels)
    runs.append((score, ari, labels, centroids))
    print(f"seed {seed}: tightness {score:6.2f}  agreement with my topics (ARI) {ari:.2f}")

score, ari, labels, centroids = max(runs, key=lambda r: r[0])  # pick tightest run
print(f"\nBEST by tightness: ARI {ari:.2f}\n")
for j in range(K):
    members = [i for i in range(len(ids)) if labels[i] == j]
    counts = Counter(true_labels[i] for i in members)
    top, n = counts.most_common(1)[0]
    rep = max(members, key=lambda i: X[i] @ centroids[j])  # closest to the centre
    print(f"cluster {j}: {len(members):2d} papers, mostly {top} ({n}/{len(members)})")
    print(f"    representative: {texts[rep][:75]}")
    others = {t: c for t, c in counts.items() if t != top}
    if others:
        print(f"    mixed in: {others}")
