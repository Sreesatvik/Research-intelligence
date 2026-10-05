"""Export the paper collection as 3D points for the UI's Corpus map (ui/public/corpus_map.json)."""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from clustering.kmeans import kmeans
from clustering.naming import cluster_keywords
from clustering.projection import pca_project

K = 7
OUT = Path("ui/public/corpus_map.json")

paths = sorted(Path("data/corpus_arxiv").glob("*.txt"))
texts = [p.read_text(encoding="utf-8") for p in paths]
ids = [p.stem for p in paths]
topics = json.load(open("data/eval/topics.json", encoding="utf-8"))

X = SentenceTransformer("all-MiniLM-L6-v2").encode(texts, normalize_embeddings=True)

# k-means depends on its random start: try 10 and keep the tightest, as we did in Stage 6
labels, _, _ = max((kmeans(X, K, seed=s) for s in range(10)), key=lambda run: run[2])
keywords = cluster_keywords(texts, labels, K)

coords, explained = pca_project(X, 3)
coords = coords / np.abs(coords).max()  # scale into [-1, 1] so the 3D scene has a predictable size

points = [{
    "id": ids[i],
    "title": texts[i].split(". ", 1)[0][:110],  # files start with "<title>. <abstract>"
    "x": round(float(coords[i, 0]), 4), "y": round(float(coords[i, 1]), 4), "z": round(float(coords[i, 2]), 4),
    "cluster": int(labels[i]),
    "topic": topics.get(ids[i], "unlabelled"),
} for i in range(len(ids))]

clusters = [{"id": j, "keywords": keywords[j], "size": int((labels == j).sum())} for j in range(K)]
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"points": points, "clusters": clusters,
                           "explained_variance": [round(float(v), 4) for v in explained]}), encoding="utf-8")
print(f"wrote {OUT} | {len(points)} points, {K} clusters")
print("variance kept by the 3 axes:", [f"{v:.1%}" for v in explained], f"(total {explained.sum():.1%})")
for c in clusters:
    print(f"  cluster {c['id']} ({c['size']:2d} papers): {', '.join(c['keywords'][:5])}")
