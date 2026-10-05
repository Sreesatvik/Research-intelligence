import json
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from classification.models import NearestCentroid, SoftmaxRegression

truth = json.load(open("data/eval/topics.json", encoding="utf-8"))
paths = sorted(Path("data/corpus_arxiv").glob("*.txt"))
y = np.array([truth[p.stem] for p in paths])
X = SentenceTransformer("all-MiniLM-L6-v2").encode(
    [p.read_text(encoding="utf-8") for p in paths], normalize_embeddings=True)
print("class sizes:", dict(Counter(y)))

class Majority:
    def fit(self, X, y):
        self.label = Counter(y).most_common(1)[0][0]
        return self
    def predict(self, X):
        return np.array([self.label] * len(X))

models = {
    "always-majority (baseline)": Majority,
    "nearest centroid": NearestCentroid,
    "softmax (ours)": SoftmaxRegression,
    "sklearn LogisticRegression": lambda: LogisticRegression(C=100, max_iter=2000),
}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
labels = sorted(set(y))
print("\nmodel                        | accuracy | macro-F1 | train acc")
oof = {}
for name, make in models.items():
    accs, f1s, train_accs, true_all, pred_all = [], [], [], [], []
    for tr, te in cv.split(X, y):
        m = make().fit(X[tr], y[tr])
        pred = m.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
        train_accs.append(accuracy_score(y[tr], m.predict(X[tr])))
        true_all += list(y[te]); pred_all += list(pred)
    oof[name] = (true_all, pred_all)
    print(f"{name:28s} |  {np.mean(accs):.3f}   |  {np.mean(f1s):.3f}   |  {np.mean(train_accs):.3f}")

true_all, pred_all = oof["softmax (ours)"]
print("\nConfusion matrix, softmax (rows = true topic, cols = predicted), summed over 25 test folds:")
short = [l[:6] for l in labels]
print(" " * 22 + " ".join(f"{s:>6}" for s in short))
for t in labels:
    row = [sum(1 for a, b in zip(true_all, pred_all) if a == t and b == p) for p in labels]
    print(f"{t:22s}" + " ".join(f"{v:6d}" for v in row))
