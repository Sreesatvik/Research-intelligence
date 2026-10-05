import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import RepeatedStratifiedKFold

from classification.models import SoftmaxRegression
from evaluation.router_questions import QUESTIONS, TOKEN_COST
from rag.cost_router import route_by_rules

texts = [q for q, _ in QUESTIONS]
y = np.array([label for _, label in QUESTIONS])
X = SentenceTransformer("all-MiniLM-L6-v2").encode(texts, normalize_embeddings=True)
ORDER = {"retrieval": 0, "rag": 1, "agent": 2}  # cheap -> expensive


def report(name: str, truth, pred) -> None:
    truth, pred = list(truth), list(pred)
    acc = np.mean([t == p for t, p in zip(truth, pred)])
    under = sum(ORDER[p] < ORDER[t] for t, p in zip(truth, pred))  # sent somewhere too cheap
    over = sum(ORDER[p] > ORDER[t] for t, p in zip(truth, pred))   # sent somewhere too costly
    tokens = np.mean([TOKEN_COST[p] for p in pred])
    print(f"{name:26s} | acc {acc:.2f} | UNDER-routed {under:3d}/{len(truth)} | over-routed {over:3d} | avg tokens {tokens:6.0f}")


print(f"{len(QUESTIONS)} labeled questions\n")
report("always agent", y, ["agent"] * len(y))
report("always rag", y, ["rag"] * len(y))
report("oracle (perfect)", y, y)
report("keyword rules", y, [route_by_rules(q) for q in texts])

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
true_all, pred_all, texts_all = [], [], []
for tr, te in cv.split(X, y):
    model = SoftmaxRegression().fit(X[tr], y[tr])
    true_all += list(y[te]); pred_all += list(model.predict(X[te])); texts_all += [texts[i] for i in te]
report("learned (5x5-fold CV)", true_all, pred_all)

print("\nQuestions the RULES get wrong:")
for q, label in QUESTIONS:
    got = route_by_rules(q)
    if got != label:
        print(f"  wanted {label:9s} got {got:9s} | {q[:80]}")
