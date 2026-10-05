import numpy as np
from sentence_transformers import SentenceTransformer

from classification.models import SoftmaxRegression


class DocumentClassifier:
    """text -> (topic label, confidence). Embedder + softmax regression."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.model = SoftmaxRegression()

    def fit(self, texts: list[str], labels: list[str]):
        X = self.embedder.encode(texts, normalize_embeddings=True)
        self.model.fit(X, np.array(labels))
        return self

    def classify(self, text: str) -> tuple[str, float]:
        x = self.embedder.encode([text], normalize_embeddings=True)
        probs = self.model.predict_proba(x)[0]
        best = int(np.argmax(probs))
        return str(self.model.classes[best]), float(probs[best])
