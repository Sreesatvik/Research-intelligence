import numpy as np


class NearestCentroid:
    """Predict the class whose average (centroid) vector is most similar."""

    def fit(self, X, y):
        self.classes = np.unique(y)
        cents = np.array([X[y == c].mean(axis=0) for c in self.classes])
        self.centroids = cents / np.linalg.norm(cents, axis=1, keepdims=True)
        return self

    def predict(self, X):
        return self.classes[np.argmax(X @ self.centroids.T, axis=1)]


class SoftmaxRegression:
    """Multiclass logistic regression, trained with plain gradient descent."""

    def __init__(self, lr: float = 5.0, epochs: int = 500, l2: float = 1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def _probs(self, X):
        logits = X @ self.W + self.b
        logits = logits - logits.max(axis=1, keepdims=True)  # numerical stability
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        self.classes = np.unique(y)
        idx = np.searchsorted(self.classes, y)
        Y = np.eye(len(self.classes))[idx]  # one-hot targets
        n, d = X.shape
        self.W = np.zeros((d, len(self.classes)))
        self.b = np.zeros(len(self.classes))
        self.loss_history = []
        for _ in range(self.epochs):
            P = self._probs(X)
            self.loss_history.append(float(-np.log(P[np.arange(n), idx] + 1e-12).mean()))
            self.W -= self.lr * (X.T @ (P - Y) / n + self.l2 * self.W)  # gradient step
            self.b -= self.lr * (P - Y).mean(axis=0)
        return self

    def predict_proba(self, X):
        return self._probs(X)

    def predict(self, X):
        return self.classes[np.argmax(self._probs(X), axis=1)]
