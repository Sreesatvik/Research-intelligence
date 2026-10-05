import numpy as np


def kmeans(X: np.ndarray, k: int, n_iter: int = 100, seed: int = 0):
    """Spherical k-means on unit-length vectors.

    Returns (labels, centroids, score). `score` is the total similarity between
    every point and its centroid: higher means tighter clusters.
    """
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), size=k, replace=False)]  # step 1: random start
    for _ in range(n_iter):
        labels = np.argmax(X @ centroids.T, axis=1)  # step 2: nearest centroid
        new_centroids = centroids.copy()
        for j in range(k):
            members = X[labels == j]
            if len(members):  # an empty cluster keeps its old centroid
                mean = members.mean(axis=0)  # step 3: move to the pile's average
                new_centroids[j] = mean / np.linalg.norm(mean)
        if np.allclose(new_centroids, centroids):  # nothing moved: converged
            break
        centroids = new_centroids
    score = float((X @ centroids.T)[np.arange(len(X)), labels].sum())
    return labels, centroids, score
