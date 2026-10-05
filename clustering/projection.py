import numpy as np


def pca_project(X: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Project points onto the directions along which they vary the most (PCA via SVD).

    Returns (coordinates with shape (n_points, n_components), fraction of total variance
    each component keeps). The kept fractions say how much of the original structure a
    low-dimensional picture can show.
    """
    centered = X - X.mean(axis=0)  # PCA works around the centre of the data
    _, singular_values, directions = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ directions[:n_components].T
    variance = singular_values**2
    return coordinates, (variance / variance.sum())[:n_components]
