import numpy as np

from clustering.projection import pca_project


def test_output_shapes():
    X = np.random.default_rng(0).normal(size=(20, 10))
    coords, explained = pca_project(X, 3)
    assert coords.shape == (20, 3) and explained.shape == (3,)


def test_explained_variance_is_sorted_and_at_most_one():
    X = np.random.default_rng(1).normal(size=(40, 8))
    _, explained = pca_project(X, 3)
    assert list(explained) == sorted(explained, reverse=True) and explained.sum() <= 1.0 + 1e-9


def test_finds_the_dominant_direction():
    rng = np.random.default_rng(2)
    X = np.column_stack([rng.normal(scale=10, size=200),   # big spread along x
                         rng.normal(scale=0.1, size=200),
                         rng.normal(scale=0.1, size=200)])
    coords, explained = pca_project(X, 1)
    assert explained[0] > 0.99
    assert abs(np.corrcoef(coords[:, 0], X[:, 0])[0, 1]) > 0.99  # component 1 tracks x (sign is arbitrary)


def test_projection_is_centred():
    X = np.random.default_rng(3).normal(loc=5.0, size=(30, 6))
    coords, _ = pca_project(X, 3)
    assert np.allclose(coords.mean(axis=0), 0, atol=1e-9)
