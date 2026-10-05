import numpy as np

from clustering.kmeans import kmeans


def unit(v):
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)


def test_two_obvious_groups_are_separated():
    X = np.array([unit([1, 0.05]), unit([1, 0.1]), unit([1, 0.0]),
                  unit([0.05, 1]), unit([0.1, 1]), unit([0.0, 1])])
    labels, _, _ = kmeans(X, k=2, seed=0)
    assert len(set(labels[:3])) == 1 and len(set(labels[3:])) == 1
    assert labels[0] != labels[3]


def test_same_seed_gives_same_result():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 5))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    a, _, _ = kmeans(X, k=3, seed=4)
    b, _, _ = kmeans(X, k=3, seed=4)
    assert (a == b).all()
