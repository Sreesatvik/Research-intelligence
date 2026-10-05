import numpy as np

from classification.models import NearestCentroid, SoftmaxRegression


def two_blobs():
    rng = np.random.default_rng(0)
    a = rng.normal([3, 0], 0.3, size=(20, 2))
    b = rng.normal([0, 3], 0.3, size=(20, 2))
    X = np.vstack([a, b])
    return X, np.array(["a"] * 20 + ["b"] * 20)


def test_softmax_learns_separable_data():
    X, y = two_blobs()
    model = SoftmaxRegression(lr=0.5, epochs=300).fit(X, y)
    assert (model.predict(X) == y).mean() > 0.95


def test_softmax_loss_goes_down():
    X, y = two_blobs()
    model = SoftmaxRegression(lr=0.5, epochs=100).fit(X, y)
    assert model.loss_history[-1] < model.loss_history[0]


def test_probabilities_sum_to_one():
    X, y = two_blobs()
    probs = SoftmaxRegression(lr=0.5, epochs=50).fit(X, y).predict_proba(X)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_nearest_centroid_on_blobs():
    X, y = two_blobs()
    assert (NearestCentroid().fit(X, y).predict(X) == y).mean() > 0.95
