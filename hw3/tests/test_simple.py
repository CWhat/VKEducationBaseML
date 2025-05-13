import time

import numpy as np
import pytest
from scipy.special import xlogy

from hw3.solution import SoftmaxRegression


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "penalty": "l2",
            "alpha": 0.0001,
            "max_iter": 1000,
            "tol": 0.001,
            "eta0": 0.01,
            "random_state": None,
            "early_stopping": False,
            "validation_fraction": 0.1,
            "n_iter_no_change": 5,
            "batch_size": 64
        },
        {
            "penalty": "l1",
            "alpha": 0.1,
            "max_iter": 100,
            "tol": 0.01,
            "eta0": 0.001,
            "random_state": 1234,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "n_iter_no_change": 6,
            "batch_size": 8
        }
    ]
)
def test_arguments(parameters: dict):
    obj = SoftmaxRegression(**parameters)
    for key, value in parameters.items():
        assert obj.__getattribute__(key) == value


def log_loss(y_true: np.array, y_pred: np.array):
    y_true = np.array(y_true.reshape(-1, 1) == np.arange(y_pred.shape[1]), dtype=int)
    eps = np.finfo(y_pred.dtype).eps
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -xlogy(y_true, y_pred).sum(axis=1).mean()


def make_blobs(n_samples, n_features, n_classes, centers_range, divs_range):
    x = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)

    mu = centers_range[0] + np.random.rand(n_classes, n_features) * (centers_range[1] - centers_range[0])
    sigma = divs_range[0] + np.random.rand(n_classes, n_features) * (divs_range[1] - divs_range[0])
    for i, (m, s) in enumerate(zip(mu, sigma)):
        start, end = i * n_samples // n_classes, (i + 1) * n_samples // n_classes
        size = end - start
        x[start:end] = np.random.randn(size, n_features) * s + m
        y[start:end] = i
    ind = np.random.permutation(n_samples)
    return x[ind], y[ind]


@pytest.mark.parametrize(
    "random_seed,train_ratio,n_samples,n_classes,n_features,centers_range,divs_range,eta0,alpha,expected_loss",
    [
        [
            1234, 0.75, 500, 2, 2, (-2, 4), (0.3, 0.7), 0.1, 0.001, 0.108
        ],
        [
            1234, 0.8, 499, 5, 2, (-5, 5), (0.3, 0.7), 0.01, 0.01, 0.16
        ],
        [
            123, 0.8, 1000, 5, 20, (-4, 4), (3.0, 6.0), 0.01, 0.01, 0.9
        ]
    ]
)
def test_fit_predict(
        random_seed: int,
        train_ratio: float,
        n_samples: int,
        n_classes: int,
        n_features: int,
        centers_range: tuple[float],
        divs_range: tuple[float],
        eta0: float,
        alpha: float,
        expected_loss: float
):
    np.random.seed(random_seed)
    x, y = make_blobs(
        n_samples=n_samples,
        n_classes=n_classes,
        n_features=n_features,
        centers_range=centers_range,
        divs_range=divs_range
    )
    split = int(x.shape[0] * train_ratio)
    x_train, y_train, x_test, y_test = x[:split], y[:split], x[split:], y[split:]

    lr = SoftmaxRegression(max_iter=1000, eta0=eta0, alpha=alpha,
                           random_state=random_seed, penalty="l2", batch_size=32)
    lr.fit(x_train, y_train)

    y_pred = lr.predict_proba(x_test)

    assert y_pred.dtype == float, "Wrong dtype for predict_proba, float expected"
    assert lr.predict(x_test).dtype == int, "Wrong dtype for predict, int expected"
    assert log_loss(y_test, y_pred) < expected_loss * 2, "Too bad loss"


def test_max_iter():
    np.random.seed(777)
    x, y = make_blobs(
        n_samples=2000,
        n_classes=5,
        n_features=20,
        centers_range=(-4, 4),
        divs_range=(3.0, 6.0)
    )

    models = [
        SoftmaxRegression(max_iter=1),
        SoftmaxRegression(max_iter=5),
        SoftmaxRegression(max_iter=15)
    ]

    totals = []
    for model in models:
        start = time.time()
        for _ in range(100):
            model.fit(x, y)
        totals.append(time.time() - start)

    assert totals[0] < totals[1] < totals[2]


def test_early_stopping():
    np.random.seed(777)
    x, y = make_blobs(
        n_samples=1000,
        n_classes=5,
        n_features=3,
        centers_range=(-4, 4),
        divs_range=(3.0, 6.0)
    )

    models = [
        SoftmaxRegression(max_iter=50, early_stopping=True),
        SoftmaxRegression(max_iter=50, early_stopping=False),
    ]

    totals = []
    for model in models:
        start = time.time()
        for _ in range(50):
            model.fit(x, y)
        totals.append(time.time() - start)

    assert totals[0] < totals[1]


@pytest.mark.parametrize(
    "random_seed",
    [1234, 4567, 386, 17482, 555]
)
def test_random_state(random_seed: int):
    x, y = make_blobs(
        n_samples=1000,
        n_classes=5,
        n_features=3,
        centers_range=(-4, 4),
        divs_range=(3.0, 6.0)
    )

    lr1 = SoftmaxRegression(max_iter=10, random_state=random_seed, shuffle=True, batch_size=8)
    lr1.fit(x, y)

    lr2 = SoftmaxRegression(max_iter=10, random_state=random_seed, shuffle=True, batch_size=8)
    lr2.fit(x, y)

    assert np.linalg.norm(lr1.coef_ - lr2.coef_) < 1e-6
    assert np.linalg.norm(lr1.intercept_ - lr2.intercept_) < 1e-6


@pytest.mark.parametrize(
    "penalty,alpha,weights,bias,expected",
    [
        (
            "l2",
            0.0001,
            [5.73740291, -1.7512518, -2.62246896, -2.97450237, -0.85847272,
             -0.53015623, -5.88052802, -8.89615508,  9.63588443, -0.06066031],
            10,
            -0.0016401816278803967 * 2
        ),
        (
            "l1",
            0.0123,
            [5.73740291, -1.7512518, -2.62246896, -2.97450237, -0.85847272,
             -0.53015623, -5.88052802, -8.89615508, 9.63588443, -0.06066031],
            -5,
            -0.0738 * 2
        ),
        (
            "l2",
            0.4056,
            [6.929, 3.885, -7.773, 6.456, -0.579, -1.183, -2.62, -4.687,
             4.151, -5.809, -3.49, 7.474, -8.813, 0.907, 0.434],
            1.234,
            -3.827241600000001 * 2
        ),
        (
            "l1",
            0.003961,
            [-6.8727, 9.8228, -3.4836, -3.8546, 0., -2.5597, -1.9072,
             0., 8.8834, -9.9594, 0., 0., -9.7112, -2.761, -9.1893],
            0.1,
            -0.027726999999999998 * 2
        ),
    ]
)
def test_penalty(penalty, alpha, weights, bias, expected):
    model = SoftmaxRegression(penalty=penalty, alpha=alpha)
    weights = np.array(weights).reshape(-1, 1)
    model.coef_ = np.hstack([weights, weights])
    model.intercept_ = np.array([bias, bias])
    assert np.abs(np.sum(model.get_penalty_grad()) - expected) < 1e-5


@pytest.mark.parametrize(
    "x,expected",
    [
        [
            [1, 2, 3], [0.09003057, 0.24472847, 0.66524096]
        ],
        [
            [[1, 2, 3], [4, 7, 6]], [[0.09003057, 0.24472847, 0.66524096], [0.03511903, 0.70538451, 0.25949646]]
        ],
        [
            [12345, 67890, 99999999], [0., 0., 1.]
        ],
        [
            [[12345, 67890, 99999999], [123, 678, 88888888]], [[0., 0., 1.], [0., 0., 1.]]
        ]
    ]
)
def test_softmax(x, expected):
    x = np.array(x)
    expected = np.array(expected)
    val = SoftmaxRegression.softmax(x)
    assert expected.shape == val.shape
    assert np.linalg.norm(expected - val) < 1e-5
