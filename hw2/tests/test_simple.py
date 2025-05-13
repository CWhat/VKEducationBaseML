import time

import numpy as np
import pytest

from hw2.solution import LinearRegression


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
    obj = LinearRegression(**parameters)
    for key, value in parameters.items():
        assert obj.__getattribute__(key) == value


def mse(y_pred: np.array, y_true: np.array):
    return np.mean((y_pred - y_true) ** 2)


@pytest.mark.parametrize(
    "random_seed,train_ratio,size,coefs,bias,noise,penalty,eta0,alpha,expected_loss",
    [
        [
            1234, 0.75, (100, 1), np.array([0.4]),
            3.0, 1.0, "l2", 0.01, 0.01, 0.31
        ],
        [
            1235, 0.7, (100, 3), np.array([0.4, 1.0, -0.7]),
            -2.0, 5.0, "l1", 0.02, 0.03, 8.03
        ],
        [
            1236, 0.8, (300, 5), np.array([0.4, 1.0, -0.7, 0.1, -0.5]),
            3.0, -3.0, "l2", 0.005, 0.02, 2.94
        ]
    ]
)
def test_fit_predict(
        random_seed: int,
        train_ratio: float,
        size: tuple[int],
        coefs: np.array,
        bias: float,
        noise: float,
        penalty: str,
        eta0: float,
        alpha: float,
        expected_loss: float
):
    np.random.seed(random_seed)
    split = int(size[0] * train_ratio)

    x = np.random.randn(*size)
    delta = np.random.uniform(-noise, noise, size=(size[0],))
    y = x @ coefs + bias + delta

    x_train, y_train, x_test, y_test = x[:split], y[:split], x[split:], y[split:]

    reg = LinearRegression(max_iter=1000, tol=1e-3, alpha=alpha, random_state=random_seed)
    reg.fit(x_train, y_train)

    y_pred_test = reg.predict(x_test)

    assert mse(y_pred_test, y_test) < expected_loss * 2


def test_max_iter():
    np.random.seed(1234)
    x = np.random.randn(1000, 50)
    coefs = np.random.uniform(-10, 10, size=(50,))
    delta = np.random.uniform(-3, 3, size=(1000,))
    y = x @ coefs + delta

    models = [
        LinearRegression(max_iter=1),
        LinearRegression(max_iter=5),
        LinearRegression(max_iter=15)
    ]

    totals = []
    for model in models:
        start = time.time()
        for _ in range(100):
            model.fit(x, y)
        totals.append(time.time() - start)

    assert totals[0] < totals[1] < totals[2]


def test_early_stopping():
    np.random.seed(12345)
    x = np.random.randn(1000, 50)
    coefs = np.random.uniform(-10, 10, size=(50,))
    delta = np.random.uniform(-3, 3, size=(1000,))
    y = x @ coefs + delta

    models = [
        LinearRegression(max_iter=50, early_stopping=True),
        LinearRegression(max_iter=50, early_stopping=False),
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
    x = np.random.randn(100, 8)
    delta = np.random.uniform(-3.0, 5.0, size=(100,))
    coefs = np.random.uniform(-10, 10, size=(8,))
    bias = np.random.uniform(-20, 20, size=1)[0]
    y = x @ coefs + bias + delta

    reg1 = LinearRegression(max_iter=10, random_state=random_seed, shuffle=True, batch_size=8)
    reg1.fit(x, y)

    reg2 = LinearRegression(max_iter=10, random_state=random_seed, shuffle=True, batch_size=8)
    reg2.fit(x, y)

    assert mse(reg1.coef_.reshape(-1), reg2.coef_.reshape(-1)) < 1e-6
    assert mse(reg1.intercept_, reg2.intercept_) < 1e-6


@pytest.mark.parametrize(
    "penalty,alpha,weights,bias,expected",
    [
        (
            "l2",
            0.0001,
            [5.73740291, -1.7512518, -2.62246896, -2.97450237, -0.85847272,
             -0.53015623, -5.88052802, -8.89615508,  9.63588443, -0.06066031],
            10,
            -0.0016401816278803967
        ),
        (
            "l1",
            0.0123,
            [5.73740291, -1.7512518, -2.62246896, -2.97450237, -0.85847272,
             -0.53015623, -5.88052802, -8.89615508, 9.63588443, -0.06066031],
            -5,
            -0.0738
        ),
        (
            "l2",
            0.4056,
            [6.929, 3.885, -7.773, 6.456, -0.579, -1.183, -2.62, -4.687,
             4.151, -5.809, -3.49, 7.474, -8.813, 0.907, 0.434],
            1.234,
            -3.827241600000001
        ),
        (
            "l1",
            0.003961,
            [-6.8727, 9.8228, -3.4836, -3.8546, 0., -2.5597, -1.9072,
             0., 8.8834, -9.9594, 0., 0., -9.7112, -2.761, -9.1893],
            0.1,
            -0.027726999999999998
        ),
    ]
)
def test_penalty(penalty, alpha, weights, bias, expected):
    model = LinearRegression(penalty=penalty, alpha=alpha)
    model.coef_ = np.array(weights)
    model.intercept_ = bias
    assert np.abs(np.sum(model.get_penalty_grad()) - expected) < 1e-5
