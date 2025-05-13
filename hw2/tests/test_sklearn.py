import logging
import time

import numpy as np
import pytest
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

from hw2.solution import LinearRegression


@pytest.mark.parametrize(
    "parameters,batch_size",
    [
        (
            {
                "penalty": "l2",
                "alpha": 0.001,
                "max_iter": 100,
                "tol": 0.01,
                "eta0": 0.01,
                "random_state": None,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "n_iter_no_change": 5
            },
            32
        ),
        (
            {
                "penalty": "l2",
                "alpha": 0.002,
                "max_iter": 100,
                "tol": 0.01,
                "eta0": 0.02,
                "random_state": None,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "n_iter_no_change": 7
            },
            64
        ),
        (
            {
                "penalty": "l2",
                "alpha": 0.0005,
                "max_iter": 100,
                "tol": 0.01,
                "eta0": 0.005,
                "random_state": None,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "n_iter_no_change": 10
            },
            128
        ),
        (
            {
                "penalty": "l1",
                "alpha": 0.01,
                "max_iter": 100,
                "tol": 0.01,
                "eta0": 0.02,
                "random_state": None,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "n_iter_no_change": 5
            },
            32
        ),
        (
            {
                "penalty": "l1",
                "alpha": 0.01,
                "max_iter": 100,
                "tol": 0.01,
                "eta0": 0.01,
                "random_state": None,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "n_iter_no_change": 7
            },
            64
        ),
        (
            {
                "penalty": "l1",
                "alpha": 0.01,
                "max_iter": 100,
                "tol": 0.01,
                "eta0": 0.005,
                "random_state": None,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "n_iter_no_change": 10
            },
            128
        ),
    ]
)
def test_sklearn(parameters: dict, batch_size: int):
    x = np.random.randn(10000, 50)
    split = 7500
    coefs = np.random.uniform(-10, 10, size=(50,))
    coefs[np.random.randint(0, 50, 20)] = 0
    delta = np.random.uniform(-3, 3, size=(10000,))
    bias = np.random.uniform(-20, 20, size=1)[0]
    y = x @ coefs + bias + delta

    x_train, y_train, x_test, y_test = x[:split], y[:split], x[split:], y[split:]

    model_time = 0
    model_best_metric = np.inf
    for _ in range(100):
        model = LinearRegression(batch_size=batch_size, **parameters)
        start = time.time()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        model_time += time.time() - start

        metric = mean_squared_error(y_test, y_pred)
        model_best_metric = min(metric, model_best_metric)

    model_sklearn_time = 0
    model_sklearn_best_metric = np.inf
    for _ in range(100):
        model_sklearn = SGDRegressor(learning_rate="constant", **parameters)
        start = time.time()
        model_sklearn.fit(x_train, y_train)
        y_pred = model_sklearn.predict(x_test)
        model_sklearn_time += time.time() - start

        metric = mean_squared_error(y_test, y_pred)
        model_sklearn_best_metric = min(metric, model_sklearn_best_metric)

    logging.info(f"Sklearn model best metric: {model_sklearn_best_metric}")
    logging.info(f"Your model best metric: {model_best_metric}")
    logging.info(f"Sklearn model time: {model_sklearn_time}")
    logging.info(f"Your model time: {model_time}")

    assert model_best_metric < model_sklearn_best_metric * 1.5
    assert model_time < model_sklearn_time * 6
