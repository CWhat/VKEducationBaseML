import logging
import time

import numpy as np
import pytest
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, accuracy_score

from hw3.solution import SoftmaxRegression
from hw3.tests.test_simple import make_blobs


@pytest.mark.parametrize(
    "parameters,batch_size",
    [
        (
            {
                "penalty": "l2",
                "alpha": 0.001,
                "max_iter": 100,
                "tol": 0.001,
                "eta0": 0.01,
                "random_state": None,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "n_iter_no_change": 10
            },
            16
        ),
        (
            {
                "penalty": "l2",
                "alpha": 0.001,
                "max_iter": 100,
                "tol": 0.001,
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
                "tol": 0.001,
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
                "penalty": "l1",
                "alpha": 0.01,
                "max_iter": 100,
                "tol": 0.001,
                "eta0": 0.02,
                "random_state": None,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "n_iter_no_change": 10
            },
            16
        ),
        (
            {
                "penalty": "l1",
                "alpha": 0.01,
                "max_iter": 100,
                "tol": 0.001,
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
                "tol": 0.001,
                "eta0": 0.01,
                "random_state": None,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "n_iter_no_change": 7
            },
            64
        ),
    ]
)
def test_sklearn(parameters: dict, batch_size: int):
    n_samples = 5000
    x, y = make_blobs(
        n_samples=n_samples,
        n_classes=5,
        n_features=20,
        centers_range=(-3, 5),
        divs_range=(3.0, 6.0)
    )
    x_random = 4 * np.random.randn(n_samples, 20)
    x = np.hstack([x, x_random])
    split = int(n_samples * 0.75)

    x_train, y_train, x_test, y_test = x[:split], y[:split], x[split:], y[split:]

    model_time = 0
    model_best_metric = np.inf
    model_best_metric2 = 0
    for _ in range(50):
        model = SoftmaxRegression(batch_size=batch_size, **parameters)
        start = time.time()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)
        model_time += time.time() - start

        metric = log_loss(y_test, y_pred_proba)
        model_best_metric = min(metric, model_best_metric)
        metric2 = accuracy_score(y_test, y_pred)
        model_best_metric2 = max(metric2, model_best_metric2)

    model_sklearn_time = 0
    model_sklearn_best_metric = np.inf
    model_sklearn_best_metric2 = 0
    for _ in range(50):
        model_sklearn = SGDClassifier(learning_rate="constant", loss="log_loss", **parameters)
        start = time.time()
        model_sklearn.fit(x_train, y_train)
        y_pred = model_sklearn.predict(x_test)
        y_pred_proba = model_sklearn.predict_proba(x_test)
        model_sklearn_time += time.time() - start

        metric = log_loss(y_test, y_pred_proba)
        model_sklearn_best_metric = min(metric, model_sklearn_best_metric)
        metric2 = accuracy_score(y_test, y_pred)
        model_sklearn_best_metric2 = max(metric2, model_sklearn_best_metric2)

    logging.info(f"Sklearn model best log_loss: {model_sklearn_best_metric}")
    logging.info(f"Your model best log_loss: {model_best_metric}")
    logging.info(f"Sklearn model best accuracy: {model_sklearn_best_metric2}")
    logging.info(f"Your model best accuracy: {model_best_metric2}")
    logging.info(f"Sklearn model time: {model_sklearn_time}")
    logging.info(f"Your model time: {model_time}")

    assert model_best_metric < model_sklearn_best_metric * 1.1
    assert model_best_metric2 > model_sklearn_best_metric2 / 1.1
    assert model_time < model_sklearn_time * 6
