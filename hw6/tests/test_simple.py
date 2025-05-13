import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from hw6.solution import GBCustomClassifier, GBCustomRegressor


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "learning_rate": 0.15,
            "n_estimators": 100,
            "criterion": "friedman_mse",
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_depth": 4,
            "random_state": None,
        },
        {
            "learning_rate": 0.09,
            "n_estimators": 200,
            "criterion": "squared_error",
            "min_samples_split": 3,
            "min_samples_leaf": 2,
            "max_depth": 2,
            "random_state": 1234,
        }
    ]
)
def test_arguments_regressor(parameters: dict):
    obj = GBCustomRegressor(**parameters)
    for key, value in parameters.items():
        assert obj.__getattribute__(key) == value


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "learning_rate": 0.15,
            "n_estimators": 100,
            "criterion": "friedman_mse",
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_depth": 4,
            "random_state": None,
        },
        {
            "learning_rate": 0.09,
            "n_estimators": 200,
            "criterion": "squared_error",
            "min_samples_split": 3,
            "min_samples_leaf": 2,
            "max_depth": 2,
            "random_state": 1234,
        }
    ]
)
def test_arguments_classifier(parameters: dict):
    obj = GBCustomClassifier(**parameters)
    for key, value in parameters.items():
        assert obj.__getattribute__(key) == value


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "learning_rate": 0.15,
            "n_estimators": 3,
            "criterion": "friedman_mse",
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_depth": 4,
            "random_state": None,
        },
        {
            "learning_rate": 0.09,
            "n_estimators": 4,
            "criterion": "squared_error",
            "min_samples_split": 3,
            "min_samples_leaf": 2,
            "max_depth": 2,
            "random_state": 1234,
        }
    ]
)
def test_fit_and_trees_regressor(parameters: dict):
    model = GBCustomRegressor(**parameters)
    x, y = make_regression(
        n_samples=200,
        n_informative=6
    )
    model.fit(x, y)
    for key, value in parameters.items():
        if key in ("learning_rate", "n_estimators"):
            continue
        for tree in model.estimators_:
            assert tree.__getattribute__(key) == value


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "learning_rate": 0.15,
            "n_estimators": 3,
            "criterion": "friedman_mse",
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_depth": 4,
            "random_state": None,
        },
        {
            "learning_rate": 0.09,
            "n_estimators": 4,
            "criterion": "squared_error",
            "min_samples_split": 3,
            "min_samples_leaf": 2,
            "max_depth": 2,
            "random_state": 1234,
        }
    ]
)
def test_fit_and_trees_classifier(parameters: dict):
    model = GBCustomClassifier(**parameters)
    x, y = make_classification(
        n_samples=200,
        n_classes=2,
        n_informative=6
    )
    model.fit(x, y)
    for key, value in parameters.items():
        if key in ("learning_rate", "n_estimators"):
            continue
        for tree in model.estimators_:
            assert tree.__getattribute__(key) == value


@pytest.mark.parametrize(
    "random_seed",
    [1234, 4567, 386, 17482, 555]
)
def test_random_state_regressor(random_seed: int):
    x, y = make_regression(
        n_samples=200,
        n_informative=6
    )

    model1 = GBCustomRegressor(random_state=random_seed)
    model1.fit(x, y)

    model2 = GBCustomRegressor(random_state=random_seed)
    model2.fit(x, y)

    assert np.linalg.norm(model1.predict(x) - model2.predict(x)) < 1e-6


@pytest.mark.parametrize(
    "random_seed",
    [1234, 4567, 386, 17482, 555]
)
def test_random_state_classifier(random_seed: int):
    x, y = make_classification(
        n_samples=200,
        n_classes=2,
        n_informative=6
    )

    model1 = GBCustomClassifier(random_state=random_seed)
    model1.fit(x, y)

    model2 = GBCustomClassifier(random_state=random_seed)
    model2.fit(x, y)

    assert np.linalg.norm(model1.predict_proba(x) - model2.predict_proba(x)) < 1e-6
