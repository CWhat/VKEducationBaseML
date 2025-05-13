import logging

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split

from hw6.solution import GBCustomClassifier, GBCustomRegressor


def test_sklearn_regressor():
    n_samples = 1000
    mse_model = []
    mse_sklearn = []

    for i in range(10):
        params = {
            "learning_rate": np.random.uniform(0.01, 0.5),
            "n_estimators": np.random.randint(10, 200),
            "criterion": np.random.choice(["friedman_mse", "squared_error"]),
            "min_samples_split": np.random.randint(2, 10),
            "min_samples_leaf": np.random.randint(2, 10),
            "max_depth": np.random.randint(2, 10)
        }

        x, y = make_regression(
            n_samples=n_samples,
            n_informative=np.random.randint(2, 15)
        )

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

        model = GBCustomRegressor(**params)
        model.fit(x_train, y_train)
        mse_model.append(mean_squared_error(y_test, model.predict(x_test)))

        model = GradientBoostingRegressor(**params)
        model.fit(x_train, y_train)
        mse_sklearn.append(mean_squared_error(y_test, model.predict(x_test)))

    logging.info(f"Sklearn model avg mse: {np.mean(mse_sklearn)}")
    logging.info(f"Your model avg mse: {np.mean(mse_model)}")

    assert np.mean(mse_model) < np.mean(mse_sklearn) * 1.1


def test_sklearn_classifier():
    n_samples = 5000
    roc_auc_model = []
    roc_auc_sklearn = []
    accuracy_model = []
    accuracy_sklearn = []

    for i in range(10):
        params = {
            "learning_rate": np.random.uniform(0.01, 0.5),
            "n_estimators": np.random.randint(10, 200),
            "criterion": np.random.choice(["friedman_mse", "squared_error"]),
            "min_samples_split": np.random.randint(2, 10),
            "min_samples_leaf": np.random.randint(2, 10),
            "max_depth": np.random.randint(2, 10)
        }

        x, y = make_classification(
            n_samples=n_samples,
            n_classes=2,
            n_informative=np.random.randint(2, 15)
        )

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

        model = GBCustomClassifier(**params)
        model.fit(x_train, y_train)
        roc_auc_model.append(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]))
        accuracy_model.append(accuracy_score(y_test, model.predict(x_test)))

        model = GradientBoostingClassifier(**params)
        model.fit(x_train, y_train)
        roc_auc_sklearn.append(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]))
        accuracy_sklearn.append(accuracy_score(y_test, model.predict(x_test)))

    logging.info(f"Sklearn model avg roc_auc: {np.mean(roc_auc_sklearn)}")
    logging.info(f"Your model avg roc_auc: {np.mean(roc_auc_model)}")
    logging.info(f"Sklearn model avg accuracy: {np.mean(accuracy_sklearn)}")
    logging.info(f"Your model avg accuracy: {np.mean(accuracy_model)}")

    assert np.mean(roc_auc_model) > np.mean(roc_auc_sklearn) / 1.1
    assert np.mean(accuracy_model) > np.mean(accuracy_sklearn) / 1.1
