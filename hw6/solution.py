import numpy as np
from sklearn.tree import DecisionTreeRegressor


class GBCustomRegressor:
    def __init__(
            self,
            *,
            learning_rate=0.1,
            n_estimators=100,
            criterion="friedman_mse",
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=3,
            random_state=None
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators = None
        self.init = None

    def fit(self, x, y):
        assert self.init is None
        assert self.estimators is None

        self.estimators = []

        # Для MSE оптимальной константой будет среднее арифметическое
        self.init = np.mean(y)

        # Градиент вычисляется как
        # grad = 2.0 * (self.predict(x) - y)
        # Т.е. при обучении каждого дерева будет много раз вычисляться predict
        # предыдущих деревьев, чего можно избежать, если хранить его.
        # grad = 2.0 * (old(x) + learning_rate * new_tree(x) - y)
        # grad = 2.0 * (old(x) - y) + 2.0 * learning_rate * new_tree(x)
        # grad = 2.0 * old_grad + 2.0 * learning_rate * new_tree(x)
        # Таким образом,
        # grad += 2.0 * learning_rate * new_tree.predict(x)
        # Замечание: в sklearn используется MSE, деленный на 2, что позволяет
        # убрать везде двойку, поэтому и здесь он будет использоваться.

        grad = np.full_like(y, self.init) - y

        for i in range(self.n_estimators):
            new_tree = DecisionTreeRegressor(
                criterion=self.criterion,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            new_tree.fit(x, -grad)
            grad += self.learning_rate * new_tree.predict(x)
            self.estimators.append(new_tree)

    def predict(self, x):
        assert self.init is not None
        assert self.estimators is not None

        res = np.ones(x.shape[0]) * self.init
        for estimator in self.estimators:
            res += self.learning_rate * estimator.predict(x)

        return res

    @property
    def estimators_(self):
        return self.estimators


class GBCustomClassifier:
    def __init__(
            self,
            *,
            learning_rate=0.1,
            n_estimators=100,
            criterion="friedman_mse",
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=3,
            random_state=None
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators = None
        self.init = None

    def fit(self, x, y):
        assert self.init is None
        assert self.estimators is None

        self.estimators = []

        # Тут такая же оптимальная константа
        self.init = np.mean(y)

        for i in range(self.n_estimators):
            grad = self._grad_loss(x, y)
            new_tree = DecisionTreeRegressor(
                criterion=self.criterion,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            new_tree.fit(x, -grad)
            self.estimators.append(new_tree)

    def _predict_proba(self, x):
        assert self.init is not None
        assert self.estimators is not None

        res = np.ones(x.shape[0]) * self.init
        for estimator in self.estimators:
            res += self.learning_rate * estimator.predict(x)

        return res

    def predict_proba(self, x):
        prob = self._predict_proba(x)
        return np.stack((1 - prob, prob), axis=1)

    def predict(self, x):
        return np.astype(self._predict_proba(x) > 0.5, np.int32)

    def _grad_loss(self, x, y):
        pred = self._predict_proba(x)
        eps = 1e-8
        np.clip(pred, eps, 1 - eps, out=pred)
        # Как и в случае с MSE в sklearn используется половинное биномиальное
        # распределение
        return -0.5 * (y - pred) / (pred * (1.0 - pred))

    @property
    def estimators_(self):
        return self.estimators
