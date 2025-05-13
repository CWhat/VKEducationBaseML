import numpy as np


class SoftmaxRegression:
    def __init__(
            self,
            *,
            penalty="l2",
            alpha=0.0001,
            max_iter=100,
            tol=0.001,
            random_state=None,
            eta0=0.01,
            early_stopping=False,
            validation_fraction=0.1,
            n_iter_no_change=5,
            shuffle=True,
            batch_size=32
    ):
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.eta0 = eta0
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.w = None
        self.rng = None
        self.classes_ = None
        self.n_classes_ = None

    def get_penalty_grad(self):
        if self.w is None or self.alpha is None or self.penalty is None:
            raise AttributeError()

        res = None
        if self.penalty == "l2":
            res = self.alpha * 2 * self.w
        elif self.penalty == "l1":
            res = self.alpha * np.sign(self.w)
        else:
            raise AttributeError()

        res[0] = 0.0
        return res

    def _log_loss(self, x, y_ind):
        """
        Должно вызываться только в fit, т.к. предполагается, что у x имеется
        фиктивный признак.

        y : np.array
            Вектор из индексов класса (не one-hot).
        """
        prob = self.softmax(x @ self.w)
        n_samples, _ = x.shape
        return - np.sum(np.log(prob[np.arange(n_samples), y_ind])) / n_samples

    def _grad_log_loss(self, x, y):
        """
        Должно вызываться только в fit, т.к. предполагается, что у x имеется
        фиктивный признак.

        y : np.array
            Матрица из one-hot векторов.
        """
        n_samples, _ = x.shape
        prob = self.softmax(x @ self.w)
        return x.T @ (prob - y) / n_samples

    def _full_grad(self, x, y):
        """
        Полный градиент с учетом регуляризации. Должно вызываться только в fit,
        т.к. предполагается, что у x имеется фиктивный признак.

        y : np.array
            Матрица из one-hot векторов.
        """
        return self.eta0 * (self._grad_log_loss(x, y) + self.get_penalty_grad())

    def _generate_index(self, n_samples):
        if self.rng is None:
            raise AttributeError()

        return np.arange(n_samples) if not self.shuffle \
            else self.rng.permutation(n_samples)

    def fit(self, x, y):
        assert len(x.shape) == 2
        assert len(y.shape) == 1

        self.rng = np.random.default_rng(seed=self.random_state)

        n_samples, n_features = x.shape

        if self.w is None:
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)

            self.w = np.zeros((n_features + 1, self.n_classes_), dtype=x.dtype)

        ind = {v: i for i, v in enumerate(self.classes_)}
        y_ind = np.vectorize(ind.get)(y)
        y_one_hot = np.zeros((n_samples, self.n_classes_), dtype=x.dtype)
        y_one_hot[np.arange(n_samples), y_ind] = 1.0

        x_dummy = np.hstack((np.ones((n_samples, 1)), x))

        train_ind = np.arange(n_samples)
        val_ind = None

        if self.early_stopping:
            val_count = int(n_samples * self.validation_fraction)
            val_ind = self.rng.choice(n_samples, val_count, replace=False)

            train_mask = ~np.isin(np.arange(n_samples), val_ind)
            train_ind = train_ind[train_mask]

            n_samples = len(train_ind)

        x_train = x_dummy[train_ind]
        y_one_hot_train = y_one_hot[train_ind]

        x_val = None
        y_ind_val = None

        if self.early_stopping:
            x_val = x_dummy[val_ind]
            y_ind_val = y_ind[val_ind]

        best_loss = self._log_loss(x_val, y_ind_val) \
            if self.early_stopping else 0.0
        no_improvement_count = 0

        for i in range(self.max_iter):
            ind = self._generate_index(n_samples)

            for b in range(0, n_samples, self.batch_size):
                batch_ind = ind[b:min(b + self.batch_size, n_samples)]
                x_batch = x_train[batch_ind, :]
                y_batch = y_one_hot_train[batch_ind]

                grad = self._full_grad(x_batch, y_batch)
                self.w -= grad

            if self.early_stopping:
                loss = self._log_loss(x_val, y_ind_val)

                if loss > best_loss - self.tol:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0

                best_loss = min(best_loss, loss)

                if no_improvement_count == self.n_iter_no_change:
                    break

    def predict_proba(self, x):
        return self.softmax(x @ self.w[1:] + self.w[[0]])

    def predict(self, x):
        prob = self.predict_proba(x)
        imax = np.argmax(prob, axis=-1)
        return self.classes_[imax]

    @staticmethod
    def softmax(z):
        """
        Calculates a softmax normalization over the last axis

        Examples:

        >>> softmax(np.array([1, 2, 3]))
        [0.09003057 0.24472847 0.66524096]

        >>> softmax(np.array([[1, 2, 3], [4, 5, 6]]))
        [[0.09003057 0.24472847 0.66524096]
         [0.03511903 0.70538451 0.25949646]]
        :param z: np.array, size: (d0, d1, ..., dn)
        :return: np.array of the same size as z
        """
        z_max = np.max(z, axis=-1, keepdims=True)
        z_exp = np.exp(z - z_max)
        return z_exp / np.sum(z_exp, axis=-1, keepdims=True)

    @property
    def coef_(self):
        return self.w[1:].copy()

    @property
    def intercept_(self):
        return self.w[0].copy()

    @coef_.setter
    def coef_(self, value):
        if self.w is None:
            v = np.asarray(value)
            self.n_classes_ = v.shape[1]
            self.w = np.vstack((np.zeros(self.n_classes_), v))
        else:
            self.w[1:] = np.asarray(value)
        pass

    @intercept_.setter
    def intercept_(self, value):
        self.w[0] = np.asarray(value)
