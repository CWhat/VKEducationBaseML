import numpy as np


class LinearRegression:
    def __init__(
            self,
            *,
            penalty="l2",
            alpha=0.0001,
            max_iter=1000,
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

    def get_penalty_grad(self):
        if self.w is None:
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

    def _train_test_split(self, x, y):
        if self.rng is None:
            raise AttributeError()

        n_samples, _ = x.shape
        val_count = int(n_samples * self.validation_fraction)
        val_ind = self.rng.choice(n_samples, val_count, replace=False)
        train_mask = ~np.isin(np.arange(n_samples), val_ind)

        x_val = x[val_ind, :]
        y_val = y[val_ind]

        x_train = x[train_mask, :]
        y_train = y[train_mask]

        return x_train, x_val, y_train, y_val

    def _loss_grad(self, x_batch, y_batch):
        n_samples, _ = x_batch.shape
        # d[(Xw-y).T (Xw-y)] = d[((Xw).T - y.T) (Xw-y)] =
        # = d[(Xw).T Xw - y.T Xw - (Xw).T y + y.T y] =
        # = d[w.T X.T X w - 2 y.T Xw + y.T y] =
        # = d[w.T X.T X w - 2 y.T Xw] = 2 (X.T X)w - 2 X.T y =
        # = 2 X.T (Xw - y)
        # Транспонируем:
        # (...)^T = 2 (Xw - y).T X
        # Т.к. (Xw - y) - вектор, то его не нужно транспонировать (при
        # произведении numpy сам это обработает)
        # Т.к. градиент - вектор, результат обратно транспонировать не нужно
        return ((x_batch @ self.w - y_batch) @ x_batch) * 2.0 / n_samples

    def _full_grad(self, x_batch, y_batch):
        return self.eta0 * \
            (self._loss_grad(x_batch, y_batch) + self.get_penalty_grad())

    def _generate_index(self, n_samples):
        if self.rng is None:
            raise AttributeError()

        return np.arange(n_samples) if not self.shuffle \
            else self.rng.permutation(n_samples)

    @classmethod
    def _add_dummy_feature(cls, x):
        n_samples, _ = x.shape
        return np.hstack((np.ones((n_samples, 1)), x))

    def fit(self, x, y):
        self.rng = np.random.default_rng(seed=self.random_state)

        n_samples, n_features = x.shape

        if self.w is None:
            self.w = np.zeros(n_features + 1, dtype=x.dtype)

        x_val = None
        y_val = None
        x_train = self._add_dummy_feature(x)
        y_train = y
        if self.early_stopping:
            x_train, x_val, y_train, y_val = self._train_test_split(
                x_train, y_train)

        n_samples, n_features = x_train.shape

        best_loss = self._loss(x_val, y_val) if self.early_stopping else 0.0
        no_improvement_count = 0

        for i in range(self.max_iter):
            ind = self._generate_index(n_samples)

            for b in range(0, n_samples, self.batch_size):
                batch_ind = ind[b:min(b + self.batch_size, n_samples)]
                x_batch = x_train[batch_ind, :]
                y_batch = y_train[batch_ind]

                grad = self._full_grad(x_batch, y_batch)
                self.w -= grad

            if self.early_stopping:
                loss = self._loss(x_val, y_val)

                if loss > best_loss - self.tol:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0

                best_loss = min(best_loss, loss)

                if no_improvement_count == self.n_iter_no_change:
                    break

    def _loss(self, x, y):
        return np.mean((x @ self.w - y)**2)

    def predict(self, x):
        return x @ self.w[1:] + self.w[0]

    @property
    def coef_(self):
        return self.w[1:]

    @property
    def intercept_(self):
        return self.w[0]

    @coef_.setter
    def coef_(self, value):
        if self.w is None:
            self.w = np.hstack((0.0, np.asarray(value)))
        elif self.w.shape[0] == 1:
            self.w = np.hstack((self.w, np.asarray(value)))
        else:
            self.w[1:] = np.asarray(value)

    @intercept_.setter
    def intercept_(self, value):
        if self.w is None:
            np.array
            self.w = np.reshape(np.asarray(value), -1)
        else:
            self.w[0] = np.asarray(value)
