import numpy as np


def product_of_diagonal_elements_vectorized(matrix: np.array):
    d = np.diag(matrix)
    return d[d != 0].prod()


def are_equal_multisets_vectorized(x: np.array, y: np.array):
    if x.shape != y.shape:
        return False

    sorted_x = np.sort(x)
    sorted_y = np.sort(y)

    return bool(np.all(sorted_x == sorted_y))


def max_before_zero_vectorized(x: np.array):
    is_zero = x == 0
    after_zero = x[1:][is_zero[:-1]]
    return after_zero.max() if len(after_zero) else -np.inf


def add_weighted_channels_vectorized(image: np.array):
    w = np.array([0.299, 0.587, 0.114])
    return image @ w


def run_length_encoding_vectorized(x: np.array):
    n = len(x)
    if not n:
        return np.array([], dtype=x.dtype), np.array([], dtype=np.int64)

    # True, если число отличается от предыдущего. Считаем True для первого
    # элемента
    is_changed = np.hstack((True, x[1:] != x[:-1]))
    nums = x[is_changed]

    start_pos, = is_changed.nonzero()
    # Кол-во того, сколько раз число необходимо повторить, равно разности
    # позиций следующего числа и текущего. Отдельно учитываем последнее число
    counts = np.hstack((start_pos[1:] - start_pos[:-1], n - start_pos[-1]))

    return nums, counts
