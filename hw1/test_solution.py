import logging
import timeit

import numpy as np
import pytest

from hw1.solution import product_of_diagonal_elements_vectorized, are_equal_multisets_vectorized, \
    max_before_zero_vectorized, add_weighted_channels_vectorized, run_length_encoding_vectorized


def product_of_diagonal_elements_non_vectorized(matrix):
    product = 1
    for i in range(min(matrix.shape)):
        if matrix[i, i] != 0:
            product *= matrix[i, i]
    return product


@pytest.mark.parametrize(
    "n,times",
    [
        (10, 0.1),
        (100, 2),
        (1000, 5),
        (10000, 10),
    ]
)
class TestProductOfDiagonalElements:
    def test_matrix(self, n, times):
        x = np.random.randint(0, 2, size=(n, n))
        x[::5, ::5] = 0  # Ensure there are zeros on the diagonal

        # Measure time for vectorized method
        vectorized_time = timeit.timeit(lambda: product_of_diagonal_elements_vectorized(x), number=100)
        logging.info(f"Vectorized: Time = {vectorized_time:.6f} seconds")

        # Measure time for non-vectorized method
        non_vectorized_time = timeit.timeit(lambda: product_of_diagonal_elements_non_vectorized(x), number=100)
        logging.info(f"Non-Vectorized: Time = {non_vectorized_time:.6f} seconds")

        assert vectorized_time < non_vectorized_time / times, \
            "Vectorized method is not faster than non-vectorized method"

        # Ensure results are the same
        assert product_of_diagonal_elements_vectorized(x) == product_of_diagonal_elements_non_vectorized(x)


def are_equal_multisets_non_vectorized(x, y):
    # Сортируем оба вектора
    sorted_x = sorted(x)
    sorted_y = sorted(y)

    # Проверяем длины
    if len(sorted_x) != len(sorted_y):
        return False

    # Проверяем каждый элемент
    for i in range(len(sorted_x)):
        if sorted_x[i] != sorted_y[i]:
            return False

    return True


@pytest.mark.parametrize(
    "n,times",
    [
        (10, 0.1),
        (100, 1),
        (1000, 5),
        (10000, 6),
    ]
)
class TestProductEqualMultisets:
    def test_matrix(self, n, times):
        x = np.random.randint(0, 10, size=n)
        y = np.random.randint(0, 10, size=n)
        # different

        # Measure time for vectorized method
        vectorized_time = timeit.timeit(lambda: are_equal_multisets_vectorized(x, y), number=100)
        logging.info(f"Vectorized on diff: Time = {vectorized_time:.6f} seconds")

        # Measure time for non-vectorized method
        non_vectorized_time = timeit.timeit(lambda: are_equal_multisets_non_vectorized(x, y), number=100)
        logging.info(f"Non-Vectorized on diff: Time = {non_vectorized_time:.6f} seconds")

        # the same
        x = np.random.randint(0, 10, size=n)
        # different

        # Measure time for vectorized method
        vectorized_time += timeit.timeit(lambda: are_equal_multisets_vectorized(x, x), number=100)
        logging.info(f"Vectorized on diff: Time = {vectorized_time / 2:.6f} seconds")

        # Measure time for non-vectorized method
        non_vectorized_time += timeit.timeit(lambda: are_equal_multisets_non_vectorized(x, x), number=100)
        logging.info(f"Non-Vectorized on diff: Time = {non_vectorized_time / 2:.6f} seconds")

        assert vectorized_time / 2 < non_vectorized_time / times / 2, \
            f"Vectorized method is not faster than {times} times than non-vectorized method"

        # Ensure results are the same
        assert are_equal_multisets_vectorized(x, x) == are_equal_multisets_non_vectorized(x, x)


def max_before_zero_non_vectorized(x):
    curr_max = -np.inf
    for i in range(1, len(x)):
        if x[i-1] == 0 and curr_max < x[i]:
            curr_max = x[i]
    return curr_max


@pytest.mark.parametrize(
    "n,times",
    [
        (10, 0.1),
        (100, 2),
        (1000, 10),
        (10000, 50),
    ]
)
class TestProductZero:
    def test_matrix(self, n, times):
        x = np.random.randint(0, 1000, size=n)
        x[np.random.choice(np.arange(n), size=n // 4, replace=True)] = 0

        # Measure time for vectorized method
        vectorized_time = timeit.timeit(lambda: max_before_zero_vectorized(x), number=100)
        logging.info(f"Vectorized: Time = {vectorized_time:.6f} seconds")

        # Measure time for non-vectorized method
        non_vectorized_time = timeit.timeit(lambda: max_before_zero_non_vectorized(x), number=100)
        logging.info(f"Non-Vectorized: Time = {non_vectorized_time:.6f} seconds")

        assert vectorized_time < non_vectorized_time / times, \
            f"Vectorized method is not faster than {times} times than non-vectorized method"

        # Ensure results are the same
        assert max_before_zero_vectorized(x) == max_before_zero_non_vectorized(x)


def add_weighted_channels_non_vectorized(image):
    height, width, num_channels = image.shape
    gray_image = np.zeros((height, width), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            gray_image[i, j] = np.dot(image[i, j, :3], np.array([0.299, 0.587, 0.114]))

    return gray_image


@pytest.mark.parametrize(
    "n,times",
    [
        (10, 9),
        (100, 80),
        (1000, 95),
    ]
)
class TestProductGrayScale:
    def test_matrix(self, n, times):
        x = np.random.randint(0, 256, size=(n, n, 3))

        # Measure time for vectorized method
        vectorized_time = timeit.timeit(lambda: add_weighted_channels_vectorized(x), number=10)
        logging.info(f"Vectorized: Time = {vectorized_time:.6f} seconds")

        # Measure time for non-vectorized method
        non_vectorized_time = timeit.timeit(lambda: add_weighted_channels_non_vectorized(x), number=10)
        logging.info(f"Non-Vectorized: Time = {non_vectorized_time:.6f} seconds")

        assert vectorized_time < non_vectorized_time / times, \
            f"Vectorized method is not faster than {times} times than non-vectorized method"

        # Ensure results are the same
        val = np.max(np.ravel(np.abs(add_weighted_channels_vectorized(x) - add_weighted_channels_non_vectorized(x))))
        assert val < 1e-4


def run_length_encoding_non_vectorized(x):
    if not len(x):
        return [], []

    encoded = []
    counts = []
    current_element = x[0]
    count = 1

    for i in range(1, len(x)):
        if x[i] == current_element:
            count += 1
        else:
            encoded.append(current_element)
            counts.append(count)
            current_element = x[i]
            count = 1

    # Добавляем последний элемент
    encoded.append(current_element)
    counts.append(count)

    return encoded, counts


@pytest.mark.parametrize(
    "n,times",
    [
        (10, 0.1),
        (100, 1.3),
        (1000, 9),
        (10000, 35),
    ]
)
class TestProductRLE:
    def test_matrix(self, n, times):
        X = np.random.randint(0, 7, size=n)

        # Measure time for vectorized method
        vectorized_time = timeit.timeit(lambda: run_length_encoding_vectorized(X), number=100)
        logging.info(f"Vectorized: Time = {vectorized_time:.6f} seconds")

        # Measure time for non-vectorized method
        non_vectorized_time = timeit.timeit(lambda: run_length_encoding_non_vectorized(X), number=100)
        logging.info(f"Non-Vectorized: Time = {non_vectorized_time:.6f} seconds")

        assert vectorized_time < non_vectorized_time / times, \
            f"Vectorized method is not faster than {times} times than non-vectorized method"

        # Ensure results are the same
        ans_vect = run_length_encoding_vectorized(X)
        ans_non_vect = run_length_encoding_non_vectorized(X)
        assert np.array_equal(ans_vect[0], ans_non_vect[0]), "Values do not match"
        assert np.array_equal(ans_vect[1], ans_non_vect[1]), "Counts do not match"
