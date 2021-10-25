from collections.abc import Collection
import numpy as np


def get_random_input(max_value, min_value, n_inputs):
    return np.random.rand(1, n_inputs) * (max_value - min_value + 1) + min_value


def get_random_expected_output(max_value, min_value):
    return np.random.rand() * (max_value - min_value + 1) + min_value
