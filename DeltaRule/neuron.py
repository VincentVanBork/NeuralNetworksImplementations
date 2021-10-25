import numpy as np


def delta_rule(weight: float, step: float, desired_output: int, neuron_output: int, neuron_input: float):
    weight = weight * step * (desired_output - neuron_output) * neuron_input
    return weight


class Neuron:
    """
    N - inputs
    weights - in init tuple (minValue, maxValue) for random generate
    """

    def __init__(self, num_inputs, weights, update_function=delta_rule):
        self.outputs = 0
        self.weights = np.random.rand(1, num_inputs) * (weights[1] - weights[0] + 1) + weights[0]
        self.update_function = update_function
        self.inputs = np.zeros((1, num_inputs))

    def calculate_output(self):
        self.outputs = np.sum(self.weights * self.inputs)

    def backward_propagate_delta(self, desired_output, training_step):
        self.weights = self.weights + training_step * (desired_output - self.outputs) * self.inputs
