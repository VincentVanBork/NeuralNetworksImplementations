import numpy as np


class LinearNeuron:
    """
       N - inputs
       weights - in init tuple (minValue, maxValue) for random generate
       """

    def __init__(self, weights):
        self.outputs = 0
        self.weights = weights
        self.inputs = np.zeros((1, len(weights)))

    def calculate_output(self):
        self.outputs = np.sum(self.weights * self.inputs)


class CopyingNeuron:
    def __init__(self, pixels_vector):
        self.inputs = pixels_vector
        self.output = pixels_vector / np.sqrt(np.count_nonzero(pixels_vector > 0))

    def forward_input(self, neuron):
        neuron.inputs = self.inputs
