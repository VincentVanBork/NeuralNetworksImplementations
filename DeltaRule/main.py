import numpy as np

from neuron import Neuron, delta_rule
from utils import get_random_input, get_random_expected_output



def main():
    N_INPUTS = 15
    K_EPOCHS = 600
    STEP = 0.01
    MIN_WEIGHT = -1
    MAX_WEIGHT = 2

    MIN_TRAINING = 0
    MAX_TRAINING = 5

    n = Neuron(N_INPUTS, (MIN_WEIGHT, MAX_WEIGHT))
    print(n.weights, n.outputs)
    training_set = []
    for i in range(1):
        training_set.append((get_random_input(MAX_TRAINING, MIN_TRAINING, N_INPUTS),
                             get_random_expected_output(MAX_TRAINING, MIN_TRAINING)))

    for k in range(K_EPOCHS):
        errors = []
        print("\n ====== NEW EPOCH ===== \n =======", k, "============\n")
        for training in training_set:
            n.inputs = training[0]
            n.calculate_output()
            n.backward_propagate_delta(desired_output=training[1], training_step=STEP)
            errors.append(n.outputs - training[1])
            # print("Error:",  n.outputs - training[1])
        print("AVERAGE ERROR:", sum(errors)/len(errors))


if __name__ == '__main__':
    main()
