from Madeline.import_images import import_all_true, import_all_ones, import_all_twos, import_all_threes
from Madeline.neuron import LinearNeuron, CopyingNeuron
import sys


def main():
    if int(sys.argv[1]) == 1:
        all_numbers = import_all_true()
    elif int(sys.argv[1]) == 2:
        all_numbers = import_all_ones()
    elif int(sys.argv[1]) == 3:
        all_numbers = import_all_twos()
    elif int(sys.argv[1]) == 4:
        all_numbers = import_all_threes()
    elif int(sys.argv[1]) == 5:
        all_numbers = import_all_true()
        all_testing = []
        all_testing += import_all_ones()
        all_testing += import_all_twos()
        all_testing += import_all_threes()
        output_layer = []
        init_network(all_numbers, output_layer)
        work_network(all_testing, output_layer)
    else:
        print("unkonwn experiment variant")
        sys.exit()

    if int(sys.argv[1]) != 5:
        output_layer = []
        input_layer = []
        build_network(all_numbers, input_layer, output_layer)

        for i, input_neuron in enumerate(input_layer):
            print(f"RECOGNIZING NUMBER {i + 1}")
            for output_neuron in output_layer:
                output_neuron.inputs = input_neuron.output
                output_neuron.calculate_output()
                print("OUTPUT_NEURON:", f"{output_neuron.outputs:.3f}")



def build_network(all_numbers, input_layer, output_layer):
    for number_data in all_numbers:
        input_neuron = CopyingNeuron(number_data)
        input_layer.append(input_neuron)
        output_neuron = LinearNeuron(input_neuron.output)
        output_neuron.weights = input_neuron.output
        output_layer.append(output_neuron)


def init_network(training_numbers, output_layer):
    for number_data in training_numbers:
        input_neuron = CopyingNeuron(number_data)
        output_neuron = LinearNeuron(input_neuron.output)
        output_neuron.weights = input_neuron.output
        output_layer.append(output_neuron)


def work_network(input_numbers, output_layer):
    for num_neuron, output_neuron in enumerate(output_layer):
        print(f"======={num_neuron}========")
        for index, input_part in enumerate(input_numbers):
            c = CopyingNeuron(input_part)
            output_neuron.inputs = c.output
            output_neuron.calculate_output()
            print("OUTPUT_NEURON:", f"{output_neuron.outputs:.3f}", f"{index}")


if __name__ == "__main__":
    main()
