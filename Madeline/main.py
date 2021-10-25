from Madeline.import_images import import_all_true
from Madeline.neuron import LinearNeuron, CopyingNeuron


def main():
    all_numbers = import_all_true()
    output_layer = []
    input_layer = []

    build_network(all_numbers, input_layer, output_layer)

    for i, input_neuron in enumerate(input_layer):
        print(f"RECOGNIZING NUMBER {i + 1}")
        for output_neuron in output_layer:
            output_neuron.inputs = input_neuron.output
            output_neuron.calculate_output()
            print("OUTPUT_NEURON:", output_neuron.outputs)


def build_network(all_numbers, input_layer, output_layer):
    for number_data in all_numbers:
        input_neuron = CopyingNeuron(number_data)
        input_layer.append(input_neuron)
        output_neuron = LinearNeuron(input_neuron.output)
        output_neuron.weights = input_neuron.output
        output_layer.append(output_neuron)


if __name__ == "__main__":
    main()
