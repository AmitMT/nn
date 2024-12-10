from itertools import zip_longest
import math
from random import random
from typing import Iterator
import matplotlib.pyplot as plt

ACTIVATION_INTENSITY = 1  # How steep is the activation curve's step
STRING_COLUMN_SPACING = 2

GRADIENT_DESCENT_FACTOR = 0.1


class NN:
    input_size: int

    weights: list[list[list[float]]]  # layer * nodes * next_nodes
    biases: list[list[float]]  # layer * nodes

    # The input layer doesnt contain weights and biases:
    # weight between layer 1 node 3 to layer 2 node 4 is weights[1][3][4]
    # bias on layer 2 node 4 is biases[1][4]

    @staticmethod
    def get_starting_value():
        return round(random() * 2 - 1, 2)

    def __init__(self, layer_sizes: Iterator[int]):
        self.input_size = layer_sizes[0]
        self.weights = [
            [
                [NN.get_starting_value() for _ in range(prev_layer_size)]
                for _ in range(curr_layer_size)
            ]
            for prev_layer_size, curr_layer_size in zip(layer_sizes, layer_sizes[1:])
        ]
        self.biases = [
            [NN.get_starting_value() for _ in range(layer_size)]
            for layer_size in layer_sizes[1:]
        ]

    def backpropagate(
        self,
        prev_layer_index: int,
        prev_layer_values: list[float,],
        layer_values: list[float],
        layer_values_partial_derivative_to_cost: list[float],
    ):
        """
        Terminology:
        layer -> current layer
        prev_layer -> the layer we want to backpropegate to
        """
        # print(
        #     prev_layer_index,
        #     prev_layer_values,
        #     layer_values,
        #     layer_values_partial_derivative_to_cost,
        # )
        prev_layer_values_partial_derivative_to_cost = []
        for j in range(len(prev_layer_values)):
            prev_layer_value_partial_derivative_to_cost = 0
            for i in range(len(layer_values)):
                prev_layer_value_partial_derivative_to_cost += (
                    layer_values_partial_derivative_to_cost[i]
                    * NN.activation_derivative(layer_values[i])
                    * self.weights[prev_layer_index][i][j]
                )
            prev_layer_values_partial_derivative_to_cost.append(
                prev_layer_value_partial_derivative_to_cost
            )
        layer_nodes_weights_partial_derivative_to_cost = []
        layer_node_biases_partial_derivative_to_cost = []
        for i in range(len(layer_values)):
            layer_node_bias_partial_derivative_to_cost = 0
            for j in range(len(prev_layer_values)):
                layer_node_bias_partial_derivative_to_cost += (
                    layer_values_partial_derivative_to_cost[i]
                    * NN.activation_derivative(layer_values[i])
                    * 1
                )
            layer_node_biases_partial_derivative_to_cost.append(
                layer_node_bias_partial_derivative_to_cost
            )
        for i in range(len(layer_values)):
            layer_node_weights_partial_derivative_to_cost = []
            for j in range(len(prev_layer_values)):
                layer_node_weights_partial_derivative_to_cost.append(
                    layer_values_partial_derivative_to_cost[i]
                    * NN.activation_derivative(layer_values[i])
                    * prev_layer_values[j]
                )
            layer_nodes_weights_partial_derivative_to_cost.append(
                layer_node_weights_partial_derivative_to_cost
            )
        return (
            prev_layer_values_partial_derivative_to_cost,
            layer_node_biases_partial_derivative_to_cost,
            layer_nodes_weights_partial_derivative_to_cost,
        )

    def train(self, training_data: Iterator[tuple[tuple[float, ...], float]]):
        # plot_data = []
        # plot_data2 = []
        for _training_index, (input_arr, expected_output) in enumerate(training_data):
            # print(f"Training data: {input_arr}. expecting result of {expected_output}")
            node_values = self.get_node_values(input_arr)
            # print(NN.stringify_2d_arr(node_values))

            output_layer_values_partial_derivative_to_cost = [
                NN.loss_derivative(node_values[-1][i], expected_output[i])
                for i in range(len(node_values[-1]))
            ]

            layer_values_partial_derivative_to_cost = (
                output_layer_values_partial_derivative_to_cost
            )
            # Iterate though all layers backwards without the input layer
            for layer_index in range(len(node_values) - 1, 0, -1):
                (
                    prev_layer_values_partial_derivative_to_cost,
                    layer_node_biases_partial_derivative_to_cost,
                    layer_nodes_weights_partial_derivative_to_cost,
                ) = self.backpropegate(
                    layer_index - 1,
                    node_values[layer_index - 1],
                    node_values[layer_index],
                    layer_values_partial_derivative_to_cost,
                )
                for (
                    node_index,
                    layer_node_weights_partial_derivative_to_cost,
                ) in enumerate(layer_nodes_weights_partial_derivative_to_cost):
                    for (
                        weight_index,
                        layer_node_weight_partial_derivative_to_cost,
                    ) in enumerate(layer_node_weights_partial_derivative_to_cost):
                        self.weights[layer_index - 1][node_index][weight_index] += (
                            layer_node_weight_partial_derivative_to_cost
                            * -GRADIENT_DESCENT_FACTOR
                        )
                for (
                    node_index,
                    layer_node_bias_partial_derivative_to_cost,
                ) in enumerate(layer_node_biases_partial_derivative_to_cost):
                    self.biases[layer_index - 1][node_index] += (
                        layer_node_bias_partial_derivative_to_cost
                        * -GRADIENT_DESCENT_FACTOR
                    )

                layer_values_partial_derivative_to_cost = (
                    prev_layer_values_partial_derivative_to_cost
                )

            # print(f"{weight_partial_derivative=}")
            # if i == 0:
            #     plot_data.append(
            #         (
            #             j,
            #             self.weights[-1][0][i],
            #             NN.loss_function(node_values[-1][0], expected_output),
            #         )
            #     )
            #     plot_data2.append(
            #         (
            #             j,
            #             -weight_partial_derivative / 1,
            #             NN.loss_function(node_values[-1][0], expected_output),
            #         )
            #     )

        print(f"{self}\n")
        print(
            f"Inputing data: {training_data[0][0]}. expecting result of {training_data[0][1]}"
        )
        node_values = self.get_node_values(input_arr)
        print(f"{NN.stringify_2d_arr(node_values)}\n")
        # fig = plt.figure()
        # ax = plt.axes(projection="3d")
        # plt.plot(*zip(*plot_data), marker="o")
        # plt.plot(*zip(*plot_data2), marker="o")
        # plt.plot(*zip(*plot_data2), marker="o")

        # ax.set_xlabel("iteration")
        # ax.set_ylabel("weight")
        # ax.set_zlabel("loss")

        # # giving a title to my graph
        # plt.title("My first graph!")
        # plt.grid()
        # # function to show the plot
        # plt.show()

    def get_node_values(self, input_arr: tuple[float, ...]):
        node_values = [[0.0] * len(layer) for layer in [list(input_arr)] + self.biases]
        node_values[0] = [input_node for input_node in input_arr]
        for prev_layer_index, (prev_layer, curr_layer) in enumerate(
            zip(node_values, node_values[1:])
        ):
            curr_layer_index = prev_layer_index + 1
            for curr_node_index in range(len(curr_layer)):
                curr_layer[curr_node_index] = NN.activation_function(
                    sum(
                        (
                            prev_node
                            * self.weights[curr_layer_index - 1][curr_node_index][
                                prev_node_index
                            ]
                            for prev_node_index, prev_node in enumerate(prev_layer)
                        )
                    )
                    + self.biases[curr_layer_index - 1][curr_node_index]
                )

        return node_values

    @staticmethod
    def loss_function(x: int, expected_x: int):
        return (x - expected_x) ** 2

    @staticmethod
    def loss_derivative(x: int, expected_x: int):
        return 2 * (x - expected_x)

    @staticmethod
    def activation_function(x: int, intensity=ACTIVATION_INTENSITY):
        return 1 / (1 + math.e ** (-intensity * x))

    @staticmethod
    def activation_derivative(x: int, intensity=ACTIVATION_INTENSITY):
        activation_value = NN.activation_function(x, intensity)
        return intensity * activation_value * (1 - activation_value)

    @staticmethod
    def transpose_2d_arr(arr: list[list]):
        return list(zip_longest(*arr))

    @staticmethod
    def stringify_2d_arr(arr: list[list]):
        # Transpose nodes 2D list and add empty strings in empty places
        nodes_strings_transposed = [
            ["" if node is None else str(node) for node in layer]
            for layer in NN.transpose_2d_arr(arr)
        ]

        # Find max lengthed string in each coumn
        layer_max_str_lengths = [0] * len(arr)
        for row in nodes_strings_transposed:
            for i, node_string in enumerate(row):
                layer_max_str_lengths[i] = max(
                    layer_max_str_lengths[i], len(node_string)
                )

        lines = []
        for row in nodes_strings_transposed:
            lines.append(
                "".join(
                    list(
                        [
                            node_string.ljust(
                                layer_max_str_lengths[i] + STRING_COLUMN_SPACING
                            )
                            for i, node_string in enumerate(row)
                        ]
                    )
                )
            )
        return "\n".join(lines)

    def __str__(self):
        weights_and_biases = [
            zip(self.biases[i], self.weights[i]) for i in range(len(self.biases))
        ]
        return NN.stringify_2d_arr(weights_and_biases)
