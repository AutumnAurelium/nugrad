"""
A module for creating neural networks.
"""

from typing import List

import numpy as np
from nugrad.value import Value, combine_scalars


class Neuron:
    """
    A single neuron using a specific 
    """
    inputs: Value
    weights: Value
    bias: Value

    output: Value

    def __init__(self, inputs: Value):
        self.inputs = inputs
        self.weights = Value(np.random.randn(inputs.data.ndim))
        self.bias = Value(np.random.randn(1))
        self.outputs = []

        self.output = ((self.inputs * self.weights).sum() + self.bias).relu()

    def parameters(self) -> List[Value]:
        """
        Returns a list of all of this neuron's parameters (the weights vector and bias scalar). aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
        """
        return [self.weights, self.bias]

class Layer:
    """
    A perceptron layer.
    """

    neurons: List[Neuron]
    output: Value

    def __init__(self, inputs: Value, n_neurons: int):
        self.neurons = [Neuron(inputs) for _ in range(n_neurons)]
        self.output = combine_scalars([n.output for n in self.neurons])

    def parameters(self) -> List[Value]:
        """
        Returns a list of all of this layer's parameters (the weights vector and bias scalar for each neuron).
        """
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    """
    A multi-layer perceptron.
    """

    layers: List[Layer]
    output: Value

    def __init__(self, inputs: Value, layer_sizes: List[int]):
        self.layers = []

        last_layer = inputs
        for size in layer_sizes:
            self.layers.append(Layer(last_layer, size))
            last_layer = self.layers[-1].output

        self.output = self.layers[-1].output
    
    def parameters(self) -> List[Value]:
        """
        Returns a list of all of this MLP's parameters (the weights vector and bias scalar for each neuron in each layer).
        """
        return [p for layer in self.layers for p in layer.parameters()]

def rmse_loss(predictions: Value, targets: Value):
    """
    Calculates the root mean squared error loss between the predictions and targets.
    """
    return ((predictions - targets) ** 2).mean().sqrt()
