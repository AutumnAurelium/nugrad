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
    input: Value
    weights: Value
    bias: Value

    output: Value

    def __init__(self, inputs: Value, activation: str, params: List[Value]):
        self.input = inputs
        self.weights = params[0]
        self.bias = params[1]
        self.outputs = []

        weighted_sum = (self.input * self.weights).sum()
        self.output = activation_from_str(weighted_sum + self.bias, activation)

    def parameters(self) -> List[Value]:
        """
        Returns a list of all of this neuron's parameters (the weights vector and bias scalar).
        """
        return [self.weights, self.bias]

class Layer:
    """
    A perceptron layer.
    """

    input: Value
    neurons: List[Neuron]
    output: Value

    def __init__(self, inputs: Value, n_neurons: int, activation: str, params: List[List[Value]]):
        self.input = inputs
        self.neurons = [Neuron(inputs, activation, params[i]) for i in range(n_neurons)]
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

    input: Value
    layers: List[Layer]
    output: Value

    def __init__(self, inputs: Value, layer_sizes: List[int], activations: List[str], params: List[List[List[Value]]]):
        self.input = inputs
        self.layers = []

        last_layer = inputs
        for i, size in enumerate(layer_sizes):
            self.layers.append(Layer(last_layer, size, activations[i], params[i]))
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
    return (((predictions - targets) ** 2).sum() / Value(predictions.data.shape[0])).sqrt()

def binary_cross_entropy_loss(predictions: Value, targets: Value) -> Value:
    """
    Calculates the binary cross-entropy loss between the predictions and targets.
    """
    epsilon = 1e-7  # Small constant to avoid log(0)
    predictions = predictions.clip(epsilon, 1 - epsilon)
    return -((targets * predictions.log()) + ((Value(1) - targets) * (Value(1) - predictions).log())).mean()

def mlp_params(n_inputs: int, neuron_counts: List[int], seed=None, pos_only=False, init_all=None) -> List[List[List[Value]]]:
    """
    Generates a list of parameters for an MLP of a given size.
    """
    if seed:
        np.random.seed(seed)
    
    params = []
    
    last_size = n_inputs
    for n_neurons in neuron_counts:
        layer = []
        
        for i in range(n_neurons):
            if init_all:
                weights = init_all * np.ones(last_size)
                bias = init_all
            else:
                weights = np.random.randn(last_size)
                bias = np.random.rand()
            
            if pos_only:
                weights = np.maximum(0, weights)
                bias = np.maximum(0, bias)
            
            layer.append([
                Value(weights),
                Value(bias)
            ])
        
        params.append(layer)
        last_size = n_neurons
    
    return params

def activation_from_str(output: Value, name: str) -> Value:
    name = name.lower()
    if name == "relu":
        return output.relu()
    elif name == "softmax":
        exp_x = (output - output.max()).exp()
        return exp_x / exp_x.sum()
    elif name == "sigmoid":
        return Value(1) / (Value(1) + (-output).exp())
    elif name == "tanh":
        return output.tanh()

def flatten_mlp_params(params: List[List[List[Value]]]) -> List[Value]:
    flat = []
    for layer in params:
        for neuron in layer:
            for param in neuron:
                flat.append(param)
    return flat