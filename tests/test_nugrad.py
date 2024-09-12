"""
Unit testing for the project as a whole.
"""

import random
from typing import List
import numpy as np
import pytest

import nugrad.nn as nn
import nugrad.viz as viz
from nugrad.value import Value

def test_exp():
    a = Value(5)
    exp = a.exp()
    
    exp.backward()
    
    assert exp.data == np.exp(5)
    assert a.grad == np.exp(5)

def test_rmse():
    output = Value([1.5, 2.5, 1.5, 3])
    target = Value([1, 2, 2, 3.5])
    
    loss = nn.rmse_loss(output, target)
    loss.forward()
    
    print(loss)
    assert loss.data == 0.5

def test_rmse_neuron():
    weights = Value([2, 4, 6])
    inputs = Value([0.5, 0.25, 1])
    bias = Value(-2)
    
    target = Value([3])
    
    neuron = nn.Neuron(inputs, "relu", [weights, bias])
    
    loss = nn.rmse_loss(neuron.output, target)
    
    loss.forward()
    loss.backward()
    
    assert neuron.output.data == 6
    assert loss.data == 3
    assert (weights.grad == np.array([0.5, 0.25, 1])).all()
    
def test_gradients():
    """
    Test gradients
    """
    
    a = Value([1, 2, 3, 4])
    b = Value(2)
    c = Value([1, 1, 2, 2])
    
    out = ((a*b) + c).sum()
    
    print(out)
    
    out.forward()
    out.backward()
    
    print(f"a: {a}")
    print(f"b: {b}")
    print(f"c: {c}")
    
    assert (a.grad == np.array([2, 2, 2, 2])).all()
    assert b.grad == 10
    assert (c.grad == np.array([1,1,1,1])).all()
    
    # Validate A's gradients.
    for i in range(4):
        old_val = out.data.copy()
        old_grad = a.grad[i].copy()
        
        a.data[i] += 1
        out.forward()
        out.backward()
        
        print(f"Grad: {old_grad}")
        print(f"Old value: {old_val}")
        print(f"New output: {out.data}")
        print(f"Right output: {old_val + old_grad}\n")
        
        assert (out.data == old_val + old_grad).all()
        
        a.data[i] -= 1
        
        out.forward()
        out.backward()

def test_neuron():
    """
    Tests a single neuron.
    """

    inputs = Value(np.array([i+1 for i in range(4)]))
    neuron = nn.Neuron(inputs, "relu", [
        Value([(i+1)*2 for i in range(4)]),
        Value(-1)
    ])

    neuron.output.forward()
    neuron.output.backward()
    
    # (1*2 + 2*4 + 3*6 + 4*8 - 1) = 59

    print(neuron.output.data)
    print(inputs.grad)
    
    assert neuron.output.data == 59
    assert (inputs.grad == np.array([(i+1)*2 for i in range(4)])).all()
    assert neuron.bias.grad == 1
    
    # Validate all input gradients
    validate_gradients(neuron.input, neuron.output)

def test_perceptron():
    """
    Tests a single perceptron layer.
    """
    
    inputs = Value(np.array([i+1 for i in range(4)]))
    layer = nn.Layer(inputs, 3, "relu", [
        [Value([(i+1)*(j+1) for j in range(4)]), Value(i+1)]
        for i in range(3)
    ])
    
    sum = layer.output.sum()
    
    print(f"inputs: {inputs.data}")
    
    sum.forward()
    sum.backward()
    
    for i, neuron in enumerate(layer.neurons):
        test = (neuron.weights.data * neuron.input.data).sum() + neuron.bias.data
        print(f"N{i} weights: {neuron.weights.data} + {neuron.bias.data}")
        print(f"N{i} correct output: {test.sum()}")
        print(f"N{i} actual output: {neuron.output.data[0]}\n")
    
    print(layer.output.data)
    print(inputs.grad)
    
    assert (layer.output.data == np.array([31, 62, 93])).all()
    assert (inputs.grad == np.array([6, 12, 18, 24])).all()
    
    # Validate all input gradients
    validate_gradients(layer.input, sum)
    
def test_mlp():
    """
    Test the full multi-layer perceptron.
    """

    for seed in range(1):
        inputs = Value(np.array([i for i in range(4)]))
        mlp_shape = [2, 1]
        mlp_activations = ["relu", "relu"]
        mlp = nn.MLP(inputs, mlp_shape, mlp_activations, nn.mlp_params(inputs.data.shape[0], mlp_shape, seed=seed, pos_only=True))
        
        mlp.output.forward()
        mlp.output.backward()
        
        correct_layers = []
        for i, layer in enumerate(mlp.layers):
            print(f"L{i} inputs: {layer.input.data}")
            
            correct_outputs = []
            for j, neuron in enumerate(layer.neurons):
                print(f"L{i}N{j} weights: {neuron.weights.data} + {neuron.bias.data}")
                output = np.maximum(0, (layer.input.data * neuron.weights.data).sum() + neuron.bias.data)
                print(f"L{i}N{j} correct output: {output}")
                print(f"L{i}N{j} actual output: {neuron.output.data}\n")
                correct_outputs.append(output[0])
                assert (output == neuron.output.data).all()
            
            assert layer.output.data.shape == (len(layer.neurons),)
            
            correct_layer_out = np.array(correct_outputs)
            assert (layer.output.data == correct_layer_out).all()
            correct_layers.append(correct_layer_out)
        
        assert mlp.output.data.shape == (1,)
        assert (mlp.output.data == correct_layers[-1]).all()
        
        # Validate input gradients.
        # 
        validate_gradients(mlp.input, mlp.output)
        for param in mlp.parameters():
            validate_gradients(param, mlp.output)
    
    # assert False

def test_training():
    np.random.seed(1234)
    
    data = [
        (np.random.randn(4), np.random.randn(1))
        for i in range(10) if i != 0
    ]
    
    mlp_shape = [4, 2, 1]
    mlp_activations = ["relu", "relu", "softmax"]
    n_inputs = 4
    
    params = nn.mlp_params(n_inputs, mlp_shape, seed=4321)
    params_flat = nn.flatten_mlp_params(params)
    
    loss = None
    
    for data_in, data_out in data:
        mlp = nn.MLP(Value(data_in), mlp_shape, mlp_activations, params)
        cur_loss = nn.rmse_loss(mlp.output, Value(data_out))
        
        if loss:
            loss = loss + cur_loss
        else:
            loss = cur_loss
        
    loss.forward()
    loss.backward()
    
    last_loss = loss.data
    
    for i in range(50):
        print(f"it{i}, l={last_loss}")
        print(f"\t{data_in} -> {mlp.output.data} ? {data_out}")
        
        for param in params_flat:
            param.data = param.data - param.grad * 0.01
        
        loss.forward()
        loss.backward()
        
        assert loss.data <= last_loss
        last_loss = loss.data
    
    assert last_loss < 1

@pytest.mark.skip("not a test")
def validate_gradients(input: Value, output: Value):
    for i in range(input.data.shape[0]):
        old_val = output.data.copy()
        old_grad = input.grad[i].copy()
        expected_value = old_val + old_grad
        
        input.data[i] += 1
        output.forward()
        output.backward()
        
        print(f"Grad: {old_grad}")
        print(f"Old value: {old_val}")
        print(f"New output: {output.data}")
        print(f"Expected output: {expected_value}\n")
        
        epsilon = 0.0001
        
        assert ((output.data - expected_value) < epsilon).all()
        
        input.data[i] -= 1
        
        output.forward()
        output.backward()
