"""
Unit testing for the project as a whole.
"""

import numpy as np

import nugrad.nn as nn
import nugrad.viz as viz
from nugrad.value import Value

def test_gradients():
    """
    Test gradients()
    """
    
    a = Value(1)
    b = Value(2)
    c = Value(3)
    
    out = (a*b) + c
    
    out.forward()
    out.backward()
    
    assert a.grad == 2
    assert b.grad == 1
    assert c.grad == 1
    
    print("Gradients test passed!")

# def test_mlp():
#     """
#     Test the full multi-layer perceptron.
#     """

#     inputs = Value(np.array([0 for _ in range(4)]))
#     mlp = nn.MLP(inputs, [2, 3, 1])

#     # Fix params to a specific value
#     for param in mlp.parameters():
#         param.data = 0.5 * np.ones(param.data.shape)
    
#     # Forward pass
#     mlp.output.recalculate()
    
#     # Backward pass
#     mlp.output.gradients()

#     print(mlp.output)