"""
The main Value node class lives here, as well as some helper functions..
"""
from typing import Callable, List, Optional, Union
import numpy as np

ForwardFunc = Callable[['Value'], np.ndarray]
BackwardFunc = Callable[['Value', int], np.ndarray]


class Value:
    """
    Represents a value or operation in the expression graph.
    """
    _forward: ForwardFunc
    _backward: BackwardFunc

    data: np.ndarray
    grad: Optional[np.ndarray]

    inputs: List['Value']

    op: str

    def __init__(self, value: Union[float, int, List, np.ndarray], label=""):
        if isinstance(value, float) or isinstance(value, int):
            value = np.array([value])
        elif isinstance(value, list):
            value = np.array(value)

        # Initialize data/grads
        self.data = value.copy()  # don't want to be randomly mutating the input array
        self.grad = None
        self.inputs = []

        self.op = label
        self._forward = lambda val : val.data  # no-op
        self._backward = lambda val, input : np.ones_like(val.data)

    def forward(self) -> np.ndarray:
        """
        Performs the forward pass recursively.
        """

        # First, we need to get the correct data for all of our parents.
        for parent in self.inputs:
            parent.forward()

        # Then, we call our forward-pass function.
        self.data = self._forward(self)

    def backward(self):
        """
        Performs the backward pass relative to this node.
        """
        
        # You have to do gradients on a scalar.
        if self.data.shape != (1,):
            raise ValueError("Gradients can only be calculated relative to scalars.")
            
        self.grad = np.ones_like(self.data)
        self._gradients()

    def _gradients(self):
        for i, parent in enumerate(self.inputs):
            print(self.grad)
            new_grad = self._backward(self, i)
            
    
    def label(self) -> str:
        name = self.op if self.op != "" else "var"
        return f"{name}\nv={self.data}\ng={self.grad}"
    
    def __str__(self) -> str:
        return f"{self.label()}(val={self.data}, grad={self.grad})"

    def sum(self) -> 'Value':
        """
        Sums the vector to a single scalar.
        """
        out = Value(self.data.sum())
        out.op = "sum"
        out.inputs = [self]
        out._forward = lambda val : val.inputs[0].data.sum()
        out._backward = lambda val, in_idx : np.ones_like(val.inputs[in_idx].data)
        return out

    def relu(self) -> 'Value':
        """
        Passes each value in the vector through the ReLU function.
        """
        out = Value(np.maximum(0, self.data))
        out.op = "relu"
        out.inputs = [self]
        out._forward = lambda val : np.maximum(0, val.inputs[0].data)
        out._backward = lambda val, in_idx : (val.inputs[0].data > 0).astype(np.float32)
        return out

    def __add__(self, rhs: 'Value') -> 'Value':
        """
        Adds two values.
        """
        out = Value(self.data + rhs.data)
        out.op = "+"
        out.inputs = [self, rhs]
        out._forward = lambda val : val.inputs[0].data + val.inputs[1].data
        out._backward = lambda val, in_idx : np.ones_like(val.inputs[in_idx].data)
        return out

    def __sub__(self, rhs: 'Value') -> 'Value':
        """
        Subtracts one value from another.
        """
        out = Value(self.data - rhs.data)
        out.op = "-"
        out.inputs = [self, rhs]
        out._forward = lambda val : val.inputs[0].data - val.inputs[1].data

        def sub_back(val: 'Value', in_idx: int):
            if in_idx == 0:
                return np.ones_like(val.inputs[in_idx].data)
            else:
                return -np.ones_like(val.inputs[in_idx].data)

        out._backward = sub_back
        return out

    def __mul__(self, rhs: 'Value') -> 'Value':
        """
        Multiplies two vectors.
        """
        out = Value(self.data * rhs.data)
        out.op = "*"
        out.inputs = [self, rhs]
        out._forward = lambda val : val.inputs[0].data * val.inputs[1].data

        def mul_back(val: 'Value', in_idx: int):
            if in_idx == 0:
                return val.inputs[1].data
            else:
                return val.inputs[0].data
        out._backward = mul_back

        return out

    def __neg__(self) -> 'Value':
        out = Value(-self.data)
        out.op = "negate"
        out.inputs = [self]
        out._forward = lambda val : -val.inputs[0].data
        out._backward = lambda val, in_idx : -np.ones_like(val.inputs[0].data)
        return out

def combine_scalars(values: List[Value], no_verify = False) -> Value:
    """
    Combines a list of scalars into a single vector.
    """
    if not no_verify:
        for val in values:
            assert val.data.shape == (1,)

    out = Value(np.array([val.data[0] for val in values]))
    out.op = "combine"
    out.inputs = values
    out._forward = lambda val : np.array([inp.data[0] for inp in val.inputs])

    def combine_back(val: 'Value', in_idx: int):
        out = np.zeros_like(val.inputs[in_idx].data)
        out[0] = 1
        return out
    out._backward = combine_back
    return out
