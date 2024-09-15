"""
The main Value node class lives here, as well as some helper functions..
"""
from typing import Callable, List, Optional, Union
import numpy as np

ForwardFunc = Callable[['Value'], np.ndarray]
BackwardFunc = Callable


class Value:
    """
    Represents a value or operation in the expression graph.
    """
    _forward: ForwardFunc
    _backward: Callable

    data: np.ndarray
    grad: Optional[np.ndarray]

    inputs: List['Value']

    op: str

    _cached_topo: Optional[List['Value']]
    

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
        
        def backward():
            pass
            
        self._backward = backward

        self._cached_topo = None

    def forward(self) -> np.ndarray:
        """
        Performs the forward pass recursively.
        """

        # First, we need to get the correct data for all of our parents.
        for parent in self.inputs:
            parent.forward()

        # Then, we call our forward-pass function.
        self.data = self._forward(self)
        if self.data.shape == ():
            self.data = np.array([self.data])

    def backward(self):
        """
        Performs the backward pass relative to this node.
        """
        if self.data.shape != (1,):
            raise ValueError("Gradients can only be calculated relative to scalars.")
        
        # Initialize gradients
        for node in self.topological_sort():
            node.grad = np.zeros_like(node.data)

        self.grad = np.ones_like(self.data)
        
        # Perform backward pass in topological order
        for node in self.topological_sort():
            node._backward()
    
    def topological_sort(self, force_recalc=False) -> List['Value']:
        if not force_recalc and self._cached_topo is not None:
            return self._cached_topo

        visited = set()
        topo_order = []

        def dfs(node):
            if node not in visited:
                visited.add(node)
                for input_node in node.inputs:
                    dfs(input_node)
                topo_order.append(node)

        dfs(self)
        self._cached_topo = topo_order[::-1]  # Reverse the order

        return self._cached_topo
    
    def label(self) -> str:
        name = self.op if self.op != "" else "var"
        return f"{name}\nv={self.data}\ng={self.grad}"
    
    def __str__(self) -> str:
        name = self.op if self.op != "" else "var"
        return f"{name}(val={self.data}, grad={self.grad})"
    
    def __repr__(self) -> str:
        return str(self)

    def sum(self) -> 'Value':
        """
        Sums the vector to a single scalar.
        """
        out = Value(np.array([self.data.sum()]))
        out.op = "sum"
        out.inputs = [self]
        out._forward = lambda val : np.array([val.inputs[0].data.sum()])
        
        def backward():
            self.grad = self.grad + out.grad
        
        out._backward = backward
        return out

    def relu(self) -> 'Value':
        """
        Passes each value in the vector through the ReLU function.
        """
        out = Value(np.maximum(0, self.data))
        out.op = "relu"
        out.inputs = [self]
        out._forward = lambda val : np.maximum(0, val.inputs[0].data)
        
        def backward():
            self.grad = self.grad + out.grad * (self.data > 0)
        
        out._backward = backward
        return out

    def sqrt(self) -> 'Value':
        return self ** 0.5
    
    def exp(self) -> 'Value':
        """
        Passes each value in the vector through the exponential function.
        """
        out = Value(np.exp(self.data))
        out.op = "exp"
        out.inputs = [self]
        out._forward = lambda val : np.exp(val.inputs[0].data)
        
        def backward():
            self.grad = self.grad + out.grad * out.data
        
        out._backward = backward
        return out
    
    def tanh(self) -> 'Value':
        """
        Applies the hyperbolic tangent function to each value in the vector.
        """
        out = Value(np.tanh(self.data))
        out.op = "tanh"
        out.inputs = [self]
        out._forward = lambda val: np.tanh(val.inputs[0].data)
        
        def backward():
            # The derivative of tanh(x) is 1 - tanh^2(x)
            self.grad = self.grad + out.grad * (1 - out.data ** 2)
        
        out._backward = backward
        return out
    
    def clip(self, min_val: float, max_val: float) -> 'Value':
        """
        Clips the values in the vector to be between min_val and max_val.
        """
        out = Value(np.clip(self.data, min_val, max_val))
        out.op = f"clip({min_val}, {max_val})"
        out.inputs = [self]
        out._forward = lambda val: np.clip(val.inputs[0].data, min_val, max_val)
        
        def backward():
            grad_mask = (self.data >= min_val) & (self.data <= max_val)
            self.grad = self.grad + out.grad * grad_mask
        
        out._backward = backward
        return out

    def log(self) -> 'Value':
        """
        Applies natural logarithm to each value in the vector.
        """
        out = Value(np.log(self.data))
        out.op = "log"
        out.inputs = [self]
        out._forward = lambda val: np.log(val.inputs[0].data)
        
        def backward():
            self.grad = self.grad + out.grad / self.data
        
        out._backward = backward
        return out
    
    def mean(self) -> 'Value':
        """
        Computes the mean of all elements in the vector.
        """
        out = Value(np.array([np.mean(self.data)]))
        out.op = "mean"
        out.inputs = [self]
        out._forward = lambda val: np.array([np.mean(val.inputs[0].data)])
        
        def backward():
            self.grad = self.grad + out.grad / self.data.size
        
        out._backward = backward
        return out
    
    def max(self) -> 'Value':
        """
        Returns the maximum value in the vector.
        """
        out = Value(np.max(self.data))
        out.op = "max"
        out.inputs = [self]
        out._forward = lambda val : np.max(val.inputs[0].data)
        
        def backward():
            self.grad = self.grad + out.grad * (self.data == out.data)
        
        out._backward = backward
        return out

    def expand(self, size: int) -> 'Value':
        """
        'Expands' a scalar value into a vector of length `size`.
        """
        return self * Value(np.ones((size,)))

    def __add__(self, rhs: 'Value') -> 'Value':
        """
        Adds two values.
        """
        assert self.data.shape == rhs.data.shape
            
        out = Value(self.data + rhs.data)
        out.op = "+"
        out.inputs = [self, rhs]
        out._forward = lambda val : val.inputs[0].data + val.inputs[1].data
        
        def backward():
            self.grad = self.grad + out.grad
            rhs.grad = rhs.grad + out.grad
        out._backward = backward
        
        return out

    def __sub__(self, rhs: 'Value') -> 'Value':
        """
        Subtracts one value from another.
        """
        assert self.data.shape == rhs.data.shape
        
        out = Value(self.data - rhs.data)
        out.op = "-"
        out.inputs = [self, rhs]
        out._forward = lambda val : val.inputs[0].data - val.inputs[1].data

        def backward():
            self.grad = self.grad + out.grad
            rhs.grad = self.grad - out.grad
        
        out._backward = backward
        return out

    def __mul__(self, rhs: 'Value') -> 'Value':
        """
        Multiplies two vectors.
        """
        out = Value(self.data * rhs.data)
        out.op = "*"
        out.inputs = [self, rhs]
        out._forward = lambda val : val.inputs[0].data * val.inputs[1].data

        def backward():
            if self.data.shape == (1,):
                self.grad = self.grad + (out.grad * rhs.data).sum()
            else:
                self.grad = self.grad + out.grad * rhs.data
            
            if rhs.data.shape == (1,):
                rhs.grad = rhs.grad + (out.grad * self.data).sum()
            else:
                rhs.grad = rhs.grad + out.grad * self.data

        out._backward = backward

        return out
    
    def __div__(self, denom: 'Value') -> 'Value':
        # I don't think there's any good definition for "vector divided by vector"
        assert denom.data.shape == (1,)
        
        return self * (denom ** -1)
    
    def __truediv__(self, denom: 'Value') -> 'Value':
        return self.__div__(denom)
        
    def __pow__(self, const: float) -> 'Value':
        assert not isinstance(const, Value)
            
        out = Value(self.data.astype("float32") ** const)
        out.op = f"^{const}"
        out.inputs = [self]
        out._forward = lambda val: val.inputs[0].data.astype("float32") ** const
        
        def backward():
            self.grad = self.grad + out.grad * (const * self.data.astype("float32") ** (const-1))
        
        out._backward = backward
        
        return out

    def __neg__(self) -> 'Value':
        out = Value(-self.data)
        out.op = "negate"
        out.inputs = [self]
        out._forward = lambda val : -val.inputs[0].data
        
        def backward():
            self.grad = self.grad + -out.grad
            
        out._backward = backward
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

    def backward():
        for i, val in enumerate(values):
            val.grad = val.grad + out.grad[i]

    out._backward = backward
    return out
