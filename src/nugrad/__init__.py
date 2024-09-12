import random
from typing import List
import numpy as np
from nugrad import nn
from nugrad.value import Value

data = [
    (np.array([i]), np.array([1 if i > 0 else 0]))
    for i in range(-50, 51) if i != 0
]

def flatten(params: List[List[List[Value]]]) -> List[Value]:
    flat = []
    for layer in params:
        for neuron in layer:
            for param in neuron:
                flat.append(param)
    return flat

def main():
    random.shuffle(data)
    
    mlp_shape = [1, 16, 4, 1]
    mlp_activations = ["tanh", "tanh", "tanh", "sigmoid"]
    params = nn.mlp_params(1, mlp_shape)
    params_flat = flatten(params)
    
    learning_rate = 0.01
    
    batch_size = 5
    
    for it in range(50):
        loss: Value = None
        mlps: List[nn.MLP] = []
        losses: List[Value] = []
        targets: List[np.ndarray] = []
        
        offset = (it*batch_size % len(data))
        
        for data_in, data_out in data[offset:offset+batch_size]:
            mlp = nn.MLP(Value(data_in), mlp_shape, mlp_activations, params)
            cur_loss = nn.binary_cross_entropy_loss(mlp.output, Value(data_out))
            
            mlps.append(mlp)
            losses.append(cur_loss)
            targets.append(data_out)
            
            if loss is None:
                loss = cur_loss
            else:
                loss = loss + cur_loss
        
        loss.forward()
        loss.backward()
        
        print(f"it{it} - loss={loss.data[0]}")
        for i, mlp in enumerate(mlps):
            print(f"{mlp.output.data[0]:.2f} vs {targets[i][0]:.2f} l={losses[i].data[0]:.2f}")
        
        for param in params_flat:
            param.data = param.data - param.grad*learning_rate
            # print(f"\t{param.data[0]:.2f}, g={param.grad[0]:.2f}")
        
        # input("> ")

if __name__ == "__main__":
    main()