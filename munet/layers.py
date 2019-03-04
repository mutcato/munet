"""
input->Linear->Tanh->Linear->output
"""
from typing import Dict, Callable, Sequence, Iterator, Tuple
import numpy as np
from munet.tensor import Tensor

class Layer(object):
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Produce output according to input
        """
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backpropogate this gradient through the layer
        """
        raise NotImplementedError

    
class Linear(Layer):
    """
    computes output layer = inputs @ w + b
    """
    def __init__(self, input_size: int, output_size: int)->None:
        """
        inputs will be (batch_size, input_size)
        output will be (batch_size, output_size)
        """
        super().__init__()
        self.params["w"] = np.random.randn(input_size,output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        returns input @ w + b
        """
        # inputs saved here to use in backpropogation later
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = L(out) and out = x @ w + b
        then dy/dw = x.T @ L'(out)
        and dy/db = L'(out)
        and dy/dx = L'(out) @ w #we leave open ended due to continue backward propogation
        -for MSE L'(out) = 2(pred-actual)-
        """
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T

F = Callable[[Tensor], Tensor]

class Activation(Layer):
    """
    1 input 1 output
    """
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs saved here to use in backpropogation later
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        """
        chain rule:
        if y = f(x) and x = g(z)
        then dy/dz = dy/dx * dx/dz
        """
        return self.f_prime(self.inputs) * grad


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y **2

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)








    