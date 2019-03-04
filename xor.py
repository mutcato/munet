import numpy as np
from munet.train import train
from munet.nn import NeuralNet
from munet.layers import Linear, Tanh


inputs = np.array([
    [0,0],
    [1,0],
    [0,1],
    [1,1]
])

targets = np.array([
    [1,0],
    [0,1],
    [0,1],
    [1,0]
])

net = NeuralNet([
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2)
])

train(net, inputs, targets)

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)