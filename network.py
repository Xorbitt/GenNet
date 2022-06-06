from lin import Lin
from sigmoid import Sigmoid
from mse import Mse
import matplotlib.pyplot as plt

class Network:

    def __init__(self, weights) -> None:
        self.layers = list()
        
        for i in range(len(weights) - 1):
            self.layers.append(Lin(weights[i][0], weights[i][1]))
            self.layers.append(Sigmoid())

        self.layers.append(Lin(weights[-1][0], weights[-1][1]))
        self.loss = Mse()
        self.predictions = None
    
    def __call__(self, x, target):
        for layer in self.layers:
            x = layer(x)

        self.predictions = x
        return self.loss(x, target)
