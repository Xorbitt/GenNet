from lin import Lin
from sigmoid import Sigmoid
from mse import Mse

class Network:

    def __init__(self, weights) -> None:
        self.layers = list()
        
        for i in range(len(weights) - 1):
            self.layers.append(Lin(weights[i][0], weights[i][1]))
            self.layers.append(Sigmoid())

        self.layers.append(Lin(weights[-1][0], weights[-1][1]))
        self.loss = Mse()
        self.predictions = None
    
    def __call__(self, input, target):
        for layer in self.layers:
            input = layer(input)

        return self.loss(input, target), 
