from genNet.lin import Lin
from genNet.sigmoid import Sigmoid
from genNet.mse import Mse

class Network:

    def __init__(self, weights) -> None:
        self.layers = list()
        
        for i in range(len(weights) - 1):
            self.layers.append(Lin(weights[i][0], weights[i][1]))
            self.layers.append(Sigmoid())

        self.layers.append(Lin(weights[-1][0], weights[-1][1]))
        self.loss = Mse()
    
    def __call__(self, inputToLayer, target):
        """
        Forward pass
        """
        for layer in self.layers:
            inputToLayer = layer(inputToLayer)

        return self.loss(inputToLayer, target), inputToLayer
