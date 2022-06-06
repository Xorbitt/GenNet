from layer import Layer
from numpy import exp

class Sigmoid(Layer):
    def forward(self, input):
        g = 1 / (1 + exp(-input))
        return g
