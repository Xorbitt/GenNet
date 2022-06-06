from layer import Layer
import numpy as np

class Mse(Layer):

    def forward(self, input, target):
        calc = np.subtract(target, input.squeeze(-1))
        return (np.square(calc)).mean()
        
    