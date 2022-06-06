from layer import Layer

class Lin(Layer):
    def __init__(self, w, b) -> None:
        self.w, self.b = w, b
    
    def forward(self, input):
        return input@self.w + self.b
