class Layer():
    def __call__(self, *args):
        self.args = args
        self.out = self.forward(*args)
        return self.out

    def forward(self):  raise Exception('Forward not implemented!')