from .layers import Layer

class MSE(Layer):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        return ((pred - target)*(pred - target)).sum(0)


class CrossEntropy(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return input.cross_entropy(target)