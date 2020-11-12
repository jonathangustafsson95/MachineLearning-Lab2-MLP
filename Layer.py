class Layer: 
    def __init__(self, n_nodes, n_inputs):
        pass

    def forward(self, x):
        pass

    def backprop(self, loss, learning_rate):
        pass


class LossFunction:
    @staticmethod
    def forward(z):
        pass

    @staticmethod
    def backward(z):
        pass


class ActivationFunction:
    @staticmethod
    def forward(z):
        pass

    @staticmethod
    def backward(z):
        pass