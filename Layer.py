import numpy as np

class Layer:
    def __init__(self, n_nodes, n_inputs):
        self.n_nodes = n_nodes
        self.n_inputs = n_inputs
        # Initialize weights and bias with randoms.
        self.w = [np.random.randn() for _ in range(n_nodes)]
        self.b = np.random.randn()

    def forward(self, x):
        return np.dot(x, self.w) + self.b

    def backprop(self, loss, learning_rate):
        # update weights
        pass

