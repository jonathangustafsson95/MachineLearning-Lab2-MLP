import numpy as np

class Layer:
    def __init__(self, n_nodes, n_inputs):
        self.n_nodes = n_nodes
        self.n_inputs = n_inputs
        # Initialize weights and bias with randoms.
        self.w = [np.random.randn() for _ in range(n_inputs)]
        self.b = np.random.randn() # random from 1 to n_nodes

    def forward(self, x):
        return np.dot(x, self.w) + self.b

    def backprop(self, loss, learning_rate):
        error = np.dot(loss, self.w)
        w_error = np.dot()
        pass

L = [8,6,7]
print(L[-1])