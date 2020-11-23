import numpy as np

class Layer:
    def __init__(self, n_nodes, n_inputs):
        self.n_nodes = n_nodes
        self.n_inputs = n_inputs
        self.x = []
        # Initialize weights and bias with randoms.
        self.w = np.random.rand(n_inputs, n_nodes)
        self.b = np.random.rand(1, n_nodes) 

    def forward(self, x):
        self.x = x
        return np.dot(x, self.w) + self.b

    # computes dE/dW, dE/dB for a given output_error=dE/dY. 
    # Returns input_error=dE/dX.
    def backprop(self, loss, learning_rate):
        dw = np.dot(self.x.T, loss) / len(self.x) # Average
        self.w -= (learning_rate * dw) 
        self.b -= (learning_rate * np.average(loss))

        return np.dot(loss, self.w.T)
        