import numpy as np

class Layer:
    def __init__(self, n_nodes, n_inputs):
        self.n_nodes = n_nodes
        self.n_inputs = n_inputs
        self.input = []
        self.output = []
        # Initialize weights and bias with randoms.
        self.w = np.random.rand(n_inputs, n_nodes)
        self.b = np.random.rand(1, n_nodes)

    def get_output(self):
        return self.output 

    def forward(self, x):
        self.input = x
        self.output = np.dot(x, self.w) + self.b 
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. 
    # Returns input_error=dE/dX.
    def backprop(self, loss, learning_rate):
        input_error = np.dot(loss, self.w.T)
        dw = np.dot(self.input.T, loss) / len(self.input) # Average

        self.w -= (learning_rate * dw) 
        self.b -= (learning_rate * np.average(loss))
        return input_error
        