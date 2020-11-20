import numpy as np

class Layer:
    def __init__(self, n_nodes, n_inputs):
        self.n_nodes = n_nodes
        self.n_inputs = n_inputs
        self.x = []
        self.input = []
        self.output = []
        # Initialize weights and bias with randoms.
        self.w = np.random.rand(n_inputs, n_nodes)
        self.b = np.random.rand(1, n_nodes) 

    def forward(self, x):
        self.input = x
        self.x = np.average(x, axis=0).reshape(1, -1)
        self.output = np.dot(x, self.w) + self.b
        print('forward output')
        print(self.output)
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. 
    # Returns input_error=dE/dX.
    def backprop(self, loss, learning_rate):
        loss = loss.reshape(1,-1)
        print('self.x')
        print(self.x)
        print('self.w')
        print(self.w.T)
        print('self.b')
        print(self.b)
        print('loss')
        print(loss)
        input_error = np.dot(loss, self.w.T)
        print('input_error')
        print(input_error)
        weights_error = np.dot(self.x.T, loss)
        print('weights_error')
        print(weights_error)
        # dBias = output_error
        print('learning_rate * weights_error')
        print((learning_rate * weights_error))
        # update parameters
        print(self.w)
        self.w = np.subtract(self.w, learning_rate * weights_error)
        print('self.w')
        print(self.w)
        self.b -= learning_rate * loss
        print('self.b')
        print(self.b)
        return input_error
        