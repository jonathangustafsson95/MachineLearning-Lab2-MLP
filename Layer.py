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
        print('loss')
        print(loss)
        dw = np.dot(self.x.T, loss) / len(self.x) # Average
        print('self.x.T')
        print(self.x.T)
        print('dw')
        print(dw)
        print('self.w')
        print(self.w)
        self.w -= (learning_rate * dw) 
        print('self.x')
        print(self.x)
        print('self.w')
        print(self.w)
        print('self.b')
        print(self.b)
        print('np.average(loss)')
        print(np.average(loss))
        self.b -= (learning_rate * np.average(loss))

        return np.dot(loss, self.w.T)
        