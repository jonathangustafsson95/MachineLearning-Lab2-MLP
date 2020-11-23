from Layer import Layer
from Loss import SquaredErrorLoss
from Activation import InputActivationFunction, SigmoidActivationFunction, LinearActivationFunction
import numpy as np

class MLP:
    def __init__(self, loss_function):
        self.layers = []
        self.activations_functions = []
        self.loss_function = loss_function

    def add_layer(self, size, activation_function):
        n_inputs = size
        if self.layers:
            n_inputs = self.layers[-1].n_nodes

        self.layers.append(Layer(size, n_inputs))
        self.activations_functions.append(activation_function)

    def _backprop(self, x, y, d_loss, learning_rate):
        
        for i in range(len(self.layers)-1, 1, -1):
            d_loss = self.activations_functions[i].backward(x).reshape(len(x), -1) * d_loss
            x = self.layers[i].backprop(d_loss, learning_rate)

    def train(self, x, y, learning_rate=0.01, n_epochs=1000):
        for i in range(n_epochs): 
            sum_error = 0
            output = x
            for j, layer in enumerate(self.layers):
                output = self.activations_functions[j].forward(layer.forward(output))

            #print('output after actv func')
            #print(output)

            # print(np.average(self.loss_function.forward(output, y)))

            #print("Y: {}".format(y))
            #print("Y shape: {}".format(y.shape))


            error = y - output
            #print("Error: {}".format(error))

            sum_error += (np.average(self.loss_function.forward(output, y)))
            #print("Sum Error: {}".format(sum_error))


            print("Error: {} at epoch {}".format(sum_error / len(x), i+1))
            
            
            d_loss = self.loss_function.backward(output, y)

            self._backprop(output, y, d_loss, learning_rate)

    def predict(self, x, y): 
        pass
        

input = np.array([[0,0], [0,1], [1,0],[1,1]])
target = np.array([[0,1,1,1]])
# Reshape .reshape(4,1)
