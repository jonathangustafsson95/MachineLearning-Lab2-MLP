from Layer import Layer
from Loss import SquaredErrorLoss
from Activation import InputActivationFunction, SigmoidActivationFunction, LinearActivationFunction
import numpy as np
from Normalizer import Normalizer


class MLP:
    def __init__(self, loss_function):
        self.layers = []
        self.activations_functions = []
        self.loss_function = loss_function
        self.normalizer = Normalizer()


    def add_layer(self, size, activation_function):
        n_inputs = size
        if self.layers:
            n_inputs = self.layers[-1].n_nodes

        self.layers.append(Layer(size, n_inputs))
        self.activations_functions.append(activation_function)

    def _backprop(self, x, d_loss, learning_rate):
        
        for i in range(len(self.layers)-1, 1, -1):
            d_loss = self.activations_functions[i].backward(x).reshape(len(x), -1) * d_loss
            x = self.layers[i].backprop(d_loss, learning_rate)

    def train(self, x, y, learning_rate=0.01, n_epochs=10):

        # Normalize data
        self.normalizer.fit(x,y)
        x, y = self.normalizer.normalize(x,y)
        # print("Data after norm:")
        # print("X: {}, Y: {}" .format(x,y))

        for i in range(n_epochs): 
            sum_error = 0
            output = x
            for j, layer in enumerate(self.layers):
                output = self.activations_functions[j].forward(layer.forward(output))
            
            sum_error += (np.average(self.loss_function.forward(output, y)))

            #print("Error: {} at epoch {}".format(sum_error / len(x), i+1))
                        
            d_loss = self.loss_function.backward(output, y)

            self._backprop(output, d_loss, learning_rate)
            

    def predict(self, x, y): 
        # Normalize data
        x, y = self.normalizer.normalize(x, y)

        output = x
        for j, layer in enumerate(self.layers):
            output = self.activations_functions[j].forward(layer.forward(output))
        
        return self.normalizer.renormalize(output)

        
