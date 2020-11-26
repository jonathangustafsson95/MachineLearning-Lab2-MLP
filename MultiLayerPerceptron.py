from Layer import Layer
from Loss import SquaredErrorLoss
from Activation import InputActivationFunction, SigmoidActivationFunction, LinearActivationFunction
import numpy as np
import matplotlib.pyplot as plt
from Normalizer import Normalizer


class MLP:
    def __init__(self, loss_function):
        self.layers = []
        self.activations_functions = []
        self.loss_function = loss_function
        self.normalizer = Normalizer()
        self.errors = []


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

            error = sum_error / len(x)
            self.errors.append(error)
            #print("Error: {} at epoch {}".format(error, i+1))
                        
            d_loss = self.loss_function.backward(output, y)

            self._backprop(output, d_loss, learning_rate)
            

    def predict(self, x, y): 
        # Normalize data
        x, y = self.normalizer.normalize(x, y)

        output = x
        for j, layer in enumerate(self.layers):
            output = self.activations_functions[j].forward(layer.forward(output))
        
        return self.normalizer.renormalize(output)


    def plot(self, dataset_name, nr_epochs, y_test, y_pred):
        plt.figure(figsize=(10,6))
        plt.scatter(np.arange(1, nr_epochs+1), self.errors, label='loss')
        plt.title('Average Loss by epoch. {}'.format(dataset_name), fontsize=20)
        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.show()

        plt.scatter(y_test, y_pred)
        xy_max = max(max(y_pred), max(y_test))
        xy_min = min(min(y_pred),  min(y_test))
        plt.xlim(xy_min, xy_max)
        plt.ylim(xy_min, xy_max)
        
        plt.xlabel("Target")
        plt.ylabel("Predicted")
        plt.title("Actual Y vs Predicted Y\nData: {}".format(dataset_name))
        plt.show()

        
