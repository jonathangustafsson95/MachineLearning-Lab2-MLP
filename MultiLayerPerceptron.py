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

    def _backprop(self, d_loss, learning_rate):
        for i in range(len(self.layers)-1, 1, -1):
            input_data = self.layers[i].get_output()
            loss = self.activations_functions[i].backward(input_data).reshape(len(input_data), -1) * d_loss
            d_loss = self.layers[i].backprop(loss, learning_rate)

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
                layer.set_output(output)
            
            sum_error += (np.average(self.loss_function.forward(output, y)))

            error = sum_error / len(x)
            self.errors.append(error)
            #print("Error: {} at epoch {}".format(error, i+1))
                        
            d_loss = self.loss_function.backward(output, y)

            self._backprop(d_loss, learning_rate)
            

    def predict(self, x, y): 
        # Normalize data
        x, y = self.normalizer.normalize(x, y)

        output = x
        for j, layer in enumerate(self.layers):
            output = self.activations_functions[j].forward(layer.forward(output))
        
        return self.normalizer.renormalize(output)


    def plot(self, dataset_name, nr_epochs, y_test, y_pred):

        xy_max = max(max(y_pred), max(y_test))
        xy_min = min(min(y_pred),  min(y_test))

        plt.figure(figsize=(10,6))
        plt.suptitle("Data: {}".format(dataset_name))
        plt.subplot(121)
        plt.scatter(np.arange(1, nr_epochs+1), self.errors, label='loss')
        plt.title("Average Loss by epoch")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.subplot(122)
        text1 = "{} {} {} {} {}".format(1, 2,
            3, 4, 5)
        text2 = "{} {} {} {} {}".format(6, 7,
            8, 9, 10)
        text = text1 + '\n' + text2

        plt.scatter(y_test, y_pred)
        plt.xlim(xy_min, xy_max)
        plt.ylim(xy_min, xy_max)
        plt.xlabel("Target")
        plt.ylabel("Predicted")
        plt.title("Actual Y vs Predicted ")
        plt.annotate(text, xy=(1, 1), xytext=(-15, -15), fontsize=10,
            xycoords='axes fraction', textcoords='offset points',
            bbox=dict(facecolor='white', alpha=0.8),
            horizontalalignment='right', verticalalignment='top')

        plt.show()
