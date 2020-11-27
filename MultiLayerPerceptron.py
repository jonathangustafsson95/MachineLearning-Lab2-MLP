from Layer import Layer
from Loss import SquaredErrorLoss
from Activation import InputActivationFunction, SigmoidActivationFunction, LinearActivationFunction
import numpy as np
import matplotlib.pyplot as plt
from Normalizer import Normalizer


class MLP:
    """
    This is the class used for processing the data and creating a Multi
       layered perceptron.
       
       ...

       Attributes
    ----------
    layers : list of the class Layer
        a list of the layers included in the MLP
    activations_functions : list of the class Activation
        a list of the activations included in the MLP
    loss_function : Loss
        the loss function defined for the MLP
    normalizer : Normalizer
        a normalizer class to normalize the data used in the MLP 
    errors : float []
        list for the average error for each training epoch in the MLP
    predict_errors : float 
        a float for the  average error of all data rows in predict

    Methods
    -------
    add_layer(self, size, activation_function)
        adds a layer to the MLP
    _backprop(self, d_loss, learning_rate)
        backpropagates through the MLP
    train(self, x, y, learning_rate=0.01, n_epochs=10)
        method for training the MLP
    predict(self, x, y)
        method for predicting given data
    plot(self, dataset_name, nr_epochs, y_test, y_pred, train_time, pred_time)
        method for plotting result from MLP
    
    """
       
    def __init__(self, loss_function):
        """
        Parameters          
        ----------
        loss_function : SquaredErrorLoss
            loss function for the MLP
        """

        self.layers = []
        self.activations_functions = []
        self.loss_function = loss_function
        self.normalizer = Normalizer()
        # All errors, returned in end for plotting
        self.errors = []
        self.predict_errors = 0


    def add_layer(self, size, activation_function):
        """Adds a layer to the MLP.

        Parameters
        ----------
        size : int
            The size of the layer to be added
        activation_function : Activation
            The activation function for the layer to be added
        """

        n_inputs = size
        if self.layers:
            n_inputs = self.layers[-1].n_nodes

        self.layers.append(Layer(size, n_inputs))
        self.activations_functions.append(activation_function)

    def _backprop(self, d_loss, learning_rate):
        """Backpropagates through the MLP.

        Parameters
        ----------
        d_loss : numpy array
            The derivated loss
        learning_rate : float
            The rate at which the MLP learns
        """

        for i in range(len(self.layers)-1, 1, -1):
            input_data = self.layers[i].get_output()
            loss = self.activations_functions[i].backward(input_data).reshape(len(input_data), -1) * d_loss
            d_loss = self.layers[i].backprop(loss, learning_rate)

    def train(self, x, y, learning_rate=0.01, n_epochs=10):
        """Backpropagates through the MLP.

        Parameters
        ----------
        n_epochs : int
            The number of epochs the MLP will run
        x : floats [][]
            The input data for training the MLP
        y : floats [][]
            The target data for each training data
        learning_rate : float
            The rate at which the MLP learns
        """

        # Normalize data
        self.normalizer.fit(x,y)
        x, y = self.normalizer.normalize(x,y)

        # Main loop, handles forward and backprop
        for _ in range(n_epochs): 
            output = x
            for j, layer in enumerate(self.layers):
                # Activation forward, sets the input and output for each layers
                output = self.activations_functions[j].forward(layer.forward(output))
                layer.set_output(output)
            
            # Current error for the epoch is saved
            error = (np.average(self.loss_function.forward(output, y)))
            self.errors.append(error)

            # Get the derivative loss from the output node
            d_loss = self.loss_function.backward(output, y)
            # Backprop
            self._backprop(d_loss, learning_rate)
            

    def predict(self, x, y): 
        """Predicts a given input with the MLP.

        Parameters
        ----------
        x : floats [][]
            The input data for training the MLP
        y : floats [][]
            The target data for each training data
        """

        # Normalize data
        x, y = self.normalizer.normalize(x, y)

        output = x
        for j, layer in enumerate(self.layers):
            output = self.activations_functions[j].forward(layer.forward(output))

        self.predict_errors = (np.average(self.loss_function.forward(output, y)))
        
        return self.normalizer.renormalize(output)


    def plot(self, dataset_name, nr_epochs, y_test, y_pred, train_time, pred_time):
        """Plots the results from a dataset trained and predicted with the MLP.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset being plot
        nr_epochs : int
            The number of epochs the MLP has been trained
        y_test : floats [][]
            The target data for each training data
        y_pred : floats [][]
            The predicted result from the MLP
        train_time : float
            The elapsed time of the MLP training on the dataset
        pred_time : float
            The elapsed time of the MLP prediction on the dataset
        """

        xy_max = max(max(y_pred), max(y_test))
        xy_min = min(min(y_pred),  min(y_test))

        plt.figure(figsize=(10,6))
        plt.suptitle("Data: {}".format(dataset_name))
        plt.subplot(121)
        plt.scatter(np.arange(1, nr_epochs+1), self.errors, label='loss')
        plt.title("Average Loss by epoch")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        pred_info = "Total pred time: {:.6f}\n".format(pred_time)
        train_info = "Total train time: {:.2f}\n".format(train_time)
        average_loss = "Pred MSE: {:.2f}\n".format(np.average(self.predict_errors))
        epochs = "Number of epochs: {}".format(nr_epochs)

        text = pred_info + train_info + average_loss + epochs

        plt.annotate(text, xy=(1, 1), xytext=(-15, -15), fontsize=10,
            xycoords='axes fraction', textcoords='offset points',
            bbox=dict(facecolor='white', alpha=0.8),
            horizontalalignment='right', verticalalignment='top')

        plt.subplot(122)
        plt.scatter(y_test, y_pred)
        plt.xlim(xy_min, xy_max)
        plt.ylim(xy_min, xy_max)
        plt.xlabel("Target")
        plt.ylabel("Predicted")
        plt.title("Actual Y vs Predicted ")

        plt.show()
