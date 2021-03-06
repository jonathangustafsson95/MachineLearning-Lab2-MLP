from numpy import genfromtxt
import operator
import numpy as np
from MultiLayerPerceptron import MLP
from Loss import SquaredErrorLoss
from Activation import LinearActivationFunction, SigmoidActivationFunction, InputActivationFunction
from Normalizer import Normalizer
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split


class Dataset:
    """
    This class is a helper class for holding necessary information 
    for presentation and such when testing this ANN.

    ... 

    Attributes 
    ----------
    name : str
        the Name of the dataset
    learning_rate : float
        algorithms rate of learning
    epochs : int
        number of epochs algorithm is going to run
    loss_f : SquaredErrorLoss 
        loss function for ANN
    layers: Layer 
        number of nodes and activation functions for layer
    X_train : float []
        set of training instances from dataset
    X_test : float []
        holdoutset of testing instances from dataset
    y_train : float []
        targets corresponding to X_train
    y_test : float []
        targets corresponding to y_train
    pred : float []
        predicted values 
    train_time : float
        elapsed time for training
    pred_time = float
        elapsed time for prediction
    
    
    Methods
    -------
    
    """
    def __init__(self, name, learning_rate, epochs, loss_f, layers): 
        """ 
        Constructor for this class.

        Parameters
        ----------
        name : str
            the name of the dataset
        learning_rate : float
            algorithms rate of learning
        epochs : int
            number of epochs algorithm is going to run
        loss_f : SquaredErrorLoss 
            loss function for ANN
        layers: Layer 
            number of nodes and activation functions for layer
        """
        self.name = name
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_f = loss_f
        self.layers = layers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.pred = None
        self.train_time = None
        self.pred_time = None


class Layer:
    """
    This class is a helper class for holding necessary information for presentation and 
    such when testing this ANN.
    """
    def __init__(self, act_f_of_layer, nodes_per_layer):
        self.act_f_of_layer = act_f_of_layer
        self.nodes_per_layer = nodes_per_layer


def main():
    """
    Main method for this program. It handles datapreperation and time meassuring.
    """    
    
    # Setting parameters for datasets thats going to be tested.
    datasets = [Dataset(name = 'boston', learning_rate = 0.04,\
        epochs = 10000, loss_f = SquaredErrorLoss, layers = Layer(
            act_f_of_layer = [InputActivationFunction, SigmoidActivationFunction, SigmoidActivationFunction, LinearActivationFunction],             
            nodes_per_layer = [13, 8, 4, 1])),

        Dataset(name = 'concrete', learning_rate = 0.08,\
            epochs = 10000, loss_f = SquaredErrorLoss, layers = Layer(
                act_f_of_layer = [InputActivationFunction, SigmoidActivationFunction, SigmoidActivationFunction, LinearActivationFunction],
                nodes_per_layer = [8, 6, 4, 1])),

        Dataset(name = 'friedm', learning_rate = 0.14, \
            epochs = 10000, loss_f = SquaredErrorLoss, layers = Layer(
                act_f_of_layer = [InputActivationFunction, SigmoidActivationFunction, LinearActivationFunction],
                nodes_per_layer = [5, 3 ,1])),

        Dataset(name = 'istanbul', learning_rate = 0.025, \
            epochs = 10000, loss_f = SquaredErrorLoss, layers = Layer(
                act_f_of_layer = [InputActivationFunction, SigmoidActivationFunction, LinearActivationFunction],
                nodes_per_layer = [7, 5 ,1])),

        Dataset(name = 'laser', learning_rate = 0.05, \
            epochs = 10000, loss_f = SquaredErrorLoss, layers = Layer(
                act_f_of_layer = [InputActivationFunction, SigmoidActivationFunction, LinearActivationFunction],
                nodes_per_layer = [4, 3 ,1])),

        Dataset(name = 'plastic', learning_rate = 0.06, \
            epochs = 10000, loss_f = SquaredErrorLoss, layers = Layer(
                act_f_of_layer = [InputActivationFunction, SigmoidActivationFunction, LinearActivationFunction],
                nodes_per_layer = [2, 2 ,1])),

        Dataset(name = 'quakes', learning_rate = 0.01, \
            epochs = 10000, loss_f = SquaredErrorLoss, layers = Layer(
                act_f_of_layer = [InputActivationFunction, SigmoidActivationFunction, LinearActivationFunction],
                nodes_per_layer = [3, 2 ,1])),

        Dataset(name = 'stock', learning_rate = 0.06, \
            epochs = 10000, loss_f = SquaredErrorLoss, layers = Layer(
                act_f_of_layer = [InputActivationFunction, SigmoidActivationFunction, SigmoidActivationFunction, LinearActivationFunction],
                nodes_per_layer = [9, 7, 5, 1])),

        Dataset(name = 'wizmir', learning_rate = 0.01, \
            epochs = 10000, loss_f = SquaredErrorLoss, layers = Layer(
                act_f_of_layer = [InputActivationFunction, SigmoidActivationFunction, SigmoidActivationFunction, LinearActivationFunction],
                nodes_per_layer = [9, 7, 5, 1]))
        ]
    
    for dataset in datasets:

        # Data prep
        data = genfromtxt(f'data/{dataset.name}.csv', delimiter=', ')

        x = data[:, :-1]
        y = data[:, -1]
        x = x[:len(x)]
        y = y[:len(y)]
        y = y.reshape(len(x),1)

        # Make test and training sets
        dataset.X_train, dataset.X_test, dataset.y_train, dataset.y_test = \
            train_test_split(x, y, test_size=0.25)

        model = MLP(dataset.loss_f)

        # Add layers to model
        for i in range(len(dataset.layers.act_f_of_layer)):
            model.add_layer(dataset.layers.nodes_per_layer[i], dataset.layers.act_f_of_layer[i])
        
        # train model and measure elapsed time 
        train_start = time.time()
        model.train(dataset.X_train, dataset.y_train, dataset.learning_rate, dataset.epochs)
        dataset.train_time = time.time() - train_start

        # predict and measure elapsed time
        pred_start = time.time()
        dataset.pred = model.predict(dataset.X_test, dataset.y_test)
        dataset.pred_time = time.time() - pred_start

        # plot 
        model.plot(dataset.name, dataset.epochs, dataset.y_test, dataset.pred, dataset.train_time, dataset.pred_time)


    # Print predictions versus targets of testing, another way of verifying our work.
    for dataset in datasets:
        print(f'{dataset.name}')
        print("\n \n Target:        Pred:\n")
        print(np.c_[dataset.y_test, dataset.pred])

        print("\nAvg TARGET")
        print(np.average(dataset.y_test))

        print("Avg PRED")
        print(np.average(dataset.pred))


if __name__ == "__main__":
    main()