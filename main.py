from numpy import genfromtxt
import operator
import numpy as np
from MultiLayerPerceptron import MLP
from Loss import SquaredErrorLoss
from Activation import LinearActivationFunction, SigmoidActivationFunction, InputActivationFunction
from Normalizer import Normalizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, name, learning_rate, epochs, loss_f, layers): 
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


class Layer:
    def __init__(self, act_f_of_layer, nodes_per_layer):
        self.act_f_of_layer = act_f_of_layer
        self.nodes_per_layer = nodes_per_layer


def main():
    # Data prep

    datasets = [Dataset(name = 'boston', learning_rate = 0.025,\
        epochs = 100, loss_f = SquaredErrorLoss, layers = Layer(
            act_f_of_layer = [InputActivationFunction, SigmoidActivationFunction, LinearActivationFunction],             
            nodes_per_layer = [13, 8 ,1])),

        Dataset(name = 'concrete', learning_rate = 0.025,\
            epochs = 100, loss_f = SquaredErrorLoss, layers = Layer(
                act_f_of_layer = [InputActivationFunction, SigmoidActivationFunction, LinearActivationFunction],
                nodes_per_layer = [8, 5 ,1])),

        Dataset(name = 'friedm', learning_rate = 0.025, \
            epochs = 100, loss_f = SquaredErrorLoss, layers = Layer(
                act_f_of_layer = [InputActivationFunction, SigmoidActivationFunction, LinearActivationFunction],
                nodes_per_layer = [5, 3 ,1])),

        Dataset(name = 'istanbul', learning_rate = 0.025, \
            epochs = 100, loss_f = SquaredErrorLoss, layers = Layer(
                act_f_of_layer = [InputActivationFunction, SigmoidActivationFunction, LinearActivationFunction],
                nodes_per_layer = [7, 5 ,1])),

        Dataset(name = 'laser', learning_rate = 0.025, \
            epochs = 100, loss_f = SquaredErrorLoss, layers = Layer(
                act_f_of_layer = [InputActivationFunction, SigmoidActivationFunction, LinearActivationFunction],
                nodes_per_layer = [4, 3 ,1])),

        Dataset(name = 'plastic', learning_rate = 0.025, \
            epochs = 100, loss_f = SquaredErrorLoss, layers = Layer(
                act_f_of_layer = [InputActivationFunction, SigmoidActivationFunction, LinearActivationFunction],
                nodes_per_layer = [2, 2 ,1])),

        Dataset(name = 'quakes', learning_rate = 0.025, \
            epochs = 100, loss_f = SquaredErrorLoss, layers = Layer(
                act_f_of_layer = [InputActivationFunction, SigmoidActivationFunction, LinearActivationFunction],
                nodes_per_layer = [3, 2 ,1])),

        Dataset(name = 'stock', learning_rate = 0.025, \
            epochs = 100, loss_f = SquaredErrorLoss, layers = Layer(
                act_f_of_layer = [InputActivationFunction, SigmoidActivationFunction, LinearActivationFunction],
                nodes_per_layer = [9, 7 ,1])),

        Dataset(name = 'wizmir', learning_rate = 0.025, \
            epochs = 100, loss_f = SquaredErrorLoss, layers = Layer(
                act_f_of_layer = [InputActivationFunction, SigmoidActivationFunction, LinearActivationFunction],
                nodes_per_layer = [9, 7 ,1]))
        ]
    
    for dataset in datasets:
        data = genfromtxt(f'data/{dataset.name}.csv', delimiter=', ')

        x = data[:, :-1]
        y = data[:, -1]
        x = x[:len(x)]
        y = y[:len(y)]
        y = y.reshape(len(x),1)

        dataset.X_train, dataset.X_test, dataset.y_train, dataset.y_test = \
            train_test_split(x, y, test_size=0.25, random_state=0)

        model = MLP(dataset.loss_f)
        for i in range(len(dataset.layers.act_f_of_layer)):
            model.add_layer(dataset.layers.nodes_per_layer[i], dataset.layers.act_f_of_layer[i])
        
        model.train(dataset.X_train, dataset.y_train, dataset.learning_rate, dataset.epochs)

        dataset.pred = model.predict(dataset.X_test, dataset.y_test)

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