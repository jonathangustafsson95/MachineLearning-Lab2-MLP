from numpy import genfromtxt
import operator
import numpy as np
from MultiLayerPerceptron import MLP
from Loss import SquaredErrorLoss
from Activation import LinearActivationFunction, SigmoidActivationFunction, InputActivationFunction
from Normalizer import Normalizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def main():
    # Data prep
    #plastic_data = genfromtxt('data/plastic.csv', delimiter=', ')
    boston_data = genfromtxt('data/boston.csv', delimiter=', ')

    x = boston_data[:, :-1]
    y = boston_data[:, -1]
    x = x[:len(x)]
    y = y[:len(y)]
    y = y.reshape(len(x),1)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    # print("Boston Data before norm:")
    # print("X: {}, Y: {}" .format(x,y))


    # Define loss function and activation function
    loss_func = SquaredErrorLoss()
    activation_func_inp = InputActivationFunction()
    activation_func_hid = SigmoidActivationFunction()
    activation_func_out = LinearActivationFunction()
    model = MLP(loss_func)

    # Add layers

    # Input layer
    model.add_layer(13, activation_func_inp)
    # Hidden layer
    model.add_layer(10, activation_func_hid)
    # Output
    model.add_layer(1, activation_func_out)

    # Train
    model.train(X_train, y_train, 0.01, 20000)

    # x1 = boston_data[:, :-1]
    # y1 = boston_data[:, -1]
    # x1 = x1[:500]
    # y1 = y1[:500]
    # y1 = y1.reshape(len(x1),1)

    #print("X1:  {}  \nY1: {}".format(x1,y1))
    # Make predict
    pred = model.predict(X_test, y_test)
    diff = y_test.round(1) - pred.round(1)

    print("\n \n Target:        Pred:       Difference:\n")
    print(np.c_[y_test, pred, diff])

    print("\nAvg TARGET")
    print(np.average(y_test))

    print("Avg PRED")
    print(np.average(pred))


if __name__ == "__main__":
    main()