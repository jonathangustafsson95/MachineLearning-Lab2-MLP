from numpy import genfromtxt
import numpy as np
from MultiLayerPerceptron import MLP
from Loss import SquaredErrorLoss
from Activation import LinearActivationFunction, SigmoidActivationFunction, InputActivationFunction
from Normalizer import Normalizer

def main():
    # Data prep
    data = genfromtxt('data/plastic.csv', delimiter=', ')
    x = data[:, :-1]
    y = data[:, -1]
    x = x[:len(x)]
    y = y[:len(y)]
    y = y.reshape(len(x),1)

    print("Data before norm:")
    print("X: {}, Y: {}" .format(x,y))


    # Define loss function and activation function
    loss_func = SquaredErrorLoss()
    activation_func_inp = InputActivationFunction()
    activation_func_hid = SigmoidActivationFunction()
    activation_func_out = LinearActivationFunction()
    model = MLP(loss_func)

    # Add layers

    # Input layer
    model.add_layer(2, activation_func_inp)
    # Hidden layer
    model.add_layer(2, activation_func_hid)
    # Output
    model.add_layer(1, activation_func_out)

    # Train
    model.train(x, y, 0.01, 500)

    x1 = data[:, :-1]
    y1 = data[:, -1]
    x1 = x1[:50]
    y1 = y1[:50]
    y1 = y1.reshape(len(x1),1)

    print("X1:  {}  \nY1: {}".format(x1,y1))
    # Make predict
    pred = model.predict(x1, y1)

    print("\n \n Network pred:\n {}".format(pred))

    print("Avg TARGET")
    print(np.average(y1))

    print("Avg PRED")
    print(np.average(pred))

if __name__ == "__main__":
    main()