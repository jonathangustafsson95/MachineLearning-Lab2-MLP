from numpy import genfromtxt
from MultiLayerPerceptron import MLP
from Loss import SquaredErrorLoss
from Activation import LinearActivationFunction, SigmoidActivationFunction, InputActivationFunction


def main():
    # Data prep
    data = genfromtxt('data/plastic.csv', delimiter=', ')
    x = data[:, :-1]
    print(x)
    y = data[:, -1]
    print(y)
    x = x[:3]
    print(x)
    y = y[:3]
    print(y)
    y = y.reshape(3,1)
    print(y)

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
    model.train(x, y)


if __name__ == "__main__":
    main()