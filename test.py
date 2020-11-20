import numpy as np 
from random import random

# https://www.youtube.com/watch?v=Z97XGNUUx9o&ab_channel=ValerioVelardo-TheSoundofAI

class MLP(object):

    def __init__(self, nr_inputs=3, nr_hidden_layers=[3,3], nr_outputs=1):

        self.nr_inputs = nr_inputs
        self.nr_hidden_layers = nr_hidden_layers
        self.nr_outputs = nr_outputs
    
        layers = [self.nr_inputs] + self.nr_hidden_layers + [self.nr_outputs]

        self.weights = []
        
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            self.weights.append(w)

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations
    
        derivatives = []
        for i in range(len(layers)-1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives


    def forward_propagate(self, inputs):

        activations = inputs
        self.activations[0] = inputs

        for i, w in enumerate(self.weights):
            # calculate net inputs
            net_inputs = np.dot(activations, w)

            # calculate activations
            activations = self._sigmoid(net_inputs)

            # a_3 = sigmoid(h_3)
            # h_3 = a_2 * w_2
            self.activations[i+1] = activations

        return activations


    def back_propagate(self, error, verbose=False):

        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            print("Activations: {}" .format(activations))
            delta = error * self._sigmoid_derivative(activations)
            print("Delta: {}" .format(delta))

            delta_reshaped  = delta.reshape(delta.shape[0], -1).T
            print("Delta_Reshaped: {}" .format(delta_reshaped))

            current_activations = self.activations[i] 
            print("Current_Act: {}" .format(current_activations))

            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1) 
            print("Current_Act_reshaped: {}" .format(current_activations_reshaped))

            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            print("Derivatives: {}" .format(self.derivatives[i]))

            error = np.dot(delta, self.weights[i].T)
            print("Error: {}" .format(error))


            if verbose:
                print("Derivatives for W{}: \n {}".format(i, self.derivatives[i]))

        return error


    def gradient_descent(self, learning_rate):
        
        for i in range(len(self.weights)):
            weights = self.weights[i]
            #print("\nOriginal Weights{} \n{}".format(i, weights))
            derivatives = self.derivatives[i]

            weights += derivatives * learning_rate
            #print("\nUpdated Weights{} \n{}".format(i, weights))


    def train(self, inputs, targets, epochs, learning_rate):
        
        for i in range(epochs):
            sum_error = 0
            for input, target in zip(inputs, targets):

                print("Input: {}, Target: {}" .format(input, target))
                output = self.forward_propagate(input)

                error = target - output
                print("Error: {}" .format(error))

                self.back_propagate(error, verbose=False)

                self.gradient_descent(learning_rate)

                sum_error += self._mse(target, output)

            print("Error: {} at epoch {}".format(sum_error / len(inputs), i))



    def _mse(self, target, output):
        return np.average((target - output) ** 2)


    def  _sigmoid_derivative(self, x):
        return x * (1.0 - x)


    def _sigmoid(self, x):
        return 1/(1+ np.exp(-x)) 


if __name__ == "__main__":

    # create MLP
    mlp = MLP(2, [5], 1)

    # create some inputs
    inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(100)])   # array of samples
    targets = np.array([[i[0] +  i[1]] for i in inputs])                         # array of target vals
    print("inputs")
    print(inputs)
    print("targets")
    print(targets)
    # train mlp
    mlp.train(inputs, targets, 10, 0.1)


    # create  dummy data
    input = np.array([0.3, 0.1])
    target = np.array([0.4])

    output = mlp.forward_propagate(input)
    print("\n \n Our network believes that {} + {} is equal to {}".format(input[0], input[1], output[0]))