from Layer import Layer
from Loss import SquaredErrorLoss
from Activation import InputActivationFunction, SigmoidActivationFunction, LinearActivationFunction

class MLP:
    def __init__(self, loss_function):
        self.layers = []
        self.activations_functions = []
        self.loss_function = loss_function

    def add_layer(self, size, activation_function):
        n_inputs = 0
        if self.layers:
            n_inputs = len(self.layers[-1:])

        self.layers.append(Layer(size, n_inputs))
        self.activations_functions.append(activation_function)

    def _backprop(self, x, y, d_loss, learning_rate):
        pass

    def train(self, x, y, learning_rate=0.01, n_epochs=100):
        samples = len(x)

        for i in range(n_epochs):
            err = 0
            
            for j in range(len(samples)):

                output = x[j]
                for k, layer in enumerate(self.layers):
                    input_data = output
                    output = []

                    for node in range(layer.n_nodes):
                        node_value = layer.forward(input_data)
                        output.append(self.activations_functions[k](node_value))

                err += self.loss_function(output ,y[j]) # display purpose
                
                # fault + backprop
                    


    def predict(self, x, y):
        pass


l = [ 
    [1,2,3,4,5],
    [1,2,3,4,5],
    [1,2,3,4,5]
    ]

net = MLP(SquaredErrorLoss)
net.add_layer(len(l[0]), InputActivationFunction)
net.add_layer(8, SigmoidActivationFunction)
net.add_layer(2, LinearActivationFunction)