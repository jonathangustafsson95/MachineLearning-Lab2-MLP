from Layer import Layer

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
        pass

    def predict(self, x, y):
        pass