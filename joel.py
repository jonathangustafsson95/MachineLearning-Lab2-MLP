class MLP:
    def init(self, loss_function):
        self.layers = []
        self.loss_function = loss_function

    def add_layer(self, size, activation_function):
        n_inputs = size
        if self.layers:
            n_inputs = self.layers[-1].n_nodes
        self.layers.append(Layer(size, n_inputs, activation_function))

    def _backprop(self, x, y, d_loss, learning_rate):
        pass

    def train(self, x, y, learning_rate=0.03, n_epochs=500):
        norm = Normalizer()
        norm.fit(x, y)
        x, y = norm.normalize(x, y)
        for i in range(n_epochs):
            input = x
            for j, layer in enumerate(self.layers):
                layer.forward(input)
                input = layer.output

            for l in reversed(self.layers):
                derror_douto = l.output - y
                duoto_dino = l.activation_function.backward(l.input)
                derror_wo = np.dot(l.input.T, derror_douto * duoto_dino)
                l.w -= learning_rate * derror_wo

            e = ((1 / 2) * (np.power((self.layers[-1].output - y), 2)))
            print(e.sum())

    def predict(self, x, y):
        pass
[10:06]
class Layer:
    def init(self, n_nodes, n_inputs, activation_function):
        self.n_nodes = n_nodes
        self.n_inputs = n_inputs
        self.activation_function = activation_function
        self.input = []
        self.output = []
        # Initialize weights and bias with randoms.
        self.w = np.random.rand(n_inputs, n_nodes)
        self.b = np.random.rand(1, n_nodes) 

    def forward(self, x):
        self.input = np.dot(x, self.w) + self.b
        self.output = self.activation_function.forward(self.input)

    def activate_forward(self, x):
        self.activation_function.forward(x)

    def activate_backward(self, x):
        self.activation_function.backward(x)

    def backprop(self, loss, learning_rate):
        # update weights
        pass
[10:06]
def main():
    # Data prep
    data = genfromtxt('data/plastic.csv', delimiter=', ')
    x = data[:, :-1]
    y = data[:, -1]
    x = x[:10]
    y = y[:10]
    y = y.reshape(10,1)

    # Define loss function and activation function
    loss_func = SquaredErrorLoss()
    activation_func_hid = SigmoidActivationFunction()
    model = MLP(loss_func)

    # Add layers

    # Layers
    model.add_layer(2, activation_func_hid)
    model.add_layer(2, activation_func_hid)
    # Output
    model.add_layer(1, activation_func_hid)

    # Train
    model.train(x, y)


if name == "main":
    main()