import numpy as np

class Layer:
    """
    Class for a single layer
    ... 

    Attributes 
    ----------
    n_nodes : int
        number of nodes in layer
    n_inputs : int
        number of input nodes to layer
    input : []
        holds a list of inputs for this layer
    output : []
        holds a list of outputs for this layer
    w : numpy array
        holds a list of weights for all inputs
    b : numpy array
        holds bias weight for this layer
    
    
    Methods
    -------

    get_output(self)

    set_output(self, output)

    def forward(self, x)

    backprop(self, loss, learning_rate)
    
    """
    def __init__(self, n_nodes, n_inputs):
        """
        Parameters
        ----------
        n_nodes : int
            number of nodes in layer
        n_inputs : int
            number of input nodes to layer

        """
        self.n_nodes = n_nodes
        self.n_inputs = n_inputs
        self.input = []
        self.output = []
        # Initialize weights and bias with randoms. np.randon.uniform
        self.w = np.random.uniform(-0.5, 0.5, (n_inputs, n_nodes))
        self.b = np.random.uniform(-0.5, 0.5, (1, n_nodes))

    def get_output(self):
        """
        A getter method to get output

        """
        return self.output 

    def set_output(self, output):
        """
        A setter method for this class parameter "output"

        """
        self.output = output    

    def forward(self, x):
        """
        Forward method of this class calculates partial output of this layers.
        Activation function is handled else where.
        
        Parameters
        ----------
        x : []
            input to this layer

        """
        self.input = x
        return np.dot(x, self.w) + self.b

    # computes dE/dW, dE/dB for a given output_error=dE/dY. 
    # Returns input_error=dE/dX.
    def backprop(self, loss, learning_rate):
        """
        Backpropogates and updates weights/bias for the layer.

        Parameters
        ----------
        loss : int
            The calculated loss, difference between target and prediction
        learning_rate : float
            The rate at which the MLP learns

        """
        input_error = np.dot(loss, self.w.T)
        dw = np.dot(self.input.T, loss) / len(self.input) # Average

        self.w -= (learning_rate * dw) 
        self.b -= (learning_rate * np.average(loss))
        return input_error
        