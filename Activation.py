import numpy as np


class ActivationFunction:
    """
    This is Mr.Henriks code. So we refuse to comment this code.
    """
    @staticmethod
    def forward(z):
        pass

    @staticmethod
    def backward(z):
        pass


class SigmoidActivationFunction:
    """
    This is Mr.Henriks code. So we refuse to comment this code.
    """
    @staticmethod
    def forward(z): 
        return 1 / (1 + np.e ** (-z))
    @staticmethod
    def backward(z): 
        return z * (1 - z)


class LinearActivationFunction:
    """
    This is Mr.Henriks code. So we refuse to comment this code.
    """
    @staticmethod
    def forward(z): 
        return z
    @staticmethod
    def backward(z): 
        return np.ones(z.size)


class InputActivationFunction:
    """
    This is Mr.Henriks code. So we refuse to comment this code.
    """
    @staticmethod
    def forward(z): 
        return z
    @staticmethod
    def backward(z): 
        return np.zeros(z.size)
