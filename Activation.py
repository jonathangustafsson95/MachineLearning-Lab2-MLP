import numpy as np


class ActivationFunction:
    @staticmethod
    def forward(z):
        pass

    @staticmethod
    def backward(z):
        pass


class SigmoidActivationFunction:
    @staticmethod
    def forward(z): 
        return 1 / (1 + np.e ** (-z))
    @staticmethod
    def backward(z): 
        return z * (1 - z)


class LinearActivationFunction:
    @staticmethod
    def forward(z): 
        return z
    @staticmethod
    def backward(z): 
        return np.ones(z.size)


class InputActivationFunction:
    @staticmethod
    def forward(z): 
        return z
    @staticmethod
    def backward(z): 
        return np.zeros(z.size)
