import numpy as np

class LossFunction:
    """
    This is Mr.Henriks code. So we refuse to comment this code.
    """
    @staticmethod
    def forward(z):
        pass

    @staticmethod
    def backward(z):
        pass

class SquaredErrorLoss:
    """
    This is Mr.Henriks code. So we refuse to comment this code.
    """
    
    '''Used for regression problems
    '''
    @staticmethod
    def forward(predictions, correct_outputs):
        return np.power((predictions - correct_outputs), 2)

    # we tend to call the gradient/derivative of the
    # function the "backward" of the function
    @staticmethod
    def backward(predictions, correct_outputs):
        result = 2 * (predictions - correct_outputs)
        return result