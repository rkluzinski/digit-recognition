"""
Written by Ryan Kluzinski
Last Edited March 30, 2018

A neural network class.
"""

import numpy as np

class NeuralNetwork:
    """
    A single-layer neural network class.
    """

    def __init__(self, num_input, num_hidden, num_output):
        """
        Initializes the network with random weights.

        TODO: args, returns
        """

        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output

        # weight matrices
        self.weights_1 = 2 * np.random.rand(num_hidden, num_input) - 1
        self.weights_2 = 2 * np.random.rand(num_output, num_hidden) - 1

        # bias vectors
        self.bias_1 = np.random.rand(num_hidden, 1)
        self.bias_2 = np.random.rand(num_output, 1)

    def _sigmoid(self, x, deriv=False):
        """
        The sigmoid function and it's derivative.

        TODO: args, returns
        """

        if deriv:
            temp = self._sigmoid(x)
            return temp * (1 - temp)

        return 1/(1 + np.exp(-x))

    def feedforward(self, vector):
        """
        Feeds a vector in to the neural network a return another vector.

        TODO: args, returns
        """

        assert vector.shape == (self.num_input, 1)

        # stores the most recent activation fore backprop
        self.activation_1 = self._sigmoid( self.weights_1.dot(vector) + self.bias_1 )
        self.activation_2 = self._sigmoid( self.weights_2.dot(self.activation_1) + self.bias_2 )

        return self.activation_2

    def backpropogate(self, vector):
        """
        Uses backpropogation to find the gradient with respect to one training example.

        TODO: args, return
        """
        pass
