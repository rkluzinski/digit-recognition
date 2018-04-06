"""
Written by Ryan Kluzinski
Last Edited April 6, 2018

A neural network class.
"""

import numpy as np

class NeuralNetwork:
    """
    A class for 3-layer neural network.
    """

    def __init__(self, sizes):
        """
        Initializes the network with random weights.
        """

        # only 3 layer-networks are supported
        assert len(sizes) == 3

        # stores the layer sizes
        self.sizes = sizes

        # initializes the weight matrices
        self.W1 = 2 * np.random.rand(sizes[1], sizes[0]) - 1
        self.W2 = 2 * np.random.rand(sizes[2], sizes[1]) - 1

        # initializes the bias vectors
        self.b1 = 2 * np.random.rand(sizes[1], 1) - 1
        self.b2 = 2 * np.random.rand(sizes[2], 1) - 1


    def sigmoid(self, x, deriv=False):
        """
        Uses the sigmoid function for the nonlinear activation
        function. Returns the derivative if deriv=True.
        """

        # return derivative
        if deriv:
            s = self.sigmoid(x)
            return s*(1-s)

        return 1/(1+np.exp(-x))


    def feedforward(self, vector):
        """
        Feeds a column vector into the neural network and returns 
        the ouput vector.
        """

        # input must be a column vector
        assert vector.shape == (self.sizes[0], 1)

        z1 = self.W1.dot(vector) + self.b1
        a1 = self.sigmoid(z1)

        z2 = self.W2.dot(a1) + self.b2
        a2 = self.sigmoid(z2)

        return a2


    def backpropagate(self, X, Y):
        """
        Locally computes the gradient of the cost function.
        """

        assert len(X) == len(Y)
        batch_size = len(X)

        # stores gradients of the weights and biases
        dW1 = np.zeros([self.sizes[1], self.sizes[0]])
        dW2 = np.zeros([self.sizes[2], self.sizes[1]])
        db1 = np.zeros([self.sizes[1], 1])
        db2 = np.zeros([self.sizes[2], 1])

        loss = 0

        for x, y in zip(X, Y):
            # convert to column vectors
            x = x.reshape(self.sizes[0], 1)
            y = y.reshape(self.sizes[2], 1)

            # computes the forward propogation
            z1 = self.W1.dot(x) + self.b1
            a1 = self.sigmoid(z1)
            z2 = self.W2.dot(a1) + self.b2
            a2 = self.sigmoid(z2)

            # computes error and loss
            error = a2 - y
            loss += np.sum(np.square(error))

            # computes the backwards propagating errors
            delta2 = np.multiply(error,
                                 self.sigmoid(z2, deriv=True))
            delta1 = np.multiply(self.W2.T.dot(delta2),
                                 self.sigmoid(z1, deriv=True))

            # updates the gradients
            dW1 += delta1.dot(x.T)
            dW2 += delta2.dot(a1.T)
            db1 += delta1
            db2 += delta2

        #average of individual gradients
        dW1 = dW1 / batch_size
        dW2 = dW2 / batch_size
        db1 = db1 / batch_size
        db2 = db2 / batch_size

        return loss, dW1, dW2, db1, db2


    def batch_grad_descent(self, X, Y, alpha, epochs):
        """
        Uses the entire set of training data to compute the
        gradient for each epoch. The weights and biases are only
        updated once per epoch.
        """

        for i in range(epochs):
            loss, dW1, dW2, db1, db2 = self.backpropagate(X, Y)

            # update parameters
            self.W1 -= alpha * dW1
            self.W2 -= alpha * dW2
            self.b1 -= alpha * db1
            self.b2 -= alpha * db2
            
            print(loss)


def main():
    # test data
    X = np.array([[1,0,1], [0,0,1], [1,0,0]])
    Y = np.array([[1,0], [0,1], [1,1]])

    # initialize and train
    testNN = NeuralNetwork([3,2,2])
    testNN.batch_grad_descent(X, Y, 1.0, 10000)

    # print results
    for x in X:
        print(testNN.feedforward(x.reshape(3,1)))


if __name__ == "__main__":
    main()

                
