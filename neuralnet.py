"""
Written by Ryan Kluzinski
Last Edited April 6, 2018

A class that implements a 3 layer neural network.
"""

from random import shuffle
from collections import deque
import numpy as np

class NeuralNetwork:
    """
    A class for initializes and training a 3-layers neural network.
    """

    def __init__(self, sizes):
        """
        Initializes the network with random weights and biases.
        """

        # only 3 layer-networks are supported currently
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
        
        Arguments:
        x: a real number.
        deriv (default = False): If true, return derivative.

        Returns:
        The value of the sigmoid function at x.
        """

        # return derivative
        if deriv:
            s = self.sigmoid(x)
            return s*(1-s)

        # calculate sigmoid(x)
        return 1/(1+np.exp(-x))


    def feedforward(self, vector):
        """
        Feeds a column vector into the neural network and returns 
        the output.

        Arguments:
        vector: the columns vector inputted to the neural network.

        Returns:
        a2: The final layer activation, the output of the network.
        """

        # input must be a column vector
        assert vector.shape == (self.sizes[0], 1)

        # calculates the hidden layer activation
        z1 = self.W1.dot(vector) + self.b1
        a1 = self.sigmoid(z1)

        # calculates the final layer activation
        z2 = self.W2.dot(a1) + self.b2
        a2 = self.sigmoid(z2)

        return a2


    def backpropagate(self, X, Y, reg_const):
        """
        Computes the gradient of the cost function using backpropagation.

        Arguments:
        X: the batch of training inputs.
        Y: the batch of training outputs.

        Returns:
        dW1, dW2, db1, db2: the gradients of the weights and biases.
        loss: the total loss for this batch.
        """

        # ensures both batches are the same size
        assert len(X) == len(Y)

        # gets the batch size
        batch_size = len(X)

        # intializes gradient vectors for the weights and biases
        dW1 = np.zeros([self.sizes[1], self.sizes[0]])
        dW2 = np.zeros([self.sizes[2], self.sizes[1]])
        db1 = np.zeros([self.sizes[1], 1])
        db2 = np.zeros([self.sizes[2], 1])

        # stores the loss
        loss = 0

        for x, y in zip(X, Y):
            # converts inputs/outputs to column vectors
            x = x.reshape(self.sizes[0], 1)
            y = y.reshape(self.sizes[2], 1)

            # computes the forward propogation
            z1 = self.W1.dot(x) + self.b1
            a1 = self.sigmoid(z1)
            z2 = self.W2.dot(a1) + self.b2
            a2 = self.sigmoid(z2)

            # computes error and loss
            error = (a2 - y)

            # computes cross entropy loss
            loss += -1 * np.sum( y * np.log(a2) + (1 - y) * np.log(1 - a2) )

            # add regularization component to loss
            loss += reg_const * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2))) / len(X) 

            # computes the backwards propagating errors
            delta2 = error
            delta1 = np.multiply(self.W2.T.dot(delta2), self.sigmoid(z1, deriv=True))

            # updates the gradients
            dW1 += delta1.dot(x.T) + reg_const / len(X) * self.W1
            dW2 += delta2.dot(a1.T) + reg_const / len(X) * self.W2
            db1 += delta1
            db2 += delta2

        #average of individual gradients
        dW1 = dW1 / batch_size
        dW2 = dW2 / batch_size
        db1 = db1 / batch_size
        db2 = db2 / batch_size

        return loss, dW1, dW2, db1, db2


    def SGD(self, X, Y, batch_size, epochs, learning_rate, reg_const, logfile=None):
        """
        Optimizes the weights and biases using stochastic gradient descent.

        Arguments:
        X: the training inputs.
        Y: the training outputs.
        batch_size: how big each backprop batch will be.
        epochs: how many times to iterate over all training data.
        learning_rate: how fast the neural learns.
        reg_const: the regularization constant, prevents overfitting with
          large networks.
        logfile (default = None): the file to log the epoch, iteration,
          and the current loss to.
        """

        # assertions to prevent errors
        assert len(X) == len(Y)
        assert batch_size <= len(X)

        # iteration count
        iterations = 0

        for i in range(epochs):
            # randomly shuffle training data
            training_order = [i for i in range( len(X) )]
            shuffle(training_order)

            # for each batch
            for j in range(0, len(X), batch_size):
                # create the training batches
                batchX = [X[k] for k in training_order[j:j+batch_size]]
                batchY = [Y[k] for k in training_order[j:j+batch_size]]

                # performs backpropagation
                loss, dW1, dW2, db1, db2 = self.backpropagate(batchX, batchY, reg_const)

                # updates the models parameters
                self.W1 -= learning_rate * dW1
                self.W2 -= learning_rate * dW2
                self.b1 -= learning_rate * db1
                self.b2 -= learning_rate * db2

                iterations += 1

                # log to file
                if logfile != None:
                    line = "epoch: {} iteration: {} loss: {}\n".format(i+1, iterations, loss)
                    logfile.write(line)


def main():
    # some sample testing data
    X = np.array([[1,0,1], [0,0,1], [1,0,0]])
    Y = np.array([[1,0], [0,1], [1,1]])

    # initialize and trains the test neural network
    testNN = NeuralNetwork([3,2,2])
    testNN.SGD(X, Y, 3, 10000, 0.1, 0)

    # print results
    for x in X:
        print(testNN.feedforward(x.reshape(3,1)))


if __name__ == "__main__":
    main()
