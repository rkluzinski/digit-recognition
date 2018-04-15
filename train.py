"""
Written by Ryan Kluzinski
Last Edited April 13, 2018

A program to train a neural network to identify handwritten digits.
"""

import pickle
from sys import stdout
import numpy as np

from mnist import load_training, load_testing
from neuralnet import NeuralNetwork
    

def evaluate(network):
    """
    Evaluates the neural network on the testing data and displays
    how many examples it identifies correctly out of the total.
    """

    # loads the testing data
    Y, X = load_testing()

    # initializes variables
    correct = 0
    total = 0

    # for each testing input and output
    for x, y in zip(X, Y):

        # reshape into column vectors
        x = x.reshape(784, 1)
        y = y.reshape(10,1)

        # if the network is correct, add to correct
        if np.argmax(y) == np.argmax(network.feedforward(x)):
            correct += 1

        total += 1

    # prints accurary
    print("Accuracy: {} / {}".format(correct, total))


def main():
    # gets the hyperparametres from the user
    hidden_size = int(input("Hidden Layer Size: "))
    epochs = int(input("Epochs: "))
    batch_size = int(input("Batch Size: "))
    learning_rate = float(input("Learning Rate: "))

    # gets file to store the neuralnetwork in
    outfile = input("Save neural network as (filename): ")
    print("Beginning training, this may take a few minutes...")

    # loads the testing data
    Y, X = load_training()

    # initializes the network
    network = NeuralNetwork([784, hidden_size, 10])

    # runs stochastic gradient descent with given hyperparametres
    network.SGD(X, Y, batch_size, learning_rate, epochs, logfile=stdout)

    # evaluates the networks performance
    evaluate(network)

    # saves the model using the pickel file format
    print("Saving model under: models/{}".format(outfile))
    pickle.dump(network, open("models/{}".format(outfile), "wb"))


if __name__ == "__main__":
    main()
