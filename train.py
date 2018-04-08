"""
Written by Ryan Kluzinski
Last Edited April 6, 2018

Trains a neural network to classify handwritten digits.
"""

import pickle
from sys import stdout
import numpy as np
from mnist import load_training, load_testing
from neuralnet import NeuralNetwork
    

def evaluate(NN):
    Y, X = load_testing()

    correct = 0
    total = 0

    for x, y in zip(X, Y):
        x = x.reshape(784, 1)
        y = y.reshape(10,1)
        if np.argmax(y) == np.argmax(NN.feedforward(x)):
            correct += 1

        total += 1

    print("Accuracy: {} / {}".format(correct, total))


def main():
    hidden_size = int(input("Hidden Layer Size: "))
    epochs = int(input("Epochs: "))
    batch_size = int(input("Batch Size: "))
    learning_rate = float(input("Learning Rate: "))

    outfile = input("Save neural network as (filename): ")

    Y, X = load_training()
    
    network = NeuralNetwork([784, hidden_size, 10])
    network.SGD(X, Y, batch_size, learning_rate, epochs, logfile=stdout)
    
    evaluate(network)

    print("Saving model under: models/{}".format(outfile))
    pickle.dump(network, open("models/{}".format(outfile), "wb"))


if __name__ == "__main__":
    main()
