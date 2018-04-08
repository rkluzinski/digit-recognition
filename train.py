"""
Written by Ryan Kluzinski
Last Edited April 6, 2018

Trains a neural network to classify handwritten digits.
"""

from sys import stdout
import numpy as np
from mnist import load_training, load_testing
from neuralnet import NeuralNetwork


def train(NN):
    learning_rate = 0.5
    epochs = 30
    batch_size = 10
    Y, X = load_training()
    NN.SGD(X, Y, batch_size, learning_rate, epochs, logfile=stdout)
    

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

    print("{} / {}".format(correct, total))


def main():
    digitClassifier = NeuralNetwork([784, 15, 10])
    train(digitClassifier)
    evaluate(digitClassifier)


if __name__ == "__main__":
    main()
