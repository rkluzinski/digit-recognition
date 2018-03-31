"""
Written by Ryan Kluzinski
Last Edited March 30, 2018

Trains the neural network to recognize handwritten digits.
"""

import numpy as np
from mnist import loadLabels, loadVectors
from neuralnet import NeuralNetwork

        
def main():
    training_labels = loadLabels("./data/train-labels.idx1-ubyte")
    training_vectors = loadVectors("./data/train-images.idx3-ubyte")

    testing_labels = loadLabels("./data/t10k-labels.idx1-ubyte")
    testing_vectors = loadVectors("./data/train-images.idx3-ubyte")
    
    digitClassifier = NeuralNetwork(784, 16, 10)

    correct = 0
    total = 0

    for label, vector in zip(testing_labels, testing_vectors):
        onehot = digitClassifier.feedforward(vector.reshape(784,1))
        prediction = np.argmax(onehot)

        if prediction == label:
            correct += 1

        total += 1

        print("{:5} / {:5}; Acc. {:.4f}".format(correct, total, correct/total))

    
if __name__ == '__main__':
    main()
