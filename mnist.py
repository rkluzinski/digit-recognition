"""
Written by Ryan Kluzinski
Last Edited April 6, 2018

Functions for loading the MNIST data.

TODO:
-Handle wrong filenames?
"""

import numpy as np

def read_int(fp):
    """
    Reads an integer from a file open in read binary mode.
    """
    return int.from_bytes(fp.read(4), byteorder='big')


def load_labels(filename):
    """
    Reads labels from the given MNIST file. Returns a list of
    one-hot vectors.
    """
    
    with open(filename, 'rb') as infile:
        # read magic number and count
        magic = read_int(infile)
        count = read_int(infile)
        
        # read entire file into a buffer
        npbuffer = np.frombuffer(infile.read(count), dtype=np.uint8)

    # creates numpy array of onehot vectors
    labels = np.zeros([count, 10])
    labels[np.arange(count), npbuffer] = 1 

    return labels


def load_vectors(filename):
    """
    Reads input data from the given MNIST file. Returns a list of
    vectors containing the input data.
    """

    with open(filename, 'rb') as infile:
        # read magic number and count
        magic = read_int(infile)
        count = read_int(infile)

        # read num of rows and columns of data
        rows = read_int(infile)
        columns = read_int(infile)

        # number of bytes to read
        bytecount = count*rows*columns

        # read entire file into a buffer
        npbuffer = np.frombuffer(infile.read(bytecount),
                                 dtype=np.uint8)

    # creates array of scaled input vectors
    vectors = npbuffer.reshape(count, rows*columns) / 255
    
    return vectors


def load_training():
    labels = load_labels("mnist/train-labels.idx1-ubyte")
    vectors = load_vectors("mnist/train-images.idx3-ubyte")

    return labels, vectors


def load_testing():
    labels = load_labels("mnist/t10k-labels.idx1-ubyte")
    vectors = load_vectors("mnist/t10k-images.idx3-ubyte")

    return labels, vectors


def load_mnist():
    return load_training() + load_testing()


# for testing
def main():
    training_labels, training_vectors, testing_labels,\
    testing_vectors = load_mnist()

    print("Training Labels:")
    print(training_labels.shape)

    print("\nTraining Images:")
    print(training_vectors.shape)

    print("\nTesting Labels:")
    print(testing_labels.shape)

    print("\nTesting Images:")
    print(testing_vectors.shape)


if __name__ == "__main__":
    main()
