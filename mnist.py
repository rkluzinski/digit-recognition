"""
Written by:
Ryan Kluzinski	1492614
Kelly Luc    	1498694
Last Edited April 15, 2018

This file contains functions for loading the mnist training
and testing data from the mnist/ directory.

TODO:
-Handle wrong filenames?
"""

import numpy as np

def read_int(fp):
    """
    Reads a big-endian integer from a file open in binary mode.

    Arguments:
    fp: file-pointer where the integer will be read from.

    Returns:
    int: the integer read from the file.
    """
    
    return int.from_bytes(fp.read(4), byteorder='big')


def load_labels(filename):
    """
    Reads labels from the given MNIST file. Returns a list of
    one-hot vectors.

    Arguments:
    filename: the filename as a string.

    Returns:
    labels: a numpy array of one hot vectors where each row is a
      new label. Each label correpsonds to the image in the same
      row in the images array.
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


def load_images(filename):
    """
    Reads the image from the mnist data. Returns a numpy array of
    vectors containing the training data.

    Arguments:
    filename: the filename as a string.

    Returns:
    images: a numpy array where each row is a 784x1 numpy array that
      represents each image.
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
        npbuffer = np.frombuffer(infile.read(bytecount), dtype=np.uint8)

    # creates array of scaled input vectors
    images = npbuffer.reshape(count, rows*columns) / 255
    
    return images


def load_training():
    """
    Loads the training labels and images from the mnist training data.

    Returns:
    images: a numpy array where each row is a 784x1 numpy array that
      represents each image.

    Returns:
    labels: a numpy array of one hot vectors where each row is a
      new label. Each label correpsonds to the image in the same
      row in the images array.
    """
    
    labels = load_labels("mnist/train-labels.idx1-ubyte")
    images = load_images("mnist/train-images.idx3-ubyte")

    return labels, images


def load_testing():
    """
    Loads the testing labels and images from the mnist training data.

    Returns:
    images: a numpy array where each row is a 784x1 numpy array that
      represents each image.

    Returns:
    labels: a numpy array of one hot vectors where each row is a
      new label. Each label correpsonds to the image in the same
      row in the images array.
    """

    labels = load_labels("mnist/t10k-labels.idx1-ubyte")
    images = load_images("mnist/t10k-images.idx3-ubyte")

    return labels, images

# TODO remove?
def load_mnist():
    return load_training() + load_testing()


# tests to ensure the loading function work as intended.
def main():
    # loads the mnist data.
    training_labels, training_images, testing_labels, testing_images = load_mnist()

    # prints the training labels
    print("Training Labels:")
    print(training_labels.shape)

    # prints the training images
    print("\nTraining Images:")
    print(training_images.shape)

    # prints the testing labels
    print("\nTesting Labels:")
    print(testing_labels.shape)

    # prints the testing images
    print("\nTesting Images:")
    print(testing_images.shape)


if __name__ == "__main__":
    main()
