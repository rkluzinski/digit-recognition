"""
Written by Ryan Kluzinski
Last Edited March 30, 2018

Functions for loading the MNIST data.
"""

import numpy as np

def read_int(fp):
    """
    Reads an integer from a file open in binary mode.

    TODO: args, returns
    """
    return int.from_bytes(fp.read(4), byteorder='big')

def loadLabels(filename):
    """
    Reads labels from the MNIST data.
    
    TODO: args, returns
    """
    
    with open(filename, 'rb') as infile:
        magic = read_int(infile)
        count = read_int(infile)

        labels = np.frombuffer(infile.read(count), dtype=np.uint8)

    return labels


def loadVectors(filename):
    """
    Reads vectors from the MNIST data.

    TODO: args, returns
    """

    with open(filename, 'rb') as infile:
        magic = read_int(infile)
        count = read_int(infile)

        rows = read_int(infile)
        columns = read_int(infile)

        vectors = np.frombuffer(infile.read(count*rows*columns), dtype=np.uint8)

    return vectors.reshape(count, rows*columns) / 255
