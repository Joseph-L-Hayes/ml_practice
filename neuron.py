""" simple neuron using numpy as outlined by Victor Zhou in his blog
https://victorzhou.com/blog/intro-to-neural-networks/ """

import numpy as np

def sigmoid(x):
    """ sigmoid activation function """
    return 1 / (1 + np.exp(-x))

class Neuron:

    def __init__(self, weights, bias):
        """ weights is an np.array """

        self.weights = weights
        self.bias = bias

    def feed_forward(self, inputs):
        """ takes the dot product of all weights and all inputs plus the bias """
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)
