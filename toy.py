"""
Written by Ryan Kluzinski
Last Edited April 5, 2018

Simple proof of concept neural network.
"""

import numpy as np

class NN:
    def __init__(self, l0, l1, l2):
        self.l0 = l0
        self.l1 = l1
        self.l2 = l2

        self.A1 = None
        self.A2 = None

        self.W1 = 2*np.random.rand(l1, l0) - 1
        self.W2 = 2*np.random.rand(l2, l1) - 1

        self.B1 = 2*np.random.rand(l1, 1) - 1
        self.B2 = 2*np.random.rand(l2, 1) - 1

        
    def sig(self, x, d=False):
        if d:
            s = self.sig(x)
            return s*(1-s)

        return 1/(1+np.exp(-x))

    
    def feedforward(self, vector):
        #assert vector.shape == (self.l0, 1)
        
        z1 = self.W1.dot(vector) + self.B1
        self.A1 = self.sig(z1)

        z2 = self.W2.dot(self.A1) + self.B2
        self.A2 = self.sig(z2)

        return self.A2

    def learn(self, X, Y, n):
        alpha = 0.05
        
        for i in range(n):
            cost = 0

            dW1 = np.zeros([self.l1, self.l0])
            dW2 = np.zeros([self.l2, self.l1])
            dB1 = np.zeros([self.l1, 1])
            dB2 = np.zeros([self.l2, 1])
            
            for x, y in zip(X, Y):
                x = x.reshape(3,1)
                yhat = self.feedforward(x)

                error = yhat - y.reshape(2,1)
                cost += np.sum(np.square(error))

                A1 = self.A1
                A2 = self.A2
                W1 = self.W1
                W2 = self.W2
                B1 = self.B1
                B2 = self.B2

                d2 = np.multiply(error,
                                 self.sig(W2.dot(A1) + B2, d=True))
                dW2 += d2.dot(A1.T)
                dB2 += d2

                d1 = np.multiply(W2.T.dot(d2),
                                 self.sig(W1.dot(x) + B1, d=True))

                dW1 += d1.dot(x.T)
                dB1 += d1

            dW2 / len(X)
            dB2 / len(X)
            dW1 / len(X)
            dB1 / len(X)

            self.W2 -= alpha * dW2
            self.B2 -= alpha * dB2
            self.W1 -= alpha * dW1
            self.B1 -= alpha * dB1

            print(cost)
                
        
def main():
    X = (np.array([1,0,1]), np.array([0,0,1]), np.array([1,0,0]))
    Y = (np.array([1,0]), np.array([0,1]), np.array([1,1]))
    
    nn = NN(3, 2, 2)
    nn.learn(X, Y, 10000)

    for x in X:
        print(nn.feedforward(x.reshape(3,1)))
    

if __name__ == "__main__":
    main()
        
