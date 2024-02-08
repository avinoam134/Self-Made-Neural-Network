import numpy as np
import Activation_Functions as af

activations = {
    "softmax" :
    {
        "calc" : af.softmax_calculate,
        "der" : af.softmax_regression_loss
    },
    "tanh":
    {
        "calc" : af.tanh_calculate,
        "der" : af.tanh_derive
    }
}


class Layer:
    def __init__(self, input_dim, output_dim, activation):
        self.W = np.random.rand(output_dim, input_dim)
        self.b = np.random.rand(output_dim,1)
        self.activation = activation
        self.X = None
        self.Y = None
        self.dW = None
        self.db = None
        self.dX = None

    def forward(self, X):
        self.X = X
        self.linear_calc = np.dot(self.W, self.X) + self.b
        self.Y = self.activation["calc"](self.linear_calc)
        return self.Y

    def backward(self, dY):
        self.db = dY * self.activation["der"](self.linear_calc)
        self.dW = np.dot(self.db, self.X.T)
        self.dX = np.dot(self.W.T, self.db)
        return self.dX

    def update(self, alpha):
        self.W -= alpha*self.dW
        self.b -= alpha*self.db


class LossLayer(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim, activations["softmax"])
        self.loss = None
        self.W = self.W.T
        self.b = self.b.T
    
    def forward(self, X):
        self.X = X

    def backward(self, Y):
        self.loss, self.dW, self.db, self.dX = self.activation["der"](self.X, Y, self.W, self.b)
        return self.dX
    

class ResisudalLayer(Layer):
    def __init__(self, input_dim, activation):
        super().__init__(input_dim, input_dim, activation)
        self.W = None
        self.W1 = np.random.rand(input_dim, input_dim)
        self.W2 = np.random.rand(input_dim, input_dim)

    def forward (self, X):
        self.X = X
        self.linear_calc = np.dot(self.W1, self.X) + self.b
        act = self.activation["calc"](self.linear_calc)
        self.Y = np.dot(self.W2, self.X) + act
        return self.Y
    
    def backward(self, dY):
        pass
    
    def update (self, alpha):
        self.W1 -= alpha*self.dW1
        self.W2 -= alpha*self.dW2
        self.b -= alpha*self.db










