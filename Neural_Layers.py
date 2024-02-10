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
    },
    "least-squares":
    {
        "calc" : af.least_squares_loss,
        "der" : af.least_squares_loss
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
        db_unsummed = dY * self.activation["der"](self.linear_calc)
        self.db = np.sum(db_unsummed)
        self.dW = np.dot(db_unsummed, self.X.T)
        self.dX = np.dot(self.W.T, db_unsummed)
        return self.dX

    def update(self, alpha):
        self.W -= alpha*self.dW
        self.b -= alpha*self.db


class LossLayer(Layer):
    def __init__(self, input_dim, output_dim, activation = activations["softmax"]):
        super().__init__(input_dim, output_dim, activation)
        self.loss = None
        self.W = self.W.T
        self.b = self.b.T
    
    def forward(self, X):
        self.X = X

    def backward(self, Y):
        self.loss, self.dW, self.db, self.dX = self.activation["der"](self.X, Y, self.W, self.b)
        return self.dX
    

class ResidualLayer(Layer):
    def __init__(self, input_dim, num_samples ,activation):
        super().__init__(input_dim, num_samples, activation)
        #self.W = None
        self.dW1 = None
        self.dW2 = None
        self.W1 = np.random.rand(num_samples, input_dim)
        self.W2 = np.random.rand(input_dim, num_samples)

    def forward (self, X):
        self.X = X
        self.linear_calc = self.W1 @ self.X + self.b
        act = self.activation["calc"](self.linear_calc)
        self.Y = self.W2 @ act + X
        return self.Y
    
    def backward(self, dY):
        lin_der = self.activation["der"](self.linear_calc)
        W2T_dY = self.W2.T @ dY
        db_unsummed = lin_der * W2T_dY
        self.db = np.sum(db_unsummed)
        self.dW1 = db_unsummed @self.X.T
        self.dW2 = dY @ self.activation["calc"](self.linear_calc).T
        self.dX = self.W1.T @ db_unsummed + (np.identity(self.W1.shape[1]) @ dY)
        return self.dX

    def update (self, alpha):
        self.W1 -= alpha*self.dW1
        self.W2 -= alpha*self.dW2
        self.b -= alpha*self.db











