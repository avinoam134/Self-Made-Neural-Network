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
        db_unsummed = dY * self.activation["der"](self.linear_calc)
        self.db = np.sum(db_unsummed)
        self.dW = np.dot(db_unsummed, self.X.T)
        self.dX = np.dot(self.W.T, db_unsummed)
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
    

class ResidualLayer(Layer):
    def __init__(self, input_dim, activation):
        super().__init__(input_dim, input_dim, activation)
        #self.W = None
        self.dW1 = None
        self.dW2 = None
        self.W1 = np.random.rand(input_dim, input_dim)
        self.W2 = np.random.rand(input_dim, input_dim)

    def forward (self, X):
        self.X = X
        self.linear_calc = np.dot(self.W1, self.X) + self.b
        act = self.activation["calc"](self.linear_calc)
        self.Y = np.dot(self.W2, act) + X
        return self.Y
    
    def backward(self, dY):
        lin_der = self.activation["der"](self.linear_calc)
        lin_der_W2T = lin_der * self.W2.T
        db_unsummed = np.dot(lin_der_W2T, dY)
        self.db = np.sum(db_unsummed)
        self.W1 = np.dot(db_unsummed, self.X.T)
        self.dW2 = np.dot(dY, self.linear_calc.T)
        self.dX = np.dot(np.dot(self.W1.T, lin_der_W2T) + np.identity(self.W1.shape[0]), dY)
        #diag = np.diag(lin_der.reshape(-1))
        #print (diag.shape)
        #self.dX = np.dot(np.dot(self.W2, np.dot(diag, self.W1)).T + np.identity(self.W1.shape[0]), dY)
        #self.dX = np.dot(np.dot(self.W2, np.sum(lin_der, axis=1)*self.W1).T + np.identity(self.W1.shape[0]), dY)
        #self.dX = np.dot(dY, self.W2.T) * lin_der
        # self.dX = np.dot((np.dot(self.W1.T, dY) * lin_der).T, self.W2.T)
        print (self.dX.shape)
        return self.dX

    def update (self, alpha):
        self.W1 -= alpha*self.dW1
        self.W2 -= alpha*self.dW2
        self.b -= alpha*self.db











