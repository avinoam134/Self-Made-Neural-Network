import numpy as np
import Activation_Functions as af

activations = {
    "softmax" :
    {
        "calc" : af.softmax_calculate,
        "der" : af.softmax_regression_loss
    },
        "least-squares":
    {
        "calc" : af.least_squares_loss,
        "der" : af.least_squares_loss
    },
    "tanh":
    {
        "calc" : af.tanh_calculate,
        "der" : af.tanh_derive
    },
    "relu":
    {
        "calc" : af.relu_calculate,
        "der" : af.relu_derive
    }

}


class Layer:
    def __init__(self, input_dim, output_dim, activation):
        self.W = np.random.rand(output_dim, input_dim)
        self.b = np.random.rand(output_dim,1)
        self.W /= np.linalg.norm(self.W)
        self.b /= np.linalg.norm(self.b)
        self.v_w = 0
        self.v_b = 0
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
    
    def forward_predict(self, X):
        return self.activation["calc"](np.dot(self.W, X) + self.b)

    def backward(self, dY):
        db_unsummed = dY * self.activation["der"](self.linear_calc)
        self.db = np.sum(db_unsummed)
        self.dW = np.dot(db_unsummed, self.X.T)
        self.dX = np.dot(self.W.T, db_unsummed)
        return self.dX

    def update(self, alpha=0.1, beta=0.9):
        self.v_w = beta*self.v_w + (1-beta)*self.dW
        self.v_b = beta*self.v_b + (1-beta)*self.db
        self.W -= alpha*self.dW
        self.b -= alpha*self.db


class LossLayer(Layer):
    def __init__(self, input_dim, output_dim, activation = activations["softmax"]):
        super().__init__(input_dim, output_dim, activation)
        self.loss = None
        self.W = self.W.T
        self.b = self.b.T
    
    
    def forward_predict(self, X):
        return self.activation["calc"](np.dot(X.T, self.W) + self.b.reshape(1,-1)).T
    
    def forward(self, X):
        self.X = X

    def backward(self, Y):
        self.loss, self.dW, self.db, self.dX = self.activation["der"](self.X, Y, self.W, self.b)
        return self.dX
    

class ResidualLayer(Layer):
    def __init__(self, input_dim, num_samples ,activation):
        super().__init__(input_dim, num_samples, activation)
        #self.W = None
        self.v_w1 = 0
        self.v_w2 = 0
        self.dW1 = None
        self.dW2 = None
        self.W1 = np.random.rand(num_samples, input_dim)
        self.W2 = np.random.rand(input_dim, num_samples)
        self.W1 /= np.linalg.norm(self.W1)
        self.W2 /= np.linalg.norm(self.W2)

    def forward (self, X):
        self.X = X
        self.linear_calc = self.W1 @ self.X + self.b
        act = self.activation["calc"](self.linear_calc)
        self.Y = self.W2 @ act + X
        return self.Y
    
    def forward_predict(self, X):
        return self.W2 @ self.activation["calc"](self.W1 @ X + self.b) + X
    
    def backward(self, dY):
        lin_der = self.activation["der"](self.linear_calc)
        W2T_dY = self.W2.T @ dY
        db_unsummed = lin_der * W2T_dY
        self.db = np.sum(db_unsummed)
        self.dW1 = db_unsummed @self.X.T
        self.dW2 = dY @ self.activation["calc"](self.linear_calc).T
        self.dX = self.W1.T @ db_unsummed + (np.identity(self.W1.shape[1]) @ dY)
        return self.dX

    def update (self, alpha=0.1, beta=0.9):
        self.v_w1 = beta*self.v_w1 + (1-beta)*self.dW1
        self.v_w2 = beta*self.v_w2 + (1-beta)*self.dW2
        self.v_b = beta*self.v_b + (1-beta)*self.db
        self.W1 -= alpha*self.dW1
        self.W2 -= alpha*self.dW2
        self.b -= alpha*self.db


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def predict(self, X):
        Y = self.predict_probs(X)
        return np.argmax(Y, axis=0)
    
    def predict_probs(self, X):
        for layer in self.layers:
            X = layer.forward_predict(X)
        return X

    def backward(self, dY):
        for layer in reversed(self.layers):
            dY = layer.backward(dY)

    def update(self, alpha):
        for layer in self.layers:
            layer.update(alpha)
    
    def train (self, X, Y, mb_size=32 ,alpha = 0.1, beta = 0.9, num_iterations = 100):
        losses = []
        for i in range(num_iterations):
            unshaped_m = X.shape[1]
            num_batches = unshaped_m // mb_size
            m = num_batches * mb_size
            X = X[:, :m]
            Y = Y[:, :m]
            indices = np.arange(m)
            np.random.shuffle(indices)
            X = X[:, indices]
            Y = Y[:, indices]
            for j in range(num_batches):
                start = j * mb_size
                end = (j + 1) * mb_size
                X_mb = X[:, start:end]
                Y_mb = Y[:, start:end]
                self.forward(X_mb)
                self.backward(Y_mb)
                self.update(alpha)
            losses.append(self.layers[-1].loss)
        return losses
                

    












