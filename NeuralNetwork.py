import numpy as np
import matplotlib.pyplot as plt

def softmax(X):  
    #print(f"X: {X}")
    #calculate the value of the linear multiplication with normalization:
    max_col = np.max(X).reshape(-1,1)
    norm = X - max_col
    #compute the softmax function:
    norm_exp = np.exp(norm)
    #print(f"norm_exp: {norm_exp}")
    softmax = norm_exp / np.sum(norm_exp).reshape(-1,1)
    return softmax


def softmax_loss_and_dx(Y, X):
    _softmax = softmax(X)
    #print (f'softmax: {_softmax}')
    num_samples = Y.shape[1]
    # compute loss:
    log_softmax = np.log(_softmax)
    # replace loss_unnorm with a vector with each entry being the inner multiplication of each column of Y with the corresponding column of log_softmax:
    loss_unnorm = Y * log_softmax.T
    loss = -np.sum(loss_unnorm) / num_samples
    # compute gradient w.r.t to X:
    dX = _softmax - Y
    return loss, dX
    




class Activation:
    def __init__(self, activation_function, activation_derivative):
        self.calc = activation_function
        self.der = activation_derivative

class Layer:
    def __init__(self, input_dim, output_dim, activation):
        self.W = np.random.rand(output_dim, input_dim)
        self.b = np.random.rand(output_dim,1)
        self.activation = activation
        self.input = None
        self.output = None
        self.dW = None
        self.db = None
        self.dX = None

    def forward(self, X):
        self.input = X
        self.linear_calc = np.dot(self.W, self.input) + self.b
        self.output = self.activation.calc(self.linear_calc)
        return self.output

    def backward(self, dY):
        self.db = dY * self.activation.der(self.linear_calc)
        self.dW = np.dot(self.db, self.input.T)
        self.dX = np.dot(self.W.T, self.db)
        return self.dX

    def update(self, alpha):
        self.W -= alpha*self.dW
        self.b -= alpha*self.db



    
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.loss_function = softmax_loss_and_dx

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dY):
        for layer in reversed(self.layers):
            dY = layer.backward(dY)

    def update(self, alpha):
        for layer in self.layers:
            layer.update(alpha)
    
    def get_network_loss_and_dx(self, X, Y):
        output = self.forward(X)
        loss, dY = self.loss_function(Y, output)
        self.backward(dY)
        dx = self.layers[0].dX
        return loss, dx
    
    def grad_verify(self):
        X = np.random.rand(self.layers[1].W .shape[0], 1)
        Y = np.zeros((self.layers[-1].W.shape[0], 1))
        Y[np.random.randint(0, Y.shape[0])] = 1
        #print (f'Y: {Y}')
        epsilons = np.array([(0.5)**i for i in range(1, 10)])
        d = np.random.rand(X.shape[0], X.shape[1])
        losses_ord1 = []
        losses_ord2 = []
        for epsilon in epsilons:
            eps_d = epsilon*d
            X_plus = X.copy() + eps_d
            loss_plus, _ = self.get_network_loss_and_dx(X_plus, Y)
            loss, dx = self.get_network_loss_and_dx(X, Y)
            print (f'loss: {loss}, dx: {dx.shape}')
            #first order:
            loss_ord1 = loss_plus - loss
            loss_ord1_abs = np.abs(loss_ord1)
            #second order:
            epsd_grad = np.sum(np.multiply(eps_d, dx))
            loss_ord2 = loss_ord1 - epsd_grad
            print(f'epsd_grad: {epsd_grad}')
            loss_ord2_abs = np.abs(loss_ord2)
            losses_ord1.append(loss_ord1_abs)
            losses_ord2.append(loss_ord2_abs)
            print (f'loss_ord1: {loss_ord1_abs}, loss_ord2: {loss_ord2_abs}')
        return epsilons, losses_ord1, losses_ord2
    
    def grad_visualizer(self):
        epsilons, losses_ord1, losses_ord2 = self.grad_verify()
        epsilons = np.sort(epsilons)
        plt.plot(epsilons, losses_ord1, label='First Order - O(\u03B5)')
        plt.plot(epsilons, losses_ord2, label='Second Order - O(\u03B5^2)')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Epsilon')
        plt.ylabel('Loss')
        plt.title('Gradient Verification - Log Scale')
        plt.legend()
        plt.show()

    def sgd(self, X, Y, alpha=0.1, num_iterations=10000):
        losses = []
        for i in range(num_iterations):
            output = self.forward(X)
            loss, dY = self.loss_function(Y, output)
            self.backward(dY)
            self.update(alpha)
            losses.append(loss)
        return losses


def tanh_calculation(X):
    return np.tanh(X)
def tanh_derivative(X):
    return 1 - np.tanh(X)**2


def main():
    tanh_activation = Activation(tanh_calculation, tanh_derivative)
    input_dim = 3
    hidden_dim = 2
    output_dim = 5
    layer1 = Layer(input_dim, hidden_dim, tanh_activation)
    layer2 = Layer(hidden_dim, output_dim, tanh_activation)
    network = NeuralNetwork([layer1, layer2])
    network.grad_visualizer()



if __name__ == "__main__":
    main()