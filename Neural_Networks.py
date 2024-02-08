class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

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
    
    def sgd(self, X, Y, alpha=0.1, num_iterations=10000):
        losses = []
        for i in range(num_iterations):
            self.forward(X)
            self.backward(Y)
            self.update(alpha)
            losses.append(self.layers[-1].loss)
        return losses