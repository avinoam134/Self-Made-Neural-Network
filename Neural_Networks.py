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
    
    def train (self, X, Y, mb_size=32 ,alpha = 0.1, num_iterations = 100):
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
                

    

