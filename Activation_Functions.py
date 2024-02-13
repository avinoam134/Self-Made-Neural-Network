import numpy as np

def softmax_calculate(X):  
    #calculate the value of the linear multiplication with normalization:
    max_col = np.max(X, axis=1).reshape(-1,1)
    norm = X - max_col
    #compute the softmax function:
    norm_exp = np.exp(norm)
    softmax = norm_exp / np.sum(norm_exp, axis=1).reshape(-1,1)
    return softmax

def softmax_regression_loss(X, Y, W, b):
    num_samples = Y.shape[1]
    #calculate the value of the linear multiplication with normalization:
    linear_calc = np.dot(X.T, W) + b.reshape(1,-1)
    max_col = np.max(linear_calc, axis=1).reshape(-1,1)
    linear_norm = linear_calc - max_col
    #compute the softmax function:
    linear_norm_exp = np.exp(linear_norm)
    softmax = linear_norm_exp / np.sum(linear_norm_exp, axis=1).reshape(-1,1)
    #compute loss:
    log_softmax = np.log(softmax)
    #replace loss_unnorm with a vector with each entry being the inner multiplication of each column of Y with the corresponding column of log_softmax:
    loss_unnorm = Y*log_softmax.T
    loss = -np.sum(np.sum(loss_unnorm, axis=0)) / num_samples
    #compute gradients:
    dsoftmax = softmax - Y.T
    dw = np.dot(X, dsoftmax) / num_samples
    db = np.sum(dsoftmax) / num_samples
    #gradient w.t.r to X:
    dx = np.dot(W, softmax.T - Y) / num_samples
    return loss, dw, db, dx

'''
1. check if db is calculated correctly
2. check if dsoftmax should be taken with Y'''

def sgd(X, Y, W, b, loss_function,  alpha = 0.1, num_iterations = 10000):
    losses = []
    for i in range(num_iterations):
        loss, dw, db, _ = loss_function(X, Y, W, b)
        losses.append(loss)
        W -= alpha*dw
        b -= alpha*db
    return W, b, losses

def least_squares_loss(A, b, x, _):
    num_samples = A.shape[0]
    loss = np.sum(np.square(x*A - b)) / num_samples
    dx = 2 * np.dot(A.T, x*A - b) / num_samples
    return loss, dx, 0, 0

def tanh_calculate(X):
    return np.tanh(X)

def tanh_derive(X):
    return 1 - np.tanh(X)**2

def relu_calculate(X):
    return np.maximum(0, X)

def relu_derive(X):
    return (X > 0).astype(int)


