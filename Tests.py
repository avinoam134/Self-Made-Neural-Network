import numpy as np
import matplotlib as plt


def jacobian_verification(X,W,Y,b, epsilons, loss_function):
    num_features = X.shape[0]
    num_samples = X.shape[1]
    d = np.random.rand(num_features, num_samples)
    losses_ord1 = []
    losses_ord2 = []   
    for epsilon in epsilons:
        eps_d = epsilon*d
        X_plus = X.copy() + eps_d
        #first order:
        loss_plus, _, _, _ = loss_function(X_plus, Y, W, b)
        loss, _,_, dx = loss_function(X, Y, W, b)
        loss_ord1 = loss_plus - loss
        loss_ord1_abs = np.abs(loss_ord1)
        #second order:
        loss_ord2 = loss_ord1 - np.sum(np.multiply(eps_d, dx))
        loss_ord2_abs = np.abs(loss_ord2)
        losses_ord1.append(loss_ord1_abs)
        losses_ord2.append(loss_ord2_abs)  
    return losses_ord1, losses_ord2


def get_network_loss_and_dx(NN, X, Y):
    NN.forward(X)
    NN.backward(Y)
    dX = NN.layers[0].dX
    loss = NN.layers[-1].loss
    return loss, dX

#following the same logic from the previous gradient/jacoian verification
def network_gradient_verification(NN, X, Y, epsilons):
    input_features = NN.layers[0].W.shape[1]
    d = np.random.rand(input_features, 1)
    losses_ord1 = []
    losses_ord2 = []
    for epsilon in epsilons:
        eps_d = epsilon*d
        X_plus = X.copy() + eps_d
        loss_plus, _ = get_network_loss_and_dx(NN, X_plus, Y)
        loss, dx = get_network_loss_and_dx(NN, X, Y)
        #first order:
        loss_ord1 = loss_plus - loss
        loss_ord1_abs = np.abs(loss_ord1)
        epsd_grad = np.sum(np.multiply(eps_d, dx))
        losses_ord1.append(loss_ord1_abs)
        #second order:
        loss_ord2 = loss_ord1 - epsd_grad
        loss_ord2_abs = np.abs(loss_ord2)
        losses_ord2.append(loss_ord2_abs)
    return losses_ord1, losses_ord2

