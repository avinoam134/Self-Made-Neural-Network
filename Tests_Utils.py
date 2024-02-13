import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

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

def jacobian_verification_visualizer(losses_ord1, losses_ord2, epsilons):
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


def get_network_loss_and_dx(NN, X, Y):
    NN.forward(X)
    NN.backward(Y)
    dX = NN.layers[0].dX
    loss = NN.layers[-1].loss
    return loss, dX

#following the same logic from the previous gradient/jacoian verification:
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

def parse_data(filename, limited_data_set=False):
    path = f"./HW1_Data/{filename}.mat"
    data = loadmat(path)
    Yt = data['Yt']
    Ct = data['Ct']
    Yv = data['Yv']
    Cv = data['Cv']
    if limited_data_set:
        Yt = Yt[:, :200]
        Ct = Ct[:, :200]
    return Yt, Ct, Yv, Cv

def test_network_on_data_set(data_set, network, limited_data_set=False):
    Yt, Ct, Yv, Cv = parse_data(data_set, limited_data_set=limited_data_set)
    #train the network:
    alpha = 0.1
    beta = 0.9
    num_iterations = 500
    batch_size = 30
    train_losses = network.train(Yt, Ct, batch_size, alpha, beta, num_iterations)
    #test the network:
    predictions = network.predict_probs(Yv)
    W_id = np.identity(network.layers[-1].W.shape[1])
    b_id = np.zeros((network.layers[-1].b.shape[1], 1))
    test_loss, _, _, _ = network.layers[-1].activation["der"](predictions, Cv, W_id, b_id)
    test_accurcy = np.sum(np.argmax(predictions, axis=0) == np.argmax(Cv, axis=0)) / Cv.shape[1] * 100
    return train_losses, test_loss, test_accurcy


