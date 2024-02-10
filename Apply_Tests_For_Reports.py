'''
This file mainly randomises Weights, inputs, labels, etc,
to test various implementations
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from Tests import jacobian_verification, network_gradient_verification
from Activation_Functions import softmax_regression_loss, least_squares_loss, sgd
from SGD import sgd_find_global_minimum
from Neural_Layers import Layer, ResidualLayer, LossLayer, activations
from Neural_Networks import NeuralNetwork

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


def P1Q1_test_softmax_regression_loss():
    num_features = 5
    num_classes = 3
    num_samples = 10
    X = np.random.rand(num_features, num_samples)
    W = np.random.rand(num_features, num_classes)
    b = np.random.rand(num_classes,1)
    #set Y to a random matrix with 1s in the correct class and 0s elsewhere:
    Y = np.zeros((num_classes, num_samples))
    for i in range(num_samples):
        Y[np.random.randint(0, num_classes), i] = 1
        # create a decending array of epsilon values:
        epsilons = np.array([(0.5)**i for i in range(1, 10)])
        loss_function = softmax_regression_loss

    losses_ord1, losses_ord2 = jacobian_verification(X , W, Y, b, epsilons, loss_function)
    jacobian_verification_visualizer(losses_ord1, losses_ord2, epsilons)


def P1Q2_test_sgd_with_least_squares():
    #create data A, b s.t the appropriate functions will make y=2x (for example):
    A = np.random.rand(100, 1)
    b = 2*A
    #initialize x to a different factor than 2:
    x = float(10)
    #perform sgd and plot the initialised x and the final x:
    alpha = 0.1
    num_iterations = 100
    loss_function = least_squares_loss
    x_final, _, losses = sgd(A, b, x, 0, loss_function, alpha, num_iterations)
    #add a fig for the loss and a fig for comparing the initial x to the final x on top of the data:
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].scatter(A, b, label='Data')
    axs[0].plot(A, x*A, label='Initial x')
    axs[0].plot(A, x_final*A, label='Final x')
    axs[0].set_xlabel('A')
    axs[0].set_ylabel('b')
    axs[0].set_title('Linear Regression')
    axs[0].legend()

    axs[1].plot(losses)
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Least Squares Loss')

    plt.tight_layout()
    plt.show()


def parse_data(filename):
    path = f"./HW1_Data/{filename}.mat"
    data = loadmat(path)
    Yt = data['Yt']
    Ct = data['Ct']
    Yv = data['Yv']
    Cv = data['Cv']
    return Yt, Ct, Yv, Cv


def P1Q3_test_sgd_with_softmax(data_set_name):
    Yt, Ct, Yv, Cv = parse_data(data_set_name)
    num_samples = Yt.shape[1]
    num_features = Yt.shape[0]
    num_classes = Ct.shape[0]
    #initialize W and b:
    W = np.random.rand(num_features, num_classes)
    b = np.random.rand(num_classes, 1)
    #perform batch sgd:
    alpha = 0.1
    num_iterations = 50
    batch_size = [20, 200, 2000, 20000]
    test_size = 5000
    loss_function = softmax_regression_loss
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i in range(4):
        W_final, b_final, losses, test_losses = sgd_find_global_minimum(Yt, Ct, Yv, Cv, loss_function, alpha, num_iterations, batch_size[i], test_size)
        #plot the train and test losses:
        axs[i//2, i%2].plot(losses, label=f'Train Loss - Batch Size: {batch_size[i]}')
        axs[i//2, i%2].plot(test_losses, label=f'Test Loss - Batch Size: {batch_size[i]}')
        axs[i//2, i%2].set_xlabel('Iteration')
        axs[i//2, i%2].set_ylabel('Loss')
        axs[i//2, i%2].set_title(f'Train and Test Losses - Batch Size: {batch_size[i]}')
        axs[i//2, i%2].legend()
    plt.tight_layout()
    plt.show()


def P2Q1_test_NN_all_layers():
    input_dim = 3
    hidden_1 = 2
    hidden_2 = 7
    output_dim = 5
    tanh_activation = activations["tanh"]
    layer1 = Layer(input_dim, hidden_1, tanh_activation)
    layer2 = Layer(hidden_1, hidden_2, tanh_activation)
    loss_layer = LossLayer(hidden_2, output_dim)
    network = NeuralNetwork([layer1, layer2, loss_layer])
    epsilons = np.array([(0.5)**i for i in range(1, 10)])
    input_features = network.layers[0].W.shape[1]
    classes = network.layers[-1].b.shape[1]
    num_samples = 10
    X = np.random.rand(input_features, num_samples)
    Y = np.zeros((classes, num_samples))
    Y[np.random.randint(0, classes)] = 1
    losses_ord1, losses_ord2 = network_gradient_verification(network, X, Y, epsilons)
    jacobian_verification_visualizer(losses_ord1, losses_ord2, epsilons)


def P2Q2_test_ResNet_all_layers():
    input_dim = 5
    num_samples = 10
    output_dim = 5
    tanh_activation = activations["tanh"]
    layer1 = ResidualLayer(input_dim, num_samples ,tanh_activation)
    layer2 = ResidualLayer(input_dim, num_samples ,tanh_activation)
    layer3 = ResidualLayer(input_dim, num_samples ,tanh_activation)
    loss_layer = LossLayer(input_dim, output_dim)
    network = NeuralNetwork([layer1, layer2, layer3, loss_layer])
    epsilons = np.array([(0.5)**i for i in range(1, 10)])
    input_features = network.layers[0].W1.shape[1]
    X = np.random.rand(input_features, num_samples)
    Y = np.zeros((output_dim, num_samples))
    Y[np.random.randint(0, input_features)] = 1
    losses_ord1, losses_ord2 = network_gradient_verification(network, X, Y, epsilons)
    jacobian_verification_visualizer(losses_ord1, losses_ord2, epsilons)

def P2Q3_test_combined_layers():
    input_dim = 5
    hidden_1 = 7
    hidden_2 = 3
    num_samples = 10
    output_dim = 5
    tanh_activation = activations["tanh"]
    layer1 = Layer(input_dim, hidden_1, tanh_activation)
    layer2 = ResidualLayer(hidden_1, num_samples ,tanh_activation)
    layer3 = Layer(hidden_1, hidden_2, tanh_activation)
    loss_layer = LossLayer(hidden_2, output_dim)
    network = NeuralNetwork([layer1, layer2, layer3, loss_layer])
    epsilons = np.array([(0.5)**i for i in range(1, 10)])
    input_features = network.layers[0].W.shape[1]
    X = np.random.rand(input_features, num_samples)
    Y = np.zeros((output_dim, num_samples))
    Y[np.random.randint(0, input_features)] = 1
    losses_ord1, losses_ord2 = network_gradient_verification(network, X, Y, epsilons)
    jacobian_verification_visualizer(losses_ord1, losses_ord2, epsilons)

def main():
    P2Q3_test_combined_layers()


if __name__ == "__main__":
    main()