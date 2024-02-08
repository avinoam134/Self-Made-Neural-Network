import numpy as np

def sgd(X, Y, W, b, loss_function,  alpha = 0.1, num_iterations = 10000):
    losses = []
    for i in range(num_iterations):
        loss, dw, db, _ = loss_function(X, Y, W, b)
        losses.append(loss)
        W -= alpha*dw
        b -= alpha*db
    return W, b, losses

def get_batch(X, Y, batch_size):
    num_samples = X.shape[1]
    batch_indices = np.random.choice(num_samples, batch_size, replace=False)
    X_batch = X[:, batch_indices]
    Y_batch = Y[:, batch_indices]
    return X_batch, Y_batch

def batch_sgd(X, Y, W, b, loss_function, alpha=0.1, num_iterations=10000, batch_size=100):
    losses = []
    for i in range(num_iterations):
        X_batch, Y_batch = get_batch(X, Y, batch_size)
        loss, dw, db, _ = loss_function(X_batch, Y_batch, W, b)
        losses.append(loss)
        W -= alpha*dw
        b -= alpha*db
    return W, b, losses

def batch_sgd_with_momentum(X, Y, W, b, loss_function, alpha=0.1, num_iterations=10000, batch_size=100, beta=0.9):
    losses = []
    v_w = 0
    v_b = 0
    for i in range(num_iterations):
        X_batch, Y_batch = get_batch(X, Y, batch_size)
        loss, dw, db, _ = loss_function(X_batch, Y_batch, W, b)
        losses.append(loss)
        v_w = beta*v_w + (1-beta)*dw
        v_b = beta*v_b + (1-beta)*db
        W -= alpha*v_w
        b -= alpha*v_b
    return W, b, losses

def batch_sgd_with_momentum_and_test_loss(Xt, Yt, Xv, Yv, W, b, loss_function, alpha=0.1, num_iterations=10000, batch_size=100, test_size=20, beta=0.9):
    losses = []
    test_losses = []
    v_w = 0
    v_b = 0
    for i in range(num_iterations):
        X_batch, Y_batch = get_batch(Xt, Yt, batch_size)
        loss, dw, db, _ = loss_function(X_batch, Y_batch, W, b)
        losses.append(loss)
        v_w = beta*v_w + (1-beta)*dw
        v_b = beta*v_b + (1-beta)*db
        W -= alpha*v_w
        b -= alpha*v_b
        X_test, Y_test = get_batch(Xv, Yv, test_size)
        test_loss, _, _, _ = loss_function(X_test, Y_test, W, b)
        test_losses.append(test_loss)
    return W, b, losses, test_losses

def batch_sgd_with_test_set_loss (Xt, Yt, Xv, Yv, W, b, loss_function, alpha=0.1, num_iterations=10000, batch_size=100, test_size=20):
    losses = []
    test_losses = []
    for i in range(num_iterations):
        X_batch, Y_batch = get_batch(Xt, Yt, batch_size)
        loss, dw, db, _ = loss_function(X_batch, Y_batch, W, b)
        losses.append(loss)
        W -= alpha*dw
        b -= alpha*db
        X_test, Y_test = get_batch(Xv, Yv, test_size)
        test_loss, _, _, _ = loss_function(X_test, Y_test, W, b)
        test_losses.append(test_loss)
    return W, b, losses, test_losses

def rand_weights(num_features, num_classes):
    W = np.random.rand(num_features, num_classes)
    b = np.random.rand(num_classes, 1)
    return W, b

def sgd_find_global_minimum(Xt, Yt, Xv, Yv, loss_function, alpha=0.1, num_iterations=30, batch_size=100, test_size=20):
    num_features = Xt.shape[0]
    num_classes = Yt.shape[0]
    final_losses_matrix = []
    final_test_losses_matrix = []
    best_loss = np.inf
    W = None
    b = None
    for i in range(num_iterations):
        rand_W, rand_b = rand_weights(num_features, num_classes)
        rand_W, rand_b, losses, test_losses = batch_sgd_with_momentum_and_test_loss(Xt, Yt, Xv, Yv, rand_W, rand_b, loss_function, alpha, 1000, batch_size, test_size)
        if test_losses[-1] < best_loss:
            best_loss = test_losses[-1]
            W = rand_W
            b = rand_b
        final_losses_matrix.append(losses)
        final_test_losses_matrix.append(test_losses)

    final_losses_matrix = np.array(final_losses_matrix)
    final_test_losses_matrix = np.array(final_test_losses_matrix)
    final_losses = np.min(final_losses_matrix, axis=0)
    final_test_losses = np.min(final_test_losses_matrix, axis=0)
    return W,b,final_losses, final_test_losses



