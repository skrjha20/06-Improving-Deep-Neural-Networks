import numpy as np
import matplotlib.pyplot as plt
import math
from opt_utils import *
from testcases import *

def update_parameters_with_gd(parameters, grads, learning_rate):
    L = len(parameters) // 2  # number of layers in the neural networks
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]
    return parameters

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def initialize_velocity(parameters):
    L = len(parameters) // 2
    v = {}
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros(parameters['W' + str(l + 1)].shape)
        v["db" + str(l + 1)] = np.zeros(parameters['b' + str(l + 1)].shape)
    return v

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        v["dW" + str(l + 1)] = beta * v['dW' + str(l + 1)] + (1 - beta) * (grads['dW' + str(l + 1)])
        v["db" + str(l + 1)] = beta * v['db' + str(l + 1)] + (1 - beta) * (grads['db' + str(l + 1)])
        parameters["W" + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * v['dW' + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * v['db' + str(l + 1)]
    return parameters, v

def initialize_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros((parameters["W" + str(l + 1)].shape[0], parameters["W" + str(l + 1)].shape[1]))
        v["db" + str(l + 1)] = np.zeros((parameters["b" + str(l + 1)].shape[0], parameters["b" + str(l + 1)].shape[1]))
        s["dW" + str(l + 1)] = np.zeros((parameters["W" + str(l + 1)].shape[0], parameters["W" + str(l + 1)].shape[1]))
        s["db" + str(l + 1)] = np.zeros((parameters["b" + str(l + 1)].shape[0], parameters["b" + str(l + 1)].shape[1]))
    return v, s

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    for l in range(L):
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]

        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - pow(beta1, t))
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - pow(beta1, t))

        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.power(grads['dW' + str(l + 1)], 2)
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.power(grads['db' + str(l + 1)], 2)

        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - pow(beta2, t))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - pow(beta2, t))

        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * np.divide(
            v_corrected["dW" + str(l + 1)], np.sqrt(s_corrected["dW" + str(l + 1)]) + epsilon)
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * np.divide(
            v_corrected["db" + str(l + 1)], np.sqrt(s_corrected["db" + str(l + 1)]) + epsilon)

    return parameters, v, s

def model(X, Y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9,
          beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000, print_cost=True):

    L = len(layers_dims)
    costs = []
    t = 0
    seed = 10

    parameters = initialize_parameters(layers_dims)
    if optimizer == "gd":
        pass  # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    for i in range(num_epochs):
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            a3, caches = forward_propagation(minibatch_X, parameters)
            cost = compute_cost(a3, minibatch_Y)
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1  # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)

        if print_cost and i % 1000 == 0:
            print("Cost after epoch %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()
    return parameters

if __name__ == "__main__":

    parameters, grads, learning_rate = update_parameters_with_gd_test_case()
    parameters = update_parameters_with_gd(parameters, grads, learning_rate)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

    X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
    mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)
    print("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
    print("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
    print("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
    print("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
    print("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape))
    print("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
    print("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))

    parameters = initialize_velocity_test_case()
    v = initialize_velocity(parameters)
    print("v[\"dW1\"] = " + str(v["dW1"]))
    print("v[\"db1\"] = " + str(v["db1"]))
    print("v[\"dW2\"] = " + str(v["dW2"]))
    print("v[\"db2\"] = " + str(v["db2"]))

    parameters = initialize_adam_test_case()
    v, s = initialize_adam(parameters)
    print("v[\"dW1\"] = " + str(v["dW1"]))
    print("v[\"db1\"] = " + str(v["db1"]))
    print("v[\"dW2\"] = " + str(v["dW2"]))
    print("v[\"db2\"] = " + str(v["db2"]))
    print("s[\"dW1\"] = " + str(s["dW1"]))
    print("s[\"db1\"] = " + str(s["db1"]))
    print("s[\"dW2\"] = " + str(s["dW2"]))
    print("s[\"db2\"] = " + str(s["db2"]))

    parameters, grads, v, s = update_parameters_with_adam_test_case()
    parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t=2)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    print("v[\"dW1\"] = " + str(v["dW1"]))
    print("v[\"db1\"] = " + str(v["db1"]))
    print("v[\"dW2\"] = " + str(v["dW2"]))
    print("v[\"db2\"] = " + str(v["db2"]))
    print("s[\"dW1\"] = " + str(s["dW1"]))
    print("s[\"db1\"] = " + str(s["db1"]))
    print("s[\"dW2\"] = " + str(s["dW2"]))
    print("s[\"db2\"] = " + str(s["db2"]))

    train_X, train_Y = load_dataset()
    # train 3-layer model
    layers_dims = [train_X.shape[0], 5, 2, 1]
    plt.figure(2)
    parameters = model(train_X, train_Y, layers_dims, optimizer="gd")
    predictions = predict(train_X, train_Y, parameters)

    plt.figure(3)
    plt.title("Model with Gradient Descent optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 2.5])
    axes.set_ylim([-1, 1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y.ravel())

    # train 3-layer model
    layers_dims = [train_X.shape[0], 5, 2, 1]
    plt.figure(4)
    parameters = model(train_X, train_Y, layers_dims, beta=0.9, optimizer="momentum")
    predictions = predict(train_X, train_Y, parameters)

    plt.figure(5)
    plt.title("Model with Momentum optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 2.5])
    axes.set_ylim([-1, 1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y.ravel())

    # train 3-layer model
    layers_dims = [train_X.shape[0], 5, 2, 1]
    plt.figure(6)
    parameters = model(train_X, train_Y, layers_dims, optimizer="adam")
    predictions = predict(train_X, train_Y, parameters)

    plt.figure(7)
    plt.title("Model with Adam optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 2.5])
    axes.set_ylim([-1, 1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y.ravel())
