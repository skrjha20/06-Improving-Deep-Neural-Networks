import numpy as np
import matplotlib.pyplot as plt
from reg_utils import *
from testcases import *

def model(X, Y, learning_rate=0.3, num_iterations=30000, print_cost=True, lambd=0, keep_prob=1):
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 20, 3, 1]

    parameters = initialize_parameters(layers_dims)
    for i in range(0, num_iterations):
        if keep_prob == 1:
            a3, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)

        if lambd == 0:
            cost = compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)

        assert (lambd == 0 or keep_prob == 1)
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters

def compute_cost_with_regularization(A3, Y, parameters, lambd):
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    cross_entropy_cost = compute_cost(A3, Y)
    L2_regularization_cost = (1./m)*(lambd/2)*(np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
    cost = cross_entropy_cost + L2_regularization_cost
    return cost


def backward_propagation_with_regularization(X, Y, cache, lambd):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    dZ3 = A3 - Y

    dW3 = 1. / m * (np.dot(dZ3, A2.T) + lambd * W3)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * (np.dot(dZ2, A1.T) + lambd * W2)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * (np.dot(dZ1, X.T) + lambd * W1)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    return gradients


def forward_propagation_with_dropout(X, parameters, keep_prob=0.5):
    np.random.seed(1)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    D1 = np.random.rand(A1.shape[0], A1.shape[1])  # Step 1: initialize matrix D1 = np.random.rand(..., ...)
    D1 = D1 < keep_prob  # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
    A1 = np.multiply(A1, D1)  # Step 3: shut down some neurons of A1
    A1 /= keep_prob  # Step 4: scale the value of neurons that haven't been shut down

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    D2 = np.random.rand(A2.shape[0], A2.shape[1])  # Step 1: initialize matrix D2 = np.random.rand(..., ...)
    D2 = D2 < keep_prob  # Step 2: convert entries of D2 to 0 or 1 (using keep_prob as the threshold)
    A2 = np.multiply(A2, D2)  # Step 3: shut down some neurons of A2
    A2 /= keep_prob  # Step 4: scale the value of neurons that haven't been shut down

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    return A3, cache

def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)
    dA2 = np.multiply(dA2, D2)  # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
    dA2 /= keep_prob  # Step 2: Scale the value of neurons that haven't been shut down

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dA1 = np.multiply(dA1, D1)  # Step 1: Apply mask D1 to shut down the same neurons as during the forward propagation
    dA1 /= keep_prob  # Step 2: Scale the value of neurons that haven't been shut down

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    return gradients

if __name__ == "__main__":

    plt.figure(1)
    train_X, train_Y, test_X, test_Y = load_2D_dataset()

    plt.figure(2)
    parameters = model(train_X, train_Y)
    print("On the training set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)

    plt.figure(3)
    plt.title("Model without regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y.ravel())

    A3, Y_assess, parameters = compute_cost_with_regularization_test_case()
    print("cost = " + str(compute_cost_with_regularization(A3, Y_assess, parameters, lambd=0.1)))

    X_assess, Y_assess, cache = backward_propagation_with_regularization_test_case()
    grads = backward_propagation_with_regularization(X_assess, Y_assess, cache, lambd=0.7)
    print("dW1 = " + str(grads["dW1"]))
    print("dW2 = " + str(grads["dW2"]))
    print("dW3 = " + str(grads["dW3"]))

    plt.figure(4)
    parameters = model(train_X, train_Y, lambd=0.7)
    print("On the train set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)

    plt.figure(5)
    plt.title("Model with L2-regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y.ravel())

    X_assess, parameters = forward_propagation_with_dropout_test_case()
    A3, cache = forward_propagation_with_dropout(X_assess, parameters, keep_prob=0.7)
    print("A3 = " + str(A3))

    X_assess, Y_assess, cache = backward_propagation_with_dropout_test_case()
    gradients = backward_propagation_with_dropout(X_assess, Y_assess, cache, keep_prob=0.8)
    print("dA1 = " + str(gradients["dA1"]))
    print("dA2 = " + str(gradients["dA2"]))

    plt.figure(6)
    parameters = model(train_X, train_Y, keep_prob=0.86, learning_rate=0.3)
    print("On the train set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)

    plt.figure(7)
    plt.title("Model with dropout")
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y.ravel())