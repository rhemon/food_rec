# import required libraries
import numpy as np
import time

def sigmoid(z):
    """
    Sigmoid activation function
    
    :param z:  Accepts the z value that is calculated through linear equation. z could be a
               matrix of real number, or could be a single real number.
    
    :returns:  Matrix/Real number (based on input), where the values have been aplied through
               sigmoid function. Range: 0 < g(z) < 1.
    """
    
    return 1/(1+np.exp(-z))

def init_params(layers):
    # Random initalize parameters
    params = {}
    nl = len(layers)
    for q in range(nl-1):
        params["W"+str(q+1)] = np.random.randn(layers[q+1], layers[q]) * 0.01
        params["b"+str(q+1)] = np.zeros((layers[q+1], 1))

    return params

def forward_prop(X, params, nl):
    cache = {"A0": X}
    for j in range(nl-2):
        cache["Z"+str(j+1)] = np.array(np.dot(params["W"+str(j+1)], cache["A"+str(j)]) + params["b"+str(j+1)])
        cache["A"+str(j+1)] = sigmoid(cache["Z"+str(j+1)])
    cache["Z"+str(nl-1)] = np.dot(params["W"+str(nl-1)], cache["A"+str(nl-2)]) + params["b"+str(nl-1)]
    cache["T"] = np.exp(cache["Z"+str(nl-1)])
    cache["A"+str(nl-1)] = (cache["T"]/np.sum(cache["T"], axis=0, keepdims=True))

    return cache

def compute_cost(cache, Y, m, nl):
    logprobs = -1 * np.sum(np.multiply(np.log(cache["A"+str(nl-1)]), Y), axis=0, keepdims=True)
    cost = np.squeeze((1/(m)) * np.sum(logprobs, axis=1, keepdims=True))

    return cost

def back_prop(cache, params, Y, m, nl):
    grads = {}
    grads["dZ"+str(nl-1)] = cache["A"+str(nl-1)] - Y
    for k in reversed(range(nl-1)):
        grads["dW"+str(k+1)] = 1/(m) * np.dot(grads["dZ"+str(k+1)], cache["A"+str(k)].T)
        grads["db"+str(k+1)] = 1/(m) * np.sum(grads["dZ"+str(k+1)], axis=1, keepdims=True, dtype='float')
        grads["dZ"+str(k)] = np.dot(params["W"+str(k+1)].T, grads["dZ"+str(k+1)]) * (cache["A"+str(k)] * (1 - cache["A"+str(k)]))

    return grads

def gradient_descent(params, grads, learning_rate, nl):
    for l in range(nl-1):
        params["W"+str(l+1)] -= (grads["dW"+str(l+1)] * learning_rate)
        params["b"+str(l+1)] -= (grads["db"+str(l+1)] * learning_rate)

    return params

def model(X, Y, params, layers, learning_rate, iteration, print_cost, iter_count):
    # Useful variables
    m = X.shape[1]
#     n_x = X.shape[0]
#     n_h = X.shape[0]
#     n_y = 4
    costs = []
    iterations = []
    nl = len(layers)
    for i in range(iteration):

        # # Shuffle Training set
        # TrainingSet = np.append(oX, oY, axis=0)
        # np.random.shuffle(np.transpose(TrainingSet))
        # X = TrainingSet[:4800, :]
        # Y = TrainingSet[4800:, :]

        # Forward Propagation
        cache = forward_prop(X, params, nl)

        # Compute cost
        cost = compute_cost(cache, Y, m, nl)

        # Backward Propagation
        grads = back_prop(cache, params, Y, m, nl)

        # Gradient descent
        params = gradient_descent(params, grads, learning_rate, nl)
        
        # Print cost
        if ((i+1) % 100 == 0) and print_cost:
            print("Cost after",(iter_count*iteration)+(i+1),"turn:",cost)
        if i > 1:
            if cost > costs[i-1]:
                print("cost increased at", str((iter_count*iteration)+i)+"th turn")
        costs.append(cost)
        iterations.append(i+1)

    return params, costs, iterations

