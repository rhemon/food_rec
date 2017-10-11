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
        # Forward Propagation
        cache = {"A0": X}
        for j in range(nl-1):
            cache["Z"+str(j+1)] = np.dot(params["W"+str(j+1)], cache["A"+str(j)]) + params["b"+str(j+1)]
            cache["A"+str(j+1)] = sigmoid(cache["Z"+str(j+1)])
        
        logprobs = np.multiply(np.log(cache["A"+str(nl-1)]), Y) + np.multiply(np.log(1-cache["A"+str(nl-1)]), 1-Y)
        cost = np.squeeze((-1/(m*layers[-1])) * np.sum(logprobs))

        # Backward Propagation
        grads = {}
        grads["dZ"+str(nl-1)] = cache["A"+str(nl-1)] - Y
        for k in reversed(range(nl-1)):
            grads["dW"+str(k+1)] = 1/(m) * np.dot(grads["dZ"+str(k+1)], cache["A"+str(k)].T)
            grads["db"+str(k+1)] = 1/(m) * np.sum(grads["dZ"+str(k+1)], axis=1, keepdims=True, dtype='float')
            grads["dZ"+str(k)] = np.dot(params["W"+str(k+1)].T, grads["dZ"+str(k+1)]) * (cache["A"+str(k)] * (1 - cache["A"+str(k)]))

        # Gradient descent
        for l in range(nl-1):
            params["W"+str(l+1)] -= (grads["dW"+str(l+1)] * learning_rate)
            params["b"+str(l+1)] -= (grads["db"+str(l+1)] * learning_rate)
        
        if ((i+1) % 20 == 0) and print_cost:
            print("Cost after",(iter_count*100)+(i+1),"turn:",cost)
        if i > 1:
            if cost > costs[i-1]:
                print("cost increased at", str((iter_count*100)+i)+"th turn")
        costs.append(cost)
        iterations.append(i+1)
    return params, costs, iterations

