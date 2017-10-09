import train_nn
import numpy as np
import sys

X = np.load("data/X.npy")
Y = np.load("data/Y.npy")

params = np.load("params.npy").item()
params, costs, iters = train_nn.model(X, Y, params, [X.shape[0], 4800, 4800, 4800, 4800, 4800, 4], 0.008, 100, True, int(sys.argv[1]))
np.save("params.npy", params)
