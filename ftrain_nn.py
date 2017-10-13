import train
import numpy as np
import sys

X = np.load("data/X.npy")
Y = np.load("data/Y.npy")

params = np.load("params.npy").item()
params, costs, iters = train.model(X, Y, params, [X.shape[0], 12, 4], 0.01, 100, True, int(sys.argv[1]))
np.save("params.npy", params)
