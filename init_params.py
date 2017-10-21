import numpy as np
import train

X = np.load("data/X.npy")
params = train.init_params([X.shape[0], 24, 12, 4])
np.save("params.npy", params)