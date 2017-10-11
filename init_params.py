import numpy as np
import train_nn

X = np.load("data/X.npy")
params = train_nn.init_params([X.shape[0], 4800, 4800, 4800, 4800, 4])
np.save("params.npy", params)