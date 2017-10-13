import numpy as np

params = np.load("params.npy").item()
X = np.load("data/X.npy")
Y = np.load("data/Y.npy")
dev_set = np.load("data/X_dev.npy")
dev_set_y = np.load("data/Y_dev.npy")

def sigmoid(z):
    """
    Sigmoid activation function
    
    :param z:  Accepts the z value that is calculated through linear equation. z could be a
               matrix of real number, or could be a single real number.
    
    :returns:  Matrix/Real number (based on input), where the values have been aplied through
               sigmoid function. Range: 0 < g(z) < 1.
    """
    return 1/(1+np.exp(-z))

def predict(X, params):
    L = int(len(list(params.keys()))/2)
    cache = {"A0": X}
    for i in range(L):
        cache["Z"+str(i+1)] = np.dot(params["W"+str(i+1)], cache["A"+str(i)]) + params["b"+str(i+1)]
        cache["A"+str(i+1)] = sigmoid(cache["Z"+str(i+1)])

    return np.where(cache["A"+str(L)]==np.max(cache["A"+str(L)], axis=0),1,0)

yHat = predict(X, params)
print("Accuracy with training set,",str(((np.sum(yHat == Y))/(Y.shape[0]*X.shape[1]))*100) + "%")

yTest = predict(dev_set, params)
print("Accuracy with dev set,",str(((np.sum(yTest == dev_set_y))/(dev_set_y.shape[0]*dev_set.shape[1])*100)) + "%")

# # check out 500 mangos how many it can recognise
# mangoExamples = np.zeros((4800, 450))
# mangoExamples[:, 0:400] = X[:,0:400]
# mangoExamples[:, 400:450] = dev_set[:, 0:50]
# mangoPredict = predict(mangoExamples, params)
# expectedOutput_mango = np.zeros((4, mangoExamples.shape[1]))
# expectedOutput_mango[0,:] = 1
# print("Accuracy in recognising mango:", (np.sum(mangoPredict == expectedOutput_mango)/(4*450)) * 100)

# # check out 500 apples how many it can recognise
# appleExamples = np.zeros((4800,450))
# appleExamples[:, 0:400] = X[:,400:800]
# appleExamples[:, 400:450] = dev_set[:,50:100]
# applePredict = predict(appleExamples, params)
# expectedOutput_apple = np.zeros((4, appleExamples.shape[1]))
# expectedOutput_apple[1,:] = 1
# print("Accuracy in recognising apple:", (np.sum(applePredict == expectedOutput_apple)/(4*450)) * 100)

# # check out 500 apples how many it can recognise
# orangeExamples = np.zeros((4800,450))
# orangeExamples[:, 0:400] = X[:,800:1200]
# orangeExamples[:, 400:450] = dev_set[:, 100:150]
# orangePredict = predict(orangeExamples, params)
# expectedOutput_orange = np.zeros((4, orangeExamples.shape[1]))
# expectedOutput_orange[2,:] = 1
# print("Accuracy in recognising orange:", (np.sum(orangePredict == expectedOutput_orange)/(4*450)) * 100)

# # check out 500 apples how many it can recognise
# pearExamples = np.zeros((4800,450))
# pearExamples[:, 0:400] = X[:,1200:1600]
# pearExamples[:, 400:450] = dev_set[:,150:]
# pearPredict = predict(pearExamples, params)
# expectedOutput_pear = np.zeros((4, pearExamples.shape[1]))
# expectedOutput_pear[3,:] = 1
# print("Accuracy in recognising pear:", (np.sum(pearPredict == expectedOutput_pear)/(4*450)) * 100)
