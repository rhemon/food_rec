import numpy as np
import glob
from scipy.misc import imread, imresize
import random

# Loading and processing dataset
data_dir = "../data/Original/"
food_dir_list = ["mango", "green_apple", "orange", "pear"]
X = np.zeros((40*40*3, 400*len(food_dir_list)))
Y = np.zeros((len(food_dir_list), 400*len(food_dir_list)))
dev_set = np.zeros((40*40*3, 50*len(food_dir_list)))
dev_set_y = np.zeros((len(food_dir_list), 50*len(food_dir_list)))
test_set = np.zeros((40*40*3, 50*len(food_dir_list)))
test_set_y = np.zeros((len(food_dir_list), 50*len(food_dir_list)))
notChosenX = []
XYi = 0
dXYi = 0
tXYi = 0
for food_dir in food_dir_list:
    food_files = glob.glob(data_dir+food_dir+"/*.jpg")
    random.shuffle(food_files)
    notChosenX.append(food_files[500:])
    food_files = food_files[:500]
    for img_name in food_files[:400]:
        try:
            im = imread(img_name, mode='RGB')
            im = imresize(im, (40,40,3))
            im = im.reshape(40*40*3,)
        except ValueError:
            print("Trouble reading in", img_name, "for training set")
        y = np.zeros((len(food_dir_list),))
        y[food_dir_list.index(food_dir)] = 1
        Y[:, XYi] = y 
        X[:, XYi] = im
        XYi+= 1
        print("Training image", img_name, "is added")
    for img_name in food_files[400:450]:
        try:
            im = imread(img_name, mode='RGB')
            im = imresize(im, (40,40,3))
            im = im.reshape(40*40*3,)
        except ValueError:
            print("Trouble reading in", img_name, "for dev set")
        y = np.zeros((len(food_dir_list),))
        y[food_dir_list.index(food_dir)] = 1
        dev_set_y[:, dXYi] = y 
        dev_set[:, dXYi] = im
        dXYi+= 1
        print("Dev set image", img_name, " is added")
    for img_name in food_files[450:500]:
        try:
            im = imread(img_name, mode='RGB')
            im = imresize(im, (40,40,3))
            im = im.reshape(40*40*3,)
        except:
            print("Trouble reading in", img_name, "for test set")
        y = np.zeros((len(food_dir_list),))
        y[food_dir_list.index(food_dir)] = 1
        test_set_y[:, tXYi] = y 
        test_set[:, tXYi] = im
        tXYi += 1
        print("Test image:", img_name, " is added")
print("Train set shape:", X.shape)
print("Y shape:", Y.shape)
print("Dev set shape:", dev_set.shape)
print("Dev set Y shape:", dev_set_y.shape)
print("Test set shape:", test_set.shape)
print("Test set Y shape:", test_set_y.shape)

# Normalising the data
# X = (X - X.mean(axis=1, keepdims=True))/X.var(axis=1, keepdims=True)
# dev_set = (dev_set - dev_set.mean(axis=1, keepdims=True))/dev_set.var(axis=1, keepdims=True)
# test_set = (test_set - test_set.mean(axis=1, keepdims=True))/test_set.var(axis=1, keepdims=True)

X = X/255
dev_set = dev_set/255
test_set = test_set/255

np.save("data/X.npy", X)
np.save("data/Y.npy", Y)
np.save("data/X_dev.npy", dev_set)
np.save("data/Y_dev", dev_set_y)
np.save("data/X_test.npy", test_set)
np.save("data/Y_test.npy", test_set_y)