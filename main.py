import numpy as np
import matplotlib.pyplot as plt
from functions.tools import *

# Load files
train_base  = np.load("basetrain.npy") / 255
train_label = np.load('labeltrain.npy')
nb_classes = 10

# Get train base dimension
dim = train_base.shape

# Define weight matrix
w = mlp1def(nb_classes, dim[0])

# Get the output of the neural
y = mlp1run(train_base[:, 0 : 5], w)
# print('y len : ' + str(len(y))) 
# print(y)

# Get score
score = getScore(y, train_label)
