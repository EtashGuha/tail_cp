import numpy as np
import random
import pickle
d = 5
n = 1000
random_vector = np.random.rand(d)
other_random_vector = np.random.rand(d)

inputs = np.random.rand(n, d)
y = []
X = []
for i in range(n):
    y.append(-1 + np.random.normal(0, .05 ,1)),
    y.append(1 + np.random.normal(0, .05 ,1))
    X.append(inputs[i] + np.random.normal(0, .05 ,inputs[i].shape))
    X.append(inputs[i])
y = np.asarray(y)
X = np.asarray(X)
with open("datasets/bimodal.pkl", "wb") as f:
    pickle.dump((X, y), f)