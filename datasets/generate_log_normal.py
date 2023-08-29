import numpy as np
import random
import pickle
d = 5
n = 1000
X = np.random.rand(n, d)
y = np.random.lognormal(1,.5,  n)

with open("datasets/tail.pkl", "wb") as f:
    pickle.dump((X, y), f)