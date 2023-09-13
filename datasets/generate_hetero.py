import numpy as np
import random
import pickle
import numpy as np
d = 1
n = 1000
X = []
y = np.asarray([])

for i in range(n):
    feat = np.random.rand(d)

    y = np.concatenate((y, np.random.normal(scale = np.abs(feat), size=10)))
    X.extend([feat] * 10)

with open("datasets/hetero.pkl", "wb") as f:
    pickle.dump((np.asarray(X), np.asarray(y)), f)