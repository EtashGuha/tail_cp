import numpy as np
import random
import pickle
d = 5
n = 1000
random_vector = np.random.rand(d)
other_random_vector = np.random.rand(d)

X = np.random.rand(n, d)
y = []
for i in range(n):
    coin_flip = random.choice(['first', 'second'])
    if coin_flip == "first":
        y.append(np.dot(random_vector, X[i]))
    else: 
        y.append(np.dot(other_random_vector, X[i]))
y = np.asarray(y)
with open("datasets/bimodal.pkl", "wb") as f:
    pickle.dump((X, y), f)