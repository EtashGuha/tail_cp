import numpy as np
import pickle
import matplotlib.pyplot as plt
# Number of samples
num_samples = 1000

# Initialize arrays to store (x, y) pairs
x_values = np.random.uniform(-1.5, 1.5, num_samples)

y_values = []

# Define the functions f(x), g(x), and sigma^2(x)
def f(x):
    return  (x - 1) ** 2 * (x + 1)

def g(x):
    return 4 * np.sqrt(x + 0.5) if x >= -0.5 else 0

def sigma_squared(x):
    return 1/4 + abs(x)

# Generate (x, y) pairs
for x in x_values:
    mean_y =  f(x) - g(x)  # Mean of the distribution
    variance = sigma_squared(x)  # Variance of the distribution

    other_mean_y = f(x) + g(x)
    # Sample y from the specified normal distribution
    weight = np.random.choice([0, 1])
    one_y = np.random.normal(loc= mean_y, scale=np.sqrt(variance))
    two_y = np.random.normal(loc= other_mean_y, scale=np.sqrt(variance))

    y_values.append(weight * one_y + (1-weight) * two_y)

with open("datasets/lei.pkl", "wb") as f:
    pickle.dump((np.expand_dims(x_values, axis=1), np.asarray(y_values)), f)