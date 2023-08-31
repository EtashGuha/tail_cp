import numpy as np
import matplotlib.pyplot as plt

# Parameters of the log-normal distribution
mu =1 # Mean of the logarithm of the values
sigma = .5  # Standard deviation of the logarithm of the values

# Number of samples
num_samples = 1000

# Generate random samples from the log-normal distribution
samples = np.random.lognormal(mean=mu, sigma=sigma, size=num_samples)

# Plot a histogram of the samples
plt.hist(samples, bins=30, density=True, alpha=0.6, color='b')

# Plot the probability density function (PDF) for comparison
x = np.linspace(0, np.max(samples), 1000)
pdf = (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma ** 2))
plt.plot(x, pdf, 'r-', lw=2, label='Log-Normal PDF')

plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.title('Random Samples from a Right-Fat-Tailed Distribution (Log-Normal)')
plt.savefig("fatty.png")