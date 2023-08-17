import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tail import get_px, get_above_quantile, max_entropy_distribution
from tqdm import tqdm

# Load the diabetes dataset
data = load_diabetes()
X = data.data
y = data.target
K = 4
alpha = .1
X_scaler = StandardScaler()
X_normalized = X_scaler.fit_transform(X)

# Normalize the target variable using StandardScaler
y_scaler = StandardScaler()
y_normalized = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)
# Normalize the features using StandardScaler


# Train a machine learning model (e.g., Linear Regression)
models = []
for index in range(K):
    model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=5000)
    model.fit(X_train, np.power(y_train, index+1))
    plt.clf()
    plt.plot(model.loss_curve_)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("losscurve_{}.png".format(index))
    models.append(model)

y_preds = []
for model in models:
    y_preds.append(model.predict(X_test))


all_moments = np.stack(y_preds).T

probs = []
for index in tqdm(range(len(all_moments))):
    try:
        probs.append(get_px(y_test[index], all_moments[index]))
    except:
        print("none")
        pass

threshold =  np.percentile(probs, alpha)
breakpoint()
def get_blah(index):
    trial_point = X_test[index]
    true_val = y_test[index]

    moments = []
    for model in models:
        moments.append(model.predict(trial_point.reshape(1, -1)))

    get_above_quantile(threshold, moments)
    vals, probs = max_entropy_distribution(moments)
    plt.clf()
    plt.plot(vals, probs)
    plt.savefig("pls.png")