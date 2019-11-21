import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1978)

X = np.random.choice(np.linspace(0, 20, 10000), size=200, replace=False).reshape(-1, 1)
y = np.sin(X) + np.random.normal(scale=0.3, size=X.shape)

X_test = np.linspace(0, 20, 100).reshape(-1, 1)
y_test = np.sin(X_test)

def GP_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    query_idx = np.argmax(std)
    return query_idx, X[query_idx]

def plot_pred(regressor, X_test, y_test, X_training, y_training, x_new=None, y_new=None):
    plt.figure()
    plt.plot(X_test, y_test, "--r", label="gt")
    mean, std = regressor.predict(X_test, return_std=True)
    mean = mean.reshape(-1)
    std = std.reshape(-1)
    plt.plot(X_test, mean, "-", label="pred")
    plt.fill_between(X_test.reshape(-1), mean + std, mean - std, alpha=0.1)
    plt.plot(X_training, y_training, 'gx', label="training")
    if x_new is not None:
        plt.plot(x_new, y_new, 'x', label="new_query")
    plt.legend()

from modAL.models import ActiveLearner
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF

n_initial = 5
initial_idx = np.random.choice(range(len(X)), size=n_initial, replace=False)
X_training, y_training = X[initial_idx], y[initial_idx]

kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
         + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))

regressor = ActiveLearner(
    estimator=GaussianProcessRegressor(kernel=kernel),
    query_strategy=GP_regression_std,
    # X_training=X_training.reshape(-1, 1), y_training=y_training.reshape(-1, 1)
    X_training=None, y_training=None
)

plot_pred(regressor, X_test, y_test, X_training, y_training)

# active learning
query_training = []
n_queries = 5
for idx in range(n_queries):
    query_idx, query_instance = regressor.query(X)
    regressor.teach(X[query_idx].reshape(1, -1), y[query_idx].reshape(1, -1))
    query_training.append(query_idx)
    plot_pred(regressor, X_test, y_test, X_training, y_training, X[query_idx], y[query_idx])
    X_training = np.concatenate([X_training, X[query_idx].reshape(1, -1)], axis=0)
    y_training = np.concatenate([y_training, y[query_idx].reshape(1, -1)], axis=0)

plt.show()
print()

plt.figure()
plt.plot(X, y, ".", label="all samples")
plt.plot(X_test, y_test, "--r", label="gt")
plt.legend()

