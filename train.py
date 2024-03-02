import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# Create a dataset
x, y = make_regression(n_samples=100, n_features=1, noise=10)
y = y.reshape(y.shape[0], 1)

#matrice X

X = np.hstack([x, np.ones(x.shape)])
theta = np.random.randn(2, 1)

#model
def model(X, theta):
    return X.dot(theta)

#cost function

def cost_function(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta) - y)**2)

#gradient descent

def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)

def gradient_descent(X, y, theta, learning_rate, n_iterations):
    cost_history = np.zeros(n_iterations)
    for i in range(0, n_iterations):
        theta = theta - learning_rate * grad(X, y, theta)
        cost_history[i] = cost_function(X, y, theta)

    return theta, cost_history

def r2_score(y, pred):
    u = ((y - pred)**2).sum()
    v = ((y - y.mean())**2).sum()
    return 1 - u/v


theta_final, cost_history = gradient_descent(X, y, theta, 0.01, 1000)

prediction = model(X, theta_final)

coef_determination = r2_score(y, prediction)

print(coef_determination)

#scatter new data
# plt.scatter(x, y)
# plt.plot(x, prediction, c='r')
# plt.plot(range(1000), cost_history, c='g')
# plt.show()
