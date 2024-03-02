import numpy as np
import matplotlib.pyplot as plt

def model(X, theta):
    return X.dot(theta)

def cost_function(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta) - y)**2)

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


def main():
    data = np.genfromtxt('./data.csv', delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    #matrice X
    mean_x = np.mean(x, axis=0)
    std_x = np.std(x, axis=0)
    x_normalized = (x - mean_x) / std_x

    X = np.hstack([x_normalized, np.ones(x.shape)])
    theta = np.zeros((2, 1))
    theta_final, cost_history = gradient_descent(X, y, theta, 0.05, 1000)
    print("Accuracy : ", r2_score(y, model(X, theta_final)))
    plt.scatter(x, y)
    plt.plot(x, model(X, theta_final), c='r')
    plt.show()
    np.save('finalTheta.npy', theta_final)
    np.save('mean_x.npy', mean_x)
    np.save('std_x.npy', std_x)

if __name__ == "__main__":
    main()