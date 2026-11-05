#!/usr/bin/env python3
import numpy as np
def mse_cost(X, y, theta):
    m = X.shape[0]
    prediction = X @ theta
    error = X @ theta - y
    return float(np.sum(error ** 2) / (2 * m))
def minibatch_gradient_descent(X, y, alpha=0.01, epochs=10, batch_size=32, seed=0, theta0=None):
    (m, n) = X.shape
    if theta0 is None:
       theta = np.zeros(n)
    else:
         theta = np.array(theta0, dtype=float)
    rng = np.random.default_rng(seed)
    costs = np.zeros(epochs + 1)
    costs[0] = mse_cost(X, y, theta)
    for e in range(epochs):
        order = rng.permutation(m)
        for start in range(0, m, batch_size):
            idx = order[start:start + batch_size]
            XB = X[idx]
            yB = y[idx]
            error = XB @ theta - yB
            gradient = (1/XB.shape[0]) * (XB.T @ error)
            theta -= alpha * gradient
        costs[e + 1] = mse_cost(X, y, theta)
    return theta, costs




         

