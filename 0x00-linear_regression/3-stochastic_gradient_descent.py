#!/usr/bin/env python3
import numpy as np
mse_cost = __import__('1-cost_gradient').mse_cost
def stochastic_gradient_descent(X, y, alpha=0.01, epochs=10, seed=0, theta0=None):
    (m, n) = X.shape
    if theta0 is None:
        theta = np.zeros(n)
    else:
        theta = np.array(theta0, dtype=float)
    rng = np.random.default_rng(seed)
    costs = np.zeros(epochs + 1)
    costs[0] = mse_cost(X, y, theta)
    for e in range (epochs):
        order = rng.permutation(m)
        for i in order:
            residual = theta @ X[i] - y[i]
            theta = theta - alpha * residual * X[i]
            costs[e + 1] = mse_cost(X, y, theta)
    return(theta, costs)

