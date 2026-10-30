#!/usr/bin/env python3
import numpy as np
def mse_cost(X, y, theta):
    m = X.shape[0]
    prediction = X @ theta
    error = prediction - y
    cost = (1 / (2 * m)) * np.sum (error ** 2)
    return float(cost)
def mse_gradient(X, y, theta):
    m = X.shape[0]
    prediction = X @ theta
    error = prediction - y
    gradient = (1/m) * (X.T @ error)
    return gradient
