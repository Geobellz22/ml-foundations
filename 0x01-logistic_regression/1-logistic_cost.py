#!/usr/bin/env python3
import numpy as np
sigmoid = __import__('0-sigmoid').sigmoid
def logistic_cost(X, y, theta):
    z = X @ theta
    m = X.shape[0]
    loss = np.logaddexp(0.0, z) - y * z
    return float(np.mean(loss))
def logistic_gradient(X, y, theta):
    m = X.shape[0]
    z = X @ theta
    h = sigmoid(z)
    return (X.T @ (h - y)) / m
