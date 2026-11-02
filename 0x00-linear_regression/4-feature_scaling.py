#!/usr/bin/env python3
import numpy as np
def standardize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma
def add_intercept(X):
    m = X.shape[0]
    ones = np.ones((m, 1), dtype=X.dtype)
    return np.hstack((ones, X))


