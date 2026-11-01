#!/usr/bin/env python3
import numpy as np
batch_gradient_descent = __import__(
    '2-batch_gradient_descent').batch_gradient_descent

X = np.array([[1., 1.], [1., 2.], [1., 3.]])
y = np.array([1., 2., 3.])

theta, costs = batch_gradient_descent(X, y, alpha=0.1, iterations=500)
print(costs[0], costs[-1])   # cost should decrease monotonically for a stable alpha
