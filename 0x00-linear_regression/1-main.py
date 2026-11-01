#!/usr/bin/env python3
import numpy as np
mse_cost = __import__('1-cost_gradient').mse_cost
mse_gradient = __import__('1-cost_gradient').mse_gradient
mse_cost(X, y, np.zeros(2))
mse_gradient(X, y, np.zeros(2))
print(mse_cost(X, y, np.zeros(2)))
print(mse_gradient(X, y, np.zeros(2)))

