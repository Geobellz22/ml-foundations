#!/usr/bin/env python3
import numpy as np
logistic_cost = __import__('1-logistic_cost').logistic_cost
logistic_gradient = __import__('1-logistic_cost').logistic_gradient

X = np.array([[1., 0.5], [1., -1.5], [1., 2.0]])
y = np.array([1., 0., 1.])
X0, y0 = X.copy(), y.copy()

# 1. theta = 0: cost is log 2 for any data, and the type is float
cost0 = logistic_cost(X, y, np.zeros(2))
print(cost0, type(cost0))

# 2. gradient at theta = 0 equals (1/m) X^T (0.5 - y)
g0 = logistic_gradient(X, y, np.zeros(2))
print(g0, g0.shape)
print(np.allclose(g0, X.T @ (0.5 - y) / 3))

# 3. extreme values must stay finite, with no warnings
with np.errstate(over='raise', divide='raise', invalid='raise'):
    big = np.array([[1.0]])
    print(logistic_cost(big, np.array([0.]), np.array([1000.])))    # ~1000
    print(logistic_cost(big, np.array([1.]), np.array([1000.])))    # ~0
    print(logistic_cost(big, np.array([1.]), np.array([-1000.])))   # ~1000
    print(logistic_gradient(big, np.array([0.]), np.array([1000.])),
          logistic_gradient(big, np.array([1.]), np.array([-1000.])))

# 4. gradient vs numerical estimate
theta = np.array([0.3, -0.7])
eps = 1e-6
num = np.zeros(2)
for j in range(2):
    d = np.zeros(2)
    d[j] = eps
    num[j] = (logistic_cost(X, y, theta + d)
              - logistic_cost(X, y, theta - d)) / (2 * eps)
print(num, logistic_gradient(X, y, theta))
print(np.allclose(num, logistic_gradient(X, y, theta), atol=1e-6))

# 5. same value as the usual cross-entropy for moderate theta
p = 1 / (1 + np.exp(-(X @ theta)))
ce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
print(np.isclose(logistic_cost(X, y, theta), ce))

# 6. inputs unchanged
print(np.array_equal(X, X0), np.array_equal(y, y0))
