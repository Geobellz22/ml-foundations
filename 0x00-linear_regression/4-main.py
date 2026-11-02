#!/usr/bin/env python3
import numpy as np
standardize = __import__('4-feature_scaling').standardize
add_intercept = __import__('4-feature_scaling').add_intercept

# Task example: standardize
X = np.array([[1., 5.], [3., 5.]])
X_copy = X.copy()
Xn, mu, sigma = standardize(X)
print(Xn)
print(mu, sigma)
print(np.array_equal(X, X_copy))      # input unchanged

# Task example: add_intercept
A = np.array([[2.], [3.]])
A_copy = A.copy()
print(add_intercept(A))
print(np.array_equal(A, A_copy))      # input unchanged

# Integer input should still give float results
Xi, _, _ = standardize(np.array([[1, 5], [3, 5]]))
print(Xi.dtype)

# On random data: mean about 0 and std about 1 per column
rng = np.random.default_rng(0)
R = rng.normal(5, 3, size=(100, 3))
Rn, _, _ = standardize(R)
print(Rn.mean(axis=0).round(6), Rn.std(axis=0).round(6))

# Shape and content of add_intercept
B = add_intercept(R)
print(B.shape, np.array_equal(B[:, 1:], R), np.all(B[:, 0] == 1))
