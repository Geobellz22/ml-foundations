#!/usr/bin/env python3
import numpy as np
normal_equations = __import__('0-normal_equations').normal_equations

rng = np.random.default_rng(0)
X = np.c_[np.ones(100), rng.normal(size=(100, 2))]
y = X @ np.array([4.0, 3.0, -2.0]) + 0.1 * rng.normal(size=100)
print(normal_equations(X, y).round(2))

