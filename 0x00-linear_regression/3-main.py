#!/usr/bin/env python3
import numpy as np
stochastic_gradient_descent = __import__(
    '3-stochastic_gradient_descent').stochastic_gradient_descent
normal_equations = __import__('0-normal_equations').normal_equations

X = np.array([[1., 1.], [1., 2.], [1., 3.]])
y = np.array([1., 2., 3.])

# Task example
theta, costs = stochastic_gradient_descent(X, y, alpha=0.01, epochs=20)
print(costs[0], costs[-1])
print(theta)

# Shapes
print(theta.shape, costs.shape)

# Final cost should be lower than the starting cost
print(costs[-1] < costs[0])

# Same seed gives identical results, a different seed gives a different path
t1, _ = stochastic_gradient_descent(X, y, epochs=20, seed=1)
t2, _ = stochastic_gradient_descent(X, y, epochs=20, seed=1)
t3, _ = stochastic_gradient_descent(X, y, epochs=20, seed=2)
print(np.array_equal(t1, t2), np.array_equal(t1, t3))

# theta0 must not be modified
theta0 = np.array([5., 5.])
stochastic_gradient_descent(X, y, epochs=5, theta0=theta0)
print(theta0)

# epochs=0 returns the starting theta and a costs array of length 1
theta, costs = stochastic_gradient_descent(X, y, epochs=0)
print(theta, costs.shape)

# Noisy data from 0-main.py: SGD should land near the normal equations answer
rng = np.random.default_rng(0)
Xn = np.c_[np.ones(100), rng.normal(size=(100, 2))]
yn = Xn @ np.array([4.0, 3.0, -2.0]) + 0.1 * rng.normal(size=100)
theta, costs = stochastic_gradient_descent(Xn, yn, alpha=0.01, epochs=200)
print(theta.round(2))
print(normal_equations(Xn, yn).round(2))
