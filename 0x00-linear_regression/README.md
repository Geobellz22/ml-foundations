Write a function def normal_equations(X, y): that returns the least-squares solution 
θ
⋆
=
arg
⁡
min
⁡
θ
∥
X
θ
−
y
∥
2
θ 
⋆
 =argmin 
θ
​
 ∥Xθ−y∥ 
2
 .

X is a numpy.ndarray of shape (m, n) — the design matrix (if you want an intercept, the caller has already added a column of ones).
y is a numpy.ndarray of shape (m,).
Returns a numpy.ndarray of shape (n,).
You must not call np.linalg.inv. Use np.linalg.solve on the normal equations 
X
⊤
X
θ
=

