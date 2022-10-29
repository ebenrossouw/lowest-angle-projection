import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd
import matplotlib.pyplot as plt

def show_data(X:np.ndarray, x:int=0, y:int=1, c:np.ndarray = None):
    plt.scatter(X[:,x], X[:,y], c = c)
    plt.show()

seed = 734658
N = 100

rng = np.random.default_rng(734658)
X = rng.normal([0.0, 2.0, 3.0], [1.0, 1.0, 0.02], (N, 3))
X[0,2] = 100
X[1,2] = 120
X[2,2] = 95
mu =  X.mean(axis = 0)
X = X - mu
show_data(X, 1, 2)
# Z = PCA(n_components = 2).fit_transform(X.numpy())
# show_data(Z)



a = 1 / norm(X, axis = 1)
# n = X.T @ a
# a = a * (-1) ** (pt.sum(X * n, dim = 1) < 0)
n = X.T @ a #- mu
n = n / norm(n)
proj_n = np.outer(np.dot(X, n), n)
Z = X - proj_n

view = np.vstack([X, Z, proj_n, n])
c = np.zeros(N * 3 + 1)
c[N:(N*2)] = 1
c[(N*2):(N*3)] = 2
c[c.size - 1] = 3
show_data(view, 0, 2, c)

show_data(np.outer(np.dot(X, n), n), 1, 2) # pay attention to scale

view = np.vstack((X, n))
plt.scatter(view[0], view[1])
plt.show()

show_data(Z, 1, 2)
show_data(np.vstack((Z, n)), 1, 2, c)

W = Z[3:5]
W[1] = W[0] - W[1] * np.dot(W[0], W[1]) / (norm(W[0])) ** 2
W /= norm(W, axis = 0)
show_data(X @ W.T)

U, D, V = svd(Z, full_matrices = False)
W = V[0:2]
show_data(Z @ W.T)

# Compute angles
theta = np.arccos(np.sum(X * Z, axis = 1) / (norm(Z, axis = 1) * norm(X, axis = 1)))
theta = theta / np.pi * 180



# def lap(X:np.ndarray) -> np.ndarray:
