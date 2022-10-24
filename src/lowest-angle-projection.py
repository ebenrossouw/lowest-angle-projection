# import random
import torch as pt
from torch import nn, optim
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

def show_data(X:pt.Tensor or np.ndarray, x:int=0, y:int=1):
    if isinstance(X, pt.Tensor):
        X = X.numpy()
    plt.scatter(X[:,x], X[:,y])
    plt.show()

seed = 734658
N = 100

pt.manual_seed(seed)
X = pt.distributions.Normal(pt.tensor([0.0, 2.0, 3.0]), pt.tensor([1.0, 1.0, 0.02])).sample((N,))
X = X - X.mean(axis = 0)
# show_data(X)

Z = PCA(n_components = 2).fit_transform(X.numpy())
# show_data(Z)

X[0,2] = 100
X[1,2] = 120
X[2,2] = 95
Z = PCA(n_components = 2).fit_transform(X.numpy())
show_data(Z)

def lap(X:pt.Tensor, epsilon:float = 0.00001, seed:int = None, max_iter:int = 1_000) -> pt.Tensor:
    K = 2
    if seed is not None:
        pt.manual_seed(seed)
    X = X - X.mean(axis = 0) # center data
    W = pt.distributions.Uniform(-1.0, 1.0).sample((X.shape[1], K))
    W.requires_grad_(True)
    opt = optim.SGD([W], lr = 1.0, momentum = 0.9)

    for t in range(max_iter):
        W_prev = W.data.clone()
        Y = X @ W @ pt.linalg.solve(W.T @ W, W.T)
        R = pt.sum(X * Y, dim = 1) / (pt.linalg.norm(Y, dim = 1) * pt.linalg.norm(X, dim = 1))
        R = R.mean()

        opt.zero_grad()
        R.backward()
        opt.step()
        if (pt.abs(W.data - W_prev) < epsilon).all():
            break

    if t + 1 == max_iter:
        print('Warning: Max iterations reached')
    W = W.detach()
    W[:,1] = W[:,0] - W[:,1] * pt.dot(W[:,0], W[:,1]) / pt.norm(W[:,0]) ** 2
    W /= pt.linalg.vector_norm(W, dim=0)
    return W

W = lap(X, epsilon=1.0e-4, seed = seed, max_iter = 1_000)
show_data(X @ W)