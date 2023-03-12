import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
import seaborn as sns

def compute_optimal_transport(M, r, c, lam, epsilon=1e-5):
    """
    Computes the optimal transport matrix and Sinkhorn distance using the
    Sinkhorn-Knopp algorithm

    Inputs:
        - M : cost matrix (n x m)
        - r : vector of marginals (n, )
        - c : vector of marginals (m, )
        - lam : strength of the entropic regularization
        - epsilon : convergence parameter

    Output:
        - P : optimal transport matrix (n x m)
        - dist : Sinkhorn distance
    """
    n, m = M.shape
    P = np.exp(-lam * M) # K
    # Avoiding poor math condition
    P /= P.sum()
    u = np.zeros(n)
    # Normalize this matrix so that P.sum(1) == r, P.sum(0) == c
    while np.max(np.abs(u - P.sum(1))) > epsilon:
        # Shape (n, )
        u = P.sum(1)
        P *= (r / u).reshape((-1, 1))
        P *= (c / P.sum(0)).reshape((1, -1))
        
    return P, np.sum(P * M)


if __name__ == '__main__':
    base = np.linspace(-4, 4, 128)
    x = np.sin(4 * base)[:, np.newaxis]
    y = np.cos(4 * base)[:, np.newaxis]
    M = pairwise_distances(x, y, metric='euclidean')

    n, m = M.shape
    # Uniform distribution
    r = np.ones(n) / n
    c = np.ones(m) / m
    P, d = compute_optimal_transport(M, r, c, lam=500, epsilon=1e-6)
    # Normalize, so each row sums to 1 (i.e. probability)
    P /= r.reshape((-1, 1))

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title('Measure')
    sns.heatmap(M, square=False, cmap='Reds')
    plt.subplot(1, 2, 2)
    plt.title('Transport')
    sns.heatmap(P, square=False, cmap='Reds')
    plt.show()

    plt.plot(x, label='curve 1')
    plt.plot(y, label='curve 2')
    plt.plot(P.T @ x, label='1 -> 2')
    plt.legend()
    plt.show()
