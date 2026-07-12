import numpy as np

def pagerank(M, d: float = 0.85):
    N = M.shape[1]
    w = np.ones(N) / N
    M_hat = d * M
    v = M_hat @ w + (1 - d) / N
    while np.linalg.norm(w - v) >= 1e-10:
        w = v
        v = M_hat @ w + (1 - d) / N
    return v

M = np.array([[0, 0, 0, .25],
              [0, 0, 0, .5],
              [1, 0.5, 0, .25],
              [0, 0.5, 1, 0]])
v = pagerank(M, 0.85)
print("PageRank vector:", v)
