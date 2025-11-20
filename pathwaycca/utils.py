import numpy as np

def compute_canonical_loadings(Xmat, Umat):
    Xstd = (Xmat - Xmat.mean(0)) / Xmat.std(0, ddof=1)
    Ustd = (Umat - Umat.mean(0)) / Umat.std(0, ddof=1)
    return (Xstd.T @ Ustd) / (Xmat.shape[0] - 1)

def empirical_corr(U, V):
    Uc = U - U.mean(axis=0)
    Vc = V - V.mean(axis=0)
    sdU = Uc.std(axis=0, ddof=1)
    sdV = Vc.std(axis=0, ddof=1)
    cov = np.sum(Uc * Vc, axis=0) / (U.shape[0] - 1)
    return cov / (sdU * sdV)

