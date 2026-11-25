import numpy as np

def compute_canonical_loadings(Xmat, Umat):
    # Constant columns get loading 0 instead of NaN
    Xmat = np.asarray(Xmat, dtype=float)
    Umat = np.asarray(Umat, dtype=float)
    sdX = Xmat.std(0, ddof=1)
    sdU = Umat.std(0, ddof=1)
    sdX = np.where(sdX == 0, 1.0, sdX)
    sdU = np.where(sdU == 0, 1.0, sdU)
    Xstd = (Xmat - Xmat.mean(0)) / sdX
    Ustd = (Umat - Umat.mean(0)) / sdU
    return (Xstd.T @ Ustd) / (Xmat.shape[0] - 1)

def empirical_corr(U, V):
    Uc = U - U.mean(axis=0)
    Vc = V - V.mean(axis=0)
    sdU = Uc.std(axis=0, ddof=1)
    sdV = Vc.std(axis=0, ddof=1)
    cov = np.sum(Uc * Vc, axis=0) / (U.shape[0] - 1)
    return cov / (sdU * sdV)

def compute_trace_scale(X):
    n, p = X.shape
    # S = np.cov(X, rowvar=False)
    # trace_scale = np.trace(S) / p  # Your original: mean variance
    dim_scale = np.sqrt(p / n)     # √(p/n)
    return dim_scale  # Combined scale

def snap_to_power10(x):
    if x <= 0:
        return 1.0
    exponent = int(np.round(np.log10(x)))
    return 10 ** exponent

def make_lambda_grid(X, min_shift=-1, max_shift=1):
    raw_scale = compute_trace_scale(X)
    base_power = snap_to_power10(raw_scale)
    grid = []
    base_exponent = int(np.log10(base_power))
    for shift in range(min_shift, max_shift + 1):
        exponent = base_exponent + shift
        grid.append(1 * 10**exponent)
        grid.append(5 * 10**exponent)
    return np.sort(np.array(grid)), base_power


