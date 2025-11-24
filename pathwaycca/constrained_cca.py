import numpy as np
import os
from scipy.stats import norm
from numpy.linalg import eigh, lstsq
from scipy.linalg import null_space

from .utils import compute_canonical_loadings
from .utils import empirical_corr


def solve_ccca(
    X_input,
    Y_input,
    C=None,
    d=None,
    lambda_x=1e-6,
    lambda_y=1e-6,
    n_components=None
):
    """
    CCCA / RCCA with optional linear equality constraint CA = d.
    If C=None or d=None or C has zero rows -> RCCA.
    """

    X = np.asarray(X_input, dtype=float)
    Y = np.asarray(Y_input, dtype=float)

    # Mean Center
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)

    n, p = X.shape
    _, q = Y.shape
    if n_components is None:
        n_components = min(p, q)

    # L2 Regularization
    Sxx = np.cov(X, rowvar=False) + lambda_x * np.eye(p)
    Syy = np.cov(Y, rowvar=False) + lambda_y * np.eye(q)
    Sxy = np.cov(X, Y, rowvar=False)[:p, p:]
    Sxx = (Sxx + Sxx.T) / 2 + 1e-10 * np.eye(p)
    Syy = (Syy + Syy.T) / 2 + 1e-10 * np.eye(q)

    if C is None or d is None or len(np.atleast_2d(C)) == 0:
        eig_x, Ux = eigh(Sxx)
        eig_y, Uy = eigh(Syy)
        eig_x = np.clip(eig_x, 1e-10, None)
        eig_y = np.clip(eig_y, 1e-10, None)
        Sxx_inv_sqrt = Ux @ np.diag(1/np.sqrt(eig_x)) @ Ux.T
        Syy_inv_sqrt = Uy @ np.diag(1/np.sqrt(eig_y)) @ Uy.T

        T = Sxx_inv_sqrt @ Sxy @ Syy_inv_sqrt
        U, s, Vt = np.linalg.svd(T)
        r = s[:n_components]

        A = Sxx_inv_sqrt @ U[:, :n_components]
        B = Syy_inv_sqrt @ Vt.T[:, :n_components]
        U_scores = X @ A
        V_scores = Y @ B
        r_emp = empirical_corr(U_scores[:, :n_components], V_scores[:, :n_components])
        return A, B, r, r_emp

    C = np.asarray(C, dtype=float)
    d = np.asarray(d, dtype=float).ravel()
    N = null_space(C) 

    # Particular solution A0
    a0, *_ = lstsq(C, d, rcond=None)
    a0 = a0.flatten()

    SxNa0 = Sxx @ a0
    alpha = N.T @ SxNa0
    gamma = np.linalg.solve(N.T @ Sxx @ N, alpha)
    a0_proj = a0 - N @ gamma 

    # Reduced covariances
    Sxx_N = N.T @ Sxx @ N
    Sxy_N = N.T @ Sxy
    eig_x, Ux = eigh(Sxx_N)
    eig_y, Uy = eigh(Syy)
    eig_x = np.clip(eig_x, 1e-10, None)
    eig_y = np.clip(eig_y, 1e-10, None)

    SxxN_inv_sqrt = Ux @ np.diag(1/np.sqrt(eig_x)) @ Ux.T
    Syy_inv_sqrt = Uy @ np.diag(1/np.sqrt(eig_y)) @ Uy.T

    # Canonical correlation via SVD
    T = SxxN_inv_sqrt @ Sxy_N @ Syy_inv_sqrt
    U, s, Vt = np.linalg.svd(T)
    r = s[:n_components]
    A_reduced = SxxN_inv_sqrt @ U[:, :n_components]
    B = Syy_inv_sqrt @ Vt.T[:, :n_components]
    A_full = a0_proj[:, None] + N @ A_reduced
    U_scores = X @ A_full
    V_scores = Y @ B
    r_emp = empirical_corr(U_scores[:, :n_components], V_scores[:, :n_components])
    return A_full, B, r, r_emp


def ccca(
    X_input,
    Y_input,
    C,
    d,
    lambda_x=1e-6,
    lambda_y=1e-6,
    n_components=None
):
    """
    Full Constrained CCA output including:
        - A, B, r
        - U, V, Z
        - U_resid, V_resid, Z_resid
        - loadings_X, loadings_Y
        - constrained_idx
    """

    X = np.asarray(X_input, float)
    Y = np.asarray(Y_input, float)

    n, p = X.shape
    _, q = Y.shape

    if n_components is None:
        n_components = min(p, q)
    n_components = int(n_components)
    assert n_components >= 1

    A, B, _, r = solve_ccca(
        X_input, Y_input,
        C=C, d=d,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        n_components=n_components
    )

    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)

    U = Xc @ A
    V = Yc @ B
    Z = U + V

    C = np.asarray(C, float)
    constrained_idx = np.where(C.sum(axis=0) != 0)[0]

    if len(constrained_idx) > 0:
        X_sub = Xc[:, constrained_idx]
        Q, _ = np.linalg.qr(X_sub) 
        P = Q @ Q.T
        U_resid = U - P @ U
        V_resid = V - P @ V
        Z_resid = Z - P @ Z
    else:
        U_resid = U.copy()
        V_resid = V.copy()
        Z_resid = Z.copy()

    loadings_X = compute_canonical_loadings(Xc, U)
    loadings_Y = compute_canonical_loadings(Yc, V)
    
    return {
        "A": A,
        "B": B,
        "r": r,
        "U": U,
        "V": V,
        "Z": Z,
        "U_resid": U_resid,
        "V_resid": V_resid,
        "Z_resid": Z_resid,
        "loadings_X": loadings_X,
        "loadings_Y": loadings_Y,
        "constrained_idx": constrained_idx,
        "n_components": n_components
    }

