import numpy as np
from numpy.linalg import eigh
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

from .utils import compute_canonical_loadings
from .utils import empirical_corr

def canonical_test_score(X_test, Y_test, A, B, n_components=3):
    """
    Mean absolute canonical correlation on held-out data.
    Kept for backward compatibility; same as canonical_score with n_components=3 default.
    """
    return canonical_score(X_test, Y_test, A, B, n_components=n_components)

def cv_prcca(
    X,
    Y,
    idx_x1,
    idx_x2,
    lambda_x1_grid,
    lambda_y_grid,
    n_components=1,
    n_folds=5,
    random_state=0
):
    """
    Cross-validation for PRCCA using out-of-sample canonical correlation.
    """

    X = np.asarray(X)
    Y = np.asarray(Y)

    assert X.shape[0] == Y.shape[0], "X and Y must have same number of patients"

    kf = KFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state
    )

    results = []

    for lx1 in lambda_x1_grid:
        lx2 = lx1*1000
        for ly in lambda_y_grid:

            fold_scores = []

            for train_idx, test_idx in kf.split(X):

                X_tr, X_te = X[train_idx, :], X[test_idx, :]
                Y_tr, Y_te = Y[train_idx, :], Y[test_idx, :]

                A, B, _, _ = solve_prcca_x(
                    X_tr,
                    Y_tr,
                    idx_x1=idx_x1,
                    idx_x2=idx_x2,
                    lambda_x1=lx1,
                    lambda_x2=lx2,
                    lambda_y=ly,
                    n_components=n_components
                )

                score = canonical_test_score(
                    X_te,
                    Y_te,
                    A,
                    B,
                    n_components=n_components
                )

                fold_scores.append(score)

            results.append({
                "lambda_x1": lx1,
                "lambda_x2": lx2,
                "lambda_y": ly,
                "mean_cv_corr": float(np.mean(fold_scores)),
                "std_cv_corr": float(np.std(fold_scores))
            })

    return results

def canonical_score(X, Y, A, B, n_components=1):
    U = X @ A[:, :n_components]
    V = Y @ B[:, :n_components]

    scores = []
    for k in range(n_components):
        if np.std(U[:, k]) == 0 or np.std(V[:, k]) == 0:
            scores.append(0.0)
        else:
            scores.append(abs(pearsonr(U[:, k], V[:, k])[0]))
    return float(np.mean(scores))

def loading_stability(A_list):
    """
    Mean pairwise cosine similarity of canonical loading vectors.
    """
    if len(A_list) < 2:
        return 1.0

    sims = []
    for i in range(len(A_list)):
        for j in range(i + 1, len(A_list)):
            a1 = A_list[i].ravel()
            a2 = A_list[j].ravel()
            denom = np.linalg.norm(a1) * np.linalg.norm(a2)
            if denom == 0:
                sims.append(0.0)
            else:
                sims.append(abs(np.dot(a1, a2) / denom))
    return float(np.mean(sims))

def cv_prcca_pareto(
    X,
    Y,
    idx_x1,
    idx_x2,
    lambda_x1_grid,
    lambda_y_grid,
    n_components=1,
    n_folds=5,
    random_state=0
):
    X = np.asarray(X)
    Y = np.asarray(Y)

    kf = KFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state
    )

    results = []

    for lx1 in lambda_x1_grid:
        lx2 = lx1 * 1000
        for ly in lambda_y_grid:

            train_scores = []
            test_scores = []
            A_folds = []
            B_folds = []

            for train_idx, test_idx in kf.split(X):
                X_tr, X_te = X[train_idx], X[test_idx]
                Y_tr, Y_te = Y[train_idx], Y[test_idx]

                A, B, _, _ = solve_prcca_x(
                    X_tr,
                    Y_tr,
                    idx_x1=idx_x1,
                    idx_x2=idx_x2,
                    lambda_x1=lx1,
                    lambda_x2=lx2,
                    lambda_y=ly,
                    n_components=n_components
                )

                train_scores.append(
                    canonical_score(X_tr, Y_tr, A, B, n_components)
                )
                test_scores.append(
                    canonical_score(X_te, Y_te, A, B, n_components)
                )

                A_folds.append(A[:, :n_components])
                B_folds.append(B[:, :n_components])
            eps = 1e-6
            train_mean = np.mean(train_scores)
            test_mean = np.mean(test_scores)
            
            gap = float((train_mean - test_mean) / (test_mean + eps))
            stab_A = loading_stability(A_folds)
            stab_B = loading_stability(B_folds)
            stab = (stab_A + stab_B) / 2
            results.append({
                "lambda_x1": lx1,
                "lambda_x2": lx2,
                "lambda_y": ly,
                "test_corr": float(np.mean(test_scores)),
                "train_corr": float(np.mean(train_scores)),
                "gap": gap,
                "stability_A": stab_A,
                "stability_B": stab_B,
                "stability": stab
            })

    return results

def pareto_select(results):
    front = []

    for i, r in enumerate(results):
        dominated = False
        for j, q in enumerate(results):
            if i == j:
                continue

            better_or_equal = (
                # q["test_corr"] >= r["test_corr"] and
                q["stability"] >= r["stability"] and
                q["gap"] <= r["gap"]
            )
            strictly_better = (
                # q["test_corr"] > r["test_corr"] or
                q["stability"] > r["stability"] or
                q["gap"] < r["gap"]
            )

            if better_or_equal and strictly_better:
                dominated = True
                break

        if not dominated:
            front.append(r)

    return front




def solve_prcca_x(
    X_input,
    Y_input,
    idx_x1,
    idx_x2,
    lambda_x1,
    lambda_x2,
    lambda_y=1e-6,
    n_components=None
):
    X = np.asarray(X_input, dtype=float)
    Y = np.asarray(Y_input, dtype=float)

    # Mean center
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    n, p = X.shape
    _, q = Y.shape
    if n_components is None:
        n_components = min(p, q)

    # === Partial regularization on X ===
    lambda_vec = np.zeros(p)
    lambda_vec[idx_x1] = lambda_x1
    lambda_vec[idx_x2] = lambda_x2
    Lambda_x = np.diag(lambda_vec)

    # Covariances
    Sxx = np.cov(X, rowvar=False) + Lambda_x
    Syy = np.cov(Y, rowvar=False) + lambda_y * np.eye(q)
    Sxy = np.cov(X, Y, rowvar=False)[:p, p:]

    # Numerical stabilization
    Sxx = (Sxx + Sxx.T) / 2 + 1e-10 * np.eye(p)
    Syy = (Syy + Syy.T) / 2 + 1e-10 * np.eye(q)

    # Whitening
    eig_x, Ux = eigh(Sxx)
    eig_y, Uy = eigh(Syy)
    eig_x = np.clip(eig_x, 1e-10, None)
    eig_y = np.clip(eig_y, 1e-10, None)

    Sxx_inv_sqrt = Ux @ np.diag(1 / np.sqrt(eig_x)) @ Ux.T
    Syy_inv_sqrt = Uy @ np.diag(1 / np.sqrt(eig_y)) @ Uy.T

    # Canonical correlations via SVD
    T = Sxx_inv_sqrt @ Sxy @ Syy_inv_sqrt
    U, s, Vt = np.linalg.svd(T)

    r = s[:n_components]
    A = Sxx_inv_sqrt @ U[:, :n_components]
    B = Syy_inv_sqrt @ Vt.T[:, :n_components]

    U_scores = X @ A
    V_scores = Y @ B
    r_emp = empirical_corr(U_scores, V_scores)

    return A, B, r, r_emp


def prcca_x(
    X_input,
    Y_input,
    idx_x1,
    idx_x2,
    lambda_x1=1e-6,
    lambda_x2=1e-6,
    lambda_y=1e-6,
    n_components=None
):
    """
    Note: the returned "r" is the singular values of the regularized
    problem, not the empirical correlations of the canonical variates.
    For calculate_p_value_prcca_x, pass the 4th output of solve_prcca_x
    (r_emp) as observed_corrs.
    """

    X = np.asarray(X_input, float)
    Y = np.asarray(Y_input, float)

    n, p = X.shape
    _, q = Y.shape

    if n_components is None:
        n_components = min(p, q)
    n_components = int(n_components)
    assert n_components >= 1

    A, B, r, _ = solve_prcca_x(
        X_input, Y_input,
        idx_x1=idx_x1,
        idx_x2=idx_x2,
        lambda_x1=lambda_x1,
        lambda_x2=lambda_x2,
        lambda_y=lambda_y,
        n_components=n_components
    )

    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)

    U = Xc @ A
    V = Yc @ B
    Z = U + V


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
        "n_components": n_components
    }

