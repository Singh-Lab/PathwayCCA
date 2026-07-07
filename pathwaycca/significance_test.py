import os
import numpy as np
from scipy.stats import norm

from .prcca_x import solve_prcca_x, prcca_x
from .prcca_x import cv_prcca_pareto, pareto_select

def pillai_bartlett_trace(rho, p, q):
    """
    Compute Pillai–Bartlett trace statistics: 
    T_r = sum_{i=0}^r  rho_i^2.
    Equivalent to R's PillaiBartlettTrace.
    """
    rho = np.asarray(rho)
    minpq = min(p, q, len(rho))
    return np.array([np.sum(rho[:r + 1] ** 2) for r in range(minpq)])

def select_best_lambda_prcca(
    X_input,
    Y_input,
    idx_x1,
    idx_x2,
    lambda_x1_grid,
    lambda_y_grid,
    n_components=3,
    n_folds=3,
    max_gap=1
):
    cv_results = cv_prcca_pareto(
        X_input,
        Y_input,
        idx_x1=idx_x1,
        idx_x2=idx_x2,
        lambda_x1_grid=lambda_x1_grid,
        lambda_y_grid=lambda_y_grid,
        n_components=n_components,
        n_folds=n_folds
    )

    pareto = pareto_select(cv_results)

    pareto_filtered = pareto # [r for r in pareto if r["gap"] <= max_gap]
    if len(pareto_filtered) == 0:
        pareto_filtered = sorted(pareto, key=lambda r: r["gap"])[:3]

    best_param = max(
        pareto_filtered,
        key=lambda r: (r["stability"], -r["gap"], r["lambda_x1"], r["lambda_y"])
    )

    return best_param

def calculate_p_value_prcca_x(
    X_input,
    Y_input,
    Full_Y_input,
    idx_x1,
    idx_x2,
    observed_corrs,
    n_perm=899,
    lambda_x1=1e-6,
    lambda_x2=1e-6,
    lambda_x1_grid = [0.01],
    lambda_y_grid = [0.01],
    cv_folds = 3,
    lambda_y=0,
    cache_base="./",
    file_name="test",
    n_components=None,
    i_choice="min",   # "min", "max", "first", or explicit integer index
    use_gaussian=True, # option to use Gaussian approx or empirical p-value
    tune_lambda=False
):
    """
    Permutation-based significance test for Constrained CCA using 
    Pillai–Bartlett trace statistics.

    Parameters
    ----------
    X_input : array (n × p)
    Y_input : array (n × q)
    Full_Y_input : array (n × Q)  -- full gene pool for sampling
    observed_corrs : array-like, canonical correlations r from original data
    n_perm : number of bootstrap permutations
    cache_base : base directory to store cached perm statistics
    file_name : subfolder name
    i_choice : statistic index to test. 
    use_gaussian : if True, use Gaussian approximation
                   if False, compute empirical permutation p-value

    Returns
    -------
    dict with
        stat0   : observed test statistic (Pillai trace)
        p_value : permutation p-value
        n_perm  : number of permutations actually used
        mu, sigma (if gaussian)
        i       : index of statistic tested
    """

    X_input = np.asarray(X_input)
    Y_input = np.asarray(Y_input)
    Full_Y_input = np.asarray(Full_Y_input)
    p, q = X_input.shape[1], Y_input.shape[1]

    stat0_vec = pillai_bartlett_trace(observed_corrs, p, q)

    Y_size = q
    cache_dir = os.path.join(cache_base, file_name)
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = os.path.join(
        cache_dir,
        f"Ysize_{Y_size}_nperm_{n_perm}_lx_{lambda_x1}_{lambda_x2}_ly_{lambda_y}.npy"
    )

    if os.path.exists(cache_file):
        print(f"Loading cached permutation statistics from {cache_file}")
        perm_stats = np.load(cache_file)
    else:
        perm_stats = np.zeros((n_perm, len(stat0_vec)))

        tune_warmup = 100
        warmup_params = []
        
        fixed_lx1 = lambda_x1
        fixed_lx2 = lambda_x2
        fixed_ly = lambda_y
        
        for i in range(n_perm):
            try:
                gene_idx = np.random.choice(
                    Full_Y_input.shape[1],
                    size=Y_size,
                    replace=False
                )
                Y_boot = Full_Y_input[:, gene_idx]
        
                if tune_lambda and len(warmup_params) < tune_warmup:
                    best_param = select_best_lambda_prcca(
                        X_input,
                        Y_boot,
                        idx_x1=idx_x1,
                        idx_x2=idx_x2,
                        lambda_x1_grid=lambda_x1_grid,
                        lambda_y_grid=lambda_y_grid,
                        n_components=n_components,
                        n_folds=cv_folds
                    )
        
                    lx1 = best_param["lambda_x1"]
                    lx2 = best_param["lambda_x2"]
                    ly = best_param["lambda_y"]
        
                    warmup_params.append((lx1, lx2, ly))
        
                    if len(warmup_params) == tune_warmup:
                        warmup_arr = np.asarray(warmup_params, dtype=float)
        
                        fixed_lx1 = float(np.median(warmup_arr[:, 0]))
                        fixed_lx2 = float(np.median(warmup_arr[:, 1]))
                        fixed_ly = float(np.median(warmup_arr[:, 2]))
        
                        print(
                            f"Using median lambdas after {tune_warmup} tuned permutations: "
                            f"lambda_x1={fixed_lx1}, lambda_x2={fixed_lx2}, lambda_y={fixed_ly}"
                        )
        
                elif tune_lambda:
                    lx1 = fixed_lx1
                    lx2 = fixed_lx2
                    ly = fixed_ly
        
                else:
                    lx1 = lambda_x1
                    lx2 = lambda_x2
                    ly = lambda_y
        
                A_full, B, _, r = solve_prcca_x(
                    X_input,
                    Y_boot,
                    idx_x1=idx_x1,
                    idx_x2=idx_x2,
                    lambda_x1=lx1,
                    lambda_x2=lx2,
                    lambda_y=ly,
                    n_components=n_components
                )
        
                perm_stats[i, :] = pillai_bartlett_trace(r, p, q)
        
            except Exception as e:
                print(f"Permutation {i+1} failed: {e}!")
                perm_stats[i, :] = np.nan
        
            if (i + 1) % 100 == 0:
                print(f"{i+1}/{n_perm} permutations computed")

    valid = ~np.isnan(perm_stats).any(axis=1)
    perm_stats = perm_stats[valid]
    if perm_stats.shape[0] == 0:
        raise RuntimeError("All permutations failed. Cannot compute p-value.")
    np.save(cache_file, perm_stats)
    print(f"Saved permutation statistics to {cache_file}")
    if i_choice == "min":
        mu_vec = perm_stats.mean(0)
        sigma_vec = perm_stats.std(0, ddof=1)
        tmp_p = 1 - norm.cdf(stat0_vec, loc=mu_vec, scale=sigma_vec)
        idx = int(np.nanargmin(tmp_p))
    elif i_choice == "max":
        idx = int(np.nanargmax(stat0_vec))
    elif isinstance(i_choice, int):
        idx = i_choice
        if idx < 0 or idx >= len(stat0_vec):
            raise ValueError(f"i_choice={i_choice} is out of range.")
    else:
        raise ValueError("i_choice must be 'first', 'min', 'max', or an integer index.")

    stat0 = stat0_vec[idx]
    perm_vec = perm_stats[:, idx]

    if use_gaussian:
        mu = perm_vec.mean()
        sigma = perm_vec.std(ddof=1)
        p_value = 1 - norm.cdf(stat0, loc=mu, scale=sigma)
        return {
            "stat0": float(stat0),
            "p_value": float(p_value),
            "n_perm": int(perm_stats.shape[0]),
            "mu": float(mu),
            "sigma": float(sigma),
            "i": idx
        }
    else:
        p_value = (1 + np.sum(perm_vec >= stat0)) / (1 + perm_vec.size)
        mu = perm_vec.mean()
        sigma = perm_vec.std(ddof=1)
        return {
            "stat0": float(stat0),
            "p_value": float(p_value),
            "n_perm": int(perm_vec.size),
            "mu": float(mu),
            "sigma": float(sigma),
            "i": idx
        }
