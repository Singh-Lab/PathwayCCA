import os
import numpy as np
from scipy.stats import norm

from constrained_cca import solve_ccca,ccca

def pillai_bartlett_trace(rho, p, q):
    """
    Compute Pillai–Bartlett trace statistics: 
    T_r = sum_{i=0}^r  rho_i^2.
    Equivalent to R's PillaiBartlettTrace.
    """
    rho = np.asarray(rho)
    minpq = min(p, q, len(rho))
    return np.array([np.sum(rho[:r + 1] ** 2) for r in range(minpq)])


def calculate_p_value_ccca(
    X_input,
    Y_input,
    Full_Y_input,
    C, d,
    observed_corrs,
    n_perm=899,
    lambda_x=0,
    lambda_y=0,
    cache_base="./",
    file_name="test",
    n_components=None,
    i_choice="min",   # "min", "max", "first", or explicit integer index
    use_gaussian=True # option to use Gaussian approx or empirical p-value
):
    """
    Permutation-based significance test for Constrained CCA using 
    Pillai–Bartlett trace statistics.

    Parameters
    ----------
    X_input : array (n × p)
    Y_input : array (n × q)
    Full_Y_input : array (n × Q)  -- full gene pool for sampling
    C, d         : constraint matrices for CCCA
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
        f"Ysize_{Y_size}_nperm_{n_perm}_lx_{lambda_x}_ly_{lambda_y}.npy"
    )

    if os.path.exists(cache_file):
        print(f"Loading cached permutation statistics from {cache_file}")
        perm_stats = np.load(cache_file)
    else:
        perm_stats = np.zeros((n_perm, len(stat0_vec)))

        for i in range(n_perm):
            try:
                # sample q genes
                gene_idx = np.random.choice(
                    Full_Y_input.shape[1], 
                    size=Y_size, 
                    replace=False
                )
                Y_boot = Full_Y_input[:, gene_idx]

                # Run constrained RCCA
                A_full, B, _,r = solve_ccca(
                    X_input, Y_boot,
                    lambda_x=lambda_x,
                    lambda_y=lambda_y,
                    C=C,
                    d=d,
                    n_components = n_components
                )

                perm_stats[i, :] = pillai_bartlett_trace(r, p, q)

            except Exception as e:
                print(f"Permutation {i+1} failed: {e}!")
                perm_stats[i, :] = np.nan

            if (i + 1) % 100 == 0:
                print(f"{i+1}/{n_perm} permutations computed")

        np.save(cache_file, perm_stats)
        print(f"Saved permutation results to {cache_file}")

    valid = ~np.isnan(perm_stats).any(axis=1)
    perm_stats = perm_stats[valid]
    if perm_stats.shape[0] == 0:
        raise RuntimeError("All permutations failed. Cannot compute p-value.")

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
        return {
            "stat0": float(stat0),
            "p_value": float(p_value),
            "n_perm": int(perm_vec.size),
            "i": idx
        }
