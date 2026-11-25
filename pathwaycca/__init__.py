"""PathwayCCA: pathway-level association analysis with group-weighted regularized CCA."""

__version__ = "0.1.0"

from .prcca_x import (
    solve_prcca_x,
    prcca_x,
    cv_prcca,
    cv_prcca_pareto,
    pareto_select,
    canonical_score,
    canonical_test_score,
    loading_stability,
)
from .significance_test import (
    calculate_p_value_prcca_x,
    select_best_lambda_prcca,
    pillai_bartlett_trace,
)
from .constrained_cca import solve_ccca, ccca
from .plot_cca import plt_var, plt_indiv, plt_cca
from .utils import (
    compute_canonical_loadings,
    empirical_corr,
    make_lambda_grid,
)
