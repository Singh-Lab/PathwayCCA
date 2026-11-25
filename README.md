# PathwayCCA
PathwayCCA is a framework for pathway-level association analysis between multi-omic driver alterations and gene-set expression programs. The framework extends regularized Canonical Correlation Analysis (RCCA) and tests whether multiple somatic alterations jointly affect the expression of a given pathway. To address confounding from categorical covariates, we introduce group-weighted regularized CCA (GW-RCCA) an extension of RCCA that penalizes driver alterations and covariates separately. The covariates, such as cancer-subtype indicators, get stronger regularization, which keeps them from dominating the canonical variates. Penalty strengths are chosen by cross-validation using a Pareto criterion that balances stability against overfitting. We further develop a novel sampling-based hypothesis test that mitigates the effect of spurious correlations common in high-dimensional data.

<p align="center">
  <img src="img/PathwayCCA_1.png" width="1000">
</p>

The repository provides a complete implementation of Regularized CCA (RCCA), Constrained CCA (CCCA) (archived), Group-weighted regularized CCA (GW-RCCA) and the proposed PathwayCCA framework with resampling-based hypothesis test. 

## Installation

PathwayCCA requires Python 3.9 or later. Install it directly from GitHub:

```bash
pip install git+https://github.com/Singh-Lab/PathwayCCA.git
```

Or clone the repository and install it in editable mode (recommended if you want to run the example notebook or modify the code):

```bash
git clone https://github.com/Singh-Lab/PathwayCCA.git
cd PathwayCCA
pip install -e .
```

The dependencies (numpy, scipy, scikit-learn, pandas, matplotlib, seaborn) are installed automatically. To also install Jupyter for the example notebook, use `pip install -e ".[examples]"`.

Quick check:

```python
import pathwaycca
from pathwaycca import solve_prcca_x, calculate_p_value_prcca_x
print(pathwaycca.__version__)
```

## PathwayCCA Source Code
`pathwaycca/prcca_x.py`
- Implements `solve_prcca_x` which solved CCA with group-weighted regularization, allowing users to force stronger regularization on some of the variables.
- Implements `pareto_select` and `cv_prcca_pareto` which uses cross validation and pareto frontier to select optimal parameters in a log-scale parameter space

`pathwaycca/constrained_cca.py` (archive): 
- Implements `ccca`, which solves CCA under user-specified linear constraints on one coefficient set as proposed in the paper. 
- Provides `solve_ccca`, a wrapper that runs the solver and returns a structured result object.

`pathwaycca/plot_cca.py`:
- Contains visualization functions for examining CCA results:
  - `plt_var`: plots correlation-circle plots that shows canonical loadings of variables (with respect to the symmetric latent canonical variate $$Z=U+V$$).
  - `plt_indiv`: plots sample projections on canonical variates.
  - `plt_cca`: a wrapper that can contain both visulization plots.

`pathwaycca/significance_test.py`
- Implements `calculate_p_value_prcca_x`, which performs resampling-based significance testing to assess the statistical strength of canonical correlations.

## Example Data and Code

### Example Inputs:

`input_data/pathway_db/msigdb_H_P53.csv` constains an example pathway gene set metadata. 

`input_data/BRCA_subset/` contains a set of example input data that are small subsets of the preprocessed data from the `TCGA-BRCA` dataset used in the paper

### Example code:

We offer a simple example code for the user to experiment with PathwayCCA in `pathwaycca_prcca_example_code.ipynb` using the provided example input data. Note that the outputs may differ slightly from those reported in the paper because the example input data are only subsets of the original dataset (due to GitHub storage limits).
