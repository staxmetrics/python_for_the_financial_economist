import numpy as np
from scipy.stats import entropy
from codelib.portfolio_optimization.risk_metrics import calculate_portfolio_variance


"""
Naive diversification measures
"""


def calculate_enc(weights: np.ndarray, alpha: int = 1, relative: bool = False):

    """
    Calculates the effective number of constituents in a long-only portfolio.

    if :math:`\\alpha>0, \\alpha \\neq 1`, then

    .. math::

        \\begin{equation}
            \\text{ENC}(w, \\alpha) = \\Vert w \\Vert_{\\alpha}^{\\frac{\\alpha}{1- \\alpha}}
        \\end{equation}

    Note that :math:`\\alpha = 2` will correspond to the inverse of the Herfindahl index.

    if :math:`\\alpha = 1` then the exponential of the entropy of the portfolio weights

    .. math::

        \\begin{equation}
            \\text{ENC}(w, \\alpha) = \\exp \\left(- \\sum_{i=1}^N w_i \\ln w_i \\right)
        \\end{equation}

    Parameters
    ----------
    weights:
        Portfolio weights.
    alpha:
        Alpha parameter.
    relative:
        Calculate diversification relative to number of assets. Default is False.

    Returns
    -------

    """

    num_constituents = len(weights)
    enc = 0.0

    if alpha <= 0:
        raise ValueError("Alpha must be larger than zero.")
    elif alpha == 1:
        enc = np.exp(entropy(weights))
    else:
        norm = np.linalg.norm(weights, ord=alpha)
        enc = norm**(alpha / (1 - alpha))

    if relative:
        return enc / num_constituents
    else:
        return enc

"""
Scientific diversification measures
"""


def calculate_glr_ratio(weights: np.ndarray, cov_mat: np.ndarray):

    """
    Calculates the diversification ratio of Goetzmann, Li, and Rouwenhorst (2005)

    .. math::

        \\begin{equation}
            \\text{GLR}(w, \\Sigma) = \\frac{w^{\\top} \\Sigma w}{\\sum_{i=1}^N w_i \\sigma_i^2}
        \\end{equation}

    Parameters
    ----------
    weights:
        Portfolio weights.
    cov_mat:
        Covariance matrix.

    Returns
    -------
    float
        Diversification ratio.

    """

    port_var = calculate_portfolio_variance(weights=weights, cov_mat=cov_mat)

    var_vec = np.diag(cov_mat)
    avg_var = np.inner(weights, var_vec)

    return port_var / avg_var


def calculate_enb(weights: np.ndarray, transition_matrix: np.ndarray, factor_variances: np.ndarray, alpha: int = 1):

    inv_transition_matrix = np.linalg.inv(transition_matrix)

    factor_cov_mat = np.diag(factor_variances)
    factor_weights = inv_transition_matrix @ weights

    port_var = calculate_portfolio_variance(weights=factor_weights, cov_mat=factor_cov_mat)

    var_contrib = (factor_variances * np.square(factor_weights)) / port_var

    return calculate_enc(weights=var_contrib, alpha=alpha)
