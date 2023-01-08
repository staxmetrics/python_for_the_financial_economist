import numpy as np
from scipy.stats import entropy
from codelib.portfolio_optimization.risk_metrics import calculate_portfolio_variance, calculate_portfolio_std
from typing import Union

"""
Naive diversification measures
"""


def calculate_enc(weights: np.ndarray, alpha: int = 1, relative: bool = False) -> float:

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


def calculate_cc_ratio(weights: np.ndarray, cov_mat: np.ndarray):

    """
    Calculates the diversification ratio of Chouefaty and Coignard (2008)

    .. math::

        \\begin{equation}
            \\text{GLR}(w, \\Sigma) = \\frac{\\sum_{i=1}^N w_i \\sigma_i}{\\sqrt{w^{\\top} \\Sigma w}}
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

    port_std = calculate_portfolio_std(weights=weights, cov_mat=cov_mat)

    vol_vec = np.sqrt(np.diag(cov_mat))
    avg_std = np.inner(weights, vol_vec)

    return avg_std / port_std


def calculate_enb(weights: np.ndarray, transition_matrix: np.ndarray, factor_variances: np.ndarray, alpha: int = 1) -> Union[float, np.ndarray]:
    """
    Calculates the effective number of uncorrelated bets of Meucci (2009) - "Managining diversification".

    Notes
    -----
    The method relies on the choice of :math:`N` uncorrelated factors whose returns can be expressed as

    .. math::

        \\begin{equation}
            \\mathbf{r}_F = \\mathbf{A}^\\top \\mathbf{r}
        \\end{equation}

    where :math:`r` is the vector of constituents returns and :math:`\\mathbf{A}` is the "transition matrix" that maps
    from the constituents returns to factor returns (square matrix of size :math:`N`). :math:`\\mathbf{A}` is assumed
    invertible. Given the assumption of uncorrelated factors and invertible matrix, we can write the portfolio variance
    as

    .. math::

        \\begin{align}
            \\mathbf{w}^\\top \\boldsymbol{\\Sigma} \\mathbf{w} &= \\mathbf{w}^\\top (\\mathbf{A}^\\top)^{-1}
            \\boldsymbol{\\Sigma}_F (\\mathbf{A})^{-1} \\mathbf{w} \\
            &= \\mathbf{w}^\\top_F \\boldsymbol{\\Sigma} \\mathbf{w}_F \\
            &= \\sum_{i=1}^N (\\sigma_{F_i} w_{F_i})^2
        \\end{align}

    We can then define the i'th factors contribution to portfolio variance

    .. math::

        \\begin{equation}
            p_i =\\frac{(\\sigma_{F_i} w_{F_i})^2}{\\mathbf{w}^\\top \\boldsymbol{\\Sigma} \\mathbf{w}}
        \\end{equation}

    which is positive and sum to one. The idea is then to calculate the effective number of uncorrelated bets using the
    norm of the factor constributions to the total variance (basically calculating Effective Number of Constituents
    using :math:`p_i, i=1, ..., N`).

    For further details see Meucci (2009) - "Managing diversification".

    Parameters
    ----------
    weights:
        Portfolio weights.
    transition_matrix:
        Transtion matrix that transforms returns to factor returns.
    factor_variances:
    alpha:

    Returns
    -------

    """

    # inverse transition matrix
    inv_transition_matrix = np.linalg.inv(transition_matrix)

    # factor covariance and weights
    factor_cov_mat = np.diag(factor_variances)
    factor_weights = inv_transition_matrix @ weights

    # port. variance
    port_var = calculate_portfolio_variance(weights=factor_weights, cov_mat=factor_cov_mat)

    # factor contribution to variance
    var_contrib = (factor_variances * np.square(factor_weights)) / port_var

    # return effective number of constituents calculated using variance contribution
    return calculate_enc(weights=var_contrib, alpha=alpha), var_contrib

