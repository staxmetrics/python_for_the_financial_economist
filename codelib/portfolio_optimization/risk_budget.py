import numpy as np
from scipy import stats
from typing import Tuple


def calculate_marginal_risks_std(weights: np.ndarray, cov_mat: np.ndarray) -> np.ndarray:
    """
    Function that calculates marginal risk using std. as portfolio risk measure
    Parameters
    ----------
    weights:
        Portfolio weights
    cov_mat:
        Covariance matrix

    Returns
    -------
    np.ndarray
        Marginal risks
    """

    total_risk = np.sqrt(weights @ cov_mat @ weights)
    inner_derivative = cov_mat @ weights

    return inner_derivative / total_risk


def calculate_risk_contributions_std(weights: np.ndarray, cov_mat: np.ndarray) -> np.ndarray:
    """
    Function that calculates risk contributions using std. as portfolio risk measure

    Parameters
    ----------
    weights:
        Portfolio weights
    cov_mat:
        Covariance matrix

    Returns
    -------
    np.ndarray
        Marginal risks
    """

    mr = calculate_marginal_risks_std(weights, cov_mat)

    return weights * mr


def calculate_marginal_sharpe(weights: np.ndarray, cov_mat: np.ndarray, mu: np.ndarray, rf: float):

    """
    Function that calculates marginal Sharpe ratio

    Parameters
    ----------
    weights:
        Portfolio weights
    cov_mat:
        Covariance matrix
    mu:
        Expected return vector.
    rf:
        Risk free rate.

    Returns
    -------
    np.ndarray
        Marginal risks
    """

    mr = calculate_marginal_risks_std(weights, cov_mat)
    excess_mu = mu - rf

    return excess_mu / mr


def weights_risk_budget_two_assets(sigma1: float, sigma2: float, rho: float, b: float) -> Tuple[float, float]:
    """
    Function that calculates the portfolio weights that result in equal risk contribution. Two asset case.

    Parameters
    ----------
    sigma1:
        Std. of asset 1
    sigma2:
        Std. of asset 2
    rho:
        Correlation between asset 1 and asset 2
    b:
        Risk contribution from asset 1

    Returns
    -------
    np.ndarray
       Optimal weights.

    """

    num = (b - 0.5) * rho * sigma1 * sigma2 - b * sigma2 ** 2 + sigma1 * sigma2 * np.sqrt(
        (b - 0.5) ** 2 * rho ** 2 + b * (1 - b))
    den = (1 - b) * sigma1 ** 2 - b * sigma2 ** 2 + 2 * (b - 0.5) * rho * sigma1 * sigma2

    w_opt = num / den

    return w_opt, 1 - w_opt


"""
Functions relevant for multivariate Gaussian distribution
"""


def calculate_marginal_risks_normal_var(weights: np.ndarray, mu: np.ndarray, cov_mat: np.ndarray,
                                        alpha: float = 0.05) -> np.ndarray:
    """
    Function that calculates marginal risk using Value-at-Risk as risk measure.
    It is assumed that returns follow a normal distribution.

    Parameters
    ----------
    weights:
        Portfolio weights
    mu:
        Vector of expected returns
    cov_mat:
        Covariance matrix
    alpha:
        Confidence level

    Returns
    -------
    np.ndarray
        Marginal risks
    """

    mr_std = calculate_marginal_risks_std(weights, cov_mat)

    mr_var = -mu + stats.norm.ppf(1.0 - alpha) * mr_std

    return mr_var


def calculate_risk_contributions_normal_var(weights: np.ndarray, mu: np.ndarray, cov_mat: np.ndarray,
                                            alpha: float = 0.05) -> np.ndarray:
    """
    Function that calculates marginal risk using Value-at-Risk as risk measure.
    It is assumed that returns follow a normal distribution.

    Parameters
    ----------
    weights:
        Portfolio weights
    mu:
        Vector of expected returns
    cov_mat:
        Covariance matrix
    alpha:
        Confidence level

    Returns
    -------
    np.ndarray
        Marginal risks
    """

    mr = calculate_marginal_risks_normal_var(weights, mu, cov_mat, alpha=alpha)

    return weights * mr


def calculate_marginal_risks_normal_cvar(weights: np.ndarray, mu: np.ndarray, cov_mat: np.ndarray,
                                         alpha: float = 0.05) -> np.ndarray:
    """
    Function that calculates marginal risk using Cond. Value-at-Risk as risk measure.
    It is assumed that returns follow a normal distribution.

    Parameters
    ----------
    weights:
        Portfolio weights
    mu:
        Vector of expected returns
    cov_mat:
        Covariance matrix
    alpha:
        Confidence level

    Returns
    -------
    np.ndarray
        Marginal risks
    """

    mr_std = calculate_marginal_risks_std(weights, cov_mat)

    qnorm = stats.norm.ppf(alpha)

    mr_cvar = -mu + stats.norm.pdf(qnorm) / alpha * mr_std

    return mr_cvar


def calculate_risk_contributions_normal_cvar(weights: np.ndarray, mu: np.ndarray, cov_mat: np.ndarray,
                                             alpha: float = 0.05) -> np.ndarray:
    """
    Function that calculates marginal risk using Cond. Value-at-Risk as risk measure.
    It is assumed that returns follow a normal distribution.

    Parameters
    ----------
    weights:
        Portfolio weights
    mu:
        Vector of expected returns
    cov_mat:
        Covariance matrix
    alpha:
        Confidence level

    Returns
    -------
    np.ndarray
        Marginal risks
    """

    mr = calculate_marginal_risks_normal_cvar(weights, mu, cov_mat, alpha=alpha)

    return weights * mr


"""
Cornish Fisher Risk budgetting
"""

# https://rdrr.io/rforge/fPortfolio/src/R/risk-budgeting.R

"""
Functions relevant for scenario based estimation 
"""