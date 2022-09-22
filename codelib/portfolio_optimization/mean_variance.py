import numpy as np


def minimum_variance_portfolio(cov_mat: np.ndarray) -> np.ndarray:

    """
    Calculates the minimum-variance portfolio weights.

    Parameters
    ----------
    cov_mat:
        The covariance matrix.

    Returns
    -------
    np.ndarray
        Minimum variance portfolio weights.

    """

    num_assets = len(cov_mat)
    vec_ones = np.ones(num_assets)

    cov_mat_inv = np.linalg.inv(cov_mat)

    w_min_var = cov_mat_inv @ vec_ones / (vec_ones @ cov_mat_inv @ vec_ones)

    return w_min_var


def tangency_portfolio(cov_mat: np.ndarray, mu: np.ndarray, rf: float) -> np.ndarray:

    """
    Calculates the maximum sharpe ratio portfolio weights.

    Parameters
    ----------
    cov_mat:
        The covariance matrix.
    mu:
        Expected return vector.
    rf:
        The risk free rate.

    Returns
    -------
    np.ndarray
        maximum sharpe ratio portfolio weights.

    """

    num_assets = len(cov_mat)
    vec_ones = np.ones(num_assets)

    excess_mu = mu - vec_ones * rf

    cov_mat_inv = np.linalg.inv(cov_mat)

    w_max_sr = cov_mat_inv @ excess_mu / (vec_ones @ cov_mat_inv @ excess_mu)

    return w_max_sr

"""
Old functions - should be deleted
"""

def portfolio_variance(weights: np.ndarray, cov_mat: np.ndarray) -> float:
    """
    Function that returns the variance of the portfolio

    Parameters
    ----------
    weights:
        Portfolio weights
    cov_mat:
        Covariance matrix

    Returns
    -------
    float
        Variance of portfolio
    """

    return weights @ cov_mat @ weights


def portfolio_mean(weights: np.ndarray, mu: np.ndarray) -> float:
    """
    Function that returns the standard deviation of the portfolio

    Parameters
    ----------
    weights:
        Portfolio weights
    mu:
        Expected return vector.

    Returns
    -------
    float
        Expected return of portfolio
    """

    return weights @ mu


def portfolio_std(weights: np.ndarray, cov_mat: np.ndarray) -> float:
    """
    Function that returns the standard deviation of the portfolio

    Parameters
    ----------
    weights:
        Portfolio weights
    cov_mat:
        Covariance matrix

    Returns
    -------
    float
        Standard deviation of portfolio
    """

    return np.sqrt(portfolio_variance(weights, cov_mat))
