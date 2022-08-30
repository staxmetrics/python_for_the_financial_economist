import numpy as np
from scipy import stats
from codelib.portfolio_optimization.mean_variance import portfolio_std, portfolio_mean
from codelib.statistics.moments import weighted_percentile

from typing import Union

def drawdown(index: np.ndarray):
    """
    Calculates the running draw down

    Parameters
    ----------
    index:
        Values of e.g. an equity index

    Returns
    -------
    Tuple(np.ndarray, np.ndarray)
        Drawdown, index of running maximum

    """

    indexmax = np.maximum.accumulate(index)
    drawdowns = (index - indexmax) / indexmax

    return drawdowns, indexmax


def maxdrawdown(index: np.ndarray):
    """
    Calculates maximum draw down

    Parameters
    ----------
    index:
        Values of e.g. an equity index

    Returns
    -------
    float
        Maximum drawdown

    """

    return drawdown(index)[0].min()


"""
Scenarios
"""


def calculate_value_at_risk(x: np.ndarray, alpha: float = 0.05, probs: Union[None, np.ndarray] = None):

    var = weighted_percentile(x, p=alpha, probs=probs)

    return var


"""
Normal distribution
"""


def calculate_normal_value_at_risk(mu: float, sigma: float, alpha: float = 0.05) -> float:

    """
    Calculates Value at Risk for a Gaussian random variable

    Parameters
    ----------
    mu:
        Expected return / value.
    sigma:
        Standard deviation.
    alpha:
        Confidence level. Default value is 5%.

    Returns
    -------
    float
        Value at Risk

    """

    qnorm = stats.norm.ppf(alpha)

    var = mu + qnorm * sigma

    return var


def calculate_normal_cond_value_at_risk(mu: float, sigma: float, alpha: float = 0.05) -> float:

    """
    Calculates Conditional Value at Risk for a Gaussian random variable

    Parameters
    ----------
    mu:
        Expected return / value.
    sigma:
        Standard deviation.
    alpha:
        Confidence level. Default value is 5%.

    Returns
    -------
    float
        Conditional Value at Risk

    """

    # percentile
    qnorm = stats.norm.ppf(alpha)
    # calculate CVaR
    cvar = mu - stats.norm.pdf(qnorm) / alpha * sigma

    return cvar


def calculate_normal_port_value_at_risk(weights: np.ndarray, mu: np.ndarray, cov_mat: np.ndarray, alpha: float = 0.05) -> float:

    """
    Calculates Value at Risk for a portfolio of Gaussian random variables

    Parameters
    ----------
    weights:
        Portfolio weights.
    mu:
        Expected return / value.
    cov_mat:
        Covariance matrix.
    alpha:
        Confidence level. Default value is 5%.

    Returns
    -------
    float
        Value at Risk

    """

    mu = portfolio_mean(weights, mu)
    std = portfolio_std(weights, cov_mat)

    return calculate_normal_value_at_risk(mu, std, alpha)


def calculate_normal_port_cond_value_at_risk(weights: np.ndarray, mu: np.ndarray, cov_mat: np.ndarray, alpha: float = 0.05) -> float:

    """
    Calculates Cond. Value at Risk for a portfolio of Gaussian random variables

    Parameters
    ----------
    weights:
        Portfolio weights.
    mu:
        Expected return / value.
    cov_mat:
        Covariance matrix.
    alpha:
        Confidence level. Default value is 5%.

    Returns
    -------
    float
        Value at Risk

    """

    mu = portfolio_mean(weights, mu)
    std = portfolio_std(weights, cov_mat)

    return calculate_normal_cond_value_at_risk(mu, std, alpha)


if __name__ == '__main__':

    var = calculate_normal_value_at_risk(0, 0.2, 0.05)
    print(var)

    cvar = calculate_normal_cond_value_at_risk(0, 0.2, 0.05)
    print(cvar)
