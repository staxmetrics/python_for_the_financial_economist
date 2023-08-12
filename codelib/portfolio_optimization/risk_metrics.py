import numpy as np
from scipy import stats
from codelib.statistics.moments import weighted_percentile
from codelib.statistics.historical_probabilities import equal_weights

from typing import Union, List


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


def calculate_portfolio_variance(weights: np.ndarray, cov_mat: np.ndarray) -> float:
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


def calculate_portfolio_mean(weights: np.ndarray, mu: np.ndarray) -> float:
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


def calculate_portfolio_std(weights: np.ndarray, cov_mat: np.ndarray) -> float:
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

    return np.sqrt(calculate_portfolio_variance(weights, cov_mat))


"""
Scenarios
"""


def calculate_value_at_risk(x: np.ndarray, p: Union[float, List[float]], probs: Union[np.ndarray, None] = None,
                            axis=0) -> Union[float, List]:

    """
    Calculates the Value-at-Risk.

    Parameters
    ----------
    x:
        Matrix of values.
    p:
        Value-at-risk level.
    probs:
        Weights for calculation of weighted Value-at-Risk. Default is equally weighted.
    axis:
        Axis to calculate value at risk over.

    Returns
    -------
        Value-At-Risk for a given level or list with Value-at-Risk for different levels.

    """

    return weighted_percentile(x, p, probs=probs, axis=axis)


def calculate_conditional_value_at_risk(x: np.ndarray, p: float, probs: np.ndarray = None, axis: int = 0) -> Union[float, List]:

    """
    Calculates the Conditional Value-at-Risk.

    Parameters
    ----------

    x:
        Matrix of values.
    p:
        Conditional Value-at-risk level.
    probs:
        Weights for calculation of weighted Conditional Value-at-Risk. Default is equally weighted.
    axis:
        Axis to calculate cond. value at risk over.

    Returns
    -------
    Union[float, List]
        Cond. Value-At-Risk for a given level or list with Cond. Value-at-Risk for different levels.

    """

    assert (p >= 0.0) and (p <= 1.0), 'Percentile must be between 0.0 and 1.0'

    sorted_idx = np.argsort(x, axis=axis)
    sorted_values = np.take_along_axis(x, sorted_idx, axis=axis)
    sorted_probs = equal_weights(x, axis=axis) if probs is None else probs[sorted_idx]

    if x.ndim == 1:
        value_at_risk_matrix = np.repeat(calculate_value_at_risk(x, p, probs, axis), x.shape[axis])
    else:
        value_at_risk_matrix = np.tile(calculate_value_at_risk(x, p, probs, axis), (x.shape[axis], 1))
        sorted_probs = np.tile(sorted_probs, (x.shape[1], 1)).T if axis == 0 else np.tile(sorted_probs, (x.shape[0], 1))

    if axis == 1:
        tail_bool = sorted_values <= np.transpose(value_at_risk_matrix)
    else:
        tail_bool = sorted_values <= value_at_risk_matrix

    unscaled_tail_probs = sorted_probs
    unscaled_tail_probs[~tail_bool] = 0

    if x.ndim == 1:
        scale_denom = np.sum(unscaled_tail_probs, axis=axis)
    else:
        scale_denom = np.atleast_2d(np.sum(unscaled_tail_probs, axis=axis))

    if axis == 1:
        scale_denom = scale_denom.T
    elif axis != 0:
        raise IndexError("Axis parameter is invalid, must be either 0 or 1")

    scaled_tail_probs = unscaled_tail_probs / scale_denom
    result = np.average(sorted_values, weights=scaled_tail_probs, axis=axis)

    return result


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

    mu = calculate_portfolio_mean(weights, mu)
    std = calculate_portfolio_std(weights, cov_mat)

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

    mu = calculate_portfolio_mean(weights, mu)
    std = calculate_portfolio_std(weights, cov_mat)

    return calculate_normal_cond_value_at_risk(mu, std, alpha)


if __name__ == '__main__':

    var = calculate_normal_value_at_risk(0, 0.2, 0.05)
    print(var)

    cvar = calculate_normal_cond_value_at_risk(0, 0.2, 0.05)
    print(cvar)
