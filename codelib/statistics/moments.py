import numpy as np
from typing import Union

"""
Moments, statistics, etc.
"""


def calculate_mean(x: np.ndarray, probs: Union[np.ndarray, None]=None, axis=0) -> Union[float, np.ndarray]:

    """
    Calculates mean.

    Parameters
    ----------
    x:
        Data to calculate mean for.
    probs:
        Probabilities.
    axis:
        Axis over which to calculate.

    Returns
    -------
    Union[float, np.ndarray]
        Mean.

    """

    m = np.average(x, weights=probs, axis=axis)

    return m


def calculate_variance(x: np.ndarray, probs: Union[np.ndarray, None] = None, axis=0) -> Union[float, np.ndarray]:

    """
    Calculates variance.

    Parameters
    ----------
    x:
        Data to calculate variance for.
    probs:
        Probabilities.
    axis:
        Axis over which to calculate.

    Returns
    -------
    Union[float, np.ndarray]
        Variance.

    """

    m = np.average(x, weights=probs, axis=axis)

    return np.average(np.square(x - m), weights=probs)


def calculate_std(x: np.ndarray, probs: Union[np.ndarray, None] = None, axis=0):

    """
    Calculates standard deviation.

    Parameters
    ----------
    x:
        Data to calculate standard deviation for.
    probs:
        Probabilities.
    axis:
        Axis over which to calculate.

    Returns
    -------
    Union[float, np.ndarray]
        Standard devation.

    """

    return np.sqrt(calculate_variance(x, probs, axis))


def calculate_covariance(x: np.ndarray, y: np.ndarray, probs: Union[np.ndarray, None] = None, axis=0):

    """
    Calculates covariance between two variables.

    Parameters
    ----------
    x:
        First variable.
    y:
        Second variable.
    probs:
        Probabilities.
    axis:
        Axis over which to calculate.

    Returns
    -------
    Union[float, np.ndarray]
        Covariance.

    """

    m_x = np.average(x, weights=probs, axis=axis)
    m_y = np.average(y, weights=probs, axis=axis)

    m_xy = np.average(x*y, weights=probs, axis=axis)

    return m_xy - m_x * m_y


def calculate_correlation(x: np.ndarray, y: np.ndarray, probs: Union[np.ndarray, None] = None, axis=0):

    """
    Calculates correlation between two variables.

    Parameters
    ----------
    x:
        First variable.
    y:
        Second variable.
    probs:
        Probabilities.
    axis:
        Axis over which to calculate.

    Returns
    -------
    Union[float, np.ndarray]
        Correlation.

    """

    c = calculate_covariance(x, y, probs, axis)
    std_x = calculate_std(x, probs, axis)
    std_y = calculate_std(y, probs, axis)

    return c / (std_x * std_y)


def calculate_skewness(x: np.ndarray, probs: Union[np.ndarray, None] = None, axis=0):

    """
    Calculates skewness.

    Parameters
    ----------
    x:
        Data to calculate skewness for.
    probs:
        Probabilities.
    axis:
        Axis over which to calculate.

    Returns
    -------
    Union[float, np.ndarray]
        skewness.

    """

    m_x = np.average(x, weights=probs, axis=axis)
    m_x_3 = np.average(x ** 3, weights=probs, axis=axis)
    std = calculate_std(x, probs=probs, axis=axis)

    return (m_x_3 - 3 * m_x * std ** 2 - m_x ** 3) / (std ** 3)


def calculate_kurtosis(x: np.ndarray, probs: Union[np.ndarray, None] = None, excess: bool = True, axis=0):

    """
    Calculates kurtosis.

    Parameters
    ----------
    x:
        Data to calculate kurtosis for.
    probs:
        Probabilities.
    excess:
        Boolean indicating wether to calculate excess kurtosis. Default is True.
    axis:
        Axis over which to calculate.

    Returns
    -------
    Union[float, np.ndarray]
        Kurtosis.

    """

    x_d = x - np.average(x, weights=probs, axis=axis)
    m_x_d_4 = np.average(x_d ** 4, weights=probs, axis=axis)

    std = calculate_std(x, probs=probs, axis=axis)

    if excess:
        return m_x_d_4 / (std ** 4) - 3.0
    else:
        return m_x_d_4 / (std ** 4)


def weighted_percentile(x: np.ndarray, p: Union[float, np.ndarray], probs: Union[np.ndarray, None] = None, axis=0):

    """
    Function that calculates weighted percentiles

    Parameters
    ----------
    x:
        Array-like data for which to calculate percentiles.
    p:
        Percentile(s) to calculate.
    probs:
        Probabilities / weights
    axis:
        Axis over which to calculate.

    Returns
    -------
    np.array
        Percentiles

    """
    x = np.asarray(x)
    ndim = x.ndim

    # make sure the probs are set
    if probs is None:
        if ndim == 1:
            probs = np.ones_like(x) / len(x)
        elif axis == 0:
            length = x.shape[0]
            probs = np.ones(length) / length
        elif axis == 1:
            length = x.shape[1]
            probs = np.ones(length) / length
        else:
            raise ValueError('probs cannot be set')

    if ndim == 1:

        # get sorted index
        index_sorted = np.argsort(x)

        # get sorted data (x)
        sorted_x = x[index_sorted]

        # sorted probs
        sorted_probs = probs[index_sorted]

        # get cumulated probs
        cum_sorted_probs = np.cumsum(sorted_probs)

        pn = (cum_sorted_probs - 0.5 * sorted_probs) / cum_sorted_probs[-1]

        return np.interp(p, pn, sorted_x, left=sorted_x[0], right=sorted_x[-1])

    else:

        return np.apply_along_axis(weighted_percentile, axis, x, p, probs)


def calculate_cov_mat(x: np.ndarray, probs: Union[np.ndarray, None] = None, axis: int = 0) -> np.ndarray:

    """
    Estimates a covariance matrix based on a historical dataset and a set of probabilities.

    Parameters
    ----------
    x:
        The dataset to estimate covariance for.
    probs:
        The probabilities to weight the observations of the dataset by.
    axis:
        The axis to estimate over.

    Returns
    -------
    np.ndarray
        The estimated covariance matrix.

    """

    x = x.T if axis == 1 else x

    if probs is None:
        probs = np.repeat(1.0 / len(x), len(x))

    expected_x_squared = np.sum(probs[:, None, None] * np.einsum('ji, jk -> jik', x, x), axis=0)
    mu = probs @ x
    mu_squared = np.einsum('j, i -> ji', mu, mu)
    cov_mat = expected_x_squared - mu_squared

    return cov_mat


def calculate_cov_mat_factor(x: np.ndarray, z: np.ndarray, probs: Union[np.ndarray, None] = None,
                            axis: int = 0) -> np.ndarray:

    """
    Estimates a covariance matrix between a number of random variables and a set of factors based on a historical
    dataset and a set of probabilities.

    Parameters
    ----------
    x:
        The dataset to estimate covariance for.
    z:
        The dataset that acts as factors
    probs:
        The probabilities to weight the observations of the dataset by.
    axis:
        The axis to estimate over.

    Returns
    -------
    np.ndarray
        The estimated covariance matrix.

    """

    x = x.T if axis == 1 else x
    z = z.T if axis == 1 else z

    if probs is None:
        probs = np.repeat(1.0 / len(x), len(x))

    expected_x_squared = np.sum(probs[:, None, None] * np.einsum('ji, jk -> jik', x, z), axis=0)
    mu_x = probs @ x
    mu_z = probs @ z
    mu_squared = np.einsum('j, i -> ji', mu_x, mu_z)
    cov_mat = expected_x_squared - mu_squared

    return cov_mat


def calculate_outer_product(x: np.ndarray, z: np.ndarray, probs: Union[np.ndarray, None] = None,
                            axis: int = 0) -> np.ndarray:

    """
    Estimates the outer product for a pair of random vectors based on a historical
    dataset and a set of probabilities.

    Parameters
    ----------
    x:
        The dataset of the first random vector.
    z:
        The dataset of the second random vector.
    probs:
        The probabilities to weight the observations of the dataset by.
    axis:
        The axis to estimate over.

    Returns
    -------
    np.ndarray
        The estimated outer product.

    """

    if x.ndim == 1:
        x = (np.atleast_2d(x) if axis == 1 else np.atleast_2d(x).T)
    if z.ndim == 1:
        z = (np.atleast_2d(z) if axis == 1 else np.atleast_2d(z).T)

    x = x.T if axis == 1 else x
    z = z.T if axis == 1 else z

    if probs is None:
        probs = np.repeat(1.0 / len(x), len(x))

    expected_xz = np.sum(probs[:, None, None] * np.einsum('ji, jk -> jik', x, z), axis=0)

    return expected_xz


def calculate_corr_mat(x: np.ndarray, probs: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Estimates a correlation matrix based on a historical dataset and a set of probabilities.

    Parameters
    ----------
    x:
        The dataset to estimate correlations for.
    probs:
        The probabilities to weight the observations of the dataset by.
    axis:
        The axis to estimate over.

    Returns
    -------
    np.ndarray
        The estimated correlation matrix.

    """

    cov_mat = calculate_cov_mat(x, probs, axis)
    corr_mat = cov_to_corr_matrix(cov_mat)

    return corr_mat


"""
Utilities
"""


def cov_to_corr_matrix(cov_mat: np.ndarray) -> np.ndarray:

    """
    Transform a covariance matrix to a correlation matrix.

    Parameters
    ----------
    cov_mat:
        Covariance matrix.

    Returns
    -------
    np.ndarray
        Correlation matrix.

    """

    vols = np.sqrt(np.diag(cov_mat))
    corr_mat = cov_mat / np.outer(vols, vols)
    corr_mat[corr_mat < -1], corr_mat[corr_mat > 1] = -1, 1  # numerical error

    return corr_mat


def corr_to_cov_matrix(corr_mat: np.ndarray, vols: np.ndarray) -> np.ndarray:

    """
    Transform a covariance matrix to a correlation matrix.

    Parameters
    ----------
    corr_mat:
        Correlation matrix.
    vols:
        Volatilities.

    Returns
    -------
    np.ndarray
        Covariance matrix.

    """

    cov_mat = corr_mat * np.outer(vols, vols)

    return cov_mat


"""
Multivariate log-normal, mean and covariance
"""


def calculate_log_norm_mean(mu: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """
    Function that calculates the expected value of X,
    when :math:`\\log(X)` is multivariate normal

    Parameters
    ----------
    mu:
        Vector of expected values of log X
    covariance:
        Covariance matrix of log X+

    Returns
    -------
    float
        Expected value of X
    """

    return np.exp(mu + 0.5 * np.diag(covariance))


def calculate_log_norm_cov_mat(mu: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """
    Function that calculates the covariance matrix of X,
    when :math:`\\log(X)` is multivariate normal

    Parameters
    ----------
    mu:
        Vector of expected values of log X
    covariance:
        Covariance matrix of log X

    Returns
    -------
    float
        Covariance matrix of X
    """

    mu_l = calculate_log_norm_mean(mu, covariance)

    return np.outer(mu_l, mu_l) * (np.exp(covariance) - 1)
