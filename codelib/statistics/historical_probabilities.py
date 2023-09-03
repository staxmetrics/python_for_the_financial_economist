import numpy as np
from typing import Union, Tuple
from scipy.interpolate import interp1d


def equal_weights(x: np.ndarray, axis: int = 0) -> np.ndarray:

    """
    Calculates equal weights

    Parameters
    ----------
    x:
    axis:

    Returns
    -------
    np.ndarray
    """

    n = x.shape[axis]

    return np.repeat(1.0 / n, n)



def calculate_time_crisp_probabilities(target_time_point: Union[int, float], time_points: np.ndarray,
                                       time_interval: Union[int, float]) -> np.ndarray:

    """
    Calculates crisp probabilities for an array of time points based on a target time point and a time interval.

    Parameters
    ----------
    target_time_point:
        The target time point.
    time_points:
        The array of time points to calculate probabilities for.
    time_interval:
        The interval used when defining points close enough to the target time point to have a probability > 0.

    Returns
    -------
    Crisp probabilities.

    """

    t_lower_bound = target_time_point - time_interval
    t_upper_bound = target_time_point

    indicator = (time_points >= t_lower_bound) & (time_points <= t_upper_bound)

    p_t = indicator / np.sum(indicator)

    return p_t


def calculate_exponential_decay_probabilities(target_time_point: Union[int, float], time_points: np.ndarray,
                                              half_life: Union[float, int]) -> np.ndarray:

    """
    Calculates exponential decay probabilities for an array of time points based on a target time point and a half life.

    Parameters
    ----------
    target_time_point:
        The target time point.
    time points:
        The array of time points to calculate probabilities for.
    half_life:
        The half life of the exponential decay.

    Returns
    -------
    Exponential decay probabilities.

    """

    numerator = np.exp(-np.log(2) / half_life * np.clip(target_time_point - time_points, 0, np.inf))
    denominator = np.sum(numerator)

    p_t = numerator / denominator

    return p_t


def crisp_conditioning_probs(x: np.ndarray, condition: Tuple = (-np.inf, np.inf), axis: int = 0) -> np.ndarray:
    """
    Creates an array of crisp condition probabilities.

    Currently only works for one state variable.

    Parameters
    ----------
    x:
        Array for setting the shape of.
    condition:
        Tuple with lower and upper bound.
    axis:
        The axis to create the weights over. Default is 0.

    Returns
    -------
    ndarray
        Array of equal weights.
    """

    probs = (x >= condition[0]) & (x <= condition[1])
    probs = probs / np.sum(probs)

    return probs
