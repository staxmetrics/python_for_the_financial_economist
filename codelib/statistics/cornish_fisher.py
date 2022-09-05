import numpy as np
import scipy.stats as stats


def calculate_cornish_fisher_percentile(alpha: float, mu: float, sigma: float, skew: float, kurt: float):
    """
    Calculates the percentile, alpha, based on the Cornish-Fisher approximation

    Parameters
    ----------
    alpha:
    mu:
    sigma:
    skew:
    kurt:

    Returns
    -------
    float
        Percentile

    """

    z = stats.norm.ppf(alpha)
    he2 = np.polynomial.hermite_e.hermeval(z, [0.0, 0.0, 1.0])
    he3 = np.polynomial.hermite_e.hermeval(z, [0.0, 0.0, 0.0, 1.0])
    he13 = np.polynomial.hermite_e.hermeval(z, [0.0, -1.0, 0.0, -2.0])

    w = (z +
         he2 * skew / 6 +
         he3 * kurt / 24 +
         he13 * (skew ** 2) / 36)

    return mu + sigma * w
