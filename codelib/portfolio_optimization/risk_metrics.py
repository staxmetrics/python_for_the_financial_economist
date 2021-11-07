import numpy as np


def drawdown(index: np.ndarray):
    """
    Calculates the running draw down

    Parameters
    ----------
    index:
        Values of e.g. at equity index

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
        Values of e.g. at equity index

    Returns
    -------
    float
        Maximum drawdown

    """

    return drawdown(index)[0].min()