from codelib.statistics.moments import calculate_cov_mat

import numpy as np
from typing import Union


class LinearRegression:

    """
    Class the implements linear regression allowing for scenario probabilities
    """

    def __init__(self, y: np.ndarray, x: np.ndarray, probs: Union[bool, None]):

        ...
