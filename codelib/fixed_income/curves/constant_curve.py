"""
Contains implementation of the Nelson-Siegel curves.
"""

# import curve interface
from codelib.fixed_income.curves.curve_interface import IRateCurve

# import other python packages
import numpy as np
import math
from scipy import optimize
from typing import Union, Dict


class ConstantCurve(IRateCurve):

    """
    Class that implements a constant curve

    """

    def __init__(self, rate: float):

        """
        Constructor

        Parameters
        ----------
        rate:
            Constant rate  (cont.).
        """

        self.rate = rate

    def zero_rate(self, tenor: float) -> float:

        """
        Gets the zero rate for a specific tenor.

        Parameters
        ----------
        tenor:
            Relevant tenor, :math:`\\tau`

        Returns
        -------
        float
            Zero rate for the specific tenor.
        """
        return self.rate

    def discount_factor(self, tenor: float) -> float:

        """
        Gets the discount factor for a specific tenor.

        Parameters
        ----------
        tenor:
            Relevant tenor, :math:`\\tau`

        Returns
        -------
        float
            Discount factor.
        """

        return np.exp(-self.rate * tenor)

    def zero_rate_vector(self, tenors: np.ndarray) -> np.ndarray:

        """
        Gets the zero rates based for specific tenors.

        Parameters
        ----------
        tenors:
            Relevant tenors.

        Returns
        -------
        np.ndarray
            Zero rates for specific tenors.
        """
        return np.ones_like(tenors) * self.rate

    def discount_factor_vector(self, tenors: np.ndarray) -> np.ndarray:

        """
        Gets the discount factors for specific tenors.

        Parameters
        ----------
        tenors:
            Relevant tenors.

        Returns
        -------
        np.ndarray
            Discount factors.
        """

        return np.exp(-self.rate * tenors)
