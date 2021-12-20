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


class CIRCurve(IRateCurve):

    """
    Class that implements a CIR Curve (cont. compounding)
    """

    def __init__(self,
                 short_rate: float,
                 theta: float,
                 kappa: float,
                 beta: float):

        """
        Constructor

        Parameters
        ----------
        short_rate:
            Initial short rate.
        theta:
            Long-run level of the short rate (under Q).
        kappa:
            Speed of mean reversion.
        beta:
            Volatility of short rate.
        """

        if math.isclose(beta, 0.0):
            raise ZeroDivisionError('theta is not allowed to be zero')

        self.short_rate = short_rate
        self.theta = theta
        self.kappa = kappa
        self.beta = beta
        self.gamma = np.sqrt(self.kappa**2 + 2 * self.beta**2)

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

        temp1 = np.exp(self.gamma * tenor) - 1.0

        b = 2 * temp1 / ((self.gamma + self.kappa) * temp1 + 2 * self.gamma)
        a = -2 * self.kappa * self.theta / (self.beta**2) * (np.log(2 * self.gamma) +
                                                             0.5 * (self.kappa + self.gamma) * tenor -
                                                             np.log((self.kappa + self.gamma) * temp1 + 2 * self.gamma))

        rate = a / tenor + b / tenor * self.short_rate

        return rate

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

        rate = self.zero_rate(tenor)

        return np.exp(-rate * tenor)

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

        temp1 = np.exp(self.gamma * tenors) - 1.0

        b = 2 * temp1 / ((self.gamma + self.kappa) * temp1 + 2 * self.gamma)
        a = -2 * self.kappa * self.theta / (self.beta**2) * (np.log(2 * self.gamma) +
                                                             0.5 * (self.kappa + self.gamma) * tenors -
                                                             np.log((self.kappa + self.gamma) * temp1 + 2 * self.gamma))

        rate = a / tenors + b / tenors * self.short_rate

        return rate

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

        rates = self.zero_rate_vector(tenors)

        return np.exp(-rates * tenors)
