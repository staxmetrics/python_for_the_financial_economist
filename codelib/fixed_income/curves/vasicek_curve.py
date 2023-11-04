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


class VasicekCurve(IRateCurve):

    """
    Class that implements a Vasicek Curve (cont. compounding)

    The zero rate for a given maturity, :math:`\\tau`,  is defined by

    .. math::

        r_{\\tau} = y_\\infty + \\frac{b(\\tau)}{\\tau} \\left(\\frac{\\beta^2}{4 \\kappa} b(\\tau) + r_t -
        y_\\infty \\right)

    where

    .. math::

        y_\\infty  = \\theta - \\frac{\\beta^2}{2 \\kappa^2}


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
            raise ZeroDivisionError('beta is not allowed to be zero')

        self.short_rate = short_rate
        self.theta = theta
        self.kappa = kappa
        self.beta = beta
        self.y_infty = theta - beta**2 / (2 * kappa**2)

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

        b = 1 / self.kappa * (1 - np.exp(- self.kappa * tenor))
        a = self.y_infty * (tenor - b) + self.beta**2 / (4 * self.kappa) * b**2

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

        b = 1 / self.kappa * (1 - np.exp(- self.kappa * tenors))
        a = self.y_infty * (tenors - b) + self.beta**2 / (4 * self.kappa) * b**2

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
