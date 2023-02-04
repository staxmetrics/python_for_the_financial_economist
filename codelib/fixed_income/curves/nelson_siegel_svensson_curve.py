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


class NelsonSiegelSvenssonCurve(IRateCurve):

    """
    Class that implements a Nelson-Siegel-Svensson yield curve (cont. compounding)

    The zero rate for a given maturity, :math:`\\tau`,  is defined by

    .. math::

        r_{\\tau} = \\beta_0 + \\beta_1 \\frac{1 - \\exp(- \\tau / \\theta_1)}{\\tau / \\theta_1} +
        \\beta_2 \\left[ \\frac{1 - \\exp(- \\tau / \\theta_1)}{\\tau / \\theta_1}
        - \\exp(-\\tau /\\theta_1) \\right] + \\beta_3 \\left[ \\frac{1 - \\exp(- \\tau / \\theta_2)}{\\tau / \\theta_2}
        - \\exp(-\\tau /\\theta_2) \\right]

    The instantaneous forward rate is given by

    .. math::

        f_{\\tau} = \\beta_0 + \\beta_1 \\exp(- \\tau / \\theta_1) +
        \\beta_2 \\tau / \\theta_1 \\exp(-\\tau /\\theta_1) + \\beta_3 \\tau / \\theta_2 \\exp(-\\tau /\\theta_2)

    """

    def __init__(self,
                 theta1: float,
                 theta2: float,
                 beta0: float,
                 beta1: float,
                 beta2: float,
                 beta3: float):

        """
        Constructor

        Parameters
        ----------
        theta1:
            The decay parameter. The larger, the faster decay.
            Governs where the curvature component reaches its maximum.
        theta2:
            The decay parameter for the added component.
        beta0:
            Long-run level of interest rates.
        beta1:
            Part of the slope component.
        beta2:
            Part of the medium-term or curvature component.
        beta3:
            Part of the added component.
        """

        if math.isclose(theta1, 0.0):
            raise ZeroDivisionError('theta1 is not allowed to be zero')

        if math.isclose(theta2, 0.0):
            raise ZeroDivisionError('theta2 is not allowed to be zero')

        self.theta1 = theta1
        self.theta2 = theta2
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3

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

        factors = self.calculate_factors(tenor, self.theta1, self.theta2)

        # calculate zero as factor loadings multiplied with factors
        zr = self.beta0 * factors[0] + self.beta1 * factors[1] + self.beta2 * factors[2] + self.beta3 * factors[3]

        return zr

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

        if tenor < 1.0e-6:
            return 1.0

        zero_rate = self.zero_rate(tenor)

        return np.exp(-zero_rate * tenor)

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

        # calculate factors
        factors = self.calculate_factors(tenors, self.theta1, self.theta2)

        # calculate zero as factor loadings multiplied with factors
        zr = self.beta0 * factors[0] + self.beta1 * factors[1] + self.beta2 * factors[2] + self.beta3 * factors[3]

        return zr

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

        zero_rates = self.zero_rate_vector(tenors)

        return np.exp(-zero_rates * tenors)

    @staticmethod
    def calculate_factors(tenors: Union[float, int, np.array], theta1: float, theta2: float) -> np.ndarray:

        """
        Calculate factors.

        Parameters
        ----------
        tenors:
            Tenors for which to calculate factors.
        theta1:
            Theta for the first 3 components.
        theta2:
            Theta for added component.

        Returns
        -------
        np.ndarray

        """

        # level
        first_factor = np.ones_like(tenors)
        # slope
        second_factor = (1 - np.exp(-tenors / theta1)) / (tenors / theta1)
        # curvature
        third_factor = second_factor - np.exp(-tenors / theta1)
        #
        fourth_factor = (1 - np.exp(-tenors / theta2)) / (tenors / theta2) - np.exp(-tenors / theta2)

        return np.array([first_factor, second_factor, third_factor, fourth_factor])

    def get_coefficients(self) -> Dict:

        """
        Gets the coefficients.

        Returns
        -------
        Dict
            Dictionary with theta, beta0, beta1 and beta2.

        """

        return dict(theta1=float(self.theta1), theta2=float(self.theta2), beta0=self.beta0, beta1=self.beta1,
                    beta2=self.beta2, beta3=self.beta3)
