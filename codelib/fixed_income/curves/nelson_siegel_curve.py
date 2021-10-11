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


class NelsonSiegelCurve(IRateCurve):

    """
    Class that implements a Nelson-Siegel yield curve (cont. compounding)

    The zero rate for a given maturity, :math:`\\tau`,  is defined by

    .. math::

        r_{\\tau} = \\beta_0 + \\beta_1 \\frac{1 - \\exp(- \\tau / \\theta)}{\\tau / \\theta} +
        \\beta_2 \\left[ \\frac{1 - \\exp(- \\tau / \\theta)}{\\tau / \\theta}
        - \\exp(-\\tau /\\theta) \\right]

    The instantaneous forward rate is given by

    .. math::

        f_{\\tau} = \\beta_0 + \\beta_1 \\exp(- \\tau / \\theta) +
        \\beta_2 \\tau / \\theta \\exp(-\\tau /\\theta)

    """

    def __init__(self,
                 theta: float,
                 beta0: float,
                 beta1: float,
                 beta2: float):

        """
        Constructor

        Parameters
        ----------
        theta:
            The decay parameter. The larger, the faster decay.
            Governs where the curvature component reaches its maximum.
        beta0:
            Long-run level of interest rates.
        beta1:
            Part of the slope component.
        beta2:
            Part of the medium-term or curvature component.
        """

        if math.isclose(theta, 0.0):
            raise ZeroDivisionError('theta is not allowed to be zero')

        self.theta = theta
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2

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

        factors = self.calculate_factors(tenor, self.theta)

        # calculate zero as factor loadings multiplied with factors
        zr = self.beta0 * factors[0] + self.beta1 * factors[1] + self.beta2 * factors[2]

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
        factors = self.calculate_factors(tenors, self.theta)

        # calculate zero as factor loadings multiplied with factors
        zr = self.beta0 * factors[0] + self.beta1 * factors[1] + self.beta2 * factors[2]

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
    def calculate_factors(tenors: Union[float, int, np.array], theta: float) -> np.ndarray:

        """
        Calculate factors.

        Parameters
        ----------
        tenors:
            Tenors for which to calculate factors.
        theta:


        Returns
        -------
        np.ndarray

        """

        # level
        first_factor = np.ones_like(tenors)
        # slope
        second_factor = (1 - np.exp(-tenors / theta)) / (tenors / theta)
        # curvature
        third_factor = second_factor - np.exp(-tenors / theta)

        return np.array([first_factor, second_factor, third_factor])

    def get_coefficients(self) -> Dict:

        """
        Gets the coefficients.

        Returns
        -------
        Dict
            Dictionary with theta, beta0, beta1 and beta2.

        """

        return dict(theta=float(self.theta), beta0=self.beta0, beta1=self.beta1, beta2=self.beta2)

    @classmethod
    def objective_function(cls, params: np.ndarray, tenors: np.ndarray, cash_flow_matrix: np.ndarray,
                           market_prices: np.ndarray) -> Union[float, np.ndarray]:

        """
        Calculates the objective function when fitting to zero rates.

        Parameters
        ----------
        params
            Parameter vector.
        tenors:
            Vector of tenors.
        cash_flow_matrix:
            Cash flow matrix.
        market_prices:
            Market prices.

        Returns
        -------
        np.ndarray
            Objective function.

        """

        fitted_zero_rates = params[1:]@cls.calculate_factors(tenors, params[0])
        discount_factors = np.exp(- tenors * fitted_zero_rates)

        return market_prices - cash_flow_matrix @ discount_factors

    @classmethod
    def objective_function_known_theta(cls, params: np.ndarray, tenors: np.ndarray, cash_flow_matrix: np.ndarray,
                                       market_prices: np.ndarray, theta: float) -> Union[float, np.ndarray]:

        """
        Calculates the objective function when fitting to zero rates.

        Parameters
        ----------
        params
            Parameter vector.
        tenors:
            Vector of tenors.
        cash_flow_matrix:
            Cash flow matrix.
        market_prices:
            Market prices.
        theta:
            Theta parameter.

        Returns
        -------
        np.ndarray
            Objective function.

        """

        fitted_zero_rates = params@cls.calculate_factors(tenors, theta)
        discount_factors = np.exp(-tenors * fitted_zero_rates)

        return market_prices - cash_flow_matrix @ discount_factors

    @classmethod
    def objective_function_grid(cls, theta: float, tenors: np.ndarray, cash_flow_matrix: np.ndarray,
                                market_prices: np.ndarray, starting_params: np.array = (0.1, 0.1, 0.1)):

        """
        Calculates the objective function when fitting to zero rates, using grid search.

        Parameters
        ----------
        theta:
            The theta value.
        tenors:
            Vector of tenors.
        cash_flow_matrix:
            Cash flow matrix.
        market_prices:
            Market prices.
        starting_params:
            The starting beta parameters.

        Returns
        -------
        np.ndarray
            Vector of deviations between observed and fitted zero rates.

        """

        curve = cls.create_from_cash_flow_matrix(tenors,
                                                 cash_flow_matrix,
                                                 market_prices,
                                                 theta,
                                                 starting_params)

        x_new = np.array(list(curve.get_coefficients().values()))

        residuals = cls.objective_function(x_new, tenors, cash_flow_matrix, market_prices)

        sosr = sum(residuals ** 2)

        return sosr

    @classmethod
    def create_from_cash_flow_matrix(cls,
                                     payments_dates: np.ndarray,
                                     cash_flow_matrix: np.ndarray,
                                     market_prices: np.ndarray,
                                     theta: float = None,
                                     starting_params: np.ndarray = None,
                                     grid: slice = None,
                                     grid_points=99999):

        if theta is not None:

            if starting_params is None:
                starting_params = np.array([0.01, 0.01, 0.01])

            ls = optimize.least_squares(cls.objective_function_known_theta,
                                        x0=starting_params, args=(payments_dates, cash_flow_matrix,
                                                                  market_prices, theta))

            beta0, beta1, beta2 = ls.x

            return NelsonSiegelCurve(theta=theta, beta0=beta0, beta1=beta1, beta2=beta2)

        else:

            if grid:

                if starting_params is None:
                    starting_params = np.array([0.01, 0.01, 0.01])

                theta = optimize.brute(cls.objective_function_grid,
                                       ranges=(grid, ),
                                       args=(payments_dates, cash_flow_matrix, market_prices, starting_params),
                                       Ns=grid_points)

                return cls.create_from_cash_flow_matrix(payments_dates, cash_flow_matrix,
                                                        market_prices, theta[0])

            else:

                if starting_params is None:
                    starting_params = np.array([2.0, 0.01, 0.01, 0.01])

                ls = optimize.least_squares(cls.objective_function, x0=starting_params,
                                            args=(payments_dates, cash_flow_matrix, market_prices))

                theta, beta0, beta1, beta2 = ls.x

                return NelsonSiegelCurve(theta=theta, beta0=beta0, beta1=beta1, beta2=beta2)
