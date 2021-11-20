import abc
from enum import Enum, unique
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from typing import List, Union


"""
Meucci - Fully Flexible Views: Theory and practice 
"""


@unique
class ViewType(Enum):
    Equality = 'EQ'
    LargerThan = 'GEQ'
    SmallerThan = 'LEQ'


class View(abc.ABC):

    def __init__(self, view_type: str):
        """
        Constructor

        Parameters
        ----------
        view_type
            Specifies view type.
        """

        self.view_type = ViewType(view_type.upper())

    @abc.abstractmethod
    def get_constraint_matrix(self):
        raise NotImplementedError("Abstract member get_constraint_matrix not implemented")

    @abc.abstractmethod
    def get_constraint_vector(self):
        raise NotImplementedError("Abstract member get_constraint_vector not implemented")


class MeanView(View):
    """
    Class that implements a mean view.
    """

    def __init__(self, view_type: str, factor: np.ndarray, mean_target: float):
        """
        Constructor

        Parameters
        ----------
        view_type
            Specifies view type.
        factor:
            Factor for which we have a view
        mean_target:
            Expected value of target under the view.
        """

        super(MeanView, self).__init__(view_type)

        self.constraint_matrix = factor
        self.constraint_vector = mean_target

    def get_constraint_matrix(self):
        return self.constraint_matrix

    def get_constraint_vector(self):
        return self.constraint_vector


class VolatilityView(View):
    """
    Class that implements a volatility view.

    """

    def __init__(self, view_type: str, factor: np.ndarray, expected_value: float, vol_target: float):
        """
        Constructor

        Parameters
        ----------
        view_type
            Specifies view type.
        factor:
            Factor for which we have a view
        expected_value:
            Expected value of factor.
        vol_target:
            Volatility of factor under view.
        """

        super().__init__(view_type)

        self.constraint_matrix = factor ** 2
        self.constraint_vector = vol_target ** 2 + expected_value ** 2

    def get_constraint_matrix(self):
        return self.constraint_matrix

    def get_constraint_vector(self):
        return self.constraint_vector


class CorrelationView(View):
    """
    Class that implements a correlation view.
    """

    def __init__(self, view_type: str, first_factor: np.ndarray, second_factor: np.ndarray,
                 first_expected_value: float, second_expected_value: float, first_vol: float,
                 second_vol: float, correlation_target: float):
        """
        Constructor

        Parameters
        ----------
        view_type
            Specifies view type.
        first_factor:
            Factor for which we have a view.
        second_factor:
            Factor for which we have a view.
        first_expected_value:
            Expected value of first factor.
        second_expected_value:
            Expected value of second factor.
        first_vol:
            Volatility of first factor.
        second_vol:
            Volatility of second factor.
        correlation_target:
            Correlation target under view.
        """

        super().__init__(view_type)

        self.constraint_matrix = first_factor * second_factor
        self.constraint_vector = correlation_target * first_vol * second_vol + first_expected_value * second_expected_value

    def get_constraint_matrix(self):
        return self.constraint_matrix

    def get_constraint_vector(self):
        return self.constraint_vector


class ProbabilitySolver:

    def __init__(self, initial_probs: np.ndarray, views: List[View]):

        """
        Constructor

        Parameters
        ----------
        initial_probs:
            Initial probabilities.
        views:
            List of Views.
        """

        self.init_probs = initial_probs
        self.init_log_probs = np.log(initial_probs)

        self.total_constraint_matrix = None
        self.total_constraint_matrix_t = None
        self.total_constraint_vector = None
        self.inequality_constraint_indicators = []

        # prepare views for solving probabilities
        self.prepare_constraints(views)

    def store_inequality(self, constraint_vector, is_inequality: bool):

        """
        Functions that stores a boolean indicator for whether the constraint is inequality or not.

        Parameters
        ----------
        constraint_vector:
            Constraint vector.
        is_inequality:
            Bolean indicator.


        Returns
        -------
        NoReturn

        """

        if isinstance(constraint_vector, (float, int)):
            self.inequality_constraint_indicators.append(is_inequality)
        else:
            self.inequality_constraint_indicators.append([is_inequality for constraint in constraint_vector])

    def prepare_constraints(self, views: List[View]):

        """
        Function that prepares constraints.

        Parameters
        ----------
        views:
            List of views.

        Returns
        -------
        NoReturn

        """

        size = len(self.init_probs)
        self.total_constraint_matrix = np.ones((1, size))
        self.total_constraint_vector = np.ones(1)
        self.store_inequality(1.0, False)

        for view in views:
            if view.view_type == ViewType.Equality:
                self.total_constraint_matrix = np.vstack((self.total_constraint_matrix, view.get_constraint_matrix()))
                self.total_constraint_vector = np.vstack((self.total_constraint_vector, view.get_constraint_vector()))
                self.store_inequality(view.get_constraint_vector(), False)
            elif view.view_type == ViewType.LargerThan:
                self.total_constraint_matrix = np.vstack((self.total_constraint_matrix, -view.get_constraint_matrix()))
                self.total_constraint_vector = np.vstack((self.total_constraint_vector, -view.get_constraint_vector()))
                self.store_inequality(view.get_constraint_vector(), True)
            elif view.view_type == ViewType.SmallerThan:
                self.total_constraint_matrix = np.vstack((self.total_constraint_matrix, view.get_constraint_matrix()))
                self.total_constraint_vector = np.vstack((self.total_constraint_vector, view.get_constraint_vector()))
                self.store_inequality(view.get_constraint_vector(), True)
            else:
                raise Exception('Unknown ViewType')

        self.total_constraint_matrix_t = np.transpose(self.total_constraint_matrix)
        self.total_constraint_vector = self.total_constraint_vector.flatten()

    def objective_function(self, params: np.ndarray) -> float:

        """
        Objective function.

        See https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1213325 for further details.

        Parameters
        ----------
        params:
            Parameter vector to optimize over (lagrange multipliers).

        Returns
        -------
        float
            The Lagrange dual function (negative).

        """

        x = self.calculate_probabilities(params)
        x = np.maximum(x, 10 ** (-32))

        lagrangian = x.T @ (np.log(x) - self.init_log_probs) + params @ (self.total_constraint_matrix @ x -
                                                                         self.total_constraint_vector)

        return -lagrangian

    def calculate_probabilities(self, params: np.ndarray) -> np.ndarray:

        """
        Calculates probabilities

        Parameters
        ----------
        params:
            Parameter vector to optimize over (lagrange multipliers).

        Returns
        -------
        np.ndarray
            Vector of probabilities.


        """

        x = self.init_log_probs - 1.0 - (self.total_constraint_matrix_t @ params).flatten()

        return np.exp(x)

    def minimize_entropy(self):

        """
        Minimize entropy to obtain probabilities under the views.

        Returns
        -------
        np.ndarray
            Probabilities.

        """

        # initial parameters
        init_params = np.repeat(0.01, len(self.inequality_constraint_indicators))

        # add zero bound for inequality constraints
        bounds = [(0.0, None) if is_inequality else (None, None)
                  for is_inequality in self.inequality_constraint_indicators]

        # optimization
        # TODO:: Add analytical gradient and hessian
        opt_params, f, d = fmin_l_bfgs_b(self.objective_function, init_params, bounds=bounds,
                                         approx_grad=True)

        # calculate probs
        probs = self.calculate_probabilities(opt_params)

        if np.abs(1.0 - np.sum(probs)) > 0.001:
            raise Exception('Entropy optimization did not converge')

        return probs


def effective_sample_size_entropy(probabilities: np.ndarray, relative: bool = False):
    """
    Parameters
    ----------
    probabilities:
        vector of probabilities
    relative:
        if True returns the relative number of effective probabilities.

    Returns
    -------
    number of effective probabilities
    """

    j = len(probabilities)
    j_hat = np.exp(-np.sum(probabilities * np.log(probabilities)))

    return j_hat / j if relative else j_hat


"""
Moments, statistics, etc.
"""


def calculate_variance(x: np.ndarray, probs=None, axis=0):

    m = np.average(x, weights=probs, axis=axis)

    return np.average(np.square(x - m), weights=probs)


def calculate_std(x: np.ndarray, probs=None, axis=0):

    return np.sqrt(calculate_variance(x, probs, axis))


def calculate_covariance(x, y, probs=None, axis=0):

    m_x = np.average(x, weights=probs, axis=axis)
    m_y = np.average(y, weights=probs, axis=axis)

    m_xy = np.average(x*y, weights=probs, axis=axis)

    return m_xy - m_x * m_y


def calculate_correlation(x, y, probs=None, axis=0):

    c = calculate_covariance(x, y, probs, axis)
    std_x = calculate_std(x, probs, axis)
    std_y = calculate_std(y, probs, axis)

    return c / (std_x * std_y)


def weighted_percentile(x: np.array, p: Union[float, np.ndarray], probs=None, axis=0):

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
    axis
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
