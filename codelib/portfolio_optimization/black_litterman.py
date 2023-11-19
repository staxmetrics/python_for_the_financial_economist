import numpy as np
from scipy.stats import chi2
from typing import Union, NoReturn


def calculate_capm_expected_return(cov_mat: np.ndarray, w_m: np.ndarray, lam: float) -> np.ndarray:
    """
    Calculates the equilibrium return for the market model.

    Parameters
    ----------
    cov_mat:
        Covariance matrix of returns.
    w_m:
        Market cap weights.
    lam
        Risk aversion parameters. We can think of it as the excess return of the market divided by its variance.

    Returns
    -------
    np.ndarray
        Expected returns implied by the market model.

    """

    return lam * cov_mat @ w_m


def calculate_bl_mu(pi: float, tau: float, cov_mat: np.ndarray, view_mat: np.ndarray,
                    view_vec: np.ndarray, view_cov_mat: np.ndarray = None, confidence: float = 1.0)\
        -> np.ndarray:

    """
    Calculates the posterior expected returns in the Black-Litterman model

    Parameters
    ----------
    pi:
        The expected return from the market model.
    tau:
        Parameter that scales the covariance matrix. The smaller, the less uncertainty.
    cov_mat:
        The covariance matrix of the market model.
    view_mat:
        The "pick" matrix which defines the views.
    view_vec:
        The vector that defines the views.
    view_cov_mat:
        Covariance matrix of the views (prior). If None, then it is calculated by the formula suggested
        by Meucci (2008)
    confidence:
        The confidence level. Only necessary if view_cov_mat is not provided.

    Returns
    -------
    np.ndarray
        The posterior expected value.

    """

    if view_cov_mat is None:
        view_cov_mat = (view_mat @ cov_mat @ view_mat.T) / confidence

    part_2 = np.linalg.inv(tau * view_mat @ cov_mat @ view_mat.T + view_cov_mat)
    part_3 = view_vec - view_mat @ pi

    mu_bl = pi + tau * cov_mat @ view_mat.T @ part_2 @ part_3

    return mu_bl


def calculate_bl_sigma(tau: float, cov_mat: np.ndarray, view_mat: np.ndarray, view_cov_mat: np.ndarray = None,
                       confidence: float = 1.0) -> np.ndarray:
    """
    Calculates the posterior covariance matrix in the Black-Litterman model

    Parameters
    ----------
    tau:
        Parameter that scales the covariance matrix. The smaller, the less uncertainty.
    cov_mat:
        The covariance matrix of the market model.
    view_mat:
        The "pick" matrix which defines the views.
    view_cov_mat:
        Covariance matrix of the views (prior). If None, then it is calculated by the formula suggested
        by Meucci (2008)
    confidence:
        The confidence level. Only necessary if view_cov_mat is not provided.

    Returns
    -------
    np.ndarray
        The posterior covariance matrix.

    """

    if view_cov_mat is None:
        view_cov_mat = (view_mat @ cov_mat @ view_mat.T) / confidence

    part_1 = (1 + tau) * cov_mat
    part_2 = np.linalg.inv(tau * view_mat @ cov_mat @ view_mat.T + view_cov_mat)

    sigma_bl = part_1 - tau**2 * cov_mat @ view_mat.T @ part_2 @ view_mat @ cov_mat

    return sigma_bl


"""
Class that implements Black-Litterman a la Meucci (2008)
"""


class BlackLitterman:

    """
    Class the implements an alternative version of Black-Litterman, see

        Meucci (2008) - The Black-Litterman Approach: Original Model and Extensions

    """

    def __init__(self, cov_mat, pi=None, market_weights=None, lam=2.4, view_mat=None, view_vec=None):

        """
        Constructor

        Parameters
        ----------
        cov_mat:
            Covariance matrix of reference model.
        pi:
            Expected return vector of reference model.
        market_weights:
            Market weights of the market portfolio. This will be used to calculate pi if it is not provided.
            Either pi or market_weights must be provided.
        lam:
            Lambda parameter.
        view_mat:
            View or pick matrix.
        view_vec
            View vectors.

        Returns
        -------
        BlackLitterman

        """

        self.num_assets = len(cov_mat)

        # market / reference model
        self.cov_mat = cov_mat
        self.pi = self.calculate_eq_expected_return(cov_mat, market_weights, lam) if pi is None else pi
        self.market_weights = market_weights

        # the views
        self.view_mat = view_mat
        self.view_vec = view_vec
        self.view_cov_mat = None

        # the posterior
        self.mean_posterior = None
        self.cov_mat_posterior = None

    def calculate_posterior_distribution(self, confidence=1.0) -> NoReturn:

        """
        Calculates the posterior distribution.

        Parameters
        ----------
        confidence:
            Confidence level.

        Returns
        -------
        NoReturn

        """

        if self.view_mat is None or self.view_vec is None:
            self.mean_posterior = self.pi
            self.cov_mat_posterior = self.cov_mat
        else:
            # calculate covariance matrix of views
            self.view_cov_mat = self.calculate_view_cov_mat(self.cov_mat, self.view_mat, confidence)
            # calculate posterior expected return
            self.mean_posterior = self.calculate_posterior_mean(self.pi, self.cov_mat, self.view_mat,
                                                                self.view_vec, self.view_cov_mat)
            # calculate posterior covariance matrix
            self.cov_mat_posterior = self.calculate_posterior_cov_mat(self.cov_mat, self.view_mat, self.view_cov_mat)

    def clear_views(self) -> NoReturn:

        """
        Sets all the views to None.

        Returns
        -------
        NoReturn

        """

        self.view_cov_mat = None
        self.view_mat = None
        self.view_vec = None

    def add_view(self, view: np.ndarray, target: np.ndarray) -> NoReturn:

        """
        Adds a view.

        Parameters
        ----------
        view:
            Vector defining the view.
        target:
            Target for the view.

        Returns
        -------
        NoReturn
        """

        if self.view_mat is None or self.view_vec is None:
            self.view_mat = view
            self.view_vec = target
        else:
            self.view_mat = np.vstack((self.view_mat, view))
            self.view_vec = np.concatenate((self.view_vec, target))

    def add_equality_view(self, index: int, target: float) -> NoReturn:
        """
        Adds an equality view.

        Parameters
        ----------
        index:
            Index specifying which asset for which we have a view.
        target:
            Target for the view.

        Returns
        -------
        NoReturn
        """

        view = np.zeros((1, self.num_assets))
        view[0, index] = 1.0

        target = np.atleast_1d(target)

        self.add_view(view, target)

    def add_diff_view(self, index_first: int, index_second: int, target: float) -> NoReturn:
        """
        Adds a difference view.

        Parameters
        ----------
        index_first:
            Index specifying asset assigned 1.0
        index_second:
            Index specifying asset assigned -1.0
        target:
            Target for the view.

        Returns
        -------
        NoReturn
        """

        view = np.zeros((1, self.num_assets))
        view[0, index_first] = 1.0
        view[0, index_second] = -1.0

        target = np.atleast_1d(target)

        self.add_view(view, target)

    def calculate_consistency_index(self) -> float:

        """
        Calculates a consistency index of the views using the reference model as null hypothesis.

        Returns
        -------
        float
            Consistency index.
        """

        distance = BlackLitterman.calculate_view_distance(self.pi, self.mean_posterior, self.cov_mat)

        return 1 - chi2.cdf(distance, self.num_assets)

    def calculate_sensitivity_index(self) -> np.ndarray:

        """
        Calculates a sensitivity of the views using the reference model as null hypothesis.

        Returns
        -------
        np.ndarray
            Consistency index.
        """

        p_mat = self.view_mat
        q_mat = self.view_cov_mat

        distance = BlackLitterman.calculate_view_distance(self.pi, self.mean_posterior, self.cov_mat)

        part1 = -2 * chi2.pdf(distance, self.num_assets)
        part2 = np.linalg.inv(p_mat @ self.cov_mat @ p_mat.T + q_mat) @ p_mat @ (self.mean_posterior - self.pi)

        return part1 * part2

    @staticmethod
    def calculate_posterior_mean(pi: float, cov_mat: np.ndarray, view_mat: np.ndarray, view_vec: np.ndarray,
                                 view_cov_mat: np.ndarray = None, confidence: float = 1.0) -> np.ndarray:

        """
        Calculates the posterior expected returns in the Black-Litterman model

        Parameters
        ----------
        pi:
            The expected return from the market model.
        cov_mat:
            The covariance matrix of the market model.
        view_mat:
            The "pick" matrix which defines the views.
        view_vec:
            The vector that defines the views.
        view_cov_mat:
            Covariance matrix of the views (prior). If None, then it is calculated by the formula suggested
            by Meucci (2008)
        confidence:
            The confidence level. Only necessary if view_cov_mat is not provided.

        Returns
        -------
        np.ndarray
            The posterior expected value.

        """

        if view_cov_mat is None:
            view_cov_mat = BlackLitterman.calculate_view_cov_mat(cov_mat, view_mat, confidence)

        part_2 = np.linalg.inv(view_mat @ cov_mat @ view_mat.T + view_cov_mat)
        part_3 = view_vec - view_mat @ pi

        return pi + cov_mat @ view_mat.T @ part_2 @ part_3

    @staticmethod
    def calculate_posterior_cov_mat(cov_mat: np.ndarray, view_mat: np.ndarray, view_cov_mat: np.ndarray = None,
                                    confidence: float = 1.0) -> np.ndarray:
        """
        Calculates the posterior covariance matrix in the Black-Litterman model

        Parameters
        ----------
        cov_mat:
            The covariance matrix of the market model.
        view_mat:
            The "pick" matrix which defines the views.
        view_cov_mat:
            Covariance matrix of the views (prior). If None, then it is calculated by the formula suggested
            by Meucci (2008)
        confidence:
            The confidence level. Only necessary if view_cov_mat is not provided.

        Returns
        -------
        np.ndarray
            The posterior covariance matrix.

        """

        if view_cov_mat is None:
            view_cov_mat = BlackLitterman.calculate_view_cov_mat(cov_mat, view_mat, confidence)

        part_2 = np.linalg.inv(view_mat @ cov_mat @ view_mat.T + view_cov_mat)

        return cov_mat - cov_mat @ view_mat.T @ part_2 @ view_mat @ cov_mat

    @staticmethod
    def calculate_view_cov_mat(cov_mat: np.ndarray, view_mat: np.ndarray, confidence=1.0):

        return (view_mat @ cov_mat @ view_mat.T) / confidence

    @staticmethod
    def calculate_eq_expected_return(cov_mat: np.ndarray, market_weights: np.ndarray, lam=2.4) -> np.ndarray:

        """
        Calculate the equilibrium expected return given the covariance matrix, market weights and lambda

        Parameters
        ----------
        cov_mat:
            The covariance matrix.
        market_weights
            The market cap weights.
        lam
            Lambda related to the risk aversion.

        Returns
        -------
        np.ndarray
            Vector of equilibrium expected returns.

        """

        if market_weights is None:
            raise ValueError("Market weights cannot be None")

        return lam * cov_mat @ market_weights

    @staticmethod
    def calculate_view_distance(pi: np.ndarray, mu_view: np.ndarray, cov_mat: np.ndarray) -> float:

        return (mu_view - pi) @ np.linalg.inv(cov_mat) @ (mu_view - pi)
