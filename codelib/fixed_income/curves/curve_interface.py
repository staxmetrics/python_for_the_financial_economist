import abc
import numpy as np
from typing import Union, List


class IRateCurve(abc.ABC):

    """
    Abstract class/interface for discount curve.

    Serve as base class for different curve implementations.
    """

    @abc.abstractmethod
    def zero_rate(self, tenor: float) -> float:
        """
        Gets the cont. comp. zero rate based for a specific tenor.

        Parameters
        ----------
        tenor:
            Relevant tenor.

        Returns
        -------
        float
            Zero rate for the specific tenor.
        """

        raise NotImplementedError("Abstract member zero_rate not implemented")

    @abc.abstractmethod
    def discount_factor(self, tenor: float) -> float:

        """
        Gets the discount factor for a specific tenor.

        Parameters
        ----------
        tenor:
            Relevant tenor.

        Returns
        -------
        float
            Discount factor.
        """

        raise NotImplementedError("Abstract member discount_factor not implemented")

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

        return np.array([self.zero_rate(t) for t in tenors])

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

        return np.array([self.discount_factor(t) for t in tenors])

