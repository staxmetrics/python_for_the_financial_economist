"""
Contains cash flow objects.
"""

import numpy as np
from codelib.fixed_income.curves.curve_interface import IRateCurve
from codelib.fixed_income.curves.constant_curve import ConstantCurve
from scipy.optimize import bisect

from typing import Union, List


class CashFlow:

    """
    Class that makes it easy to work with cash flows
    """

    def __init__(self, time_points: Union[np.ndarray, List], flows: Union[np.ndarray, List]):

        """
        Constructor

        Parameters
        ----------
        time_points:
            Time points for cash flows.
        flows:
            Cash flows.
        """

        if not isinstance(time_points, (List, np.ndarray)):
            raise TypeError('time_points and flows must be a list or a np.ndarray')
        if not isinstance(flows, (List, np.ndarray)):
            raise TypeError('time_points and flows must be a list or a np.ndarray')

        time_points = np.asarray(time_points)
        flows = np.asarray(flows)

        self.time_points = time_points
        self.flows = flows
        self.number_of_time_points = len(self.time_points)

        if self.number_of_time_points != len(self.flows):
            raise ValueError('Number of time points and cash flows does not match')

    def present_value(self,
                      curve: IRateCurve,
                      time_shift: float = 0.0,
                      include_time_zero_value: bool = False) -> float:

        """
        Calculates the present value.

        Parameters
        ----------
        curve:
            Discount curve used to calculate the present value of the cash flow.
        time_shift:
            Start time for present value calculation.
        include_time_zero_value:
            Indicates whether time 0 value is included.

        Returns
        -------
        float
            Present value of cash flow.

        """

        if not isinstance(curve, IRateCurve):
            raise TypeError("Curve is not derived from IRateCurve")

        time_points = self.time_points - time_shift

        if include_time_zero_value:
            indicator = (time_points >= 0.0)
        else:
            indicator = (time_points > 0.0)

        flows = self.flows[indicator]
        time_points = time_points[indicator]

        if time_points.size == 0:
            return 0.0
        else:
            discount_factors = curve.discount_factor_vector(tenors=time_points)
            pv = np.sum(flows * discount_factors)
            return float(pv)

    def get_cash_flows(self, time_shift: float = 0.0) -> np.ndarray:
        """
        Gets cash flows after a given time point.

        Parameters
        ----------
        time_shift:
            Time shift.

        Returns
        -------
        np.ndarray
            Array of cash flows.

        """

        return self.flows[self.time_points > time_shift]

    def get_time_points(self, time_shift: float = 0.0) -> np.ndarray:

        """
        Gets time points for cash flows after a given time point.

        Parameters
        ----------
        time_shift:
            Time shift.

        Returns
        -------
        np.ndarray
            Array of time points for cash flows.

        """

        if np.any(self.time_points > time_shift):
            return self.time_points[self.time_points > time_shift] - time_shift
        else:
            return np.array([])

    def market_value_deviation(self, y: float, market_value: float, time_shift: float) -> float:

        """
        Calculates the deviation between the provided market value of the cash flow and the present value obtained using
        the constant yield, :math:`y`:

        .. math::

            \\text{MV deviation} = MV - \\sum_{i=1}^n t_i C_i \\cdot D_i

        where :math:`i` is the cash-flow index, :math:`C_i` is the :math:`i`'th cash flow, :math:`D_i` is the discount
        factor calculated using the yield to maturity, :math:`y`, e.g.  :math:`D_i=e^{- t_i \\cdot y}` using cont.
        discounting, and :math:`t_i` is the time in years until the :math:`i`'th cash flow.

        Parameters
        ----------
        y: float
            Yield to maturity.
        market_value: float
            Market value of cash flow.
        time_shift: float
            Time shift.

        Returns
        -------
        float
            Deviation between market value and present value af cash flow with constant yield, :math:`y`.

        """

        const_curve = ConstantCurve(y)

        mv_deviation = market_value - self.present_value(curve=const_curve, time_shift=time_shift)

        return mv_deviation

    def internal_rate_of_return(self, market_value: float, time_shift: float = 0.0, lower_limit: float = -1.0,
                                upper_limit: float = 2.0):

        """
        Calculates the internal rate of return for a cash flow.

        Parameters
        ----------
        market_value:
            Market value of cash flow.
        time_shift:
            Time shift.
        lower_limit:
            Lower limit to use in bisection method.
        upper_limit:
            Upper limit to use in bisection method.

        Returns
        -------
        float
            Internal rate of return/yield to maturity.

        """

        irr = bisect(self.market_value_deviation, lower_limit, upper_limit, args=(market_value, time_shift))

        return irr

    def fisher_weil_duration(self, curve: IRateCurve, time_shift: float = 0.0) -> float:

        """
        Calculates Fisher-Weil duration

        .. math::

            \text{Duration}_{FW} = \\sum_{i=1}^n t_i \\frac{PV_i}{V}

        where :math:`i` is the cash-flow index, :math:`PV_i` is the present value of the :math:`i`'th cash flow,
        :math:`t_i` is the time in years until the :math:`i`'th cash flow and :math:`V` is the present value of all
        future cash flows

        Parameters
        ----------
        curve:
            Rate curve that is used to discount the cash flows.
        time_shift:
            Time shift.

        Returns
        -------
        float
            Fisher-Weil duration.

        """

        if not isinstance(curve, IRateCurve):
            raise TypeError("Curve is not derived from IRateCurve")

        disc_time = 0.0
        present_value = 0.0
        for i in range(self.number_of_time_points):
            t = self.time_points[i] - time_shift
            if t >= 0.0:
                disc = curve.discount_factor(t)
                disc_time += t * self.flows[i] * disc
                present_value += self.flows[i] * disc

        duration = 0.0
        if present_value > 0:
            duration = disc_time / present_value

        return duration

    def macaulay_duration(self, yield_to_maturity: float, time_shift: float = 0.0) -> float:

        """
        Calculates Macaulay duration:

        .. math::

            \text{Duration}_{M} = \\sum_{i=1}^n t_i \\frac{C_i \\cdot D_i}{V}

        where :math:`i` is the cash-flow index, :math:`C_i` is the :math:`i`'th cash flow, :math:`D_i` is the discount
        factor calculated using the yield to maturity, :math:`y`, e.g.  :math:`D_i=e^{-t_i \\cdot y}` using cont.
        discounting, :math:`t_i` is the time in years until the :math:`i`'th cash flow and :math:`V` is the present
        value of all future cash flows

        Parameters
        ----------
        yield_to_maturity:
            Yield to maturity.
        time_shift:
            Time shift.

        Returns
        -------
        float
            Macaulay duration.

        """

        ytm_const_curve = ConstantCurve(yield_to_maturity)

        duration = self.fisher_weil_duration(ytm_const_curve, time_shift=time_shift)

        return duration

    def fisher_weil_convexity(self, curve: IRateCurve, time_shift: float = 0.0) -> float:

        """
        Calculates Fisher-Weil type convexity

        .. math::

            \text{Convexity}_{FW} = \\sum_{i=1}^n t_i^2 \\frac{PV_i}{V}

        where :math:`i` is the cash-flow index, :math:`PV_i` is the present value of the :math:`i`'th cash flow,
        :math:`t_i` is the time in years until the :math:`i`'th cash flow and :math:`V` is the present value of all
        future cash flows

        Parameters
        ----------
        curve:
            Rate curve that is used to discount the cash flows.
        time_shift:
            Time shift.

        Returns
        -------
        float
            Fisher-Weil type convexity.

        """

        if not isinstance(curve, IRateCurve):
            raise TypeError("Curve is not derived from IRateCurve")

        disc_time = 0.0
        present_value = 0.0
        for i in range(self.number_of_time_points):
            t = self.time_points[i] - time_shift
            if t >= 0.0:
                disc = curve.discount_factor(t)
                disc_time += t**2 * self.flows[i] * disc
                present_value += self.flows[i] * disc

        convexity = 0.0
        if present_value > 0.0:
            convexity = disc_time / present_value

        return convexity

    def macaulay_convexity(self, yield_to_maturity: float, time_shift: float = 0.0) -> float:

        """
        Calculates Macaulay type convexity:

        .. math::

            \text{Convexity}_{M} = \\sum_{i=1}^n t_i^2 \\frac{C_i \\cdot D_i}{V}

        where :math:`i` is the cash-flow index, :math:`C_i` is the :math:`i`'th cash flow, :math:`D_i` is the discount
        factor calculated using the yield to maturity, :math:`y`, e.g.  :math:`D_i=e^{-t_i \\cdot y}` using cont.
        discounting, :math:`t_i` is the time in years until the :math:`i`'th cash flow and :math:`V` is the present
        value of all future cash flows

        Parameters
        ----------
        yield_to_maturity:
            Yield to maturity.
        time_shift:
            Time shift.

        Returns
        -------
        float
            Macaulay type convexity.

        """

        ytm_const_curve = ConstantCurve(yield_to_maturity)

        convexity = self.fisher_weil_convexity(ytm_const_curve, time_shift=time_shift)

        return convexity

    """
    Methods for adding and multiplying cash flows, etc. 
    """

    def __mul__(self, other: Union[float, int, np.ndarray]) -> 'CashFlow':

        if not isinstance(other, (float, int, np.ndarray)):
            raise ValueError("Cash Flow can only be multiplied with scalar or ndarray of same size")

        if isinstance(other, np.ndarray):
            if len(other) != len(self.flows):
                raise ValueError('ndarray must have the same dimension')

        return CashFlow(self.time_points, self.flows * other)

    def __rmul__(self, other: Union[float, int, np.ndarray]) -> 'CashFlow':

        if not isinstance(other, (float, int)):
            raise ValueError("Cash Flow can only be multiplied with scalar or ndarray of same size")

        if isinstance(other, np.ndarray):
            if len(other) != len(self.flows):
                raise ValueError('ndarray must have the same dimension')

        return CashFlow(self.time_points, self.flows * other)

    def __truediv__(self, other: Union[float, int, np.ndarray]) -> 'CashFlow':

        if not isinstance(other, (float, int)):
            raise ValueError("Cash Flow can only be multiplied with scalar or ndarray of same size")

        if isinstance(other, np.ndarray):
            if len(other) != len(self.flows):
                raise ValueError('ndarray must have the same dimension')

        return CashFlow(self.time_points, self.flows * 1 / other)

    def __add__(self, other: Union[float, int, 'CashFlow']) -> 'CashFlow':

        if isinstance(other, float) or isinstance(other, int):

            return CashFlow(self.time_points, self.flows + other)

        elif isinstance(other, CashFlow):

            time_points = np.union1d(self.time_points, other.time_points)
            time_points.sort()

            flows = np.array([])

            i, j = 0, 0
            for t in time_points:
                flow = 0.0
                if t in self.time_points:
                    flow += self.flows[i]
                    i += 1
                if t in other.time_points:
                    flow += other.flows[j]
                    j += 1

                flows = np.r_[flows, flow]

            return CashFlow(time_points, flows)
        else:
            raise ValueError("Addition with Cash Flow only allowed for CashFlow or float")

    def __sub__(self, other: Union[float, int, 'CashFlow']) -> 'CashFlow':

        return self + (other * -1.0)
