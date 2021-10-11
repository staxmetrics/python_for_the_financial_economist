"""
Contains cash flow objects.
"""

import numpy as np
from codelib.fixed_income.curves.curve_interface import IRateCurve
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
