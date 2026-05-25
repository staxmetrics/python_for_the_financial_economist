from codelib.dal.fred_yield_data import get_nominal_yield_data, get_real_yield_data

import pandas as pd
import numpy as np

# time series models
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

# plotting
import matplotlib.pyplot as plt

# typing
from typing import Union

"""
Import real and nominal rates
"""

df_nominal_yield_data = get_nominal_yield_data(output_type='zero_yields')
df_real_yields_data = get_real_yield_data(output_type='zero_yields')
df_beir_data = get_real_yield_data(output_type='zero_beir')


"""
Data preprocessing
"""

# select common dates and drop NaNs
common_dates = df_nominal_yield_data.index.intersection(df_real_yields_data.index).intersection(df_beir_data.index)
df_nominal_yield_data = df_nominal_yield_data.loc[common_dates].dropna()
df_real_yields_data = df_real_yields_data.loc[common_dates].dropna()
df_beir_data = df_beir_data.loc[common_dates].dropna()

# replace column names with maturities in years
df_nominal_yield_data.columns = np.arange(1, 31)
df_beir_data.columns = np.arange(2, 21)
df_real_yields_data.columns = np.arange(2, 21)


"""
Resample data to monthly frequency
"""
df_beir_monthly = df_beir_data.resample('ME').last() / 100.0  # convert to decimals
df_real_yields_monthly = df_real_yields_data.resample('ME').last() / 100.0  # convert to decimals

"""
Plot data for visual inspection
"""

(df_beir_monthly[[2, 5, 10, 20]] * 100).plot(title='BEIR historical data', ylabel='BEIR (%)', xlabel='Date')
plt.show()

(df_real_yields_monthly[[2, 5, 10, 20]] * 100).plot(title='Real yield historical data', ylabel='Yield (%)', xlabel='Date')
plt.show()


"""
Functions for fitting Nelson-Siegel model
"""


def calculate_slope(time_to_maturity: Union[float, np.ndarray], l: float):
    return (1 - np.exp(-time_to_maturity * l)) / (time_to_maturity * l)


def calculate_curvature(time_to_maturity: Union[float, np.ndarray], l: float):
    return (1 - np.exp(-time_to_maturity * l)) / (time_to_maturity * l) - np.exp(-(time_to_maturity * l))


def ols_ns(observed_yields: np.ndarray, tenors: np.ndarray, l):
    x_mat = np.c_[np.ones_like(tenors),
    calculate_slope(tenors, l),
    calculate_curvature(tenors, l)]

    est_params = np.linalg.lstsq(x_mat, observed_yields, rcond=None)

    return est_params[0]


"""
Estimate Nelson-Siegel model for real yields
"""

tenors = np.arange(2.0, 21.0, 1.0)
real_yield_beta_estimates = df_real_yields_monthly.apply(ols_ns, axis=1, args=(tenors, 0.72), result_type="expand")
real_yield_beta_estimates.columns = ['beta1', 'beta2', 'beta3']


real_yield_beta_estimates.plot()
plt.show()

"""
Estimate Nelson-Siegel for BEIR
"""

tenors = np.arange(2.0, 21.0, 1.0)
beir_beta_estimates = df_beir_monthly.apply(ols_ns, axis=1, args=(tenors, 0.72), result_type="expand")
beir_beta_estimates.columns = ['beir beta1', 'beir beta2', 'beir beta3']

beir_beta_estimates['beir beta1'].plot()
plt.show()

"""
Estimate time series dynamics for factors using VAR(1) model
"""

# join real yield and BEIR factors
factors_df = pd.concat([real_yield_beta_estimates, beir_beta_estimates], axis=1).dropna()

# fit VAR model
var_model = VAR(factors_df)
var_results = var_model.fit(maxlags=1)
print(var_results.summary())

"""
Estimate seperate AR(1) models for each factor
"""

# dictionary to hold results
ar_results_dict = {}

# fit AR(1) models
for column in factors_df.columns:
    ar_model = sm.tsa.ARIMA(factors_df[column], order=(1, 0, 0))
    ar_results = ar_model.fit(maxiter=200)
    ar_results_dict[column] = ar_results
    print(f"AR(1) results for {column}:")
    print(ar_results.summary())

# create a dataframe with parameter estimates
ar_params_df = pd.DataFrame({col: ar_results_dict[col].params for col in factors_df.columns})
print("AR(1) parameter estimates:")
print(ar_params_df)

# create a dataframe with fitted residuals
df_residuals = pd.DataFrame({col: ar_results_dict[col].resid for col in factors_df.columns})

df_residuals['beta1'].plot(title='Residuals from AR(1) models')

df_residuals.corr().to_clipboard()