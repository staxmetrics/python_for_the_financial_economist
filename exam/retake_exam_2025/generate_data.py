import numpy as np
import pandas as pd

# typing
from typing import Union

"""
Define helper functions
"""

def calculate_slope(time_to_maturity: Union[float, np.ndarray], l: float):
    return (1 - np.exp(-time_to_maturity * l)) / (time_to_maturity * l)


def calculate_curvature(time_to_maturity: Union[float, np.ndarray], l: float):
    return (1 - np.exp(-time_to_maturity * l)) / (time_to_maturity * l) - np.exp(-(time_to_maturity * l))

def nelson_siegel_yield(maturities: np.ndarray, beta0: float, beta1: float, beta2: float, lam: float) -> np.ndarray:
    """
    Calculate Nelson-Siegel yield curve values.

    Parameters
    ----------
    maturities : np.ndarray
        Array of maturities (in years).
    beta0 : float
        Level parameter.
    beta1 : float
        Slope parameter.
    beta2 : float
        Curvature parameter.
    lam : float
        Decay factor.

    Returns
    -------
    np.ndarray
        Calculated yield values for the given maturities.
    """

    term1 = beta0
    term2 = beta1 * ((1 - np.exp(-maturities * lam)) / (maturities * lam))
    term3 = beta2 * (((1 - np.exp(-maturities * lam)) / (maturities * lam)) - np.exp(-maturities * lam))

    yields = term1 + term2 + term3

    return yields

def simulate_multiple_ar_process(mu: np.ndarray,
                                 phi: np.ndarray,
                                 sigma: np.ndarray,
                                 corr_matrix: np.ndarray,
                                 initial_values: np.ndarray,
                                 num_periods: int) -> np.ndarray:

    sim_data = np.zeros((num_periods + 1, len(mu)))
    sim_data[0, :] = initial_values

    # define covariance matrix
    cov_mat = np.diag(sigma) @ corr_matrix @ np.diag(sigma)

    # simulate normal shocks
    shocks = np.random.multivariate_normal(mean=np.zeros(len(mu)), cov=cov_mat, size=num_periods)

    # simulate AR(1) processes
    for t in range(0, num_periods):
        sim_data[t + 1, :] = mu + phi * (sim_data[t, :] - mu) + shocks[t , :]

    return sim_data


"""
General settings
"""

dt = 1.0 / 12.0 # time step in years (monthly)
sample_length = 40 # total sample length in years
num_periods = int(sample_length / dt) # number of time periods
time_steps = np.linspace(0, sample_length, num_periods) # time grid

np.random.seed(42) # set random seed for reproducibility

"""
Generate real and nominal yield data
"""

# lambda parameter for Nelson-Siegel model
lam_ns = 0.72 # assume same value for both real and beir yield curves

# intercept in AR(1) processes for 6 factors (3 real yield factors, 3 BEIR factors)
mu = np.array([0.020375, -0.008466, -0.033998, 0.024626, 0.003454, -0.027280])

# AR(1) coefficients for 6 factors
phi = np.array([0.981741, 0.912298, 0.923033, 0.934770, 0.868409, 0.913204])

# volatility of innovations for 6 factors
sigma = np.array([0.001932, 0.012579, 0.017234, 0.001705, 0.011960, 0.015692])

# correlation matrix for innovations
corr_matrix = np.array([
    [1.0, -0.06985138216921263, -0.021123261109324755, 0.047866398512132285, -0.07041322824275792, 0.04276975423873937],
    [-0.06985138216921263, 1.0, -0.7886772712545194, -0.34457128140968046, -0.848980027497746, 0.7015057200313112],
    [-0.021123261109324755, -0.7886772712545194, 1.0, 0.2585820535910715, 0.6276571109295496, -0.7089815956771305],
    [0.047866398512132285, -0.34457128140968046, 0.2585820535910715, 1.0, 0.2136797386717812, -0.38230343485045876],
    [-0.07041322824275792, -0.848980027497746, 0.6276571109295496, 0.2136797386717812, 1.0, -0.8039407655035898],
    [0.04276975423873937, 0.7015057200313112, -0.7089815956771305, -0.38230343485045876, -0.8039407655035898, 1.0]
])

std_measure_error = 2.5 / 10_000  # standard deviation of measurement error (2.5 bps)

# simulate beta factors
initial_values = mu  # start at unconditional mean
simulated_factors = simulate_multiple_ar_process(mu, phi, sigma, corr_matrix, initial_values, num_periods)

# set maturities for yield curves
maturities = np.array([1, 2, 3, 5, 7, 10, 15, 20]) # in years

# calculate slope and curvature loadings
slope_loadings = calculate_slope(maturities, lam_ns)
curvature_loadings = calculate_curvature(maturities, lam_ns)

# calculate real yields using Nelson-Siegel model, add some random noise
real_yields = simulated_factors[:, 0][:, np.newaxis] + \
              simulated_factors[:, 1][:, np.newaxis] * slope_loadings[np.newaxis, :] + \
              simulated_factors[:, 2][:, np.newaxis] * curvature_loadings[np.newaxis, :] + \
              np.random.normal(0, std_measure_error, size=(num_periods + 1, len(maturities))) # 2.5 bps noise

# calculate BEIR using Nelson-Siegel model, add some random noise
beir = simulated_factors[:, 3][:, np.newaxis] + \
       simulated_factors[:, 4][:, np.newaxis] * slope_loadings[np.newaxis, :] + \
       simulated_factors[:, 5][:, np.newaxis] * curvature_loadings[np.newaxis, :] + \
       np.random.normal(0, std_measure_error, size=(num_periods + 1, len(maturities))) # 2.5 bps noise

# calculate nominal yields
nom_yields = real_yields + beir




"""
Generate market variables

    - inflation rate
    - real estate cost index
    - real estate rent index
    - equity index
"""

# expected monthly changes in market variables: inflation (log-change), real estate cost index (log-change),
# real estate rent index (log-change), equity index (log-return)

mean_market = np.array([0.0, 0.0, 10 / 10_000, 0.08 / 12])  # mean of market variables

# standard deviations of market variables
std_market = np.array([0.25 / 100, 0.25 / 100, 0.25 / 100, 0.14 / np.sqrt(12)])

# correlation
corr_mat_market = np.array([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.7, 0.2],
                            [0.0, 0.7, 1.0, 0.6],
                            [0.0, 0.2, 0.6, 1.0]])

# covariance matrix of market variables
cov_mat_market = np.diag(std_market) @ corr_mat_market @ np.diag(std_market)

# simulate random shocks
shocks = np.random.multivariate_normal(mean_market, cov_mat_market, size=num_periods)

# simulate inflation rate
inflation = np.zeros(num_periods + 1)
inflation[1:] = beir[1:, 0] / 12 + shocks[:, 0]  # assume inflation linked to 1-year BEIR

# calculate CPI index
cpi_index = np.exp(np.cumsum(inflation))

# real estate cost index
cost_infl = inflation.copy()
cost_infl[1:] += shocks[:, 1]
real_estate_cost_index = np.exp(np.cumsum(cost_infl))

# real estate rent index
rent_infl = inflation.copy()
rent_infl[1:] += shocks[:, 2]
real_estate_rent_index = np.exp(np.cumsum(rent_infl))


"""
Generate a equity index, assume log-normal returns
"""

log_ret = np.zeros(num_periods + 1)
log_ret[1:] = shocks[:, 3]
equity_index = np.exp(np.cumsum(log_ret))


"""
Save generated data to csv files
"""

# time index
time = np.arange(0, num_periods + 1) * dt

# define dataframe
df_data = pd.DataFrame(index=time)

# add nom. yield data
for i, mat in enumerate(maturities):
    df_data[f'nominal_yield_{mat}y'] = nom_yields[:, i]

# add real yield data
for i, mat in enumerate(maturities):
    df_data[f'real_yield_{mat}y'] = real_yields[:, i]

# add market variables
df_data['cpi_index'] = cpi_index
df_data['real_estate_cost_index'] = real_estate_cost_index
df_data['real_estate_rent_index'] = real_estate_rent_index
df_data['equity_index'] = equity_index

# save to csv
df_data.to_csv('simulated_yield_and_market_data.csv', index_label='time')