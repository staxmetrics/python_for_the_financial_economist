
# load python libraries
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from typing import Union

# set random seed for reproducibility
np.random.seed(55) # 55


"""
Helper functions
"""


def simulate_vasicek(initial_short_rate: float,
                     kappa: float,
                     theta: float,
                     sigma: float,
                     horizon: float,
                     dt: float = 1.0 / 52.0,
                     num_sim: int = 10000):
    """
    simulates short rate processes in a Vasicek setting until a given horizon

    Parameters
    ----------

    initial_short_rate:
        initial short rate
    kappa:
        speed of mean reversion.
    theta:
        long term mean of the short rate.
    sigma:
        volatility of the short rate.
    horizon:
        time until maturity/expiry (horizon).
    dt:
        increments in time
    num_sim:
        number of simulations.
    """
    std_rates = np.sqrt(sigma ** 2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt)))

    num_periods = int(horizon / dt)
    short_rates = np.empty((num_sim, num_periods))
    short_rates[:, 0] = initial_short_rate

    error_terms = np.random.normal(scale=std_rates, size=(num_sim, num_periods))

    for i in range(1, num_periods):
        short_rates[:, i] = theta + (short_rates[:, i - 1] - theta) * np.exp(-kappa * dt) + error_terms[:, i - 1]

    return short_rates


def calculate_zero_coupon_price(time_to_maturity: Union[float, np.ndarray],
                                initial_short_rate: float,
                                kappa: float,
                                theta: float,
                                sigma: float):

    """
    Computes the zero coupon yield for a given maturity and short rate using the Vasicek model.

    Parameters
    ----------
    time_to_maturity:
        time to maturity of the zero coupon bond.
    initial_short_rate:
        initial short rate.
    kappa:
        speed of mean reversion.
    theta:
        long term mean of the short rate under Q.
    sigma:
        volatility of the short rate.

    Returns
    -------
    zero_coupon_price:
        price of the zero coupon bond.
    """

    y_infty = theta - sigma ** 2 / (2 * kappa ** 2)

    b = 1 / kappa * (1 - np.exp(-kappa * time_to_maturity))
    a = y_infty * (time_to_maturity - b) + sigma ** 2 / (4 * kappa) * b ** 2

    return np.exp(- a - b * initial_short_rate)

def calculate_zero_coupon_yield(time_to_maturity: Union[float, np.ndarray],
                                initial_short_rate: float,
                                kappa: float,
                                theta: float,
                                sigma: float):

    """
    Computes the zero coupon yield for a given maturity and short rate using the Vasicek model.

    Parameters
    ----------
    time_to_maturity:
        time to maturity of the zero coupon bond.
    initial_short_rate:
        initial short rate.
    kappa:
        speed of mean reversion.
    theta:
        long term mean of the short rate under Q.
    sigma:
        volatility of the short rate.

    Returns
    -------
    zero_coupon_yield:
        yield of the zero coupon bond.
    """


    price = calculate_zero_coupon_price(time_to_maturity=time_to_maturity,
                                        initial_short_rate=initial_short_rate,
                                        kappa=kappa,
                                        theta=theta,
                                        sigma=sigma)

    return - np.log(price) / time_to_maturity



"""
General settings
"""

dt = 1.0 / 52.0 # time step in years (weekly)
sample_length = 40 # total sample length in years
num_periods = int(sample_length / dt) # number of time periods
time_steps = np.linspace(0, sample_length, num_periods) # time grid

"""
Fixed income data 

    1. generate data for short rate using Vasicek model
    2. generate zero-coupon bond yields with measurement error 
    3. generate bullet bonds (price for a given maturity and coupon rate) using the provided yield curve
    
"""


# define true parameters
init_short_rate = 0.03  # initial short rate
kappa = 0.04         # speed of mean reversion
theta_q = 0.129         # long-term mean level under q
sigma = 0.0175          # volatility
lam = -0.0035          # market price of risk
theta_p = theta_q + lam / kappa

# simulate short rate under physical measure
short_rates = simulate_vasicek(initial_short_rate=init_short_rate,
                               kappa=kappa,
                               theta=theta_p,
                               sigma=sigma,
                               horizon=sample_length,
                               dt=dt,
                               num_sim=1).flatten()

# simulate zero coupon bond yields with measurement error
tenors = np.array([0.25, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])  # in years
df_zcb_yields = pd.DataFrame(data=np.zeros((len(short_rates) , len(tenors))), columns=tenors, index=time_steps)
df_discount_factors = pd.DataFrame(data=np.zeros((len(short_rates) , len(tenors))), columns=tenors, index=time_steps)
for i, t in enumerate(time_steps):

    # calculate "true" zero coupon yields
    zcb_yields = calculate_zero_coupon_yield(time_to_maturity=tenors,
                                             initial_short_rate=short_rates[i],
                                             kappa=kappa,
                                             theta=theta_q,
                                             sigma=sigma)

    # add measurement error
    zcb_yields[1:] += np.random.normal(loc=0.0, scale=0.002, size=len(tenors)-1)

    # store in dataframe
    df_zcb_yields.iloc[i, :] = zcb_yields
    df_discount_factors.iloc[i, :] = np.exp(- zcb_yields * tenors)

# price bullet bonds using the generated yield curve
# assume 6% coupon rate, annual coupon payments, maturity 1 to 10 years
bullet_bond_cash_flow_matrix = np.tri(10) * 0.06 + np.eye(10) * 1
df_bullet_bond_prices = df_discount_factors.iloc[:, 1:].values @ bullet_bond_cash_flow_matrix.T

# define output dataframe
df_fixed_income_data = pd.DataFrame(data=None,
                                     columns= ['3M ZCB'] + [f'Bullet_Bond_{i+1}Y' for i in range(10)],
                                     index=time_steps)

df_fixed_income_data.iloc[:, 1:] = df_bullet_bond_prices
df_fixed_income_data.iloc[:, 0] = df_discount_factors.iloc[:, 0]


"""
Equity universe

    1. excess return follows a multivariate log normal distribution
    
    
We assume independence between the shocks of the short rate process and equity returns.
"""

num_equity_assets = 9 # number of equity assets
num_factors = 3 # number of risk factors

# factor parameters
factor_vols = np.array([0.1, 0.05, 0.025])
factor_premia = np.array([0.03, 0.015, 0.01])
factor_corr_mat = np.array([[1.0, 0.3, 0.2],
                            [0.3, 1.0, 0.1],
                            [0.2, 0.1, 1.0]])

factor_cov_mat = np.outer(factor_vols, factor_vols) * factor_corr_mat

# parameters for equity returns
beta_matrix = np.zeros((num_equity_assets, num_factors))
beta_matrix[:3, 0] = 1.0
beta_matrix[3:6, 1] = 1.0
beta_matrix[6:, 2] = 1.0
beta_matrix += np.random.uniform(low=0.2, high=0.4, size=(num_equity_assets, num_factors)) # factor exposures
equity_premia = beta_matrix.dot(factor_premia) # expected excess returns
idiosyncratic_vols = np.random.uniform(low=0.05, high=0.06, size=num_equity_assets) # idiosyncratic volatilities
equity_cov_mat = beta_matrix @ factor_cov_mat @ beta_matrix.T + np.diag(idiosyncratic_vols ** 2) # covariance matrix

# simulate equity log returns
equity_log_excess_returns = np.random.multivariate_normal(mean=(equity_premia - 0.5 * np.diag(equity_cov_mat)) * dt,
                                                          cov=equity_cov_mat * dt,
                                                          size=num_periods - 1)

equity_log_returns = equity_log_excess_returns + short_rates[:-1, np.newaxis] * dt

equity_index = np.ones((num_periods, num_equity_assets), dtype=float)
equity_index[1:, :] = np.exp(np.cumsum(equity_log_returns, axis=0))

df_equity_index = pd.DataFrame(data=equity_index,
                               columns=[f'Equity_Asset_{i+1}' for i in range(num_equity_assets)],
                               index=time_steps)

if True: # plot generated data
    #df_equity_index.plot()
    #plt.show()

    np.log(df_equity_index).plot()
    plt.show()

    plt.plot(short_rates)
    plt.show()

    #plt.plot(equity_log_returns)
    #plt.show()

    df_zcb_yields.plot()
    plt.show()

    (df_zcb_yields[10.0] - df_zcb_yields[0.25]).plot()
    plt.title('10Y-3M Yield Spread')
    plt.show()

    (df_zcb_yields[10.0] - short_rates).plot()
    plt.title('10Y Yield minus short rate')
    plt.show()


# save generated data
df_fixed_income_data.to_csv('fixed_income_data.csv')
df_equity_index.to_csv('equity_index_data.csv')


"""
################## Check optimal portfolio holdings #########################
"""

if True:

    import cvxpy as cp

    """
    Helper functions
    """

    def _simulate_vasicek(initial_short_rate: float,
                         kappa: float,
                         theta: float,
                         sigma: float,
                         horizon: float,
                         dt: float = 1.0 / 12,
                         num_sim: int = 10000):
        """
        simulates short rate processes in a Vasicek setting until a given horizon

        Parameters
        ----------

        initial_short_rate:
            initial short rate
        kappa:
            speed of mean reversion.
        theta:
            long term mean of the short rate.
        sigma:
            volatility of the short rate.
        horizon:
            time until maturity/expiry (horizon).
        dt:
            increments in time
        num_sim:
            number of simulations.
        """
        std_rates = np.sqrt(sigma ** 2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt)))

        num_periods = int(horizon / dt)
        short_rates = np.empty((num_sim, num_periods + 1))
        short_rates[:, 0] = initial_short_rate

        error_terms = np.random.normal(scale=std_rates, size=(num_sim, num_periods))

        for i in range(1, num_periods + 1):
            short_rates[:, i] = theta + (short_rates[:, i - 1] - theta) * np.exp(-kappa * dt) + error_terms[:, i - 1]

        return short_rates

    def calculate_zero_coupon_price_vector(time_to_maturity: Union[float, np.ndarray],
                                    initial_short_rate: float,
                                    kappa: float,
                                    theta: float,
                                    sigma: float):

        """
        Computes the zero coupon yield for a given maturity and short rate using the Vasicek model.

        Parameters
        ----------
        time_to_maturity:
            time to maturity of the zero coupon bond.
        initial_short_rate:
            initial short rate.
        kappa:
            speed of mean reversion.
        theta:
            long term mean of the short rate.
        sigma:
            volatility of the short rate.

        Returns
        -------
        zero_coupon_price:
            price of the zero coupon bond.
        """

        time_to_maturity = np.atleast_1d(time_to_maturity)
        initial_short_rate = np.atleast_1d(initial_short_rate)

        y_infty = theta- sigma ** 2 / (2 * kappa ** 2)

        b = 1 / kappa * (1 - np.exp(-kappa * time_to_maturity))
        a = y_infty * (time_to_maturity - b) + sigma ** 2 / (4 * kappa) * b ** 2

        return np.exp(- a[:, None] - b[:, None] * initial_short_rate[None, :])


    # Define a function to optimize portfolio weights using CVaR minimization.
    def calculate_mean_cvar_optimization(pnl_matrix: np.ndarray,
                                         beta: float,
                                         initial_prices: np.ndarray,
                                         probs: None,
                                         pnl_target: float = None,
                                         wealth: float = 100_000_000,
                                         solver: str = None,
                                         verbose: bool = False) -> np.ndarray:

        """
        Optimize portfolio weights using CVaR minimization. The optimization problem can handle optional return constraints.

        Parameters
        ----------
        pnl_matrix : np.ndarray
            Simulated returns of shape (num_simulations, num_assets).
        beta : float
            Confidence level for CVaR (e.g., 0.95 for 95% CVaR).
        initial_prices : np.ndarray
            Initial prices of the assets.
        probs : np.ndarray or None
            Probabilities associated with each simulation. If None, equal probabilities are assumed.
        pnl_target : float or None
            Target return for the portfolio. If None, no return constraint is applied.
        verbose : bool
            If True, print solver output.

        Returns
        -------
        h : np.ndarray
            Optimized portfolio holdings.
        alpha : float
            Value at Risk at the specified confidence level.
        """

        if solver is None:
            solver = cp.SCS

        num_assets = pnl_matrix.shape[1]
        num_sim = pnl_matrix.shape[0]

        if probs is None:
            probs = np.ones(num_sim) / num_sim

        # Define variables
        h = cp.Variable(num_assets)  # portfolio weights
        u = cp.Variable(num_sim)  # auxiliary variables for CVaR
        alpha = cp.Variable()  # Value at Risk variable

        # Define the objective function (minimize CVaR)
        objective = cp.Minimize(alpha + (1 / (1 - beta)) * cp.sum(cp.multiply(probs, u)))

        # Define constraints
        constraints = [
            cp.sum(h @ initial_prices) == wealth,  # holdings multiplied by initial prices sum to wealth
            h[0] >= -wealth * 0.2,  # max shorting constraint
            h[1:] >= 0,          # no short selling for all but bank account
            u >= 0,          # auxiliary variables non-negative
            u >= -pnl_matrix @ h - alpha  # definition of u
        ]

        if pnl_target is not None:
            constraints.append(cp.sum(cp.multiply(probs, pnl_matrix @ h)) >= pnl_target)

        # Solve the optimization problem
        prob = cp.Problem(objective, constraints)
        prob.solve(verbose=verbose, solver=solver)

        return h.value, alpha.value


    """
    Generate data for portfolio optimization
    """

    esg = {}

    dt = 1.0 / 52.0  # weekly time step
    horizon = 1.0  # 1 year
    num_sim = 100_000
    time_points = np.arange(0, horizon + dt, dt)
    num_per = int(horizon / dt)

    short_rates_sim = _simulate_vasicek(initial_short_rate=short_rates[-1],
                                       kappa=kappa,
                                       theta=theta_p,
                                       sigma=sigma,
                                       horizon=horizon,
                                       dt=dt,
                                       num_sim=num_sim)

    esg['SHORT_RATE'] = short_rates_sim

    # bank account
    bank_account_values = np.c_[np.ones(num_sim), np.exp(np.cumsum(short_rates_sim[:, :-1] * dt, axis=1))]
    esg['BANK_ACCOUNT'] = bank_account_values # total return index of bank account



    bullet_cash_flow_time_points = np.array(list(range(1, 11))) # exclude 3M ZCB
    bullet_bond_cash_flow_matrix = np.tri(10) * 0.06 + np.eye(10) * 1

    # allocate tri for bullet bonds
    all_bullet_bonds = np.empty((num_sim, bullet_bond_cash_flow_matrix.shape[0], bank_account_values.shape[1]))

    for i, t in enumerate(time_points):
        # calculate time to maturities
        time_to_maturities = bullet_cash_flow_time_points - t
        time_to_maturities = np.where(time_to_maturities < 0, 0.0, time_to_maturities)

        # calculate zero coupon prices at time t
        zero_coupon_prices = calculate_zero_coupon_price_vector(time_to_maturities,
                                                                short_rates_sim[:, i],
                                                                kappa,
                                                                theta_q,
                                                                sigma)

        # calculate bond prices at time t
        bond_prices = bullet_bond_cash_flow_matrix @ zero_coupon_prices

        # store bond prices
        all_bullet_bonds[:, :, i] = bond_prices.T

    for i in range(1, 11, 1):
        esg[f'BULLET_BOND_{i}Y'] = all_bullet_bonds[:, i - 1, :]


    # simulate equity excess log returns
    equity_log_excess_returns_sim = np.random.multivariate_normal(mean=(equity_premia - 0.5 * np.diag(equity_cov_mat)) * dt,
                                                                  cov=equity_cov_mat * dt,
                                                                  size=(num_sim, num_per))

    # calculate equity log returns by adding short rates
    equity_log_returns_sim = equity_log_excess_returns_sim + short_rates_sim[:, :-1, np.newaxis] * dt

    # calculate equity index
    equity_index_sim = np.ones((num_sim, num_per + 1, equity_premia.shape[0]), dtype=float)
    equity_index_sim[:, 1:, :] = np.exp(np.cumsum(equity_log_returns_sim, axis=1))

    # store in dictionary
    for i in range(equity_premia.shape[0]):
        esg[f'EQUITY_ASSET_{i+1}'] = equity_index_sim[:, :, i]

    """
    Calculate PnL matrix for portfolio optimization
    """

    # get list of investable assets
    assets_names = list(esg.keys())[1:]

    # obtain initial prices
    initial_prices = np.array([esg[asset][0, 0] for asset in assets_names])

    # allocate array for PnL
    pnl_matrix = np.empty((num_sim, len(assets_names)))

    # calculate PnL for each asset
    for i, asset in enumerate(assets_names):
        final_prices = esg[asset][:, -1]
        pnl_matrix[:, i] = final_prices - initial_prices[i]

    """
    Define parameters for portfolio optimization
    """

    wealth = 1.0
    avg_pnl = np.average(pnl_matrix, axis=0) * wealth
    min_avg_pnl = (avg_pnl / initial_prices).min()
    max_avg_pnl = (avg_pnl / initial_prices).max()

    pnl_targets = np.linspace(min_avg_pnl, max_avg_pnl*1.2 - 0.2*avg_pnl[0], 25)

    h_optimal = np.array([calculate_mean_cvar_optimization(pnl_matrix,
                                                           beta=0.95,
                                                           initial_prices=initial_prices,
                                                           probs=None,
                                                           pnl_target=target,
                                                           wealth=wealth,
                                                           verbose=False)[0] for target in pnl_targets])


    """
    Optimal holdings
    """

    cmap = plt.get_cmap('jet')
    asset_colors = cmap(np.linspace(0, 1, 20))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(pnl_targets, h_optimal[:, 0], color=asset_colors[0], label=assets_names[0])
    ax.stackplot(pnl_targets, (h_optimal[:, 1:] * initial_prices[1:]).T, colors=asset_colors[1:], labels=assets_names[1:])
    ax.legend(ncol=5, bbox_to_anchor=(1.01, -0.12));
    ax.set_xlabel('PnL Target')
    ax.set_ylabel('$h$')
    ax.set_title('Optimal holdings')
    plt.tight_layout()
    plt.show()