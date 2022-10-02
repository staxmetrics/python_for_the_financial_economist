import numpy as np
from scipy.linalg import block_diag
from sklearn.neighbors import KernelDensity
from typing import Tuple

from codelib.statistics.moments import corr_to_cov_matrix, cov_to_corr_matrix


"""
Marcencko pastur density 
"""


def marchencko_pastur_bounds(sigma: float, ratio: float) -> Tuple[float, float]:

    """
    Calculates Marcencko-Pastur bounds

    Parameters
    ----------
    sigma:
        Sigma parameter.
    ratio:
        Ratio between num. of variables and num. of observations

    Returns
    -------
    Tuple[float, float]
        Bounds.

    """

    sigma2 = sigma ** 2

    lower = sigma2 * (1 - np.sqrt(ratio)) ** 2
    upper = sigma2 * (1 + np.sqrt(ratio)) ** 2

    bounds = (lower, upper)

    return bounds


def marchencko_pastur_density(x: np.ndarray, sigma: float, ratio: float) -> np.ndarray:

    """
    Calculates Marcencko-Pastur density

    Parameters
    ----------
    x:
        Points at which to evaluate the density.
    sigma:
       Sigma parameter.
    ratio:
       Ratio between num. of variables and num. of observations

    Returns
    -------
    np.ndarray
       Densitiy.

    """

    lower, upper = marchencko_pastur_bounds(sigma, ratio)

    indicator = np.int64(x >= lower) * np.int64(x <= upper)
    dv = np.zeros(indicator.shape)

    for i in range(len(x)):
        temp = x[i]
        idx = indicator[i]
        if idx:
            dv[i] = 1.0 / (2.0 * np.pi * sigma ** 2) * np.sqrt((upper - temp) * (temp - lower)) / (ratio * temp)

    return dv


"""
Simulate covariance matrix 
Code adapted from Marcos M. Lopéz de Prado (2020), "Machine Learning for Asset Managers"
"""


def simulate_random_cov_mat(num_var: int, num_factors: int) -> np.ndarray:

    """
    Simulates a random covariance matrix with some dependence

    Parameters
    ----------
    num_var:
        Number of variables.
    num_factors:
        Number of factors

    Returns
    -------
    np.ndarray
        Covariance matrix.

    """

    x = np.random.normal(size=(num_var, num_factors))
    cov_mat = x @ x.T
    cov_mat += np.diag(np.random.uniform(size=num_var))

    return cov_mat


def simulate_cov_mat_noise_and_signal(num_var: int, num_factors: int, num_obs: int, alpha: float):

    """
    Simulate a covariance matrix with noise and signal

    Parameters
    ----------
    num_var:
        Number of variables.
    num_factors:
        Number of factors.
    num_obs:
        Number of obervations.
    alpha:
        Weight on noisy covariance matrix.

    Returns
    -------
    np.ndarray
        Covariance matrix.

    """

    cov_mat_noise = np.cov(np.random.normal(size=(num_obs, num_var)), rowvar=False)
    cov_mat_signal = simulate_random_cov_mat(num_var, num_factors)
    cov_mat = alpha * cov_mat_noise + (1 - alpha) * cov_mat_signal

    return cov_mat


def form_block_corr_matrix(num_blocks: int, block_size: int, block_corr: float) -> np.ndarray:

    """
    Create a block correlation matrix with a number of equal size blocks.
    Each block have the same inter block correlation.

    Parameters
    ----------
    num_blocks:
        Number of blocks
    block_size:
        Block size.
    block_corr
        Inter block correlation.

    Returns
    -------
    np.ndarray
        Correlation matrix.

    """

    block = np.ones((block_size, block_size)) * block_corr
    np.fill_diagonal(block, 1.0)

    corr_mat = block_diag(*([block] * num_blocks))

    return corr_mat


def form_true_cov_and_mean(num_blocks: int, block_size: int, block_corr: float) -> Tuple[np.ndarray, np.ndarray]:

    """
    Creates a covariance matrix and mean vector

    Parameters
    ----------
    num_blocks:
        Number of blocks
    block_size:
        Block size.
    block_corr
        Inter block correlation.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Covariance matrix, mean vector
    """

    # generate corr mat
    corr_mat = form_block_corr_matrix(num_blocks, block_size, block_corr)
    num_assets = corr_mat.shape[0]

    idx = np.arange(num_assets)
    np.random.shuffle(idx)
    idx_selector = np.tile(idx.reshape(num_assets, 1), num_assets)

    corr_mat = corr_mat[idx_selector, idx_selector.T]

    # generate vols
    vols = np.random.uniform(0.05, 0.2, num_assets)

    # generate cov mat
    cov_mat = corr_to_cov_matrix(corr_mat, vols)

    # generate means (all assets have same sharpe ratio)
    mu = np.random.normal(vols, vols, size=num_assets)

    return mu, cov_mat


"""
Denoising covariance matrix 
Code adapted from Marcos M. Lopéz de Prado (2020), "Machine Learning for Asset Managers"
"""


def fit_kde(eigenvalues: np.ndarray, kernel: str = "gaussian", bandwidth: float = 0.01, x: np.ndarray = None):

    """
    Fits a kernel density to the eigenvalues

    Parameters
    ----------
    eigenvalues:
        Eigenvalues.
    kernel:
        Kernel type.
    bandwidth:
        Bandwidth.
    x:
        Points where to eveluate the pdf.

    Returns
    -------
    np.ndarray
        Densities.

    """

    if len(eigenvalues.shape) == 1:
        eigenvalues = eigenvalues.reshape(-1, 1)

    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(eigenvalues)

    if x is None:
        x = np.unique(eigenvalues)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    log_prob = kde.score_samples(x)
    return np.exp(log_prob).flatten()


def fitting_error(sigma: float, ratio: float, x: np.ndarray, eigenvalues: np.ndarray,
                  kernel: str = "gaussian", bandwidth: float = 0.01):

    """
    Calculates the fitting error of the Marchencko-Pastur distribution

    Parameters
    ----------
    sigma:
        Sigma paramter in Marchencko-Pastur distribution
    ratio:
        Ratio between dimension and num. obs.
    x:
        Points where to evaluate the pdf.
    eigenvalues:
        Eigenvalues.
    kernel:
        Kernel type.
    bandwidth
        Bandwith.

    Returns
    -------

    """

    mp_density = marchencko_pastur_density(x, sigma, ratio)

    kde_density = fit_kde(eigenvalues, kernel=kernel, bandwidth=bandwidth, x=x)

    return np.mean(np.square(mp_density - kde_density))


def crem_denoised_corr_mat(eigenvalues: np.ndarray, eigenvectors: np.ndarray, num_factors: int) -> np.ndarray:

    """
    Denoises correlation matrix by fixing random eigenvalues

    Parameters
    ----------
    eigenvalues:
        Sorted eigenvalues.
    eigenvectors:
        Sorted eigenvectors
    num_factors:
        Number of factors.

    Returns
    -------
    np.ndarray
        Denoised correlation matrix.

    """

    eig_vals = eigenvalues.copy()
    eig_vals[num_factors:] = eig_vals[num_factors:].sum() / float(len(eig_vals) - num_factors)
    eig_vals = np.diag(eig_vals)

    corr_mat = eigenvectors @ eig_vals @ eigenvectors.T
    corr_mat = cov_to_corr_matrix(corr_mat)

    return corr_mat


def tsm_denoised_corr_mat(eigenvalues: np.ndarray, eigenvectors: np.ndarray, num_factors: int,
                          alpha: float = 0.0) -> np.ndarray:

    """
    Denoises correlation matrix by shrinkage on eigenvectors

    Target shrinkage method.

    Parameters
    ----------
    eigenvalues:
        Sorted eigenvalues.
    eigenvectors:
        Sorted eigenvectors
    num_factors:
        Number of factors.
    alpha:
        Weight.

    Returns
    -------
    np.ndarray
        Denoised correlation matrix.

    """

    eig_vals_l = np.diag(eigenvalues[:num_factors])
    eig_vals_r = np.diag(eigenvalues[num_factors:])

    eig_vecs_l = eigenvectors[:, :num_factors]
    eig_vecs_r = eigenvectors[:, num_factors:]

    part1 = eig_vecs_l @ eig_vals_l @ eig_vecs_l.T
    part2 = eig_vecs_r @ eig_vals_r @ eig_vecs_r.T
    part3 = np.diag(np.diag(part2))

    corr_mat = part1 + alpha * part2 + (1 - alpha) * part3

    return corr_mat


"""
Check and fix positive semi-definite matrix
"""


def check_positive_semi_definite(matrix: np.ndarray):
    """
    Checks that a symmetric square matrix is positive semi-deinite.

    See https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy

    Parameters
    ----------

    matrix:
        Symmetric, Square matrix.


    Returns
    -------
    bool
        Boolean indicating whether the matrix is positive semi-definite

    """

    try:
        np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
        return True
    except np.linalg.LinAlgError:
        return False


def fix_nonpositive_semidefinite(matrix: np.ndarray, method="spectral", scale_diag=False):
    # Eigendecomposition
    q, V = np.linalg.eigh(matrix)

    if method == "spectral":
        # Remove negative eigenvalues
        q = np.where(q > 0, q, 0)

        if scale_diag:
            # scaling matrix
            scale_diag_mat = np.diag(1.0 / (V ** 2 @ q))

            # scaled eigenvector
            V = np.sqrt(scale_diag_mat) @ V

        # Reconstruct matrix
        fixed_matrix = V @ np.diag(q) @ V.T

    elif method == "diag":

        min_eig = np.min(q)
        fixed_matrix = matrix - 1.1 * min_eig * np.eye(len(matrix))

        if scale_diag:
            sqrt_diag = np.sqrt(np.diag(fixed_matrix))
            fixed_matrix = fixed_matrix / np.outer(sqrt_diag, sqrt_diag)
            fixed_matrix[fixed_matrix < -1], fixed_matrix[fixed_matrix > 1] = -1, 1

    else:
        raise NotImplementedError("Method not implemented")

    return fixed_matrix


"""
Covariance shrinkage estimators
"""


def ledoit_wolf_constant_variance(data: np.ndarray, demean: bool = True) -> np.ndarray:
    """
    Computes the Ledoit and Wolf (2004) shrinkage covariance matrix

    See https://www.sciencedirect.com/science/article/pii/S0047259X03000964

    Parameters
    ----------
    data:
        Num. observations x Num. variables
    demean:
        Boolean indicating whether to demean data.

    Returns
    -------
    np.ndarray
        Covariance matrix


    """

    if demean:
        data = data - np.mean(data, axis=0)[None, :]

    T = data.shape[0]

    # calculate sample covariance matrix
    S = data.T @ data / T  # np.cov(data, rowvar=False)

    # get dimension
    N = S.shape[0]

    # calculate constant variance
    m = np.trace(S) / N

    # shrinkage target
    shrink_target = m * np.eye(N)

    # delta
    d2 = np.linalg.norm(S - shrink_target, ord='fro') ** 2 / N

    # beta
    b2_bar = np.sum([np.linalg.norm(np.outer(data[t, :], data[t, :]) - S, ord='fro') ** 2 / N
                     for t in range(T)]) / (T ** 2)

    b2 = np.min([d2, b2_bar])

    # alpha
    a2 = d2 - b2

    # calculate new covariance matrix
    shrinkage_cov_mat = (b2 / d2) * shrink_target + (a2 / d2) * S

    return shrinkage_cov_mat


def ledoit_wolf_single_index(returns: np.ndarray, market_return: np.ndarray, demean: bool = True) -> Tuple[np.ndarray, float]:
    """
    Computes the Ledoit and Wolf (2003) shrinkage covariance estimator

    See https://www.sciencedirect.com/science/article/abs/pii/S0927539803000070

    Parameters
    ----------
    returns:
        Num. observations x Num. variables
    market_return:
        Market return, 1-D array
    demean:
        Boolean indicating whether to demean data.

    Returns
    -------
    np.ndarray
        Covariance matrix

    """

    # get dimensions
    T, N = np.shape(returns)

    # demean returns if necessary
    if demean:
        x = returns - np.mean(returns, axis=0)[None, :]
        x_mkt = market_return - np.mean(market_return)
    else:
        x = returns
        x_mkt = market_return

    # sample covariance matrix
    S = x.T @ x / T

    # shrinkage target
    var_mkt = x_mkt @ x_mkt / T
    cov_mkt = x_mkt @ x / T
    beta = cov_mkt / var_mkt
    F = np.outer(beta, beta) * var_mkt
    np.fill_diagonal(F, np.diag(S))

    # shrinkage intensity
    # estimate of pi
    x2 = x ** 2
    p = 1 / T * np.sum(x2.T @ x2) - np.sum(S ** 2)

    # estimate rho
    # diagonal elements
    r_diag = 1 / T * np.sum(x2 ** 2) - sum(np.diag(S) ** 2)

    # off diagonal elements
    z = x * x_mkt[:, None]
    v1 = x2.T @ z / T
    r_off_diag_1 = np.sum(v1 * cov_mkt[None, :]) / var_mkt - np.sum(np.diag(v1) * cov_mkt) / var_mkt

    v2 = z.T @ z / T
    r_off_diag_2 = np.sum(v2 * np.outer(cov_mkt,  cov_mkt)) / var_mkt**2 \
                   - np.sum(np.diag(v2) * cov_mkt**2) / var_mkt**2

    r_off_diag_3 = np.sum(F*S) - np.sum(np.diag(F*S))

    r_off_diag = 2 * r_off_diag_1 - r_off_diag_2 - r_off_diag_3

    r = r_diag + r_off_diag

    # estimate gamma
    c = np.linalg.norm(S - F, "fro") ** 2

    # compute shrinkage constant
    k = (p - r) / c
    alpha = max(0, min(1, k / T))

    # return shrinkage estimator
    return alpha * F + (1 - alpha) * S, alpha


if __name__ == "__main__":

    import numpy as np

    T = 1000
    N = 10

    mkt_return = np.random.normal(loc=0.05, scale=0.2, size=T)
    betas = np.random.uniform(low=-1, high=1.5, size=N)
    returns = mkt_return[:, None] * betas[None, :] + np.random.normal(loc=0, scale=0.05, size=(T, N))

    cov1, alpha1 = ledoit_wolf_single_index(returns, mkt_return)