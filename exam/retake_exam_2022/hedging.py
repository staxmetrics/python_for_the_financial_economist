import numpy as np
from cvxopt import matrix
from cvxopt.solvers import qp


def solve_hedging_problem(h_matrix: np.ndarray, d_vector: np.array, delta: float = 3.0):

    """
    Solves hedging problem with leverage constraint. Leverage is only allowed in first asset.
    
    See "Improving Interest Rate Risk Hedging Strategies through Regularization", 
    Mantilla-Garcia, Martellini, Milhau, Ramirez-Garrido (2022)
    
    """

    lambda_mat = h_matrix @ h_matrix.T
    theta_vec = - h_matrix @ d_vector
    n = h_matrix.shape[0]

    """
    Define vectors and matrices
    """

    P = matrix(np.block([[lambda_mat, np.zeros((n, n)), np.zeros((n, n))],
                         [np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))],
                         [np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))]]) + np.eye(3 * n) * 0.0000001)

    q = matrix(np.r_[theta_vec, np.zeros(n), np.zeros(n)])

    """
    Define inequality constraints
    """

    G = matrix(np.r_[np.block([[np.zeros((1, n)), np.ones((1, n)), np.ones((1, n))]]),
                     -np.eye(3 * n)])

    h = matrix(np.r_[delta, 999999, np.zeros(3 * n - 1)])

    """
    Define equality constraints
    """

    A = matrix(np.r_[np.block([[np.ones((1, n)), np.zeros((1, n)), np.zeros((1, n))]]),
                     np.block([[np.eye(n), np.eye(n), -np.eye(n)]])])

    b = matrix(np.r_[np.ones((1, 1)), np.zeros((n, 1))])

    """
    Solve problem
    """
    sol = np.array(qp(P, q, G, h, A, b)['x']).flatten()

    return sol[:n]
