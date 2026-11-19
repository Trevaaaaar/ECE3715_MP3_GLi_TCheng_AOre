
import numpy as np


def sample_mean_cov(X):
    X = np.asarray(X, dtype=float)

    mean_vector = np.mean(X, axis=0)
    centered_samples = X - mean_vector

    covariance_matrix = (
        centered_samples.T @ centered_samples
    ) / (X.shape[0] - 1)

    return mean_vector, covariance_matrix


def is_valid_covariance(S, tol=1e-10):
    """
    Check whether S is symmetric and positive semidefinite.
    """
    S = np.asarray(S, dtype=float)

    if S.ndim != 2:
        return False

    if S.shape[0] != S.shape[1]:
        return False

    if not np.allclose(S, S.T, atol=tol, rtol=0):
        return False

    eigenvalues, _ = np.linalg.eigh(S)

    return np.all(eigenvalues >= -tol)

def gaussian_pdf(x, mu, Sigma):
    return

def whitening_matrix(Sigma):
    return

def lmmse_weights(Sigma_xx, Sigma_sx):
    return