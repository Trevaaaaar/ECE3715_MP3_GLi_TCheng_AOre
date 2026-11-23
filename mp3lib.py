
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


def _symmetric_covariance(Sigma):
    covariance = np.asarray(Sigma, dtype=float)
    if (
        covariance.ndim != 2
        or covariance.shape[0] == 0
        or covariance.shape[0] != covariance.shape[1]
        or not np.all(np.isfinite(covariance))
        or not np.allclose(covariance, covariance.T)
    ):
        raise ValueError("Sigma must be a finite, symmetric square matrix.")
    return covariance


def gaussian_pdf(x, mu, Sigma):
    """Evaluate a Gaussian density for points whose last axis is the feature axis."""
    covariance = _symmetric_covariance(Sigma)
    dimension = covariance.shape[0]
    mean = np.asarray(mu, dtype=float)
    points = np.asarray(x, dtype=float)
    if mean.shape != (dimension,) or points.ndim == 0 or points.shape[-1] != dimension:
        raise ValueError("mu and the last axis of x must match the size of Sigma.")

    # Cholesky requires positive definiteness, which the full-dimensional PDF needs.
    # If Sigma = L @ L.T, sqrt(det(Sigma)) is the product of L's diagonal.
    factor = np.linalg.cholesky(covariance)
    centered_points = (points - mean).reshape(-1, dimension)
    # Solve L z = x - mu directly to avoid forming an inverse.
    z = np.linalg.solve(factor, centered_points.T)
    squared_distances = np.sum(z**2, axis=0)
    log_normalizer = (
        0.5 * dimension * np.log(2 * np.pi) + np.log(np.diag(factor)).sum()
    )
    density = np.exp(-0.5 * squared_distances - log_normalizer)
    return density.reshape(points.shape[:-1])


def whitening_matrix(Sigma):
    """Return W such that W @ Sigma @ W.T is the identity."""
    covariance = _symmetric_covariance(Sigma)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = np.finfo(float).eps * covariance.shape[0] * np.max(np.abs(eigenvalues))
    if np.any(eigenvalues <= tolerance):
        raise np.linalg.LinAlgError("Whitening requires a nonsingular positive definite Sigma.")
    return eigenvectors.T / np.sqrt(eigenvalues)[:, None]


def lmmse_weights(Sigma_xx, Sigma_sx):
    """Return the linear MMSE weights w that solve Sigma_xx @ w = Sigma_sx.

    For a zero-mean target s and a zero-mean measurement vector x, the
    estimate is s_hat = w.T @ x. Sigma_xx = E[x x^T] is the (d, d)
    measurement covariance and Sigma_sx = E[x s] is the length-d vector of
    covariances between each measurement and the target. Setting the
    gradient of E[(s - w^T x)^2] to zero gives the normal equations
    Sigma_xx w = Sigma_sx. A (d, k) Sigma_sx returns (d, k) weights, one
    column per target.
    """
    covariance = _symmetric_covariance(Sigma_xx)
    cross_covariance = np.asarray(Sigma_sx, dtype=float)
    if (
        cross_covariance.ndim not in (1, 2)
        or cross_covariance.shape[0] != covariance.shape[0]
        or not np.all(np.isfinite(cross_covariance))
    ):
        raise ValueError("Sigma_sx must be finite with one row per measurement in Sigma_xx.")
    # Solve the normal equations instead of forming Sigma_xx^{-1}.
    # np.linalg.solve factors Sigma_xx once (LU) and back-substitutes. That
    # takes fewer operations than computing the inverse and then multiplying,
    # and it avoids the extra rounding error an explicit inverse adds as
    # Sigma_xx approaches singularity. In Step 8 that happens near the edge of
    # the admissible noise-correlation range.
    return np.linalg.solve(covariance, cross_covariance)
