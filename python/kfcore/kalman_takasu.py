import numpy as np

def kalman_takasu(x, P, dz, R, H, Josephsform=True):
    """
    Kalman Filter update using T. Takasu's formulation with Cholesky decomposition.

      (1) D = P * H'           symm     |   Matrix dimensions:
      (2) S = H * D + R        gemm     |   D = n x m
      (3) L = chol(S) (L*L'=S) potrf    |   S = m x m
      (4) E = D * L^-T         trsm     |   L = m x m
      (5) P = P - E*E'         syrk     |   E = n x m
      (6) K = E * L^-1         trsm     |   K = n x m

    Parameters
    ----------
    x : ndarray
        A priori state vector (n x 1) at epoch k.
    P : ndarray
        Covariance matrix of state vector x (n x n) at epoch k.
    dz : ndarray
        Measurement difference vector (m x 1) at epoch k, where dz = z - H @ x.
    R : ndarray
        Covariance matrix of measurement vector z (m x m).
    H : ndarray
        Observation matrix (m x n) such that z = H @ x.
    Josephsform : bool, optional
        Ignored/unused, but kept to have the same interface as kalman_vanilla

    Returns
    -------
    x : ndarray
        A posteriori state vector at epoch k.
    P : ndarray
        A posteriori covariance matrix of x at epoch k.
    """
    # Ensure inputs are numpy arrays
    x = np.atleast_1d(x)
    P = np.atleast_2d(P)
    dz = np.atleast_1d(dz)
    R = np.atleast_2d(R)
    H = np.atleast_2d(H)

    n = x.shape[0] # state vector size
    m = dz.shape[0] # measurement vector size
    x = x.reshape(n, 1) # Reshape to column vectors
    dz = dz.reshape(m, 1)

    # Input validation
    assert P.shape == (n, n), "P must be a square matrix of shape (n, n)"
    assert R.shape == (m, m), "R must be a square matrix of shape (m, m)"
    assert H.shape == (m, n), "H must have shape (m, n)"
    assert dz.shape == (m, 1), "dz must be a column vector of shape (m, 1)"

    D = P @ H.T # could use "symm"
    S = H @ D + R  # S = H @ P @ H.T + R

    try:
        U = np.linalg.cholesky(S, upper=True)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError("Innovation covariance matrix S is singular.")
    E = np.linalg.solve(U.T, D.T).T # Solve for E: E * U = D
    K = np.linalg.solve(U, E.T).T # Solve for K: K * U.T = E

    x = x + K @ dz
    P = P - E @ E.T # Symmetric rank update
    P = 0.5 * (P + P.T) # Ensure P is symmetric
    x = x.flatten()

    return x, P

