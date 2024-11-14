import numpy as np

def kalman_vanilla(x, P, dz, R, H, Josephsform=True):
    """
    Simple Kalman Filter implementation.

    Parameters:
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
        Set to True to use the Joseph's form update of P. Default is True.

    Returns:
    x : ndarray
        A posteriori state vector at epoch k (corrected by measurements).
    P : ndarray
        A posteriori covariance of x at epoch k.
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

    S = H @ P @ H.T + R # Innovation covariance

    # Kalman Gain Matrix K
    # K = (P @ H.T)*inv(S)
    # Solving S.T @ K.T = (P @ H.T).T
    try:
        K = np.linalg.solve(S.T, np.dot(P, H.T).T).T
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError("Innovation covariance matrix S is singular.")

    # State update
    dx = K @ dz
    x = x + dx

    # Covariance update
    if Josephsform:
        W = np.eye(n) - K @ H
        P = W @ P @ W.T + K @ R @ K.T  # Joseph's form
    else:
        P = (np.eye(n) - K @ H) @ P

    P = 0.5 * (P + P.T) # Ensure P is symmetric
    x = x.flatten()

    return x, P
