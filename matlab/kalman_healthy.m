function [healthy] = kalman_healthy(x, P, deep_check)
%KALMAN_HEALTHY Check the numerical health status of a Kalman filter.
%   The state vector x must not contain NaNs and Infs.
%   The covariance matrix P must be positive-semidefinite.
%
% Jan Zwiener (jan@zwiener.org)
%
% Inputs
% x (n x 1) A priori state vector (size=n) at epoch k
% P (n x n) Covariance matrix of state vector x at epoch k

healthy =   all(isfinite(x)) && ...
            issymmetric(P) && ...
            all(diag(P) > 10*eps);

if (deep_check)
    healthy = healthy && all(eig(P) > 0);
end

end

