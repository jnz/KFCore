function [x,U,d] = kalman_udu(z,R,H,x,U,d)
% UDU' Bierman Filter observation step for linear systems
%
% Jan Zwiener (jan@zwiener.org)
%
% Inputs:
%  z   - measurement vector (m x 1)
%  R   - variance of measurement error (m x m)
%  H   - measurement sensitivity matrix (m x n)
%  x   - a priori estimate of state vector (n x 1)
%  U   - unit upper triangular factor of covariance matrix of a priori state uncertainty (n x n)
%  d   - diagonal vector with factor of covariance matrix of a priori state uncertainty (n x 1)
%
% Outputs:
%  x   - a posteriori estimate of state vector (n x 1)
%  U   - upper unit triangular UD factor of a posteriori state uncertainty covariance (n x n)
%  d   - diagonal UD factor vector of a posteriori state uncertainty covariance (n x 1)
%
% References:
%   1. Grewal, Weill, Andrews. "Global positioning systems, inertial
%      navigation, and integration". 1st ed. John Wiley & Sons, New York, 2001.

isdiagonal = isequal(R, diag(diag(R)));
if (isdiagonal == false)
    [G] = chol(R); % G'*G = R
    zdecorr = (G')\z;
    Hdecorr = (G')\H;
    Rdecorr = eye(length(z));
else
    zdecorr = z;
    Hdecorr = H;
    Rdecorr = R;
end

% Process measurements independently:
for i=1:size(H,1)
    dz = zdecorr(i)-Hdecorr(i,:)*x;
    [x,U,d] = kalman_udu_scalar(dz,Rdecorr(i,i),Hdecorr(i,:),x,U,d);
end

end

