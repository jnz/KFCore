function [x, P] = kalman_decorr(x, P, z, R, H)
%KALMAN_DECORR Kalman filter update routine based on measurement
%   decorrelation. Source section 8.1.3.1:
%   [1] Grewal, Weill, Andrews (2001): Global positioning systems, inertial
%   navigation, and integration. 1st Ed. John Wiley & Sons, New York.
%
% Jan Zwiener (jan@zwiener.org)
%
% x (n x 1) A priori state vector (size=n) at epoch k
% P (n x n) Covariance matrix of state vector x at epoch k
% z (m x 1) Measurement vector (size=m) at epoch k
% R (m x m) Covariance matrix of measurement vector z
% H (m x n) Observation matrix so that z = H*x
%
% Return value:
% x nx1 A posteriori state vector at epoch k (corrected by measurements)
% P nxn A posteriori covariance of x at epoch k
%
%
assert( ( (size(P,1)==size(P,2) ) && ... % symmetric covariance matrix
          (size(P,1)==size(x,1) ) && ...
          (size(x,2)==1 ) && ... % make sure x is a column vector
          (size(z,2)==1 ) && ... % make sure z is a column vector
          (size(z,1)==size(R,1) ) && ...
          (size(R,1)==size(R,2) ) && ...
          (size(H,1)==size(z,1) ) && ...
          (size(H,2)==size(x,1) ) && ...
          (nargin == 5) ), 'Invalid arguments');

[G] = chol(R); % G'*G = R

zdecorr = (G')\z;
Hdecorr = (G')\H;

for i=1:length(zdecorr)
    Hline = Hdecorr(i, :);

    % Vanilla form:
    % Rf = 1; % std. dev. is now exactly 1
    % K = P*Hline'/(Hline*P*Hline' + Rf);
    % dx = K*(zdecorr(i) - Hline*x);
    % x = x + dx;
    % P = P - K*Hline*P;

    s = 1.0 / (Hline*P*Hline' + 1);
    K = (P*Hline')*s;
    dz = zdecorr(i) - Hline*x;

    dx = K*dz;
    x = x + dx;
    % [1] sec. 8.1.4 "Joseph stabilized implementation".
    W = eye(length(x)) - K*Hline;
    P = W*P*W' + K*K';
end

P = 0.5*(P + P');

end

