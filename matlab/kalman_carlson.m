function [x,Cxx] = kalman_carlson(z,R,H,x,Cxx)
% Carlson Cholesky filter observation step for linear systems.
% Square Root Kalman Filter with Carlson Update and Schmidt-Householder
% Temporal Update (kalman_carlson_predict.m).
%
% Jan Zwiener (jan@zwiener.org)
%
% Inputs:
%  z     - scalar measurement
%  R     - variance of measurement error
%  H     - measurement sensitivity matrix
%  x     - a priori estimate of state vector
%  Cxx   - upper triangular Cholesky factor of covariance matrix of a priori
%          state uncertainty
%
% Outputs:
%  x   - a posteriori estimate of state vector
%  Cxx - upper triangular Cholesky factor of covariance matrix of a posteriori
%        state uncertainty covariance
%
% References:
%   1. Grewal, Weill, Andrews. "Global positioning systems, inertial
%      navigation, and integration". 1st ed. John Wiley & Sons, New York, 2001.
%
% Jan Zwiener (jan@zwiener.org)
%
% Start by a given state covariance matrix Qxx.
% Perform
%
%     Cxx = chol(Qxx)'; % note the transpose
%
% where Cxx*Cxx' = Qxx;
%
% Call
%
%     [x,Cxx] = kalman_carlson(z,R,H,x,Cxx);
%
% to update with a scalar measurement 'z'.
% If you have multiple measurements, process them individually.
% If the measurements are correlated, they are decorrelated inside
% kalman_carlson(...) first with:
%
%     [G] = chol(R); % G'*G = R
%     zdecorr = (G')\z;
%     Hdecorr = (G')\H;
%     Rdecorr = eye(length(z));
%
% The prediction step/temporal update of the Cxx matrix is done with the
%
%     [Cxx] = kalman_carlson_predict(Cxx, phi, G, Cq)
%
% Normal form:
%     Qxx^(-) = phi*Qxx^(+)*phi' + G*Qnoise*G'
%
%     Cq*Cq' = Qnoise -> from chol(...)' (note the transpose)
%
%
% Example:
%
%     % Init
%     dt_sec = 1.0;
%     x = [0 0 1]'; % state vector: position, vel, acceleration
%     Qxx = diag([0.5 0.1 0.00001].^2); % initial state covariance
%     Cxx = chol(Qxx)'; % Cxx*Cxx' = Qxx
%
%     % Prediction
%     phi = [1 dt_sec 0.5*dt_sec^2; 0 1 dt_sec; 0 0 1];
%     x = phi*x; % state prediction
%     % Add some noise to acceleration
%     G = [0 0 dt_sec]';
%     Cq = 0.001; % as std. dev. in m/s^2
%     [Cxx] = kalman_carlson_predict(Cxx, phi, G, Cq); % predict Cxx
%
%     % Fusion
%     A = [1 0 0];
%     z = 0.5; % position measurement
%     R = 0.1^2; % measurement covariance
%     [x,Cxx] = kalman_carlson(z,R,A,x,Cxx);
%
%     % Calculate Qxx if required
%     Qxx = Cxx*Cxx';


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
    [x,Cxx] = kalman_carlson_scalar(zdecorr(i),Rdecorr(i,i),Hdecorr(i,:),x,Cxx);
end

end

function [x,Cxx] = kalman_carlson_scalar(z,R,H,xin,CxxIn)
C     = CxxIn;
alpha = R;
delta = z;
w = zeros(length(xin),1); % buffer

for j=1:length(xin)
    delta  = delta - H(j)*xin(j);
    sigma  = 0;
    for i=1:j
        sigma  = sigma + C(i,j)*H(i);
    end
    beta   = alpha;
    alpha  = alpha + sigma^2;
    gamma  = sqrt(alpha*beta);
    eta    = beta/gamma;
    zeta   = sigma/gamma;
    w(j)   = 0;
    for i=1:j
        tau    = C(i,j);
        C(i,j) = eta*C(i,j) - zeta*w(i);
        w(i)   = w(i) + tau*sigma;
    end
end
Cxx     = C;
epsilon = delta/alpha;
x       = xin + epsilon*w; % multiply by unscaled Kalman gain

end
