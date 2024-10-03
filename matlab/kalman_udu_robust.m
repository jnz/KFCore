function [x,U,d] = kalman_udu_robust(z,R,H,x,U,d,chi2_threshold,reject_outliers)
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
%  chi2_threshold - Scalar threshold (>0) Threshold for outlier detection (1 x 1)
%                   Set to 0 to disable. 3.8415 for 95%, 6.6349 for 99%.
%                   Calculate with chi2inv(0.95, 1) and chi2inv(0.99, 1)
%  reject_outliers - If set to true, measurements classified as outliers are skipped (true/false)
%
% Outputs:
%  x   - a posteriori estimate of state vector (n x 1)
%  U   - upper unit triangular UD factor of a posteriori state uncertainty covariance (n x n)
%  d   - diagonal UD factor vector of a posteriori state uncertainty covariance (n x 1)
%
% References:
%   1. Grewal, Weill, Andrews. "Global positioning systems, inertial
%      navigation, and integration". 1st ed. John Wiley & Sons, New York, 2001.
%
% If reject_outliers is set to false:
% When the data is potentially corrupted, then the variance of the measurement
% is downgraded instead of being discarded, so that the filter will still
% incorporate this measurement in our estimation process, but it is more
% resistant to outliers.

% check if measurements need to be decorrelated:
isdiagonal = isequal(R, diag(diag(R)));
if (isdiagonal == false)
    [G] = chol(R); % G'*G = R
    zdecorr = (G')\z;
    Hdecorr = (G')\H;
    Rdecorr = eye(length(z));
else
    % measurements are uncorrelated, process as is
    zdecorr = z;
    Hdecorr = H;
    Rdecorr = R;
end

if (nargin < 8)
    reject_outliers = false; % by default downgrade potential outliers
end
if (nargin < 7)
    chi2_threshold = 0; % disable robust module below
end

% Process measurements independently:
for i=1:size(H,1)

    Rf = Rdecorr(i,i); % Accept the observation directly
    dz = zdecorr(i) - Hdecorr(i,:)*x;

    % <robust>
    if (chi2_threshold > 0)
        HPHT = Hdecorr(i,:)*U*diag(d)*U'*Hdecorr(i,:)';
        S = HPHT + Rdecorr(i,i);

        % square of the Mahalanobis distance from observation to the
        % predicted observation squared: (= dz'*inv(P_predicted)*dz)
        mahalanobis_dist_squared = (dz*dz)/S;
        if (mahalanobis_dist_squared > chi2_threshold)
            if reject_outliers
                continue % skip this scalar measurement
            end

            % scale by "f" to lower the influence of the observation
            f = mahalanobis_dist_squared / chi2_threshold;
            Rf = (f - 1)*HPHT + f*Rf; % Rf is the new scaled obs. covariance R(i,i)
            % P_predicted = f*P_predicted; % equiv. to P_predicted = H*P*H' + Rf;
        end
    end
    % </robust>

    [x,U,d] = kalman_udu_scalar(dz,Rf,Hdecorr(i,:),x,U,d);
end

end

