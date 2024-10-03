function [x, P] = kalman_robust(x, P, dz, R, H, chi2_threshold,reject_outliers)
%KALMAN_ROBUST Robust Kalman Filter based on:
%   Chang, G. (2014). Robust Kalman filtering based on Mahalanobis distance
%   as outlier judging criterion. Journal of Geodesy, 88(4), 391-401.
%
% Jan Zwiener (jan@zwiener.org)
%
% Inputs:
%    x    - nx1 A priori state vector (size=n) at epoch k
%    P    - nxn Covariance matrix of state vector x at epoch k
%    dz   -  mx1 Measurement difference vector (size=m) at epoch k
%         -      dz = measurement - predicted measurement
%         -      dz = y - H*x
%    R    - mxm Covariance matrix of measurement vector y
%    H    - mxn Observation matrix so that y = H*x
%
% Optional:
%    chi2_threshold 1x1 Threshold (>0) to scale down measurements.
%                   Can be computed with the chi-square inverse cumulative
%                   distribution:

%                   function ("chi2inv") for e.g. alpha=0.05:
%                     chi2_threshold = chi2inv(1-alpha, length(dz));
%
%    reject_outliers - If set to true, measurements classified as outliers
%                      are skipped (true/false).
%
% By default a table for significance level alpha = 5% is added to this
% function.  If the measurement difference vector is larger than 20, a
% chi2_threshold must be supplied by the caller.
%
% Outputs:
%    x    - nx1 A posteriori state vector at epoch k (corrected by measurements)
%    P    - nxn A posteriori covariance of x at epoch k
%
assert( ( (size(P,1)==size(P,2) ) && ... % symmetric covariance matrix
          (size(P,1)==size(x,1) ) && ...
          (size(x,2)==1 ) && ... % make sure x is a column vector
          (size(dz,2)==1 ) && ... % make sure dz is a column vector
          (size(dz,1)==size(R,1) ) && ...
          (size(R,1)==size(R,2) ) && ...
          (size(H,1)==size(dz,1) ) && ...
          (size(H,2)==size(x,1) ) && ...
          (nargin >= 5 && nargin <= 7) ), 'Invalid arguments');

if (nargin < 6)
    % precomputed chi2 table for up to m=20 observations
    % (degree of freedom) by chi2inv(0.95, 1:20) function with
    % significance level alpha = 0.05 (5% - 95%)
    table = [  3.8415  5.9915  7.8147  9.4877 11.0705 12.5916 14.0671 ...
              15.5073 16.9190 18.3070 19.6751 21.0261 22.3620 23.6848 ...
              24.9958 26.2962 27.5871 28.8693 30.1435 31.4104 ];
    if (length(dz) > length(table))
        error('Measurement vector y too big.');
    end
    chi2_threshold = table(length(dz));
else
    assert(chi2_threshold > 0);
end
if (nargin < 7)
    reject_outliers = false; % by default downgrade potential outliers
end

% Covariance of the observation prediction (= H*x in the linear case)
HPHT = H*P*H';
P_predicted = HPHT + R;

% square of the Mahalanobis distance from observation to the
% predicted observation squared: (= dz'*inv(P_predicted)*dz)
mahalanobis_dist_squared = (dz'/P_predicted)*dz;
% The probability P that mahalanobis_dist_squared is larger than
% chi2_threshold is alpha percent:
%   P(mahalanobis_dist_squared > chi2_threshold) = alpha%
if (mahalanobis_dist_squared > chi2_threshold)
    if reject_outliers
        return % don't update the state with this measurement vector
    end

    % scale by "f" to lower the influence of the observation
    f = mahalanobis_dist_squared / chi2_threshold;
    Rf = (f - 1)*HPHT + f*R; % Rf is the new scaled obs. covariance R
    % P_predicted = f*P_predicted; % equiv. to P_predicted = H*P*H' + Rf;
else
    Rf = R; % Accept the observation directly
end

[x, P] = kalman_takasu(x, P, dz, Rf, H);

end

% chi2inv(0.99, 1:20)'
%
% ans =
%
%     6.6349
%     9.2103
%    11.3449
%    13.2767
%    15.0863
%    16.8119
%    18.4753
%    20.0902
%    21.6660
%    23.2093
%    24.7250
%    26.2170
%    27.6882
%    29.1412
%    30.5779
%    31.9999
%    33.4087
%    34.8053
%    36.1909
%    37.5662
