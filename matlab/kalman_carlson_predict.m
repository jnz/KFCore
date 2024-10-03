function [x, CxxOut] = kalman_carlson_predict(CxxIn, Phi, G, Cq)
% For "Carlson Filter": Schmidt-Householder Temporal Update Step.
%
% Jan Zwiener (jan@zwiener.org)
%
% This modification of the Householder algorithm performs upper
% triangularization of the partitioned matrix [phi*CxxIn, G*Cq] by modifying
% phi*CxxIn and G*Cq in place using Householder transformations of the
% (effectively) partitioned matrix.  Source: [1].
%
% Non-square root update would be:
% QxxOut = phi*QxxIn*phi' + G*Qnoise*G'
% QxxOut = phi*CxxIn*CxxIn'*phi' + G*Cq*Cq'*G'
%
% Inputs:
%      x      - a posteriori state vector with size (n x 1)
%      Phi    - state transition matrix (n x n)
%      CxxIn  - upper triangular factor (Cxx) of the corrected state estimation
%               uncertainty P^{+} so that CxxIn*CxxIn' = P^{+} (n x n)
%      Gin    - process noise distribution matrix (n x r)
%      Cq     - diagonal covariance matrix of process noise
%               in the stochastic system model (r x r)
% Outputs:
%      x      - predicted state vector (a priori)
%      CxxIn  - upper triangular factor (Cxx) of the predicted state estimation
%               uncertainty P^{-} so that Cxx*Cxx' = P^{-} (n x n)
%
% References:
%   1. Grewal, Weill, Andrews. "Global positioning systems, inertial
%      navigation, and integration". 1st ed. John Wiley & Sons, New York, 2001.
%
% Complexity: n^3*r + 1/2*(n+1)^2*r + 5 + 1/3*(2*n + 1) flops

x = Phi*x;
A = Phi*CxxIn; % (n x n) matrix
B = G*Cq; % (n x r) matrix
n = size(A, 1);
r = size(B, 2);
v = zeros(n,1); % temp. vector
w = zeros(n+r, 1); % temp. vector

for k = n: -1:1
    sigma = 0;
    for j = 1: r
        sigma = sigma + B(k,j)^2;
    end
    for j = 1: k
        sigma = sigma + A(k,j)^2;
    end
    alpha = sqrt(sigma);
    sigma = 0;
    for j = 1: r
        w(j) = B(k,j);
        sigma = sigma + w(j)^2;
    end
    for j = 1: k
        if j == k
            v(j) = A(k,j) - alpha;
        else
            v(j) = A(k,j);
        end
        sigma = sigma + v(j)^2;
    end
    alpha = 2/sigma;
    for i = 1:k
        sigma = 0;
        for j = 1: r
            sigma = sigma + B(i,j)*w(j);
        end
        for j = 1: k
            sigma = sigma + A(i,j)*v(j);
        end
        beta = alpha*sigma;
        for j = 1: r
            B(i,j) = B(i,j) - beta*w(j);
        end
        for j = 1: k
            A(i,j) = A(i,j) - beta*v(j);
        end
    end
end

CxxOut = A;

end

