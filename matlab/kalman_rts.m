function [x_smooth, P_smooth] = kalman_rts(x_apriori, x_aposteriori, P_apriori, P_aposteriori, phi)
%KALMAN_RTS Rauch-Tung-Striebel (RTS) fixed-interval smoothing.
%
% Jan Zwiener (jan@zwiener.org)
%
% After a forward pass regular Kalman filter update with N epochs, run the
% filter again with a backwards pass.
% During the forward pass the state vector a-priori and a-posteriori states and
% covariances are saved for the RTS smoother.
%
% n: Number of elements in column state vector
% N: Number of total epochs
%
% x_apriori (n by 1 by N) by k A priori state vector (size=n) at epoch 1..N
% x_aposteriori (n by 1 by N) by k A posteriori state vector (size=n) at epoch 1..N
% P_apriori (n by n by N) Apriori covariance matrix of state vector x at epoch 1..N
% P_aposteriori (n by n by N) Aposteriori covariance matrix of state vector x at epoch 1..N
% phi (n by n by N) State transition matrix at epoch 1..N
%
% Source:
% - H. Rauch, F. Tung, and C. Striebel. Maximum likelihood estimates of 
%   lineardynamic systems. AIAA Journal, 3(8):1445 – 1450, 1965
% - P. D. Groves, Principles of GNSS, Inertial, and Multisensor 
%   Integrated Navigation Systems. Norwood, MA: Artech House, 2008.

N = size(x_apriori, 2); % epochs N
n = size(x_apriori, 1); % column state vector length

assert(size(x_aposteriori,1) == n);
assert(size(x_aposteriori,2) == N);
assert(size(P_apriori,1) == n);
assert(size(P_apriori,2) == n);
assert(size(P_apriori,3) == N);
assert(size(P_aposteriori,1) == n);
assert(size(P_aposteriori,2) == n);
assert(size(P_aposteriori,3) == N);
assert(size(phi,1) == n);
assert(size(phi,2) == n);
assert(size(phi,3) == N);

x_smooth = zeros(n, N);
P_smooth = zeros(n, n, N);

x_smooth(:,N) = x_aposteriori(:,N);
P_smooth(:,:,N) = P_aposteriori(:,:,N);

for k=(N-1):-1:1
    K = (P_aposteriori(:,:,k)*phi(:,:,k)')/(P_apriori(:,:,k+1));
    x_smooth(:,k) = x_aposteriori(:, k) + K*(x_smooth(:,k+1) -  x_apriori(:, k+1));
    P_smooth(:,:,k) = P_aposteriori(:,:,k) + K*(P_smooth(:,:,k+1) - P_apriori(:,:,k+1))*K';
end

end
