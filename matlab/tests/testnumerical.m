function [] = testnumerical()
% TESTNUMERICAL Stress test the Kalman filter update routine numerically.

rng(42);

epsilon = 1e-6;
epsilon_R = 1e-6;

H = [ 1 1 1; 1 1 1+epsilon];
fprintf('cond(H) = %.5e, rank(H) = %f\n', cond(H), rank(H));

P = eye(3);
x = [1e3; 2e3; 3e3 ];

A = [1 0.5; 0 1];
sigma = epsilon_R^2;
R = A*(eye(2)*sigma^2)*A';

S = H*P*H' + R;
fprintf("cond(H*P*H' + R) = %.5e, rank(S) = %f\n", cond(S), rank(S));

ztrue = H*x;
z = ztrue + A*randn(2,1)*sigma;

dz = z - H*x;

% Vanilla Test Numerical Stability

[x_vanilla, P_vanilla ] = kalman_vanilla(x, P, dz, R, H, false);

% Vanilla + Joseph's form Test Numerical Stability

[x_joseph, P_joseph ] = kalman_vanilla(x, P, dz, R, H);

% Test Takasu Numerical Stability

[x_takasu, P_takasu ] = kalman_takasu(x, P, dz, R, H);

% UDU Test Numerical Stability

[U, d] = udu(P);

[x_udu,U_udu,d_udu] = kalman_udu(z,R,H,x,U,d);
P_udu = U_udu*diag(d_udu)*U_udu';

% Carlson Test Numerical Stability

Cxx_carlson = chol(P)';
[x_carlson,Cxx_carlson] = kalman_carlson(z,R,H,x,Cxx_carlson);
P_carlson = Cxx_carlson*Cxx_carlson';

checkfilter(x_vanilla, P_vanilla, "Vanilla");
checkfilter(x_joseph,  P_joseph,  "Joseph");
checkfilter(x_takasu,  P_takasu,  "Takasu");
checkfilter(x_carlson, P_carlson, "Carlson");
checkfilter(x_udu,     P_udu,     "UDU");

end

function [] = checkfilter(x, P, name)
chatty = true;
[healthy] = filterhealthy(x, P, chatty);
if (healthy == false)
    fprintf(2, '[ ] Filter %s is not healthy\n', name);
else
    fprintf('[x] Filter %s is healthy\n', name);
end
end

