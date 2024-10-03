function [] = testcasegen()
% TESTCASEGEN This function helps to generate test cases and numbers for
% the C/C++ unit tests.

kalman_test();
kalman_udu_testcase();
kalman_udu_robust_testcase();
decorr_testcase();
kalman_takasu_robust();
thornton_test();
predict_test();

gnss_benchmark();

end

function [] = kalman_test()

fprintf('<default test>\n');
R = eye(3)*0.25
dz = [0.2688; 0.9169; -1.1294]
H = [8 1 6 1 ; 3 5 7 2; 4 9 2 3 ]
x = ones(4,1)
P = eye(4) * 0.04
[xnew, Pnew, chi2_takasu] = kalman_takasu(x, P, dz, R, H);

xnew
Pnew
fprintf('</default test>\n');

end

function [] = kalman_udu_testcase()

fprintf('<udu test>\n');
R = eye(3)*0.25;
z = [16.2688, 17.9169, 16.8706]';
H = [ 8 1 6 1 ; 3 5 7 2; 4 9 2 3];
x = [1;1;1;1];

dz = z - H*x;

P = eye(4)*0.04;

[xexp, Pexp, chi2exp] = kalman_vanilla(x, P, dz, R, H);

[Uexp, dexp] = udu(Pexp);
R
z
H
x
xexp
Uexp'
dexp

fprintf('</udu test>\n');

end

function [] = kalman_udu_robust_testcase()

fprintf('<robust udu test>\n');

A = [1 1.2; 1.2 1];

x = [10; -5];
P = A*diag([0.1^2, 10.0^2])*A';
H = [1 -0.5; 0.1 5.0];
z = H*x - [0; 100]; % add outlier

B = [1.0 0.75; 0.2 5.0];
R = B*diag([0.25^2 1.5^2])*B';
dz = z - H*x;

[U, d] = udu(P);
chi2_threshold = 3.8415;
[x_robust_exp,U_exp,d_exp] = kalman_udu_robust(z,R,H,x,U,d,chi2_threshold,false);
P
H
x
z
R
x_robust_exp
U_exp'
d_exp

fprintf('</robust udu test>\n');

end

function [] = decorr_testcase()

fprintf('<decorrelation test>\n');

x = [15; -2.5; 0];
H = [1 -0.5 0.25; 0.1 5.0 -2];
z = H*x;

B = [1.0 0.75; 0.2 5.0];
R = B*diag([0.25^2 1.5^2])*B';

R
z
H

[G] = chol(R); % G'*G = R
zdecorr = (G')\z
Hdecorr = (G')\H

fprintf('</decorrelation test>\n');

end

function [] = kalman_takasu_robust()

fprintf('<kalman_takasu outlier test>\n');

R = eye(3) * 0.25;
dz = [0.2688, 0.9169, -100.1294 ]';
H = [8 1 6 1 ; 3 5 7 2; 4 9 2 3 ];
x = ones(4,1);
P = eye(4) * 0.04;
[x, P, chi2_takasu] = kalman_takasu(x, P, dz, R, H);
chi2_takasu

fprintf('</kalman_takasu outlier test>\n');

end


function [] = thornton_test()

    Q = diag([0.1 0.2]);

    G = [1 0; 0 1; 0.5 0.5];

    B = [1 0.1 -0.2; 0.1 1.1 0.2; 0 0.2 1.0];
    P = B*eye(3)*B';

    x = [1; 2; 3];

    Phi = [1 0.5 0.25; 0 1 0.1; 0 0 1];

    [U,d] = udu(P);

    [x_exp,U_exp,d_exp] = kalman_udu_predict(x,Phi,U,d,G,Q);

    x_ref = Phi*x;
    P_ref = Phi*P*Phi' + G*Q*G';
    [U_ref,d_ref] = udu(P_ref);
    assert(max(max(abs(U_ref-U_exp))) < 1e-10);

    x_exp
    U_exp
    d_exp

end

function [] = gnss_benchmark()

roll = deg2rad(10);
pitch = deg2rad(-20);
yaw = deg2rad(30);

sinr = sin(roll);
sinp = sin(pitch);
siny = sin(yaw);
cosr = cos(roll);
cosp = cos(pitch);
cosy = cos(yaw);
R_b_to_n = [ cosp*cosy  sinr*sinp*cosy-cosr*siny  cosr*sinp*cosy+sinr*siny;
             cosp*siny  sinr*sinp*siny+cosr*cosy  cosr*sinp*siny-sinr*cosy;
            -sinp       sinr*cosp                 cosr*cosp ];

F = zeros(15, 15);
F(1:3, 4:6) = eye(3);
f_n = [0.5;-0.3;-9.81];
x = f_n(1); y = f_n(2); z = f_n(3);

F_N = [ 0 -z  y ; ...
        z  0 -x ; ...
       -y  x  0 ];

F(4:6, 7:9) = F_N;
F(4:6, 10:12) = -R_b_to_n;
F(7:9, 13:15) = -R_b_to_n;

P = eye(15);
P(1:3, 1:3) = eye(3)*10.5^2;
P(4:6, 4:6) = eye(3)*0.2^2;
P(7:9, 7:9) = eye(3)*deg2rad(0.1)^2;
P(10:12, 10:12) = eye(3)*0.2^2;
P(13:15, 13:15) = eye(3)*deg2rad(5)^2;

G = zeros(15, 3);
G(1:3, 1:3) = eye(3);
Q = eye(3)*0.001;
Noise = G*Q*G';

dt_sec = 0.1;

Phi = eye(15) + F*dt_sec;

for i=1:100
    P = Phi*P*Phi';
end

H = zeros(3, 15);
H(1:3, 1:3) = eye(3);
l = [0; 0; 0]; % GNSS antenna leverarm
L = [   0   -l(3)  l(2) ; ...
      l(3)   0    -l(1) ; ...
     -l(2)   l(1)   0 ];
H(1:3, 7:9) = L;

A = [1 0.5 0.2; 0.1 1 0.1; 0.1 0.2 1];
R = eye(3)*1.5^2;
R = A*R*A';
x = zeros(15,1);
pos = [1024.0; 508.0; 20.0];
x(1:3) = pos;

mat2colmajor(A, "A", "%9.3f");
mat2colmajor(H', "Ht", "%9.3f");
mat2colmajor(P, "P", "%9.3f");
mat2colmajor(R, "R", "%9.3f");
mat2colmajor(x, "x", "%9.3f");
mat2colmajor(pos, "pos", "%9.3f");

Phi = eye(15);
mat2colmajor(Phi, "Phi", "%9.6f");
mat2colmajor(G, "G", "%9.6f");
mat2colmajor(Q, "Q", "%9.6f");

end

function [] = predict_test()

    Q = diag([0.1 0.2]);

    G = [1 0; 0 1; 0.5 0.5];

    B = [1 0.1 -0.2; 0.1 1.1 0.2; 0 0.2 1.0];
    P = B*eye(3)*B';

    x = [1; 2; 3];

    Phi = [1 0.5 0.25; 0 1 0.1; 0 0 1];

    x_exp = Phi*x;
    P_exp = Phi*P*Phi' + G*Q*G';

    mat2colmajor(P, "P", "%9.6f");

    mat2colmajor(x_exp, "x_exp", "%9.6f");
    mat2colmajor(P_exp, "P_exp", "%9.6f");

end

function [] = mat2colmajor(M, name, format)

[n,m] = size(M);
fprintf('float %s[%i*%i] = {\n', name, n, m);
for j=1:m % for each col
   for i=1:n %for each row
       fprintf(format, M(i, j));
       fprintf('f');
       if (j==m && i==n) == false
           fprintf(", ");
       end
   end
   fprintf("\n");
end
fprintf('};\n');


end
