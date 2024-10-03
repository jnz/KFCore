function [ q2 ] = attitude_rotationrate_update( q, omega, dt_sec )
% Rotate a quaternion q for dt_sec with the 3x1 rotation rate vector omega (rad/s).
% Source: Wendel, J. Integrierte Navigationssysteme.
%         Oldenbourg Wissenschaftsverlag, 2009. (page 47)
%
% Jan Zwiener (jan@zwiener.org)
%
% Inputs:
%   q      - 4x1 quaternion at epoch k (Hamilton, real part at q(1))
%   omega  - 3x1 constant rotation rate between epoch k and k+1 [rad/s]
%   dt_sec - time interval between epoch k and k+1 [sec].
%
% Output:
%   q2     - 4x1 quaternion q at epoch k+1.
%
assert(( (nargin == 3) && (length(q) == 4) && ...
         (length(omega) == 3) && (length(dt_sec) == 1) ), 'Invalid arguments');

delta_angle_rad = [omega(1); omega(2); omega(3)] * dt_sec; % make sure omega is a 3x1 column vector
x = sqrt(delta_angle_rad(1)^2 + delta_angle_rad(2)^2 + delta_angle_rad(3)^2);

if ( x > 2000*pi/180 ) % Taylor series below is tested for up to 2000 deg/s with double precision
    qr = [ cos(x/2); (delta_angle_rad / x) * sin(x/2) ];
else
    % Taylor series expansion to avoid division by zero for x=0: (source below)
    % MATLAB Source for Taylor series expansion (toolbox required):
    %   syms x
    %   sympref('PolynomialDisplayStyle','ascend');
    %   f = (1/x)*sin(x/2);
    %   Tsin = taylor(f, x, 'Order', 16)
    %   f2 = cos(x/2);
    %   Tcos = taylor(f2, x, 'Order', 16)

    x2 = x*x; x4 = x2*x2; x6 = x4*x2; x8 = x4*x4; x10 = x6*x4;
    x12 = x6*x6; x14 = x8*x6; x16 = x8*x8; x18 = x10*x8;
    qr = [1 - x2/8 + x4/384 - x6/46080 + x8/10321920 - x10/3715891200 + x12/1961990553600 ...
            - x14/1428329123020800 + x16/1371195958099968000 - x18/1678343852714360832000; ...
          delta_angle_rad * (1/2 - x2/48 + x4/3840 - x6/645120 + x8/185794560 ...
                                 - x10/81749606400 + x12/51011754393600 ...
                                 - x14/42849873690624000 + x16/46620662575398912000 ...
                                 - x18/63777066403145711616000)];
end

% Equation: qnew = q * qr
% with qr = [ cos(x/2); delta_angle_rad/x * sin(x/2) ];
q2 = [ q(1) -q(2) -q(3) -q(4) ; ...
       q(2)  q(1) -q(4)  q(3) ; ...
       q(3)  q(4)  q(1) -q(2) ; ...
       q(4) -q(3)  q(2)  q(1) ]*qr;

end
