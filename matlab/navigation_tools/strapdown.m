function [ qk1, vel_n_k1, dpos_n, dvdt_n ] = strapdown( dt_sec, specific_force_b, omega_b_ib, qk, vel_n_k, latitude_rad, height_m, fastmath)
% Strapdown algorithm. Integrate from epoch k (k0) to k+1 (k1)
%
% Jan Zwiener (jan@zwiener.org)
%
% n-frame = NED = North/East/Down system for X/Y/Z components.
%
% Inputs:
%   dt_sec           - Time interval between epoch k and k+1 [s]
%   specific_force_b - 3x1 specific force measurement in epoch k in body frame
%                      (North/East/Down) [m/s^2]
%   omega_b_ib       - 3x1 rotation rate [rad/s] of body relative to n-frame,
%                      measurement in epoch k in body frame coordinates (NED)
%   qk               - 4x1 orientation quaternion (Hamilton, q(1) is scalar)
%                      to transform from body to n-frame (NED) in epoch k
%   vel_n_k          - 3x1 velocity of body relative to Earth in n-frame (NED)
%                      at epoch k [m/s]
%   latitude_rad     - latitude on ellipsoid. Must be less than |pi/2| as
%                      tan(latitude_rad) is calculated [rad]
%   height_m         - Altitude/height above ellipsoid [m]
%   fastmath         - true/false boolean. If true, ignore transport rate,
%                      coriolis force and Earth rotation rate
%
% Outputs:
%   qk1      - New 4x1 orientation quaternion for epoch k+1
%   vel_n_k1 - New 3x1 velocity for epoch k+1
%   dpos_n   - 3x1 Change (delta) in position (pos) between epoch k and k+1
%              Change in ECEF coordinates can be simply computed as:
%              dpos_ecef = R_n_to_e*dpos_n;
%   dvdt_n   - 3x1 Body acceleration if n-frame [m/s/s] (read as: delta
%              velocity over delta time in n-frame)

assert( ( (length(dt_sec) == 1) && ...
          (length(specific_force_b) == 3) && ...
          (size(specific_force_b, 1) == 3) && ...
          (length(omega_b_ib) == 3) && ...
          (size(omega_b_ib, 1) == 3) && ...
          (length(qk) == 4) && ...
          (norm(qk) > 0.99) && (norm(qk) < 1.01) && ...
          (length(vel_n_k) == 3) && ...
          (size(vel_n_k, 1) == 3) && ...
          (length(latitude_rad) == 1) && ...
          (length(height_m) == 1) && ...
          (latitude_rad > -pi/2) && ...
          (latitude_rad <  pi/2) && ...
          (islogical(fastmath)) && ...
          (nargin == 8) ), 'Invalid arguments');

% CONSTANTS
WGS84_a = 6378137.0; % WGS84 ellipsoid
WGS84_e_squared = 0.00669437999014; % WGS84 excentricity squared e*e (sqrt(f(2-f)))
WGS84_omega_rps = 7.2921151467E-5; % Earth rotation rate in rad/s
WGS84_gravity = 9.81; % Earth gravity in m/s/s
gravity_n = [0;0;WGS84_gravity]; % gravity (incl. centrifugal force) in n-frame

% 3x3 rotation matrix (at epoch k) from qk
R_b_to_n = [ qk(1)*qk(1)+qk(2)*qk(2)-qk(3)*qk(3)-qk(4)*qk(4),  2*(qk(2)*qk(3)-qk(1)*qk(4)),  2*(qk(2)*qk(4)+qk(1)*qk(3))   ; ...
             2*(qk(2)*qk(3)+qk(1)*qk(4)),  qk(1)*qk(1)-qk(2)*qk(2)+qk(3)*qk(3)-qk(4)*qk(4),  2*(qk(3)*qk(4)-qk(1)*qk(2))   ; ...
             2*(qk(2)*qk(4)-qk(1)*qk(3)),  2*(qk(3)*qk(4)+qk(1)*qk(2)),  qk(1)*qk(1)-qk(2)*qk(2)-qk(3)*qk(3)+qk(4)*qk(4) ] ;

if (fastmath == false)
    sin_lat = sin(latitude_rad);
    cos_lat = cos(latitude_rad);
    tan_lat = tan(latitude_rad);

    % Source 1: Jan Wendel, Integrierte Navigationssysteme Oldenburg Verlag p. 31 (1st edition)
    % Source 2: G. Seeber, Satellite Geodesy, de Gruyter, p. 25 (Eq. 2.41 u. 2.42)
    % denom = (1-WGS84_e_squared*sin_lat^2)^(1.5)
    denom = (1-WGS84_e_squared*sin_lat^2);
    Rn = WGS84_a*(1 - WGS84_e_squared)/sqrt(denom*denom*denom); % North/South radius
    Re = WGS84_a/sqrt(1-WGS84_e_squared*sin_lat^2); % East/West radius

    % the rotation rate required for the strapdown algorithm
    % is not exactly omega_b_ib, as the rotation rate
    % relative to the navigation frame (n-frame) is required (omega_b_nb),
    % not the rotation rate relative to the inertial frame (omega_b_ib).
    % compensate the earth rotation rate (15 deg/h) and the
    % transport rate of the n-frame (depends on the current velocity).

    % <begin computation of omega_b_nb>

    % Earth rotation rate in n-frame (get_omega_n_ie)
    omega_n_ie = [ WGS84_omega_rps*cos_lat; 0; -WGS84_omega_rps*sin_lat ];
    % Rotation rate of the n-frame (get_omega_n_en):
    % latlonh(3) is the height above the ellipsoid.
    % FIXME: make sure this works close to lat=90 deg.

    omega_n_en = [  vel_n_k(2)/(Re + height_m)  ;... % vel_n_k(2) = velocity in East direction
                   -vel_n_k(1)/(Rn + height_m)  ;... % vel_n_k(1) = velocity in North direction
                   -vel_n_k(2)*tan_lat/(Re + height_m) ];

    omega_n_in = omega_n_ie + omega_n_en;
    omega_b_nb = omega_b_ib - R_b_to_n'*omega_n_in; % omega_b_nb is required
    % <end computation of omega_b_nb>
else
    omega_n_ie = omega_b_ib*0;
    omega_n_en = omega_b_ib*0;
    omega_b_nb = omega_b_ib;
end

% Attitude
% --------

% update attitude quaternion from epoch k (qk) to epoch k+1 with the
% (assumed) constant rotation rate omega_b_nb (rad/s) over dt_sec (s):
qk1 = attitude_rotationrate_update(qk, omega_b_nb, dt_sec);

% Acceleration
% ------------

% Algorithm J. Wendel Strapdown chapter (start: p. 45).
% and Titterton chapter 11.3

% coriolis force in n-frame (m/s/s)
f_coriolis_n = -cross(2*omega_n_ie + omega_n_en, vel_n_k);

% Simply integrating the constant force over the time is possible but not
% completely correct, as the basic assumption is that there is also a constant
% rotation, so R_b_to_n is constantly changing over time.  Integrating over
% R_b_to_n(t)*specific_force_b is a bit tricky.  An approximation is to apply a
% correction term for the rotation of R_b_to_n over dt_sec. See Titterton 2nd
% ed. p. 326 and Wendel 2nd ed. chapter 3, p. 58.
% By including the following correction term, we can approx. correct for the
% constant rotation rate:
rotation_correction = cross(0.5*omega_b_nb, specific_force_b);

% basic differential equation (Wendel, 2nd ed. equation 3.151):
% dvdt_n is the acceleration of the body in n-frame (m/s/s). The centrifugal force
% is assumed to be included in the gravity_n vector:
dvdt_n = R_b_to_n*(specific_force_b + rotation_correction) + ...
         f_coriolis_n + gravity_n;

% Velocity change
% ---------------

% the velocity for the next epoch k+1:
vel_n_k1 = vel_n_k + dvdt_n*dt_sec;

% Position change
% ---------------

% Trapezoid integration, change in position in n-frame (meter) is:
dpos_n = 0.5*dt_sec*(vel_n_k + vel_n_k1);

% In ECEF coordinates:
% dpos_ecef = R_n_to_e*dpos_n;

end

