function [U, d] = udu(M)
% udu - Perform matrix decomposition to find upper triangular matrix U and
% diagonal matrix D.
%
% Jan Zwiener (jan@zwiener.org)
%
%  This function performs a specific matrix decomposition of a given square
%  matrix M into an upper triangular matrix U and a diagonal vector d such that
%  M = U * diag(d) * U'.
%
% Inputs:
%   M - A square, positive semidefinite matrix (m x m). This is the input
%   matrix that needs to be decomposed.
%
% Outputs:
%   U - An unit upper triangular matrix (m x m).
%   d - Diagonal vector (m x 1). matrix D = diag(d) so that: U*D*U' = M
%
%  The algorithm iterates from the bottom-right to the top-left of the matrix,
%  computing elements of U and d. This approach is sometimes referred to as a
%  variant of the Cholesky or LU decomposition with modifications for specific
%  applications.
%
% References:
%   1. Grewal, Weill, Andrews. "Global positioning systems, inertial
%      navigation, and integration". 1st ed. John Wiley & Sons, New York, 2001.
%   2. Golub, Gene H., and Charles F. Van Loan. "Matrix Computations." 3rd ed.,
%      Johns Hopkins University Press, 1996.
%   3. Higham, Nicholas J. "Accuracy and Stability of Numerical Algorithms."
%      2nd ed., SIAM, 2002.
%
% Example usage:
%   M = [4, 12, -16; 12, 37, -43; -16, -43, 98];
%   [U, d] = udu(M);

    [m, ~] = size(M);
    assert(size(M,2)==m);

    U = zeros(m, m); % Initialize U as a zero matrix
    d = zeros(m, 1); % Initialize d as a zero vector

    for j = m:-1:1
        for i = j:-1:1
            sigma = M(i, j);
            for k = (j+1):m
                sigma = sigma - U(i, k) * d(k) * U(j, k);
            end
            if i == j
                d(j) = sigma;
                U(j, j) = 1; % U is a unit triangular matrix
            else
                assert(d(j) > 0);
                U(i, j) = sigma / d(j); % off-diagonal elements of U
            end
        end
    end

end
