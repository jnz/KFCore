function [healthy] = filterhealthy(x, P, chatty)
%FILTERHEALTHY Check if there are issues in the current filter health.
%
% Jan Zwiener (jan@zwiener.org)
%
% @param[in] x - State vector (n x 1)
% @param[in] P - Covariance matrix of the state vector (n x n)
% @param[in] chatty - If set to true, emit debug messages via fprintf.
% @return True if everything seems to be OK

assert(length(x) == size(P,1));
assert(size(P,2) == size(P,1));
if nargin < 3
    chatty = false;
end

healthy = true;

if all(all(isfinite(P))) ~= true
    healthy = false;
    if chatty
        fprintf('Matrix P is not finite\n');
    end
end

if all(isfinite(x)) ~= true
    healthy = false;
    if chatty
        fprintf('Vector x is not finite\n');
    end
end

smallest_entry = min(diag(P));
if (smallest_entry <= eps)
    healthy = false;
    if chatty
        fprintf('Diagonal elements of P close or below zero\n');
    end
end

lower_tri = tril(P, -1);
if all(lower_tri(:) == 0)
    if chatty
        fprintf('Matrix P lower triangular part is zero\n');
    end
    P = P + triu(P,1)';
    try
        chol(P);
    catch
        healthy = false;
        if chatty
            fprintf('Matrix P is not positive definite\n');
        end
    end
else
    if issymmetric(P) == false
        healthy = false;
        if chatty
            fprintf('Matrix P is not symmetric\n');
        end
    end
    try
        chol(P);
    catch
        healthy = false;
        if chatty
            fprintf('Matrix P is not positive definite\n');
        end
    end
end


end

