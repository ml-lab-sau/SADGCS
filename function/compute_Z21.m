function Z = compute_Z21(Xl, Xu, Fl, Fu, L, lambda)
    % Initialize P as the identity matrix
    P = eye(size(Xl, 1));

    % Initialize convergence criteria
    max_iterations = 100;
    epsilon = 1e-6;
    prev_Z = zeros(size(Xl, 1), size(Fl, 2)); % Initial guess for Z

    for iter = 1:max_iterations
        % Step 1: Calculate N using Eq. (15)
        N = calculate_N(Xl, Xu, Fl, Fu, P);

        % Step 2: Compact SVD decomposition of N
        [U, S, V] = svd(N, 'econ');
        M = U * sqrt(S);
        N = sqrt(S) * V';

        % Step 3: Update Z
        Z = inv(Xl * Xl' + lambda * P) * M;

        % Step 4: Update P
        Z_norms = sqrt(sum(Z.^2, 1)) + epsilon;
        P = diag(1 ./ (2 * Z_norms));

        % Check for convergence
        if norm(Z - prev_Z, 'fro') < epsilon
            break;
        end
        prev_Z = Z;
    end
end

function N = calculate_N(Xl, Xu, Fl, Fu, P)
    % Concatenate labeled and unlabeled data
    X = [Xl, Xu];
    F = [Fl; Fu];

    % Calculate N using the definition in Eq. (15)
    E = X - F * Z;
    N = E * E';
end
