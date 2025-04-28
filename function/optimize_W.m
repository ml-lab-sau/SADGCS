
function W = optimize_W(X, Y, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, iterations, threshold, eta)
    % Initialize variables
    [n, d] = size(X);
    [~, c] = size(Y);
   % [~, c] = size(E);
    U1= zeros(n);
    E=ones(n,c);
    sigma=1;
    L=computeLaplacian(X,sigma);
    large_value = 1e10;  
    % Initialization
    W = rand(d, c);
    D = rand(n, d);
    N = X - D;
    %F = rand(n, c);
    F = ones(n, c);
for i = 1:n
    if any(Y(i, :) ~= 0)
        U1(i,i) = large_value;
    else
        U1(i,i) = 0;
    end
end
    % Main loop
    for t = 1:iterations
        % Update W
        gradient_W = - D' * (max(0, abs(E - F .* (D * W))) .* (-F)) + lambda1 * W + 2 * lambda6 * D' * D * W;
        W = W - eta * gradient_W;

        % Update D
        gradient_D = (max(0, abs(E - F .* (D * W))) .* (-F)) * W' - lambda4 * (X - D - N) + 2 * lambda6 * D * (W * W');
        D = D - eta * gradient_D;

        % Update N
        gradient_N = lambda2 * proximal_operator(N, lambda2 / eta) - lambda5 * (X - D - N);
        N = N - eta * gradient_N;

        % Update F
        gradient_F = lambda3 * L * F + lambda5 * U1 * (F - Y) * (W * W');
        F = F - eta * gradient_F;

        % Check convergence
        if norm(gradient_W, 'fro') / norm(W, 'fro') < threshold
            break;
        end
    end

    % Proximal operator for nuclear norm
    function result = proximal_operator(matrix, tau)
        [U, S, V] = svd(matrix);
        S = max(S - tau, 0);
        result = U * S * V';
    end
end
function Laplacian = computeSampleLaplacian(A)
    % Input:
    %   A - Affinity matrix (similarity matrix) between samples
    
    % Compute the degree matrix
    D = diag(sum(A, 2));

    % Compute the Laplacian matrix
    Laplacian = D - A;

    % If desired, you can use the normalized Laplacian
    % Uncomment the next line if you want the normalized Laplacian
    % Laplacian = eye(size(A, 1)) - D^(-0.5) * A * D^(-0.5);
end
