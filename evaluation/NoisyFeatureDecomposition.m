function W = NoisyFeatureDecomposition(X, Y, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, Iterations, Threshold)
    % Input: 
    %   X - Training dataset (n x d)
    %   Y - Labels (n x l)
    %   lambda1, lambda2, lambda3, lambda4, lambda5, lambda6 - Non-negative trade-off parameters
    %   Iterations - Maximum number of iterations
    %   Threshold - Convergence threshold
[n,d]=size(X);
[~,c]=size(Y);
    % Initialization
    XTX=X'*X;
    XTY=X'*Y;
    W = (XTX + eye(d)) \ (XTY);
    %rand(size(X, 2), size(Y, 2));
    D = rand(size(X, 1), size(X, 2));
    N = X - D;
    F = rand(size(X, 1), size(Y, 2));
    K=10;
U = zeros(n);
large_value = 1e10;  % Choose a suitable large value
alpha=0.1;
% Set U values based on labels
for i = 1:n
    if any(Y(i, :) ~= 0)
        U(i,i) = large_value;
    else
        U(i,i) = 0;
    end
end
W_prev = W;
E=ones(n,c);
%A = computeAffinityMatrix(X, K);
sigma=1;
L=computeLaplacian(X,sigma);
%L=computeSampleLaplacian(A);
tau=0.1;
    % Iterative optimization
    for t = 1:Iterations
        % Update W
        gradient_W = computeGradient_W(D, E, F, W, lambda1, lambda3, lambda6, L);
        W = W - alpha * gradient_W;  % Use an appropriate step size (alpha)

        % Update D
        gradient_D = computeGradient_D(E, F, W, D, N, X, lambda3, lambda4, lambda6, L);
        D = D - alpha * gradient_D;  % Use an appropriate step size (alpha)

        % Update N
        gradient_N = computeGradient_N(N, X, D, lambda2, lambda5, tau);
        N = N - alpha * gradient_N;  % Use an appropriate step size (alpha)

        % Update F
        gradient_F = computeGradient_F(D, E, F, W, Y, lambda5, U);
        F = F - alpha * gradient_F;  % Use an appropriate step size (alpha)

        % Check for convergence
        if norm(W - W_prev, 'fro') / norm(W, 'fro') < Threshold
            break;
        end
        
        % Save the current W for the next iteration
        W_prev = W;
    end
end


function gradient_W = computeGradient_W(D, E, F, W, lambda1, lambda3, lambda6, L)
    abs_term = max(0, E - F .* (D * W));
    gradient_W = D' * (abs_term .* (-F)) + lambda1 * W + lambda3 * D' * L * (D * W) + lambda3 * D' * (L' * (D * W)) + 2 * lambda6 * D' * D * W;
end

function gradient_D = computeGradient_D(E, F, W, D, N, X, lambda3, lambda4, lambda6, L)
    abs_term = max(0, E - F .* (D * W));
    gradient_D = (abs_term .* (-F)) * W' + lambda3 *(L + L')* D *W * W' - lambda4 * (X - D - N) + 2 * lambda6 * D * W * W';
end

function gradient_N = computeGradient_N(N, X, D, lambda2, lambda5, tau)
    prox_N = prox_nuclear(N, tau * lambda2);
    gradient_N = lambda2 * prox_N - lambda5 * (X - D - N);
end

function gradient_F = computeGradient_F(D, E, F, W, Y, lambda5, U)
   abs_term = max(0, E - F .* (D * W));
    gradient_F = (-D * W) .* (abs_term) + 2 * lambda5 * U * (F - Y);
    
end

function prox_value = prox_nuclear(M, tau)
       if any(isnan(M(:))) || any(isinf(M(:)))
        % Handle NaN or Inf values
        % Replace or remove problematic values based on your specific scenario
        % For example, you can replace them with zeros
        M(isnan(M) | isinf(M)) = 0;
    end
    [U, S, V] = svd(M, 'econ');
    S_soft = soft_threshold(S, tau);
    prox_value = U * S_soft * V';
end
function A = computeAffinityMatrix(X, K)
    % Input:
    %   X - Data matrix with samples in rows (n x d)
    %   K - Number of nearest neighbors for affinity
    
    % Compute pairwise Euclidean distances
    distances = pdist2(X, X);

    % Find K-nearest neighbors for each sample
    [~, sorted_indices] = sort(distances, 2);
    neighbors_indices = sorted_indices(:, 2:(K + 1)); % Exclude the sample itself

    % Construct the affinity matrix based on the K-nearest neighbors
    A = zeros(size(X, 1), size(X, 1));
    for i = 1:size(X, 1)
        A(i, neighbors_indices(i, :)) = 1;
    end

    % Make the affinity matrix symmetric
    A = max(A, A');

    % If desired, you can further modify the affinity values (e.g., Gaussian kernel)
    % Uncomment and modify the next line accordingly
    % A = exp(-(distances.^2) / (2 * sigma^2));
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


function S_soft = soft_threshold(S, tau)
    S_soft = max(0, S - tau) .* sign(S);
end

function L = computeLaplacian(X, sigma)
    % Step 1: Construct the similarity matrix S
    n = size(X, 1);
    S = zeros(n, n);
    for i = 1:n
        for j = 1:n
            S(i, j) = exp(-norm(X(i, :) - X(j, :))^2 / (2 * sigma^2));
        end
    end

    % Step 2: Compute the diagonal matrix Lambda
    Lambda = diag(sum(S, 2));

    % Step 3: Calculate the Laplacian matrix L
    L = Lambda - S;
end


