function labels = assignLabelsToHighestValueColumnwise(matrix)
    % Assigns 1 to the highest value and -1 to the rest in each column of the matrix
    
    % Input:
    % matrix: Matrix of numerical values
    
    % Output:
    % labels: Matrix of labels (1 for highest value, -1 for rest)
    
    [~, maxIndices] = max(matrix); % Find indices of maximum values in each column
    [m, n] = size(matrix);
    labels = -ones(m, n); % Initialize labels with -1
    
    linearIndices = sub2ind([m, n], maxIndices, (1:n)); % Convert subscripts to linear indices
    labels(linearIndices) = 1; % Assign 1 to the maximum values
end
