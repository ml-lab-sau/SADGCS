function weightedF1 = WeightedF1(prdtn, tru_Labels)
    % Check input dimensions
    [N, C] = size(prdtn);
    if size(tru_Labels, 1) ~= N || size(tru_Labels, 2) ~= C
        error('The dimensions of prdtn and tru_Labels must be the same.');
    end
    
    % Initialize arrays to store precision, recall, and F1 scores
    precision = zeros(C, 1);
    recall = zeros(C, 1);
    f1 = zeros(C, 1);
    numTrueInstances = sum(tru_Labels == 1, 1); % Number of true instances for each class
    
    % Compute TP, FP, TN, FN for each class
    [TP, FP, TN, FN] = TpFp_TnFnMultiClass(prdtn, tru_Labels);
    
    % Compute precision, recall, and F1 score for each class
    for i = 1:C
        if TP(i) + FP(i) > 0
            precision(i) = TP(i) / (TP(i) + FP(i));
        else
            precision(i) = 0;
        end
        
        if TP(i) + FN(i) > 0
            recall(i) = TP(i) / (TP(i) + FN(i));
        else
            recall(i) = 0;
        end
        
        if precision(i) + recall(i) > 0
            f1(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
        else
            f1(i) = 0;
        end
    end
   
    % Calculate the weighted F1 score
    if sum(numTrueInstances) > 0
        weightedF1 = (numTrueInstances * f1) / sum(numTrueInstances);
    else
        weightedF1 = 0;
    
    end  
    % Display the weighted F1 score
    fprintf('Weighted F1 Score: %.4f\n', weightedF1);
end

function [TP, FP, TN, FN] = TpFp_TnFnMultiClass(prdtn, Y)
    [N, C] = size(prdtn);
    
    TP = zeros(C, 1);
    FP = zeros(C, 1);
    TN = zeros(C, 1);
    FN = zeros(C, 1);
    
    for i = 1:C
        TP(i) = sum((Y(:, i) == 1) & (prdtn(:, i) == 1));
        FP(i) = sum((Y(:, i) == -1) & (prdtn(:, i) == 1));
        TN(i) = sum((Y(:, i) == -1) & (prdtn(:, i) == -1));
        FN(i) = sum((Y(:, i) == 1) & (prdtn(:, i) == -1));
    end
end
