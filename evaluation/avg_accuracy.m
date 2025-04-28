function [avg_accuracy] = avg_accuracy(pred, target)
    avg_accuracy = 0;
    [c, ~] = size(target); % Number of classes
    for i = 1:c
        y = pred(i, :); % Extract predictions for the i-th class
        t = target(i, :); % Extract labels for the i-th class
        % Compute accuracy
        accuracy_tmp = sum(y == t) / length(t);
        avg_accuracy = avg_accuracy + accuracy_tmp / c;
    end
end
