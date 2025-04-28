clear;
clc;
addpath(genpath('.'));

% Parameters
cv_num = 5;
% lambda1   = 1;
% lambda2   = 0.01;
% lambda3   = 0.0001;
% lambda4   = 10;
% lambda6=100;
% lambda5   = 0.1;
% lambda1 = 1;%0.21; 
% lambda2 = 1.64; 
% lambda3 = 0.59; 
% lambda4 = 0.27; 
% lambda5 = 0.5; 
% lambda6 = 35;
lambda1 = 1; 
lambda2 = 1; 
lambda3 = 100; 
lambda4 = 10; 
lambda5 = 1; 
lambda6 = 0.01; 

percentages = [0.9, 0.7, 0.5];

% List of dataset filenames
%dataset_files = {'brain_can.mat'};%, 'chowdary.mat', 'gravier.mat', 'alon.mat', 'procancer.mat', 'sun.mat', 'ovary_can.mat', 'endocrinecancer.mat', 'breast_can.mat', 'nakayam.mat', 'ship.mat', 'bur.mat'};
% dataset_files = {'dna.mat','ecoli.mat','satimage.mat','YaleB.mat','Yale1.mat','segment.mat'};
%dataset_files = {'procancer.mat','breast_can.mat','alon.mat'};
dataset_files={'breast_can.mat'};

% Initialize results storage
results = cell(length(dataset_files) * length(percentages) * 2, 8);
result_idx = 1;

for dataset_index = 1:length(dataset_files)
    dataset_file = dataset_files{dataset_index};

    fprintf('Processing dataset: %s\n', dataset_file);

    load(dataset_file);
    target = target';

    if exist('train_data', 'var') == 1
        data = [train_data; test_data];
        target = [train_target, test_target];
    end
    clear train_data test_data train_target test_target

    target(target == 0) = -1;
    target = double(target);
    data = double(data);

    num_data = size(data, 1);
    temp_data = data + eps;
    temp_data = temp_data ./ repmat(sqrt(sum(temp_data.^2, 2)), 1, size(temp_data, 2));
    if sum(sum(isnan(temp_data))) > 0
        temp_data = data + eps;
        temp_data = temp_data ./ repmat(sqrt(sum(temp_data.^2, 2)), 1, size(temp_data, 2));
    end
    temp_data = [temp_data, ones(num_data, 1)];
    randorder = randperm(num_data);

    for per = percentages
        fprintf('Processing percentage: %.2f\n', per);

        cvResult = zeros(4, cv_num);

        for j = 1:cv_num
            tic;
            fprintf('- Cross Validation - %d/%d \n', j, cv_num);

            [cv_train_data, cv_train_target, cv_test_data, cv_test_target] = generateCVSet(temp_data, target', randorder, j, cv_num);
            IncompleteTarget = mask_target_entries(cv_train_target, per);

            [out, ~] = model(cv_train_data, IncompleteTarget, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6);
            Outputs = (cv_test_data * out)';

            Pre_Labels = assignLabelsToHighestValue(Outputs);
            time = toc;

            tmpResult(1, 1) = time;
            tmpResult(2, 1) = Average_precision(Outputs, cv_test_target');
            tmpResult(3, 1) = WeightedF1(Pre_Labels', cv_test_target);
            tmpResult(4, 1) = avgauc(Outputs, cv_test_target');
            cvResult(:, j) = tmpResult;
        end

        Avg_Result = mean(cvResult, 2);
        Std_Result = std(cvResult, 0, 2);

        % Store average results in cell array
        results{result_idx, 1} = dataset_file;
        results{result_idx, 2} = per;
        results{result_idx, 3} = 'Average';
        results{result_idx, 4} = Avg_Result(1); % Average Time
        results{result_idx, 5} = Avg_Result(2); % Average Precision
        results{result_idx, 6} = Avg_Result(3); % Weighted F1 Score
        results{result_idx, 7} = Avg_Result(4); % AUC
        results{result_idx, 8} = 'Std Dev';
        
        result_idx = result_idx + 1;
        
        results{result_idx, 1} = dataset_file;
        results{result_idx, 2} = per;
        results{result_idx, 3} = 'Std Dev';
        results{result_idx, 4} = Std_Result(1); % Std Time
        results{result_idx, 5} = Std_Result(2); % Std Precision
        results{result_idx, 6} = Std_Result(3); % Std Weighted F1 Score
        results{result_idx, 7} = Std_Result(4); % Std AUC
        
        result_idx = result_idx + 1;

        % Print results
        fprintf('Time                  %.4f  %.4f\n', Avg_Result(1), Std_Result(1));
        fprintf('Average Precision     %.4f  %.4f\n', Avg_Result(2), Std_Result(2));
        fprintf('Weighted F1 Score     %.4f  %.4f\n', Avg_Result(3), Std_Result(3));
        fprintf('AUC                   %.4f  %.4f\n', Avg_Result(4), Std_Result(4));
    end
end

% Write results to Excel
resultsTable = cell2table(results, 'VariableNames', {'Dataset', 'Percentage', 'Measure', 'Time', 'Average_Precision', 'Weighted_F1_Score', 'AUC', 'Std_Dev'});
writetable(resultsTable, 'WeightedF1Scores.xlsx');
fprintf('Results written to %s\n', 'WeightedF1Scores.xlsx');
