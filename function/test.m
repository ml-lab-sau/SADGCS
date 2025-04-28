clear;
clc;
addpath(genpath('.'));

%mis = 7;
per=0.9;

load('procancer.mat');
target=target';
rng(42);

% ²ÎÊý
cv_num = 5;

% lambda1   = 1;
% lambda2   = 0.01;
% lambda3   = 0.01;
% lambda4   = 0.0001;
% lambda5   = 10;
rng(42);
lambda1=1; lambda2=1e-2; lambda3=1e-2; lambda4=10; lambda5=1; lambda6=0.1; %Iterations=5; Threshold=1e-2;
if exist('train_data','var')==1
    data    = [train_data;test_data];
    target  = [train_target,test_target];  
end
clear train_data test_data train_target test_target
target(target == 0) = -1;

target      = double (target);
data      = double (data);

num_data  = size(data,1);
temp_data = data + eps;
temp_data = temp_data./repmat(sqrt(sum(temp_data.^2,2)),1,size(temp_data,2));
if sum(sum(isnan(temp_data)))>0
    temp_data = data+eps;
    temp_data = temp_data./repmat(sqrt(sum(temp_data.^2,2)),1,size(temp_data,2));
end
temp_data = [temp_data,ones(num_data,1)];


randorder = randperm(num_data);

cvResult  = zeros(4, cv_num);


for j = 1:5
    
    tic;
    fprintf('- Cross Validation - %d/%d \r', j, cv_num);
    [cv_train_data,cv_train_target,cv_test_data,cv_test_target] = generateCVSet( temp_data,target',randorder,j,cv_num );

    %IncompleteTarget = getIncompleteTarget_lsml(cv_train_target, mis * 0.1, 1);
   IncompleteTarget = mask_target_entries(cv_train_target, per);
   %out = NoisyFeatureDecomposition(cv_train_data,IncompleteTarget,lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, Iterations, Threshold);
     
   [out, ~] = model(cv_train_data, IncompleteTarget, lambda1, lambda2, lambda3, lambda4, lambda5,lambda6);
    Outputs = (cv_test_data * out)';
    
     thr = TuneThreshold(Outputs, cv_test_target');
     %Pre_Labels = Predict(Outputs,thr);
     %Pre_Labels=sign(Outputs);
     Pre_Labels=assignLabelsToHighestValue(Outputs);
     Outputs=Pre_Labels;
    time=toc;
    tmpResult(1, 1) = Hamming_loss(Outputs, cv_test_target');
    tmpResult(2, 1) = Average_precision(Outputs, cv_test_target');
   % tmpResult(3, 1) = coverage(Outputs, cv_test_target');
   % tmpResult(4, 1) = One_error(Outputs, cv_test_target');
  %  tmpResult(5, 1) = Ranking_loss(Outputs, cv_test_target');
   % tmpResult(3, 1) = avgauc(Outputs,cv_test_target');
   tmpResult(3, 1) = mlauc(Outputs',cv_test_target);
    tmpResult(4, 1)=time;

    cvResult(:, j) = cvResult(:, j) + tmpResult;
    
end

Avg_Result      = zeros(4, 2);
Avg_Result(:, 1) = mean(cvResult, 2);
Avg_Result(:, 2) = std(cvResult, 1, 2);
% Calculate average AUC across cross-validation folds
auc_values = cvResult(3, :);
%disp(['AUC values: ', num2str(auc_values)]);

% Filter out NaN values
valid_auc_values = auc_values(~isnan(auc_values));
%disp(['Valid AUC values: ', num2str(valid_auc_values)]);

if isempty(valid_auc_values)
    disp('Error: No valid AUC values to compute average');
else
    % Compute average and standard deviation
    Avg_Result(3, 1) = mean(valid_auc_values);
    Avg_Result(3, 2) = std(valid_auc_values);
end

%fprintf('AUC           %.4f  %.4f\r', Avg_Result(4,1), Avg_Result(4,2));
% fprintf('\nEvaluation Metric                 \n');
% fprintf('-----------------avg----------------\n');
 fprintf('HammingLoss           %.4f  %.4f\r', Avg_Result(1,1), Avg_Result(1,2));
 fprintf('Averge Prec           %.4f  %.4f\r', Avg_Result(2,1), Avg_Result(2,2));
% %fprintf('Coverage           %.4f  %.4f\r', Avg_Result(3,1), Avg_Result(3,2));
% %fprintf('One_error           %.4f  %.4f\r', Avg_Result(4,1), Avg_Result(4,2));
% %fprintf('Ranking_loss           %.4f  %.4f\r', Avg_Result(5,1), Avg_Result(5,2));
% fprintf('AUC           %.4f  %.4f\r', Avg_Result(3,1), Avg_Result(3,2));
% %fprintf('F1score           %.4f  %.4f\r', Avg_Result(7,1), Avg_Result(7,2));
fprintf('NewAUC          %.4f  %.4f\r', Avg_Result(3,1), Avg_Result(3,2));
 fprintf('Time           %.4f  %.4f\r', Avg_Result(4,1), Avg_Result(4,2));
 
%disp(Avg_Result);


fprintf('end.\n');
