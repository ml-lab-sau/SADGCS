function[W_rest, F_list]=model(train_data, train_target, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6,Q)
% m 标签数量
[n, m]=size(train_target);
% d 特征维度
[~, d]=size(train_data);
Ut=zeros(n);
large_value = 1e10;  % Choose a suitable large value
% Set U values based on labels
%Y=train_target;
for i = 1:n
    if any(train_target(i, :) ~= 0)
        Ut(i,i) = large_value;
    else
        Ut(i,i) = 0;
    end
end

% 初始化W, D, N
XTX = train_data' * train_data;
XTY = train_data' * train_target;
Wt = (XTX + 0.01*eye(d)) \ (XTY);
Ft=rand(n,m);
Dt = rand(n, d);
Nt = train_data - Dt;

% 计算标签相关性       ---要改
% Rx = corr(train_target);

Rx = zeros(n, n);
sigma = 1;
%mix = train_target';
mix=train_data;
for i = 1 : n
    for j = 1 : n
        norm2 = norm(mix(i,:) - mix(j,:));
        Rx(i, j) = exp(-1 *  norm2 * norm2 / (2 * sigma * sigma));
    end
end

% 余弦相似度
% Rx = squareform(1-pdist(train_target','cosine')) + eye(size(train_target',1));


% 计算样本相关性
% 只计算特征相似度，不在迭代中更新
Rl = zeros(n, n);
% mix = [train_data, train_target];
% mix = train_data;
% mix = train_target;
% for i = 1 : n
%     for j = 1 : n
%         norm2 = norm(mix(i, :) - mix(j, :));
%         Rl(i, j) = exp(-1 *  norm2 * norm2 / (2 * sigma * sigma));
%     end
% end


% FunctionValue存储
F_list = zeros(2, 100);



% 迭代
Flag = true;
iteration = 1;
while Flag && iteration <= 25
% while Flag && iteration <= 100

    [Wtplus, Dtplus, Ntplus,Ftplus] = opt(train_data, train_target, Rx, Rl, Wt, Dt, Nt,Ft,Ut,lambda1, lambda2, lamb
%     [Wtplus, Dtplus, Ntplus, Fvalue] = opt(train_data, train_target, Rx, Rl, Wt, Dt, Nt, lambda1, lambda2, lambda3da3, lambda4, lambda5,lambda6);
    
    % 收敛性分析, lambda4, lambda5);
%     F_list(1, iteration) = iteration;
%     F_list(2, iteration) = Fvalue;
%     if norm(Wtplus - Wt, 'fro')/norm(Wt, 'fro') < 0


    if norm(Wtplus - Wt, 'fro')/norm(Wt, 'fro') < 10^-5
        Flag=false;
    else
        iteration = iteration + 1; 
        Wt = Wtplus;
        Dt = Dtplus;
        Nt = Ntplus;
        Ft = Ftplus;
    end
%     fprintf("iteration ---- %d\r", iteration);
end
row_sums_squared = sum(Wt.^2, 2);
% Sort the rows of W based on the sum of squares
[sorted_sums_squared, sorted_indices] = sort(row_sums_squared, 'descend');

% Select the top 100 rows
top_100_indices = sorted_indices(1:Q);

% Introduce zeros in the respective indices for the rest of the rows
W_rest = W;
W_rest(setdiff(1:D, top_100_indices), :) = 0;
end