function [f,g] = optd(x,Data)

[n, ~] = size(Data.Dt);
[d,m] = size(Data.Wt);
Wt = Data.Wt;
Dt = reshape(x, n, d);
Nt = Data.Nt;
Ft=Data.Ft;
Ut=Data.Ut;
%L1=Data.L1;
E = ones(size(Data.train_target));

HingeL = max((E - Ft .* (Dt * Wt)) , 0);
term1 = 1/2 * norm((HingeL.^2),1);

% term1 = 1/2 * norm(Data.train_target - Dt * Wt, 'fro')^2;

term2 = Data.lambda1 * 1/2 * norm(Wt, 'fro')^2;

term3 = Data.lambda2 * sum(abs(svd(Nt, 'econ')));
% term3 = Data.lambda2 * norm(Nt, 1);

L1 = diag(sum(Data.Rx, 2)) - Data.Rx;
% term4 = Data.lambda3 * trace(Wt * L1 * Wt');
term4 = Data.lambda3 * trace((Dt * Wt)' * L1 * (Dt * Wt));
term5 = Data.lambda4 * 1/2 * norm(Data.train_data - Dt - Nt, 'fro')^2;
term6=Data.lambda5*trace((Ft-Data.train_target)'*Ut*(Ft-Data.train_target));
term7=Data.lambda6*trace((Dt*Wt)'*(Dt*Wt)-eye(m));
% L2 = diag(sum(Data.Rl, 2)) - Data.Rl;
% term5 = Data.lambda4 * trace(Dt' * L2 * Dt);

%term5 = Data.lambda4 * 1/2 * norm(Dt - Data.train_target * Wt', 'fro')^2;

%term7 = Data.lambda5 * 1/2 * norm(Data.train_data - Dt - Nt, 'fro')^2;

f = term1 + term2 + term3 + term4 + term5 + term6 + term7;


% Çóµ¼
gterm1 =  (HingeL .* (-(Ft))) * Wt';

% gterm1 = -(Data.train_target - Dt * Wt) *  Wt';

%gterm4 = Data.lambda3 * (Dt * Wt) * (L1 + L1') * Wt';
gterm4=  Data.lambda3 *((L1 + L1')* Dt * Wt * Wt');

% gterm5 = Data.lambda4 * (L2 + L2') * Dt;

%gterm5 = Data.lambda4 * (Dt - Data.train_target * Wt');

gterm5 = -Data.lambda4 * (Data.train_data - Dt - Nt);

gterm7=Data.lambda6 * 2 * Dt * Wt * Wt';

g = gterm1 + gterm4 + gterm5 + gterm7;

% g = gterm1 + gterm5 + gterm7;

g=reshape(g, n*d, 1);

end