function[Wtplus, Dtplus, Ntplus,Ftplus] = opt(train_data, train_target, Rx, Ry, Wt, Dt, Nt,Ft,Ut,lambda1, lambda2, lambda3, lambda4, lambda5,lambda6)
Data.train_data = train_data;
Data.train_target = train_target;
Data.Rx = Rx;
Data.Ry = Ry;
Data.Wt = Wt;
Data.Dt = Dt;
Data.Nt = Nt;
Data.Ft = Ft;
Data.Ut = Ut;
Data.lambda1 = lambda1;
Data.lambda2 = lambda2;
Data.lambda3 = lambda3;
Data.lambda4 = lambda4;
Data.lambda5 = lambda5;
Data.lambda6 = lambda6;
[d, c] = size(Wt);
x0 = reshape(Wt, d*c, 1);
out=ncg(@(x) optw(x, Data), x0, 'MaxIters', 1, 'Display', 'off');
Wtplus = reshape(out.X, d, c);
Data.Wt = Wtplus;

[n, m] = size(Dt);
x1=reshape(Dt, m*n ,1);
out=ncg(@(x) optd(x, Data), x1, 'MaxIters', 1, 'Display', 'off');
Dtplus=reshape(out.X, n, m);
Data.Dt = Dtplus;

x2=reshape(Nt, m*n ,1);

out=ncg(@(x) optn(x, Data), x2, 'MaxIters', 1, 'Display', 'off');
Ntplus=reshape(out.X, n, m);

[n,c] = size(Ft);
x3=reshape(Ft,n*c,1);
out=ncg(@(x) optF(x, Data), x3, 'MaxIters', 1, 'Display', 'off');
Ftplus = reshape(out.X, n, c);

%Ftvalue = out.F;

end