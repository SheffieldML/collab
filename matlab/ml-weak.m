% variables: strong{train,test}, weak{train,test}
load 'ml-split.mat'
clear strongtrain strongtest;
regvals = sqrt(sqrt(10)).^[8 7.5 7 6.5 6 5.5 5 4.5 4 3.5 3];
objgrad = @m3fshc_norm;
tol = 1e-3;
[n,m] = size(weaktrain{1});
p = 500;
l = 5;
i = 3;
maxiter = 100;
fprintf('p=%d maxiter=%d i=%d\n',p,maxiter,i);
fn = sprintf('../result/WEAK_r%d_c%d_p%d_x%d_i%d',n,m,p,maxiter,i);
v = randn(n*p+m*p+n*(l-1),1);
for i3=1:length(regvals)
  fprintf(1,'Begin conjgrad: regval=%.1e\n',regvals(i3));
  [v] = conjgrad(v,@cgLineSearch,{'c2',1e-2},objgrad,{weaktrain{i},regvals(i3),l,'verbose',0},'tol',tol,'maxiter',maxiter,'verbose',2);
  U = reshape(v(1:n*p),n,p);
  V = reshape(v(n*p+1:n*p+m*p),m,p);
  theta = reshape(v(n*p+m*p+1:n*p+m*p+n*(l-1)),n,l-1);
  [U,V] = normCols(U,V);  
  X = U*V';
  [y] = m3fSoftmax(X,theta);
  Xrank = rank(X);
  clear U V theta X;
  fprintf(1,'%d %s xi=%d p=%d tol=%.0e rank=%d %.2e ZOE: %.2f %.4f  MAE: %.2f %.4f\n',i,func2str(objgrad),maxiter,p,tol,Xrank,regvals(i3),zoe(y,weaktrain{i}),zoe(y,weaktest{i}),mae(y,weaktrain{i}),mae(y,weaktest{i}));
  fh = fopen(fn,'a');
  fprintf(fh,'%d %s xi=%d p=%d tol=%.0e rank=%d %.2e ZOE: %.2f %.4f  MAE: %.2f %.4f\n',i,func2str(objgrad),maxiter,p,tol,Xrank,regvals(i3),zoe(y,weaktrain{i}),zoe(y,weaktest{i}),mae(y,weaktrain{i}),mae(y,weaktest{i}));
  fclose(fh);
end
