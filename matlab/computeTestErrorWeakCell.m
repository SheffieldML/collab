function [L2_error,NMAE_error,NMAE_round_error] = computeTestErrorWeakCell(model,Y,Ytest)
% 
% COMPUTETESTERRORWEAKCELL Compute the weak test error for data stored in a cell array.
% FORMAT
% DESC computes the test error for the weak generalization.
% ARG model : the model.
% ARG Y : the training data.
% ARG Ytest : the test data.
% RETURN L2_error : the l2 error.
% RETURN NMAE_error : the NMAE error.
%
% 
% COPYRIGHT : Raquel Urtasun, 2009

% COLLAB
 
val_L2 = 0;
tot_L2 = 0;
val_NMAE = 0;
tot_NMAE = 0;
val_round_NMAE = 0;
tot_round_NMAE = 0;
accum = [];



for i = 1:size(Y, 1)   
  ind = Ytest{i,1};
  elim = find(ind>size(model.X, 1));
  tind = ind;
  tind(elim) = [];
  
  if (length(ind)<1)
    disp(['No test data for ',num2str(i),]);
    continue;
  end
  [mu, varsig] = collabPosteriorMeanVarCell(model, Y{i,1}, double(Y{i,2}), model.X(tind, :));
  % normalize the values
  
  a = double(Ytest{i,2}) - mu; 
  %a = [a; Ytest(elim, i)];
  val_L2 = val_L2 + a'*a;
  tot_L2 = tot_L2 + length(a);
  val_NMAE = val_NMAE + sum(abs(a));
  tot_NMAE = tot_NMAE + length(a);
  val_round_NMAE = val_round_NMAE + sum(abs(round(a)));
  tot_round_NMAE = tot_round_NMAE + length(a);
  %accum = [accum; abs(a)];
end
L2_error = sqrt(val_L2/tot_L2);
NMAE_error = (val_NMAE/tot_NMAE)/1.6;
NMAE_round_error = (val_round_NMAE/tot_round_NMAE)/1.6;


