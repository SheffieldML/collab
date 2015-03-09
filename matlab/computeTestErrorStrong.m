function [error_L2,error_NMAE,error_NMAE_round] = computeTestErrorStrong(model,Ytest)
% COMPUTETESTERRORSTRONG Compute the strong test error.
% FORMAT
% DESC computes the test error for the strong generalization.
% ARG model : the model.
% ARG Ytest : the test data.
% RETURN L2_error : the l2 error.
% RETURN NMAE_error : the NMAE error.
% RETURN NMAE_round_error : the NMAE error with rounding on the outputs.
%
% SEEALSO : computeTestErrorWeak
% 
% COPYRIGHT : Raquel Urtasun, 2009

% COLLAB
??? this doesn't work

val_L2 = 0;
tot_L2 = 0;
val_NMAE = 0;
tot_NMAE = 0;
val_NMAE_round = 0;
tot_NMAE_round = 0;

for i = 1:size(Ytest, 2)       
  ind = find(Ytest(:, i));
  elim = find(ind>size(model.X, 1));
  tind = ind;
  tind(elim) = [];
  
  if (length(tind)==0)
      continue;
  end
  % in the case of STRONG experiments, the user is new, so we have to
  % compute the prediction using the test data
  % compute random (LOO --> leave one out)
  indexRand = randperm(length(tind));
  Y_train_user = Ytest(:,i);
  Y_test_user = Y_train_user(tind(indexRand(end)));
  Y_train_user(tind(indexRand(end)),:) = 0;
  [mu, varsig] = collabPosteriorMeanVar(model, Y_train_user, model.X(tind(indexRand(end)), :));
  
  %mu = mu*model.sd(tind);
  %mu = mu+model.mu(tind);
  
  a = Y_test_user - mu; 
  a = [a; Ytest(elim, i)];
  val_L2 = val_L2 + a'*a;
  tot_L2 = tot_L2 + length(a);
  val_NMAE = val_NMAE + sum(abs(a));
  tot_NMAE = tot_NMAE + length(a);
  val_NMAE_round = val_NMAE_round + sum(abs(round(a)));
  tot_NMAE_round = tot_NMAE_round + length(a);
end
error_L2 = sqrt(val_L2/tot_L2);
error_NMAE = (val_NMAE/tot_NMAE)/1.6;
error_NMAE_round = (val_NMAE_round/tot_NMAE_round)/1.6;
