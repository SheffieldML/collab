
function [mu_T] = computePredictionsErrorWeak(model,Y,Ytest)
%
% [error_L2,error_NMAE,error_NMAE_round] = computePredictionsErrorWeak(model,Y,Ytest)
 
val_L2 = 0;
tot_L2 = 0;
val_NMAE = 0;
tot_NMAE = 0;
val_round_NMAE = 0;
tot_round_NMAE = 0;
accum = [];
mu_T = [];

for i = 1:size(Y, 2)       
    ind = find(Ytest(:, i));
    elim = find(ind>size(model.X, 1));
    tind = ind;
    tind(elim) = [];
    [mu, varsig] = collabPosteriorMeanVar(model, Y(:, i), model.X(tind, :));
    % normalize the values

    if (length(mu)>0)
        mu = mu.*model.sd(tind);
        mu = mu+model.mu(tind);
    end
	mu_T = [mu_T; mu];
end
