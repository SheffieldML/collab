function [mae] = generateCteNMAE(num_ordinals)

% [mae] = generateCteNMAE(num_ordinals)
%
% generate the cte for NMAE normalization
% num_ordinals is 5 for movielens and 6 for eachmovie

  size_data = 100000;

% first generate a uniformly distributed random data set
Y = ceil(rand(size_data,1)*num_ordinals);

% generate predictions for the data
pred = ceil(rand(size_data,1)*num_ordinals);

% predict the mean absolute error
		     mae = mean(abs(Y - pred));
