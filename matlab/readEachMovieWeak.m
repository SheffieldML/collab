function [Y, Ytest] = readEachMovieWeak(partNo)

% READEACHMOVIEWEAK Read in EachMovie users with over 20 ratings.
% FORMAT
% DESC reads in the EachMovie users with over 20 ratings and saves them
% to a mat file for later use.
% ARG partNo : the partition number.
% RETURN Y : the training data.
% RETURN Ytest : the test data.
% 
% SEEALSO : readEachMovieMarlinWeak, readEachMovieMarlinStrong
%
% COPYRIGHT : Raquel Urtasun, 2009

% COLLAB

baseDir = datasetsDirectory;
dirSep = filesep;

% load the ratings

try
  fileName = [baseDir dirSep 'eachmovie' dirSep 'Vote_more_20.mat'];
  load(fileName);
catch
  
  fileName = [baseDir dirSep 'eachmovie' dirSep 'Vote.txt'];
  
  disp(['Reading ... ',fileName]);
  
  [users, films, ratings, weights, dates, hours, minutes, seconds] = textread(fileName, '%n\t%n\t%n\t%n\t%s %n:%n:%n');
  ind = randperm(size(users, 1));
  users = users(ind, :);
  films = films(ind, :);
  ratings = ratings(ind, :);
  numUsers = max(users);
  numFilms = max(films);
  
  activeUsers = [1:numUsers];
  % erase the users with less than 20 films
  disp('Removing users with less than 20 ratings');
  mapUsers = -ones(numUsers,1);
  numActiveUsers = 0;
  indTotal = [];
  for i=1:numUsers
    ind = find(users==i);
    if (length(ind)<20)
      % remove the user
      [indTotal] = [indTotal; ind];
    else
      numActiveUsers = numActiveUsers+1;
      mapUsers(i) = numActiveUsers;
    end
  end
  users(indTotal) = [];
  films(indTotal) = [];
  ratings(indTotal) = [];
  weights(indTotal) = [];
  dates(indTotal) = [];
  hours(indTotal) = [];
  minutes(indTotal) = [];
  second(indTotal) = [];
  users = mapUsers(users);
  fileName = [baseDir dirSep 'eachmovie' dirSep 'Vote_more_20.mat'];
  save(fileName,'users','films','ratings','weights','dates','hours','minutes','seconds');
end

numUsers = max(users);
numFilms = max(films);

numRatings = size(users, 1);
numUsersTrain = 30000;
numUsers = max(users);
for i=1:partNo
  % partition the users at random
  randIndexUsers = randperm(numUsers);

end
% get the films for those users
numTrainRatings = 0;
indexTrain = [];
indexTest = [];
for i=1:numUsersTrain
  indexUsers = find(users==randIndexUsers(i));
  
  indexTest = [indexTest; indexUsers(end)];
  
  % use one for testing and one for training
  indexUsers(end) = [];
  
  numTrainRatings = numTrainRatings + length(indexUsers);
  indexTrain = [indexTrain; indexUsers]; % ?? this takes too much time
end
numTestRatings = numUsersTrain;
Y = spalloc(numFilms, numUsers, numTrainRatings);
Ytest = spalloc(numFilms, numUsers, numTestRatings);
numRatings = numTrainRatings + numTestRatings;


indTrain = sub2ind(size(Y), films(indexTrain), users(indexTrain));
indTest = sub2ind(size(Ytest), films(indexTest), users(indexTest));

Y(indTrain) = ratings(indexTrain);
Ytest(indTest) = ratings(indexTest);

