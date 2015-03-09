function [Y, lbls, Ytest] = readMovieLens(perc_train,partNo,if_random)

% READMOVIELENS Read in a given percentage of the movielens data.
% FORMAT
% DESC reads the MovieLens 1M Marlin data.
% ARG perc_train : the percentage to use as training.
% ARG partNo : the partition number.
% RETURN Y : the data.
% RETURN lbls : the lables of the training data. 
% RETURN Ytest : the test data.
%
% SEEALSO : collabLoadData, readMovieLensMarlinStrong
%
% COPYRIGHT : Raquel Urtasun, 2009

% COLLAB

  lbls = [];


  baseDir = datasetsDirectory;
  dirSep = filesep;
  
  % load the ratings
  
  fileName = [baseDir dirSep 'movielens' dirSep 'large' dirSep 'ratings.dat'];
  [users, films, ratings, timeStamp] = textread(fileName, '%n::%n::%n::%n');
  ind = randperm(size(users, 1));
  users = users(ind, :);
  films = films(ind, :);
  ratings = ratings(ind, :);
  numUsers = max(users);
  numFilms = max(films);
  
  numRatings = size(users, 1);
  numTrainRatings = ceil(perc_train*numRatings);
  Y = spalloc(numFilms, numUsers, numTrainRatings);
  Ytest = spalloc(numFilms, numUsers, numRatings-numTrainRatings);
  
  
  
  % this depends on the partition number, and it is ordered
  numTestRatings = numRatings - numTrainRatings;
  if (if_random)
    
    randn('seed', 1e5);
    rand('seed', 1e5);
    for i=1:partNo
      index = randperm(numRatings);
    end
    indexTrain = index(1:numTrainRatings);
    indexTest = index(1+numTrainRatings:end);
    
    
  else
    index_rand = 1:numRatings;
    maxPartNo = 1./(1-perc_train);
    indexTrain = [];
    
    indexTrain = [1:(partNo-1)*numTestRatings]; 
    indexTrain = [indexTrain, 1+(partNo)*numTestRatings:numRatings];
    indexTest = [1+(partNo-1)*numTestRatings:partNo*numTestRatings];
  end
  indTrain = sub2ind(size(Y), films(indexTrain), users(indexTrain));
  indTest = sub2ind(size(Ytest), films(indexTest), users(indexTest));
  
  Y(indTrain) = ratings(indexTrain);
  Ytest(indTest) = ratings(indexTest);
  
  
  % save the additional information
  
  fileName = [baseDir dirSep 'movielens' dirSep 'large' dirSep 'movies.dat'];
  %[id, films, Type] = textread(fileName, '%n::%s::%s');
  
  % create the structure
  lbls = zeros(size(Y,1),18);
  
  fid = fopen(fileName);
  readLine = 0;
  counter = 0;
  data = [];
  all_genres = [{'Action'},{'Adventure'},{'Animation'},{'Children''s'}, ...
                {'Comedy'},{'Crime'},{'Documentary'},{'Drama'},{'Fantasy'},{'Film-Noir'}, ...
                {'Horror'},{'Musical'},{'Mystery'},{'Romance'},{'Sci-Fi'},{'Thriller'},{'War'},{'Western'}];
  
  
  readLine = fgets(fid);
  while readLine ~= -1
    
    parts = stringSplit(readLine,':');
    id = str2num(parts{1});
    title = parts(3);
    genre = parts{5};
    % createMovieLensExtra(genre);
    
    for i=1:length(all_genres)
      if (strfind(genre,all_genres{i}))
        lbls(id,i) = 1;
      end
    end
    
    readLine = fgets(fid);
    
  end
end
  
  
