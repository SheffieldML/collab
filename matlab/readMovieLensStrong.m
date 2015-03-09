function [Y,lbls,Ytest] = readMovieLensStrong(partNo)

% READMOVIELENSSTRONG Read in the strong partitions for the Movielens.
% FORMAT
% DESC reads the MovieLens 1M Marlin weak partitions.
% ARG partNo : the part of the 1M MovieLens data to read in. 
% RETURN Y : the data.
% RETURN lbls : addiitonal information.
% RETURN Ytest : the test data.
%
% SEEALSO : collabLoadData, readMovieLensMarlinStrong
%
% COPYRIGHT : Raquel Urtasun, 2009

% COLLAB



  
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
  numUsersTrain = 5000;
  numUsers = max(users);
  for i=1:partNo
    % partition the users at random
    randIndexUsers = randperm(numUsers);
    
  end
  % get the films for those users
  numTrainRatings = 0;
  indexTrain = [];
  for i=1:numUsersTrain
    indexUsers = find(users==randIndexUsers(i));
    numTrainRatings = numTrainRatings + length(indexUsers);
    indexTrain = [indexTrain; indexUsers]; % ?? this takes too much time
  end
  Y = spalloc(numFilms, numUsers, numTrainRatings);
  Ytest = spalloc(numFilms, numUsers, numRatings-numTrainRatings);
  
  indexTest = 1:length(users);
  indexTest(indexTrain) = [];
  
  % this depends on the partition number, and it is ordered
  numTestRatings = numRatings - numTrainRatings;
  
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

        
