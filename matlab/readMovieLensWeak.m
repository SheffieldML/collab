function [Y, lbls, Ytest] = readMovieLensWeak(partNo)

% READMOVIELENSWEAK Read in the weak partitions for the Movielens.
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

%indexTest = 1:length(users);
%indexTest(indexTrain) = [];

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


        
