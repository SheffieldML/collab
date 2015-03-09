function [Y, lbls, Ytest] = readMovieLens10M(partNo)

% READMOVIELENS10M Read in a partition of the movielens 10M data.
% FORMAT
% DESC reads the MovieLens 10M Marlin data.
% ARG partNo : the partition number.
% RETURN Y : the data.
% RETURN lbls : the lables of the training data. 
% RETURN Ytest : the test data.
%
% SEEALSO : collabLoadData, readMovieLens
%
% COPYRIGHT : Raquel Urtasun, 2009

% COLLAB

  lbl = [];

  baseDir = datasetsDirectory;
  dirSep = filesep;
  
  % load the ratings
  
  fileName = [baseDir dirSep 'movielens' dirSep '10M' dirSep 'r',num2str(partNo),'.train'];
  [users, films, ratings, timeStamp] = textread(fileName, '%n::%n::%n::%n');
  
  numUsers = max(users);
  numFilms = max(films);
  
  %keyboard;
  
  numTrainRatings = size(users,1);
  Y = spalloc(numFilms, numUsers, numTrainRatings);
  
  
  indTrain = sub2ind(size(Y), films, users);
  
  %keyboard;
  Y(indTrain) = ratings;
  
  %keyboard;
  
  
  users = [];
  films = [];
  ratings = [];
  timeStamp = [];
  
        
  fileNameTest = [baseDir dirSep 'movielens' dirSep '10M' dirSep 'r',num2str(partNo),'.test'];
  [users_test, films_test, ratings_test, timeStamp_test] = textread(fileNameTest, '%n::%n::%n::%n');
  
  numTestRatings = size(users_test,1);
  numRatings = numTrainRatings + numTestRatings;
  Ytest = spalloc(numFilms, numUsers, numTestRatings);
  
  
  
  % this depends on the partition number, and it is ord;
  
  
  indTest = sub2ind(size(Ytest), films_test, users_test);
  Ytest(indTest) = ratings_test;
  
  
  % save the additional information
  
  %keyboard
  
  fileName = [baseDir dirSep 'movielens' dirSep 'large' dirSep 'movies.dat'];
  
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
  
  
  