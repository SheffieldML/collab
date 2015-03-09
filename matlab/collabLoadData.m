function [Y, lbls, Ytest, lblstest] = collabLoadData(dataset, seedVal)

% COLLABLOADDATA Load a collaborative filtering dataset.
% FORMAT
% DESC loads a data set for a collaborative filtering problem.
% ARG dataset : the name of the data set to be loaded. 
% RETURN Y : the training data loaded in.
% RETURN lbls : a set of labels for the data (if there are no
% labels it is empty).
% RETURN Ytest : the test data loaded in. If no test set is
% available it is empty.
% RETURN lblstest : a set of labels for the test data (if there are
% no labels it is empty).
%
% SEEALSO : mapLoadData, datasetsDirectory
%
% COPYRIGHT : Neil D. Lawrence, 2009
%
% COPYRIGHT : Raquel Urtasun, 2009

% COLLAB

  
  if nargin > 1
    randn('seed', seedVal)
    rand('seed', seedVal)
  end

  % get directory

  baseDir = datasetsDirectory;
  dirSep = filesep;
  lbls = [];
  lblstest = [];
  switch dataset
   
       case 'mixtoydata'
    numItems = 200;
    numUsers = 200;
    X = randn(numItems, 2);
    kern = kernCreate(X, {'rbf', 'bias', 'white'});
    kern.comp{1}.variance = 1;
    kern.comp{2}.variance = 0.4;
    kern.comp{3}.variance = 0.4;
    K = kernCompute(kern, X);
    Y0 = gsamp(zeros(numItems, 1), K, numUsers)';
    Y1 = gsamp(zeros(numItems, 1), K, numUsers)';
    lbls = rand(numItems, numUsers)>0.5;
    observed = rand(numItems, numUsers)>0.8;
    Yfull = zeros(numItems, numUsers);
    Yfull(find(lbls)) = Y0(find(lbls));
    Yfull(find(~lbls)) = Y1(find(~lbls));
    Y = zeros(numItems, numUsers);
    Ytest = zeros(numItems, numUsers);
    Y(find(observed)) = Yfull(find(observed));
    Ytest(find(~observed)) = Yfull(find(~observed));
    Y = sparse(Y);
    Ytest = sparse(Ytest);
    lblstest = X;

   case 'movielens'
    try 
      load([baseDir 'movielens.mat']);
      
    catch
      [void, errid] = lasterr;
      if strcmp(errid, 'MATLAB:load:couldNotReadFile');

        
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
        numTrainRatings = ceil(0.8*numRatings);
        Y = spalloc(numFilms, numUsers, numTrainRatings);
        Ytest = spalloc(numFilms, numUsers, numRatings-numTrainRatings);
        indTrain = sub2ind(size(Y), films(1:numTrainRatings), users(1:numTrainRatings));
        indTest = sub2ind(size(Ytest), films(numTrainRatings+1:numRatings), users(numTrainRatings+1:numRatings));
        Y(indTrain) = ratings(1:numTrainRatings);
        Ytest(indTest) = ratings(numTrainRatings+1:numRatings);
        
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

        save([baseDir 'movielens.mat'], 'Y', 'lbls', 'Ytest', 'lblstest');
      else
        error(lasterr);
      end
    end
    
   case {'movielens_80_f_1','movielens_80_f_2','movielens_80_f_3','movielens_80_f_4','movielens_80_f_5'}
    perc_train = 0.8;
    try 
      load([baseDir, dataset, '.mat']);
      
    catch
      [void, errid] = lasterr;
      if strcmp(errid, 'MATLAB:load:couldNotReadFile');
        partNo = str2num(dataset(end));
        if_random = 0;
        
        [Y, lbls, Ytest] = readMovieLens(perc_train, partNo, if_random);

        save([baseDir, dataset, '.mat'], 'Y', 'lbls', 'Ytest');
      else
        error(lasterr);
      end
    end
    
   case {'movielens_50_1','movielens_50_2','movielens_50_3','movielens_50_4','movielens_50_5'}
    perc_train = 0.5;
    if_random = 1;
    try 
      load([baseDir, dataset, '.mat']);
      
    catch
      [void, errid] = lasterr;
      if strcmp(errid, 'MATLAB:load:couldNotReadFile');
        partNo = str2num(dataset(end));

        [Y, lbls, Ytest] = readMovieLens(perc_train, partNo,if_random);

        save([baseDir, dataset, '.mat'], 'Y', 'lbls', 'Ytest');
      else
        error(lasterr);
      end
    end
   case {'movielens_60_1','movielens_60_2','movielens_60_3','movielens_60_4','movielens_60_5'}
    perc_train = 0.6;
    if_random = 1;
    try 
      load([baseDir, dataset, '.mat']);
      
    catch
      [void, errid] = lasterr;
      if strcmp(errid, 'MATLAB:load:couldNotReadFile');
        partNo = str2num(dataset(end));

        [Y, lbls, Ytest] = readMovieLens(perc_train, partNo,if_random);

        save([baseDir, dataset, '.mat'], 'Y', 'lbls', 'Ytest');
      else
        error(lasterr);
      end
    end
   case {'movielens_70_1','movielens_70_2','movielens_70_3','movielens_70_4','movielens_70_5'}
    perc_train = 0.7;
    if_random = 1;
    try 
      load([baseDir, dataset, '.mat']);
      
    catch
      [void, errid] = lasterr;
      if strcmp(errid, 'MATLAB:load:couldNotReadFile');
        partNo = str2num(dataset(end));

        [Y, lbls, Ytest] = readMovieLens(perc_train, partNo,if_random);

        save([baseDir, dataset, '.mat'], 'Y', 'lbls', 'Ytest');
      else
        error(lasterr);
      end
    end
    
   case {'movielens_80_1','movielens_80_2','movielens_80_3','movielens_80_4','movielens_80_5'}
    perc_train = 0.8;
    if_random = 1;
    try 
      load([baseDir, dataset, '.mat']);
      
    catch
      [void, errid] = lasterr;
      if strcmp(errid, 'MATLAB:load:couldNotReadFile');
        partNo = str2num(dataset(end));

        [Y, lbls, Ytest] = readMovieLens(perc_train, partNo,if_random);

        save([baseDir, dataset, '.mat'], 'Y', 'lbls', 'Ytest');
      else
        error(lasterr);
      end
    end
    
   case {'movielens_90_1','movielens_90_2','movielens_90_3','movielens_90_4','movielens_90_5'}
    perc_train = 0.9;
    if_random = 1;
    try 
      load([baseDir, dataset, '.mat']);
      
    catch
      [void, errid] = lasterr;
      if strcmp(errid, 'MATLAB:load:couldNotReadFile');
        partNo = str2num(dataset(end));

        [Y, lbls, Ytest] = readMovieLens(perc_train, partNo,if_random);

        save([baseDir, dataset, '.mat'], 'Y', 'lbls', 'Ytest');
      else
        error(lasterr);
      end
    end
    
   case {'movielens_30_1','movielens_30_2','movielens_30_3','movielens_30_4','movielens_30_5'}
    perc_train = 0.3;
    if_random = 1;
    try 
      load([baseDir, dataset, '.mat']);
      
    catch
      [void, errid] = lasterr;
      if strcmp(errid, 'MATLAB:load:couldNotReadFile');
        partNo = str2num(dataset(end));

        [Y, lbls, Ytest] = readMovieLens(perc_train, partNo,if_random);

        save([baseDir, dataset, '.mat'], 'Y', 'lbls', 'Ytest');
      else
        error(lasterr);
      end
    end
    
    %%%

   case {'movielens_strong_1','movielens_strong_2','movielens_strong_3','movielens_strong_4','movielens_strong_5'}
    % this is the database strong
    %perc_train = -1;
    %if_random = 1;
    try 
      load([baseDir, dataset, '.mat']);
      
    catch
      [void, errid] = lasterr;
      if strcmp(errid, 'MATLAB:load:couldNotReadFile');
        partNo = str2num(dataset(end));

        [Y, lbls, Ytest] = readMovieLensStrong(partNo);

        save([baseDir, dataset, '.mat'], 'Y', 'lbls', 'Ytest');
      else
        error(lasterr);
      end
    end
    
   case {'movielens_weak_1','movielens_weak_2','movielens_weak_3','movielens_weak_4','movielens_weak_5'}
    % this is the database strong
    %perc_train = -1;
    %if_random = 1;
    try 
      load([baseDir, dataset, '.mat']);
      
    catch
      [void, errid] = lasterr;
      if strcmp(errid, 'MATLAB:load:couldNotReadFile');
        partNo = str2num(dataset(end));

        [Y, Ytest] = readMovieLensWeak(partNo);

        save([baseDir, dataset, '.mat'], 'Y', 'Ytest');
      else
        error(lasterr);
      end
    end
    
   case {'eachmovie_weak_1','eachmovie_weak_2','eachmovie_weak_3','eachmovie_weak_4','eachmovie_weak_5'}
    % this is the database weak
    %perc_train = -1;
    %if_random = 1;
    try 
      load([baseDir, dataset, '.mat']);
      
    catch
      [void, errid] = lasterr;
      if strcmp(errid, 'MATLAB:load:couldNotReadFile');
        partNo = str2num(dataset(end));

        [Y, Ytest] = readEachMovieWeak(partNo);

        save([baseDir, dataset, '.mat'], 'Y', 'Ytest');
      else
        error(lasterr);
      end
    end
    
   case {'eachmovie_marlin_weak_1','eachmovie_marlin_weak_2','eachmovie_marlin_weak_3'}
    % this is the database strong
    %perc_train = -1;
    %if_random = 1;
    try 
      load([baseDir, dataset, '.mat']);
    catch
      [void, errid] = lasterr;
      if strcmp(errid, 'MATLAB:load:couldNotReadFile');
        partNo = str2num(dataset(end));

        [Y, Ytest] = readEachMovieMarlinWeak(partNo); 

        save([baseDir, dataset, '.mat'], 'Y', 'Ytest');
      else
        error(lasterr);
      end
    end
    
   case {'eachmovie_marlin_strong_1','eachmovie_marlin_strong_2','eachmovie_marlin_strong_3'}
    % this is the database strong
    %perc_train = -1;
    %if_random = 1;
    try 
      load([baseDir, dataset, '.mat']);
      
    catch
      [void, errid] = lasterr;
      if strcmp(errid, 'MATLAB:load:couldNotReadFile');
        partNo = str2num(dataset(end));

        [Y, lbls, Ytest] = readEachMovieMarlinStrong(partNo);

        save([baseDir, dataset, '.mat'], 'Y', 'lbls', 'Ytest');
      else
        error(lasterr);
      end
    end
    
   case {'movielens_marlin_weak_1','movielens_marlin_weak_2','movielens_marlin_weak_3'}
    % this is the database strong
    %perc_train = -1;
    %if_random = 1;
    try 
      load([baseDir, dataset, '.mat']);
      % get the extra info
      load([baseDir, 'movielens_metadata.mat']);
      
    catch
      [void, errid] = lasterr;
      if strcmp(errid, 'MATLAB:load:couldNotReadFile');
        partNo = str2num(dataset(end));

        [Y, Ytest] = readMovieLensMarlinWeak(partNo);
        % get the extra info
        load([baseDir, 'movielens_metadata.mat']);
        

        save([baseDir, dataset, '.mat'], 'Y', 'lbls', 'Ytest', 'lblstest');
      else
        error(lasterr);
      end
    end
    
   case {'movielens_marlin_strong_1','movielens_marlin_strong_2','movielens_marlin_strong_3'}
    % this is the database strong
    %perc_train = -1;
    %if_random = 1;
    try 
      load([baseDir, dataset, '.mat']);
      % get the extra info
      %load([baseDir, 'movielens_metadata.mat']);
      
      
    catch
      [void, errid] = lasterr;
      if strcmp(errid, 'MATLAB:load:couldNotReadFile');
        partNo = str2num(dataset(end));

        [Y, lbls, Ytest] = readMovieLensMarlinStrong(partNo);
        % get the extra info
        kk = load([baseDir, 'movielens_metadata.mat']);
        lblstest = kk.lbls;

        save([baseDir, dataset, '.mat'], 'Y', 'lbls', 'Ytest', 'lblstest');
      else
        error(lasterr);
      end
    end

   case {'movielens_10M_a','movielens_10M_b'}
    try 
      load([baseDir, dataset, '.mat']);
      
    catch
      [void, errid] = lasterr;
      if strcmp(errid, 'MATLAB:load:couldNotReadFile');
        partLetter = dataset(end);

        [Y, Ytest] = readMovieLens10MCellLetter(partLetter);

        save([baseDir, dataset, '.mat'], 'Y', 'Ytest');
      else
        error(lasterr);
      end
    end


    
   case {'movielensSmall1', 'movielensSmall2', 'movielensSmall3', 'movielensSmall4', 'movielensSmall5'}

    partNo = str2num(dataset(end));
    uTrain = load([baseDir dirSep 'movielens' dirSep 'small' dirSep 'u' num2str(partNo) '.base']);
    numUsers = max(uTrain(:, 1));
    numFilms = max(uTrain(:, 2));
    numRatings = size(uTrain, 1);
    Y = spalloc(numFilms, numUsers, numRatings);
    
    for i = 1:size(uTrain, 1);
      Y(uTrain(i, 2), uTrain(i, 1)) = uTrain(i, 3);
    end
    meanY = mean(Y(find(Y)));
    Y(find(Y)) = (Y(find(Y))-meanY);
    uTest = load([baseDir dirSep 'movielens' dirSep 'small' dirSep 'u' num2str(partNo) '.test']);
    numTestRatings = size(uTest, 1);
    Ytest = spalloc(numFilms, numUsers, numTestRatings);
    for i = 1:size(uTest, 1);
      Ytest(uTest(i, 2), uTest(i, 1)) = uTest(i, 3);
    end
    Ytest(find(Ytest)) = (Ytest(find(Ytest))-meanY);
    
    
   otherwise
    error('Unknown data set requested.')
    
  end
end
