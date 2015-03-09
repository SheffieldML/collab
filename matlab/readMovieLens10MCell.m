function [Y, Ytest] = readMovieLens10MCell(partNo)

% READMOVIELENS10MCELL Read the 10M Movielens into a cell array.
% FORMAT
% DESC reads the 10M MovieLens data into a cell array.
% ARG partNo : the part of the 10M MovieLens data to read in. 
% RETURN Y : the data in a cell array.
% RETURN Ytest : the test data in a cell array.
% read the 10M movielens in a cell array. It is too big to do the regular way
%
% SEEALSO : collabLoadData
%
% COPYRIGHT : Raquel Urtasun, 2009

% COLLAB
  

  baseDir = datasetsDirectory;
  dirSep = filesep;
  
  % load the ratings
  
  fileName = [baseDir dirSep 'movielens' dirSep '10M' dirSep 'r',num2str(partNo),'.train'];
  [users, films, ratings, timeStamp] = textread(fileName, '%n::%n::%n::%n');
  
  
  [Y] = loadSparse10M(users,films,ratings);
  

  fileName = [baseDir dirSep 'movielens' dirSep '10M' dirSep 'r',num2str(partNo),'.test'];
  [users_test, films_test, ratings_test, timeStamp] = textread(fileName, '%n::%n::%n::%n');
  
  
  [Ytest] = loadSparse10M(users_test,films_test,ratings_test);
  
