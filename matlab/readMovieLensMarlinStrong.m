function [Y, lbls, Ytest] = readMovieLensMarlinStrong(partNo)

% READMOVIELENSMARLINSTRONG Read in Marlin's strong partitions for movielens 1M.
% FORMAT
% DESC reads the Movielens 1M Marlin strong partitions.
% ARG partNo : the part of the Movielens data to read in. 
% RETURN Y : the data.
% RETURN lbls : the labels associated with the movies.
% RETURN Ytest : the test data.
%
% SEEALSO : collabLoadData, readMovieLensMarlinWeak
%
% COPYRIGHT : Raquel Urtasun, 2009

% COLLAB


lbls = [];

baseDir = datasetsDirectory;
dirSep = filesep;

% load the ratings


fileName = [baseDir dirSep 'collab' dirSep 'project' dirSep '1mml-mmmf' dirSep 'data' dirSep 'marlin.mat'];

disp(['Reading ... ',fileName]);

load(fileName);

Y = weaktrain{partNo}';
lbls = strongtrain{partNo}';
Ytest = strongtest{partNo}';


        
