function [Y, Ytest] = readMovieLensMarlinWeak(partNo)

% READMOVIELENSMARLINWEAK Read in Marlin's weak partitions for movielens 1M.
% FORMAT
% DESC reads the Movielens 1M Marlin weak partitions.
% ARG partNo : the part of the Movielens data to read in. 
% RETURN Y : the data.
% RETURN Ytest : the test data.
%
% SEEALSO : collabLoadData, readMovieLensMarlinStrong
%
% COPYRIGHT : Raquel Urtasun, 2009

% COLLAB


baseDir = datasetsDirectory;
dirSep = filesep;

% load the ratings


fileName = [baseDir dirSep 'collab' dirSep 'project' dirSep '1mml-mmmf' dirSep 'data' dirSep 'marlin.mat'];

disp(['Reading ... ',fileName]);

load(fileName);

Y = weaktrain{partNo}';
Ytest = weaktest{partNo}';


        
