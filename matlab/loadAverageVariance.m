
meanVals = zeros(17700, 1);
varVals = zeros(17700, 1);
for i = 1:17700
  vals = zeros(Y{i, 3}, 1);
  fileNameBase = num2str(filmNum);
  fileName = ['mv_' repmat('0', 1, 7-length(fileNameBase)) fileNameBase ...
      '.txt'];
  fid = fopen(fileName);
  void = fgetl(fid);
  while 1
    nextLine = fgetl(fid);
    if ~ischar(nextLine), break, end
    commas = find(nextLine==44);
    vals(count) = str2num(nextLine(commas(1)+1:commas(2)-1));
  end
  meanVals(i) = mean(vals);
  varVals(i) = var(vals);
end
