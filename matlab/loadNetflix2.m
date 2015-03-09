Y = cell(2649429, 3);
userCount = spalloc(2649429, 1, 480189);
userSquareSum  = spalloc(2649429, 1, 480189);
userSum  = spalloc(2649429, 1, 480189);
ratingSum = zeros(17700, 1);
ratingSquareSum = zeros(17700, 1);
ratingCount = zeros(17700, 1);
oldTotalRating = 0;
totalRating = 0;
tic
for filmNumDouble = 1:17700
  filmNum = uint16(filmNumDouble);
  fileNameBase = num2str(filmNum);
  fileName = ['mv_' repmat('0', 1, 7-length(fileNameBase)) fileNameBase '.txt'];
  fid = fopen(fileName);
  void = fgetl(fid);
  while 1
    totalRating = totalRating +1;
    ratingCount(filmNumDouble) = ratingCount(filmNumDouble)  + 1; 
    nextLine = fgetl(fid);
    if ~ischar(nextLine), break, end
    commas = find(nextLine==44);
    uid = str2num(nextLine(1:commas(1)-1));
    score = uint8(str2num(nextLine(commas(1)+1:commas(2)-1)));
    ratingSum(filmNumDouble) = ratingSum(filmNumDouble)+double(score);
    ratingSquareSum(filmNumDouble) = ratingSquareSum(filmNumDouble)+double(score)*double(score);
    userCount(uid) = userCount(uid)+1;
    userSquareSum(uid) = userSquareSum(uid)+score*score;
    userSum(uid) = userSum(uid) + score;
  end
  fclose(fid);
  n = ratingCount(filmNumDouble);
  diffRating = totalRating - oldTotalRating;
  oldTotalRating = totalRating;
  rps = diffRating/toc;
  remain = (100000000 - totalRating)/rps;
  remain = remain/(3600);
  tic
  if ~rem(filmNumDouble, 1)
    fprintf('Film %d done,\t ratings %d,\t mean %2.4f,\t std %2.4f,\t rps %2.4f,\t remain %2.4f hrs,\t total %d.\n', filmNumDouble, n,...
            ratingSum(filmNumDouble)/n, ...
            sqrt(ratingSquareSum(filmNumDouble)/n- ...
                 ratingSum(filmNumDouble)*ratingSum(filmNumDouble)/(n*n)), ...
            rps, remain, totalRating);
  end
end
