Y = cell(2649429, 3);
ratingSum = zeros(17770, 1);
ratingSquareSum = zeros(17770, 1);
ratingCount = zeros(17770, 1);
oldTotalRating = 0;
totalRating = 0;
tic
for filmNumDouble = 1:17770
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
    if isempty(Y{uid, 1})
      Y{uid, 1} = uint16(zeros(40, 1));
      Y{uid, 2} = uint8(zeros(40, 1));
      Y{uid, 3} = 0;
    end
    if Y{uid, 1}(end) ~= 0
      %fprintf('Allocating memory for %d, user %d\n', filmNum, uid)
      Y{uid, 1} = [Y{uid, 1}; uint16(zeros(20, 1))];
      Y{uid, 2} = [Y{uid, 2}; uint8(zeros(20, 1))];
    end
    Y{uid, 3} = Y{uid, 3} + 1;
    Y{uid, 1}(Y{uid, 3}) = filmNum;
    Y{uid, 2}(Y{uid, 3}) = score;
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
userCount = spalloc(2649429, 1, 480189);
userSquareSum  = spalloc(2649429, 1, 480189);
userSum  = spalloc(2649429, 1, 480189);

for i = 1:size(Y, 1)
  if ~isempty(Y{i, 1})
    userCount(i) = Y{i, 3};
    userSum(i) = sum(Y{i, 2});
    userSquareSum(i) = sum(Y{i, 2}.*Y{i, 2});
  end
end

for i = 1:size(Y, 1)
  if ~isempty(Y{i, 1})
    Y{i, 1} = Y{i, 1}(1:Y{i,3});
    Y{i, 2} = Y{i, 2}(1:Y{i,3});
  end
end

save netFlixData.mat Y ratingSum ratingSquareSum ratingCount userCount userSquareSum userSum

