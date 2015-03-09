fid = fopen('probe.txt');
count = 0;
probeFilms = cell(17770, 1);
while 1
  count = count + 1;
  nextLine = fgetl(fid);
  if ~ischar(nextLine), break, end
  if nextLine(end)==':'
    film = str2num(nextLine(1:end-1));
  else
    probeFilms{film} = [probeFilms{film}; str2num(nextLine)];
  end
  if(~rem(count, 10000))
    fprintf('Loaded in %d rating locations.\n', count)
  end
end
fclose(fid);
Yprobe = cell(2649429, 3);

for i = 1:length(probeFilms)
  for j = 1:length(probeFilms{i})
    userId = probeFilms{i}(j);
    ind = find(Y{userId, 1}==i);
    
    if isempty(Yprobe{userId, 1})
      Yprobe{userId, 1} = zeros(20, 1);
      Yprobe{userId, 2} = zeros(20, 1);
      Yprobe{userId, 3} = 0;
    end
    if Yprobe{userId, 1}(end) ~= 0
      Yprobe{userId, 1} = [Yprobe{userId, 1}; zeros(20, 1)];
      Yprobe{userId, 2} = [Yprobe{userId, 2}; zeros(20, 1)];
    end
    Yprobe{userId, 3} = Yprobe{userId, 3} + 1;
    Yprobe{userId, 1}(Yprobe{userId, 3}) =  i;
    Yprobe{userId, 2}(Yprobe{userId, 3}) =  Y{userId,2}(ind);
    Y{userId, 1}(ind) = [];
    Y{userId, 2}(ind) = [];
    Y{userId, 3} = Y{userId, 3} - 1;
  end
  
  if(~rem(i, 10))
    fprintf('Done %d films.\n', i)
  end
end