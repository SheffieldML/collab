% COLLABTEST Test collaborative filtering model.

% COLLAB

rand('seed', 1e5)
randn('seed', 1e5)

numItems = 50;
numUsers = 10;
y = randn(numItems, numUsers);
y(find(rand(numItems, numUsers)>0.2)) = 0;
y = sparse(y);

options = collabOptions;

for numComps = [1 2 4 8]
  for heteroNoise = [false true]
    options.numComps = numComps;
    options.heteroNoise = heteroNoise;
    fprintf('Testing model with %d component(s).\n', options.numComps)
    if heteroNoise
      fprintf('Heteroschedastic noise used.\n')
    end
    model = collabCreate(2, numUsers, y(:, 1), options);
    params = collabExtractParam(model);
    params = randn(size(params));
    model = collabExpandParam(model, params);
    if model.M > 1
      model = collabEstep(model);
    end
    modelDisplay(model);
    modelGradientCheck(model);
  end
end
