
results = cell(3,8);
resultsPerm = cell(3,8);
%imNames = cell(3,8);

rand('state',sum(100*clock))

parfor cv=1:8
    ali = load('patchesAll.mat');
    swirler = randi(5000);
    for or=1:3
        display(['cv: '  mat2str(cv) ', ori: ' mat2str(or)])
        [results{or,cv} resultsPerm{or,cv} imNames{or,cv} p] = doRegression(ali.imData, ali.PAR, or, swirler);
    end
end

totalResults.results         = results;
totalResultsPerm.resultsPerm = resultsPerm;
totalResults.imNames         = imNames;

save('totalResults.mat', 'totalResults')