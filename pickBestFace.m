function [picPath picInd] = pickBestFace(totalResults)

numPics = numel(totalResults.imNames{1}.trainIm) +  numel(totalResults.imNames{1}.testIm);
numObs = numel(totalResults.results{1}.YhatTra) + numel(totalResults.results{1}.YhatTest);
numPat = numObs/numPics;

for or=1:3
    lol(or,:) = cellfun(@getDiff,totalResults.results(or,:), repmat({or}, 1, 8), 'UniformOutput', 0);
    
end
for cv=1:size(totalResults.results,2)
    sums(:,cv) = sum([lol{:,cv}],2);
end


%%%%% WOWWW
%plot(sort(sum(sums,2))) %same when you average all patches too!

for cv=1:size(sums, 2)
    myPicScores = reshape(sums(:,cv), numPat, []);
    ali(cv,:) = sum(myPicScores);
end
ali = ali';
[r,c] = find(ali==min(min(ali))); %find the minimum element in ali

picPath = totalResults.imNames{1,c}.testIm{r}; %imNames{1,c} is arbitrary, since eevery orientation has the same picture, 1 doesn't matter  

picInd = [r c ];
%picInd = randomizer(goodInd);
% 
% 
% if strcmp(computer, 'MACI64')
%     dataPath = '/Users/aarslan/Dropbox/Blender Faces Database/Sample_Static_Database'; 
% else
%     dataPath = '/gpfs/home/aarslan/work/PART_staticFaceDataBase';
% end
% 
% origDirs = dir([dataPath '/*orig.png']);
% 
% origPicPaths = [repmat({dataPath}, numel(origDirs),1),{origDirs.name}'];
% origPicPaths = cellfun(@(paths,pics) [paths '/' pics(1:end)], origPicPaths(:,1), origPicPaths(:,2),'uni',false);
% picPath = origPicPaths{picInd};

end

function cellDiff = getDiff(aCell, or)
    cellDiff = abs(aCell.YhatTest - aCell.Ytest(:,or));
end