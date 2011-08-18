function [picPath picInd] = pickBestFace(results, randomizer)

%NUMPAT ASSUMED TO BE 10
numPat = 10;

for or=1:3
    lol(or,:) = cellfun(@getDiff,results(or,:), repmat({or}, 1, 8), 'UniformOutput', 0);
    
end
for cv=1:size(results,2)
    sums(:,cv) = sum([lol{:,cv}],2);
end


%%%%% WOWWW
%plot(sort(sum(sums,2))) %same when you average all patches too!

ehe = reshape(sum(sums,2), numPat, []); %average over patches
[C I] = min(mean(ehe));
goodInd = numel(results{1,1}.YhatTra)/numPat+I;
picInd = randomizer(goodInd);


if strcmp(computer, 'MACI64')
    dataPath = '/Users/aarslan/Dropbox/Blender Faces Database/Sample_Static_Database';                % Path where the Caltech 101 dataset can be found (*** required ***).
else
    dataPath = '/gpfs/home/aarslan/work/PART_staticFaceDataBase';
end

origDirs = dir([dataPath '/*orig.png']);

origPicPaths = [repmat({dataPath}, numel(origDirs),1),{origDirs.name}'];
origPicPaths = cellfun(@(paths,pics) [paths '/' pics(1:end)], origPicPaths(:,1), origPicPaths(:,2),'uni',false);
picPath = origPicPaths{picInd};

end

function cellDiff = getDiff(aCell, or)
    cellDiff = abs(aCell.YhatTest - aCell.Ytest(:,or));
end