% hmax_cvpr06_run_cal101 - Demo script that performs multiclass classification.
%
% This script runs hmax against the Caltech 101 dataset using the model
% configuration of [Mutch & Lowe 2006].  With minor modifications it could run
% on any similar dataset.
%
% The script does everything in the paper (for the multiclass problem) except
% the final feature selection step.  (The feature selection step doesn't improve
% the classification score much -- it just reduces the number of features needed
% for the model.)
%
% You will have to copy and edit this script to provide the path of the Caltech
% 101 (or similar) dataset on your system.
%
% You might also want to insert "save" commands at appropriate points in this
% script, as it will take some time to run.
%
% See hmax_cvpr06_run_simple for more comments on individual command usage.
%
% See also: hmax_cvpr06_run_simple, hmax_cvpr06_run_uiuc.

%-----------------------------------------------------------------------------------------------------------------------

% while true
%     ans = lower(strtrim(input('All variables will be cleared.  Is this okay (y/n) ? ', 's')));
%     if isempty(ans), continue; end
%     if ans(1) == 'y', break; end
%     if ans(1) == 'n', return; end
% end

function face2dto3d_CSHA

fprintf('\n');

%-----------------------------------------------------------------------------------------------------------------------

% Edit this section to supply script parameters.
if strcmp(computer, 'MACI64')
    dataPath = '/Users/aarslan/Dropbox/Blender Faces Database/Sample_Static_Database';                % Path where the Caltech 101 dataset can be found (*** required ***).
else
    dataPath = '/gpfs/home/aarslan/work/PART_staticFaceDataBase';
end

%addpath('/Users/aarslan/Documents/MATLAB/cns/hmax/face_lfw')
%p = hmax_cvpr06_params_full_lfw;  % Model configuration to use.  Note that this script assumes that the only stage having


numFeatures = 4096;           % Number of S2 features to learn.
numPat = 10;
patSize = 50;
downSamSiz = [9 9];
oriMode = 1;                    % 1= mean orientation in the patch, else = mid point in the patch
numPics = 1500;                  %numel(origPicPaths)
numTrain    = 8;             % Number of training images per category.
maxTest     = inf;            % Maximum number of test images per category.
minSetSize  = 15;              % minimum number of images for a person required to include that person in the classification.
okBands     = 2:2:8;

%-----------------------------------------------------------------------------------------------------------------------

if isempty(dataPath)
    error('you must edit this script to supply the path of the dataset (variable "dataPath")');
end
%-----------------------------------------------------------------------------------------------------------------------
origDirs = dir([dataPath '/*orig.png']);
normDirs = dir([dataPath '/*normals.png']);

origPicPaths = [repmat({dataPath}, numel(origDirs),1),{origDirs.name}'];
origPicPaths = cellfun(@(paths,pics) [paths '/' pics(1:end)], origPicPaths(:,1), origPicPaths(:,2),'uni',false);

normPicPaths = [repmat({dataPath}, numel(normDirs),1),{normDirs.name}'];
normPicPaths = cellfun(@(paths,pics) [paths '/' pics(1:end)], normPicPaths(:,1), normPicPaths(:,2),'uni',false);

%-----------------------------------------------------------------------------------------------------------------------

load filters.mat
         rot = linspace(0, 180, 13);
         RF_siz    = [7:2:39];
         Div       = linspace(4,3.2,numel(RF_siz));%[4:-.05:3.2];
         display('Initializing gabor filters -- full set...');
         %creates the gabor filters use to extract the S1 layer
         [fSiz,filters,c1OL,numSimpleFilters] = init_gabor(rot, RF_siz, Div);

s1vec = cell(numPics, numPat, numel(rot));
truth = cell(numPics, numPat);
origPats  = cell(numPics, numPat);

randomizer = randperm(numPics);

parfor i = 1:numPics
    
    fprintf('%u/%u: computing s1 vector for %s\n', i, numPics, origPicPaths{i});
    
    im = imread(origPicPaths{randomizer(i)});
    imNorm = imread(normPicPaths{randomizer(i)});
    im = rgb2gray(im);
    [ s1] = getS1C1_legacy(im, fSiz,filters,c1OL,numSimpleFilters );
    
    for j=1:numPat
        [rect] = prepPatches(im, patSize);
        origPat = imcrop(im, rect);
        normPat = imcrop(imNorm, rect);
        [x y z] = getAveOri(normPat, oriMode);
        truth{i,j} = [x, y , z];
        s1vec(i,j,:) = vectorizeS1(s1,rect, okBands, downSamSiz);
        origPats{i,j} = imresize(origPat, downSamSiz);
    end
end
size(origPats);
try
save('results_lessBands.mat', 's1vec', 'truth', 'origPats', 'randomizer')
catch
    save('sictim.mat')
end

end

function [ s1vec] = getS1C1_legacy(im, fSiz,filters,c1OL,numSimpleFilters)
rot = linspace(0, 180, 13);
c1ScaleSS = [1:2:18];
RF_siz    = [7:2:39];
c1SpaceSS = [8:2:22];
minFS     = 7;
maxFS     = 39;
minFS     = 7;
maxFS     = 39;
Div       = linspace(4,3.2,numel(RF_siz));%[4:-.05:3.2];

[s1vec] = C1(im, filters, fSiz, c1SpaceSS, c1ScaleSS, c1OL, 1);

end

function [ rect] = prepPatches(im, patSize, varargin)
%patSize: the edge size of a square patch
%varargin is there for optional argument rect (when doing a specified crop on a separate image)

if nargin == 2
    imSize = size(im);
%     if any(patSize > imSize/3)
%         error('patch size too big')
%     end
    invalidCrop = 1;
    while invalidCrop
        rect = [randi(imSize(1)-patSize), randi(imSize(2)-patSize), patSize, patSize];
        pat = imcrop(im, rect);
        if (numel(find(pat == 0)) > (patSize^2)/3) || numel(pat) ~= (patSize+1)^2 %make sure that cropped section doesn't cover a lot of background.
            invalidCrop = 1;
        else
            break
        end

    end
else
    pat = imcrop(im, varargin{1});
    rect = varargin{1};
end

end

function s1vec_lol = vectorizeS1(s1, rect, okBands,downSamSiz)
hmm = cellfun(@(x)[x{:}], s1, 'UniformOutput', 0 );
for band=1:numel(hmm)
    for ori=1:numel(hmm{band})/2
         pat = hmm{band}{ori} + hmm{band}{ori+numel(hmm{band})/2};
        s1vec{ori}{band} = imresize(imcrop(pat, rect), downSamSiz);
    end
end

for band=1:numel(okBands)
    for ori=1:numel(hmm{band})/2
        temp = cellfun(@(a)reshape(a, prod(size(a)),1 ), s1vec{ori}, 'UniformOutput', 0);
        eheh =[temp{okBands}];
        s1vec_lol{ori}= (reshape(eheh, numel(eheh), 1)); %%%do I need to transpose during reshape??
    end
end


end


function [x y z] = getAveOri(normPat, Mode)
im = double(normPat);
x = (im(:, :, 1) - 128)/127;  % R positive values point right
y = (im(:, :, 2) - 128)/127;  % G positive values point up
z = (im(:, :, 3) - 128)/127;  % B positive values point toward the camera

if Mode == 1 %take the mean orientation
    x = mean(mean(x));
    y = mean(mean(y));
    z = mean(mean(z));
else % take the central orientation
    dims = floor(size(normPat));
    x = x(dims(1), dims(2));
    y = y(dims(1), dims(2));
    z = z(dims(1), dims(2));
end

end

function doRegression(normOri , trainOri)
model = svmtrain(normOri , trainOri, '-s 3 -t 2')
[y_hat,  Acc,projection] = svmpredict(valdata.y,valdata.X, model);  
end