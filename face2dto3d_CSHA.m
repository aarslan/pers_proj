
function face2dto3d_CSHA(runMode, varargin)
%runMode: extract patches for multiple face: 0
%         extract patches for one face     : 1

if ~exist('runMode', 'var')
    runMode = 0;
end

mystream = RandStream('mt19937ar','Seed',sum(100*clock));
RandStream.setDefaultStream(mystream);

%-----------------------------------------------------------------------------------------------------------------------

% Edit this section to supply script parameters.
if strcmp(computer, 'MACI64')
    dataPath = '/Users/aarslan/Dropbox/Blender Faces Database/Sample_Static_Database';                % Path where the Caltech 101 dataset can be found (*** required ***).
else
    dataPath = '/gpfs/home/aarslan/work/PART_staticFaceDataBase';
end
if ~exist(dataPath)
    error('I couldnt find anything in the datapaths provided');
end

%addpath('/Users/aarslan/Documents/MATLAB/cns/hmax/face_lfw')
%p = hmax_cvpr06_params_full_lfw;  % Model configuration to use.  Note that this script assumes that the only stage having


PAR.numPat      = 25;
PAR.downSamSiz  = 9;
PAR.oriMode     = 1;                    % 1= mean orientation in the patch, else = mid point in the patch
PAR.numPics     = 1000;                  %numel(origPicPaths)
PAR.okBands     = 2:2:8;
PAR.regularPatching = 0;
PAR.patMode = 1;
PAR.runMode = runMode;

pSpec.imSize = 0; %will be written later on
pSpec.patSize = 50;
pSpec.patMode = 0; %patch mode: 0 random; 1 regular spaced.
pSpec.numPat = PAR.numPat;

PAR.pSpec = pSpec;

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
RF_siz    = 7:2:39;
Div       = linspace(4,3.2,numel(RF_siz));%[4:-.05:3.2];
display('Initializing gabor filters -- full set...');
%creates the gabor filters use to extract the S1 layer
[fSiz,filters,c1OL,numSimpleFilters] = init_gabor(rot, RF_siz, Div);



if runMode
    PAR.numPics     = 1;  
    PAR.pSpec.patMode = 1; %patch mode: 0 random; 1 regular spaced.
    load regr_results.mat
    [patOfPic randomizer] = pickBestFace(results);
    clear results
end

origPats  = cell(PAR.numPics, PAR.numPat);
randomizer = randperm(numel(origPicPaths));

imData = struct('name', repmat({'-'}, PAR.numPics, 1), 'truth', [], 's1vec',[], 'origPats', [], 'actRects', [] );

PAR.pSpec.imSize = size(imread(origPicPaths{randomizer(1)}));
tic
parfor ii = 1:PAR.numPics
    
    fprintf('%u/%u: computing s1 vector for %s\n', ii, PAR.numPics, origPicPaths{randomizer(ii)});
    
    im = imread(origPicPaths{randomizer(ii)});
    imNorm = imread(normPicPaths{randomizer(ii)});
    im = rgb2gray(im);
    %PAR.pSpec.imSize = size(im);
    [ s1] = getS1C1_legacy(im, fSiz,filters,c1OL,numSimpleFilters );
    
    [rects] = prepAllPatches(PAR.pSpec);
    [truth s1vec origPats actRects] = prepValidPatches(im, imNorm, s1, PAR, rects);
    imData(ii).truth    = truth;
    imData(ii).s1vec    = s1vec;
    imData(ii).origPats = origPats;
    imData(ii).actRects  = actRects;
    imData(ii).name = origPicPaths{randomizer(ii)};
    %display([num2str(sum(strcmp('-', {imData.name}))) ' images to go'])
end
toc

if ~runMode
    matName = 'patchesAll.mat';
else
    [~, name, ext] = fileparts(origPicPaths{1}); 
    matName = [name '_patches.mat' ];
end

PAR.origDirs = origDirs;
PAR.normDirs = normDirs;


try
    save(matName, 'imData', 'PAR', '-v7.3')
catch
    save('sictim.mat')
end

end

function [ s1vec] = getS1C1_legacy(im, fSiz,filters,c1OL,~)
c1ScaleSS = 1:2:18;
c1SpaceSS = 8:2:22;
[s1vec] = C1(im, filters, fSiz, c1SpaceSS, c1ScaleSS, c1OL, 0);

end

function [ rects] = prepAllPatches(pSpec)
imSize  = pSpec.imSize;
patSize = pSpec.patSize;
patMode = pSpec.patMode; %patch mode: 0 random; 1 regular spaced.
numPat  = pSpec.numPat;
switch patMode
    case 0
        numPat  = pSpec.numPat*5; %create 5 times more patches than necessary
        rects = zeros(numPat,4);
        rects(:,2) = randi(imSize(1) - patSize,numPat,1);   %%%WOT, WHY DO I HAE TO CROSS W and H?
        rects(:,1) = randi(imSize(2) - patSize,numPat,1);
    case 1
        rects = zeros(numPat,4);
        facts = imSize/min(imSize); %create a series of pairs to find the ratios for edges
        veli = repmat(facts, [], prod(imSize));
        ali= repmat(1:prod(imSize), [], 2)';
        
        axsDims = ceil((find(prod(ali.*veli,2) <= PAR.numPat,1, 'last')) *facts);
        
        if (axsDims(2)+1)* axsDims(1) <= PAR.numPat
            axsDims = [axsDims(1) axsDims(2)+1];
        elseif (axsDims(1)+1)* axsDims(2) <= PAR.numPat %check if we can increase the number a bit more
            axsDims = [axsDims(1)+1 axsDims(2)];
        end
        
        [x y] = meshgrid(linspace(1, imSize(1)-patSize, axsDims(1)), linspace(1, imSize(2)-patSize, axsDims(2))); % create xy pairs
        finNumP = prod(axsDims);
        
        rects(1:finNumP,1:2) = [x(1:finNumP); y(1:finNumP)]'; %put in rect & trim
        
        if finNumP ~= PAR.numPat
            warning(sprintf('you wanted %d patches, but I could fit in only %d', PAR.numPat,  finNumP ))
        end
end
rects(:,3) = patSize;
rects(:,4) = patSize;
end


function  [truth s1vec origPats actRect] = prepValidPatches(im, imNorm, s1, PAR, rects)
numPat      = size(rects, 1);
truth       = zeros(numPat, 3);
origPats    = zeros(numPat, PAR.downSamSiz, PAR.downSamSiz);
actRects    = zeros(numPat, 4);
s1vec       = cell(numPat,1);
% DONT KNOW THE SIZE OF s1vec !!!!!!!!!!!!!!!
%warning('figure out a way to calculate s1vec size and preallocate')

cnt = 1;
joff=0;
for j=1:numPat
    
    invalidCrop = 1;
    while invalidCrop
        rect = rects(j+joff,:);
        j+joff
        origPat = imcrop(im, rect);
        joff = joff+1;
        if ~((numel(find(origPat == 0)) > (PAR.pSpec.patSize^2)/3) || numel(origPat) ~= (PAR.pSpec.patSize+1)^2) %make sure that cropped section doesn't cover a lot of background.
            normPat = imcrop(imNorm, rect);
            [x y z] = getAveOri(normPat, PAR.oriMode);
            truth(cnt,:) = [x y z];
            s1vec{cnt} = vectorizeS1(s1,rect, PAR);
            if sum(s1vec{cnt}{1}) == 0
                display('eaja')
            end
            origPats(cnt,:,:) = imresize(origPat, [PAR.downSamSiz, PAR.downSamSiz] );
            actRects(cnt,:) = rect;
            cnt = cnt+1;
            invalidCrop = 0;
            break
        end
    end
    if cnt-1 == PAR.pSpec.numPat
        break
    end
end
cnt = cnt-1;
truth    = truth(1:cnt,:);
s1vec    = s1vec(1:cnt); 
origPats = origPats(1:cnt,:,:);
actRect  = actRects(1:cnt,:);
end


function s1vec_lol = vectorizeS1(s1, rect, PAR)
hmm = cellfun(@(x)[x{:}], s1, 'UniformOutput', 0 );
for band=1:numel(hmm)
    for ori=1:numel(hmm{band})/2
        pat = hmm{band}{ori} + hmm{band}{ori+numel(hmm{band})/2};
        s1vec{ori}{band} = imresize(imcrop(pat, rect), [PAR.downSamSiz PAR.downSamSiz]);
    end
end

for band=1:numel(PAR.okBands)
    for ori=1:numel(hmm{band})/2
        temp = cellfun(@(a)reshape(a, prod(size(a)),1 ), s1vec{ori}, 'UniformOutput', 0);
        eheh =[temp{PAR.okBands}];
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
    dims = floor(size(normPat)/2);
    x = x(dims(1), dims(2));
    y = y(dims(1), dims(2));
    z = z(dims(1), dims(2));
end

end

function doRegression(normOri , trainOri)
model = svmtrain(normOri , trainOri, '-s 3 -t 2')
[y_hat,  Acc,projection] = svmpredict(valdata.y,valdata.X, model);
end