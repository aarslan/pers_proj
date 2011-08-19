
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


numPat = 1000;
patSize = 50;
downSamSiz = [9 9];
oriMode = 1;                    % 1= mean orientation in the patch, else = mid point in the patch
numPics = 1;                  %numel(origPicPaths)
okBands     = 2:2:8;
regularPatching = 0;
patMode = 1;

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

s1vec = cell(numPics, numPat, numel(rot));
truth = cell(numPics, numPat);
origPats  = cell(numPics, numPat);

randomizer = randperm(numel(origPicPaths));

if regularPatching
    results = varargin{1};
    randomizer = varargin{2};
[patOfPic randomizer] = pickBestFace(results, randomizer);
end


for i = 1:numPics
    
    fprintf('%u/%u: computing s1 vector for %s\n', i, numPics, origPicPaths{randomizer(i)});
    
    im = imread(origPicPaths{randomizer(i)});
    imNorm = imread(normPicPaths{randomizer(i)});
    im = rgb2gray(im);
    [ s1] = getS1C1_legacy(im, fSiz,filters,c1OL,numSimpleFilters );
    
    pSpec.imSize = size(im);
    pSpec.patSize = patSize;
    pSpec.patMode = patMode; %patch mode: 0 random; 1 regular spaced.
    pSpec.numPat = numPat;
    
    [rects] = prepAllPatches(pSpec);
    numPat = size(rects, 1);
    
    for j=1:numPat
        origPat = imcrop(im, rect);
        normPat = imcrop(imNorm, rect);
        [x y z] = getAveOri(normPat, oriMode);
        truth{i,j} = [x, y , z];
        s1vec(i,j,:) = vectorizeS1(s1,rect, okBands, downSamSiz);
        origPats{i,j} = imresize(origPat, downSamSiz);
        rects{i,j} = rect;
    end
    clear s1
end
size(origPats);
try
    save('results_oneFace.mat', 's1vec', 'truth', 'origPats', 'randomizer', 'rects')
catch
    save('sictim.mat')
end

end

function [ s1vec] = getS1C1_legacy(im, fSiz,filters,c1OL,~)
c1ScaleSS = 1:2:18;
c1SpaceSS = 8:2:22;
[s1vec] = C1(im, filters, fSiz, c1SpaceSS, c1ScaleSS, c1OL, 1);

end

function [ rects] = prepAllPatches(pSpec)
imSize  = pSpec.imSize;
patSize = pSpec.patSize;
patMode   = pSpec.patMode; %patch mode: 0 random; 1 regular spaced.
numPat  = pSpec.numPat;

rects = zeros(numPat,4);

switch patMode
    case 0
        rects(:,1) = randi(imSize(1) - patSize,numPat,1);
        rects(:,2) = randi(imSize(2) - patSize,numPat,1);
    case 1
        
        facts = imSize/min(imSize); %create a series of pairs to find the ratios for edges
        veli = repmat(facts, [], prod(imSize));
        ali= repmat(1:prod(imSize), [], 2)';
        
        axsDims = ceil((find(prod(ali.*veli,2) <= numPat,1, 'last')) *facts);
        
        if (axsDims(2)+1)* axsDims(1) <= numPat
            axsDims = [axsDims(1) axsDims(2)+1];
        elseif (axsDims(1)+1)* axsDims(2) <= numPat %check if we can increase the number a bit more
            axsDims = [axsDims(1)+1 axsDims(2)];
        end
        
        [x y] = meshgrid(linspace(1, imSize(1)-patSize, axsDims(1)), linspace(1, imSize(2)-patSize, axsDims(2))); % create xy pairs
        finNumP = prod(axsDims);
        
        rects(1:finNumP,1:2) = [x(1:finNumP); y(1:finNumP)]'; %put in rect & trim
        rects = rects(1:finNumP,:);
        if finNumP ~= numPat
            warning(sprintf('you wanted %d patches, but I could fit in only %d', numPat,  finNumP ))
        end
end

rects(:,3) = patSize;
rects(:,4) = patSize;

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
end

end


function  ali = preValidPatches(im, imNorm, rects)

    for j=1:numPat
        origPat = imcrop(im, rect);
        normPat = imcrop(imNorm, rect);
        [x y z] = getAveOri(normPat, oriMode);
        truth{i,j} = [x, y , z];
        s1vec(i,j,:) = vectorizeS1(s1,rect, okBands, downSamSiz);
        origPats{i,j} = imresize(origPat, downSamSiz);
        rects{i,j} = rect;
    end

end

function [thisRect rects] = prepRegularPatches(regPar, it)


pro=regPar.numPat+1;
w = regPar.imSize(1);
h = regPar.imSize(2);
y=1;
while pro >= regPar.numPat
    pro = (700-y)*(600-y);
    y=y+1;
end
wArr = ceil(linspace(regPar.patSize/2, w-regPar.patSize/2, w-y)); %use meshgrid
hArr = ceil(linspace(regPar.patSize/2, h-regPar.patSize/2, h-y));
cnt = 1;
for ws=1: w-y
    for hs=1: h-y
        rects{cnt} = [wArr(ws) hArr(hs)];
        cnt=cnt+1;
    end
end
thisRect = [rects{it} regPar.patSize regPar.patSize];
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