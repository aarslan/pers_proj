
function res = doRegression(origPats, s1vec, truth, or)

p = makeParam;

randInd = randperm(p.totalPics*size(s1vec,2));
sep = ceil(p.totalPics *size(s1vec,2)* p.ratio);
trainInd = randInd(1:sep);
testInd = randInd(sep+1:end);

%construct a 2D data array with these dimensions:
% numObservation*numpatches X numOrientation*numPixels
numPix = numel(s1vec{1,1,1});
dataMat  = zeros(p.totalPics*size(s1vec,2), size(s1vec,3)*numPix);
truthMat = zeros(p.totalPics*size(s1vec,2), 3);

count = 1;
for im=1:p.totalPics;
    for pat=1:size(s1vec,2)
        for ori=1:size(s1vec,3)
            dataMat(count, (ori-1)*numPix+1: (ori*numPix)) = mat2gray(s1vec{im, pat, ori});
            
        end
        truthMat(count,:) = truth{im,pat};
        count = count+1;
    end
    
end

for ob=1:3
    truthMat(:,ob) = mat2gray(truthMat(:,ob));    
end
    
clear s1vec truth

gam = 10;
sig2 = 0.2;
type = 'function estimation';

dataMat = [ones(size(dataMat,1),1) dataMat]; %added bias term

X = dataMat(trainInd,:);
Y =  truthMat(trainInd,:);

if ~p.singlePic
Xtest = dataMat(testInd,:);
Ytest = truthMat(testInd,:);
else
    load results_oneFace.mat
    display('loaded features for single pic');
end

if p.matRegress
    [B, bint, r, rint, STATS]=regress(Y(:,or),X, 0.001);
    regressStats.bint = bint;
    regressStats.r = r;
    regressStats.rint = rint;
    regressStats.STATS = STATS;
    res.regressStats = regressStats;
    
    YhatTest=Xtest*B;
    YhatTra=X*B;
    res.MSE = getMSE(Ytest(:,or), YhatTest);
    display(sprintf('mean sq error: %f', res.MSE))
else
    [alpha,B] = trainlssvm({X, Y, type, gam, sig2, 'RBF_kernel','preprocess'});
    YhatTest = simlssvm({X, Y,type,gam,sig2,'RBF_kernel'},{alpha,B},Xtest); 
    YhatTra = simlssvm({X, Y,type,gam,sig2,'RBF_kernel'},{alpha,B},X); 
end

res.Ymean = mean(Y);

res.CorrTra = corr(Y(:,or), YhatTra);
display(sprintf('training error: %f', res.CorrTra))

res.CorrTest = corr(Ytest(:,or), YhatTest);
display(sprintf('test error: %f', res.CorrTest))


res.YhatTra = YhatTra;
res.YhatTest = YhatTest;
res.Ytest = Ytest;
end


function  params = makeParam
params.ratio = 0.95;
params.totalPics = 1; %total number of pics to use
params.matRegress = 1;
params.singlePic = 1;
end

function errz = getMSE(dact,dpred)
errz = mean(dact - dpred).^2;
end