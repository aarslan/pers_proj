
function [res resPerm imNames p] = doRegression(imData, PAR, or, varargin)

p.ratio = 0.75; %training vs test ratio
p.totalPics = NaN; %total number of pics to use
p.matRegress = 1; %use matlab's built-in regression
p.singlePic = 0; %
p.nrmz = 1; %normalize the features between 0-1?
p.doPerm = 0; % 1 if you also want to regress on random labels

if isnan(p.totalPics)
    p.totalPics = PAR.numPics;
end
p.numPat = PAR.pSpec.numPat;
p.NumObs = p.totalPics*p.numPat;

if nargin == 4
    rand('state', varargin{1})
end
randInd = randperm(p.totalPics);
sep = ceil(p.totalPics * p.ratio);
trainInd = randInd(1:sep);
testInd = randInd(sep+1:end);
trainIndExpanded = reshape(repmat(trainInd, p.numPat, 1), 1, []);
testIndExpanded = reshape(repmat(testInd, p.numPat,1), 1, []);

%construct a 2D data array with these dimensions:
% numObservation*numpatches X numOrientation*numPixels
p.numPix = numel(imData(1,1).s1vec{1,1}{1,1});
p.numOri = numel(imData(1,1).s1vec{1,1});
dataMat  = zeros(p.NumObs, p.numOri*p.numPix);
truthMat = zeros(p.NumObs, 3);

count = 1;
for im=1:p.totalPics;
    for pat=1:p.numPat
        for ori=1:p.numOri
            if p.nrmz
                dataMat(count, (ori-1)*p.numPix+1: (ori*p.numPix)) = mat2gray(imData(im).s1vec{pat}{ori});
            else
                dataMat(count, (ori-1)*p.numPix+1: (ori*p.numPix)) = imData(im).s1vec{pat}{ori};
            end
        end
        truthMat(count,:) = imData(im).truth(pat,:);
        count = count+1;
    end
    imNames{im} = imData(im).name; %%%VERIYI KARMAN CORMAN EDERKEN ARKA ARKAYA GELEN PATCH'LERI BOLUYORUZ, SONRA NASIL BIR ARAYA GELECEKLER???
    %belki bir resimden tum patch'leri grup grup scramble etmek daha iyi,
    %hem trainin ve test arasinda bolunmemis olurlar.
end

for ob=1:3
    truthMat(:,ob) = mat2gray(truthMat(:,ob));    
end
    
clear s1vec truth

gam = 10;
sig2 = 0.2;
type = 'function estimation';

dataMat = [ones(size(dataMat,1),1) dataMat]; %added bias term

X = dataMat(trainIndExpanded,:);
Y =  truthMat(trainIndExpanded,:);

if ~p.singlePic
Xtest = dataMat(testIndExpanded,:);
Ytest = truthMat(testIndExpanded,:);
else
    load results_oneFace.mat
    display('loaded features for single pic');
end

if p.matRegress
    scrm = randperm(size(Y,1));
    res     = cannedReg(Y, Ytest, X, Xtest, or);
    if p.doPerm
    resPerm = cannedReg(Y, Ytest, X(scrm, :), Xtest, or);
    end

    % else
%     [alpha,B] = trainlssvm({X, Y, type, gam, sig2, 'RBF_kernel','preprocess'});
%     YhatTest = simlssvm({X, Y,type,gam,sig2,'RBF_kernel'},{alpha,B},Xtest); 
%     YhatTra = simlssvm({X, Y,type,gam,sig2,'RBF_kernel'},{alpha,B},X); 
end

imNames.trainIm = {imData(trainInd).name};
imNames.testIm = {imData(testInd).name};
%save('regressionResults.mat', 'res','resPerm', 'p', 'imNames')
end

function res = cannedReg(Y, Ytest, X, Xtest, or)
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
    
    res.Ymean = mean(Y);
    res.CorrTra = corr(Y(:,or), YhatTra);
    display(sprintf('training corr: %f', res.CorrTra))
    
    res.CorrTest = corr(Ytest(:,or), YhatTest);
    display(sprintf('test corr: %f', res.CorrTest))
    res.B = B;
    res.YhatTra = YhatTra;
    res.YhatTest = YhatTest;
    res.Ytest = Ytest;
end

function errz = getMSE(dact,dpred)
errz = mean(dact - dpred).^2;
end