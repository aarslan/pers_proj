
function doRegression(origPats, s1vec, truth)

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
        lello{count} = repmat({dataMat(count,:)},1,3);
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

%truthMat = zscore(truthMat,0,1);


X = dataMat(trainInd,:);
Y =  truthMat(trainInd,:);
Xtest = dataMat(testInd,:);
Ytest = truthMat(testInd,:);

[B, bint, r, rint, STATS]=regress(Y(:,1),X, 0.001);
Yhat=X*B;
corr(Y(:,1), Yhat)

Yhat=Xtest*B;
'TEST ERROR'
corr(Ytest(:,1), Yhat)

errz = getMSE(Ytest(:,1), Yhat)

% IMRI's solution
%beta = Y\X;
%Yimri = Xtest*beta';


%singe value
betaTra = Y(:,1)\X;
YTra = X*betaTra';
errzTra = getMSE(Y,YTra);
Ymean = mean(Y);

betaTest = Y(:,1)\X;
YTest = Xtest*betaTest';
errzTest = getMSE(Y,YTest);
Ymean = mean(Y);



errz = getMSE(Ytest,Yimri);
Ymean = mean(Ytest);



model = initlssvm(X,Y,type,gam, sig2,'RBF_kernel');
model = trainlssvm(model);

% [alpha,b] = trainlssvm({X, Y, type, gam, sig2, 'RBF_kernel','preprocess'});
% Ytest = simlssvm({dataMat(trainInd,end-1000:end), truthMat(trainInd,:),type,gam,sig2,'RBF_kernel'},{alpha,b},dataMat(testInd,end-1000:end));
% 



% d = data( dataMat(trainInd,end-1000:end), truthMat(trainInd,:));
% d2 = data( dataMat(testInd,end-1000:end), truthMat(testInd,:));
% 
% a=multi_reg(svr('C=10'))
% [r,a]=train(a,d);
% [sol]=test(a,d);

%model = svmtrain( truthMat(trainInd,:), dataMat(trainInd,:) , '-s 3 -t 2');


%[y_hat,  Acc,projection] = svmpredict(truthMat(testInd,:), dataMat(testInd,:), model);  

end


function  params = makeParam
params.ratio = 0.75;
params.totalPics = 50; %total number of pics to use
end

function errz = getMSE(dact,dpred)
errz = mean(dact - dpred).^2;
end