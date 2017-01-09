%% choose dataset
%load('M:\Desktop\forest.csv');
%load('M:\Desktop\optdigits.data');
%load('M:\Desktop\balance-scale.csv');
%load('M:\Desktop\winequality-red.csv');
%load('M:\Desktop\hw4\hw4\q1\Iris.mat');
load('M:\Desktop\breast-cancer-wisconsin.csv');

%assigning the dataSet to the desinated vairables
%For Forest type mapping dataset
if(exist('forest','var'))
    label = forest(:,1);
    dataSet = forest(:,2:size(forest,2));
end;

%For Optical recognition of handwritten digits - original testing data set
if(exist('optdigits','var'))
    label = optdigits(:,65);
    dataSet = optdigits(:,1:64);
end;


%For balance scale data set
if(exist('balance_scale','var'))
    label = balance_scale(:,1);
    dataSet = balance_scale(:,2:size(balance_scale,2));
end;

%For winequality-red data set
if(exist('winequality_red','var'))
    label = winequality_red(:,size(winequality_red,2));
    dataSet = winequality_red(:,1:size(winequality_red,2)-1);
end;

%For Iris.mat
if(exist('dataSet','var'))
    label = label;
    dataSet = dataSet;
end;

%For breast cancer wisconsin - original data set
if(exist('breast_cancer_wisconsin','var'))
    label = breast_cancer_wisconsin(:,11);
    dataSet = breast_cancer_wisconsin(:,2:10);
end;

% parameter
epsilon = 0.001;       % small value for numerical issue
C = 1;                  % regularization constant
T = 50;               % iteration times

sigma = 1;
iters = zeros(1,10);    % iteration times to finish
class = unique(label);  % unique list of classes
totalSamples = size(dataSet,1);        % Total number of samples
ratios = [0.1; 0.2; 0.3; 0.4; 0.5];    % training sample ratio: 10%, 20%, 30%, 40%, 50%
acc = zeros(size(ratios,1),10);      % classification accuracy


%% implementation
for ratio = 1 : size(ratios,1)
    for t = 1 : 10
        % Generating Training Dataset and Testing DataSet
        m = round(totalSamples * ratios(ratio,1));   % number of samples in the training Set
        total = 1: totalSamples;
        trainIndices = randperm(totalSamples,m);      % random indices generation to pick samples randomly for training set
        testIndices = setdiff(total, trainIndices);                     % the indices for the testing data set
        testSet = dataSet(testIndices,:);
        testLabel = label(testIndices,:);
        trainSet = dataSet(trainIndices,:);
        trainLabel = label(trainIndices,:);
        
        % initialization and computation of f value for training and testing data set
        ftrain = zeros(size(trainSet,1));
        ftest = zeros(size(testSet,1),size(trainSet,1));
        
        for i = 1 : size(trainSet,1)
            for j = 1 : size(trainSet,1)
                ftrain(i,j) = exp( - power(pdist2(trainSet(i,:),trainSet(j,:)),2) / (2 * sigma * sigma));
            end;
        end;
        
        for i = 1 : size(testSet,1)
            for j = 1 : size(trainSet,1)
                ftest(i,j) = exp( - power(pdist2(testSet(i,:),trainSet(j,:)),2) / (2 * sigma * sigma));
            end;
        end;
        
        % initialization
        for c = 1 : size(class,1)
            trainLabel = label(trainIndices,:);
            
            trainLabel(trainLabel ~= class(c,1)) = -1;
            trainLabel(trainLabel == class(c,1)) = 1;
            
            x = ftrain';
            y = trainLabel';
            
            numOfSamples = size(x,2);
            dim = size(x,1);
            w = zeros(dim, 1);
            
            % offset "b"
            w_b = [0; w];
            x = [ones(1,numOfSamples); x];
            dim = dim + 1;
            
            iter = 0;
            while(iter < T)
                iter = iter + 1;
                score = 1 - y .* (w_b'*x);
                
                z = -score;
                
                index = find(score > 0);
                z(index) = z(index) * (-1);
                z = max(z, epsilon);
                
                % iterations for w
                A = zeros(dim);
                d = zeros(dim, 1);
                for k = 1:dim
                    for j = 1:dim
                        A(j,k) = C * sum(1 ./ (2*z) .* x(k,:) .* x(j,:));
                        if j == k && j ~= 1
                            A(j,k) = A(j,k) + 1;
                        end;
                    end;
                    d(k) = C * sum((1+z) ./ (2*z) .* y .* x(k,:));
                end;
                w_new = A \ d;  % linear system
                w_b = w_new;
            end;
            theta{c} = w_new;
        end;
        
        % classification accuracy
        numOfTestSamples = size(ftest, 1);
        pLabel = zeros(numOfTestSamples, size(class,1));
        for c = 1 : size(class,1)
            SoftLabel = theta{c}' * [ones(1,numOfTestSamples); ftest'];
            tLabel = zeros(numOfTestSamples, 1);
            tLabel(SoftLabel < 0) = -1;
            tLabel(SoftLabel > 0) = 1;
            pLabel(:,c) = tLabel;
        end;
        [maxy,predictLabel] = max(pLabel,[],2);
        
        acc(ratio,t) = length(find(class(predictLabel,1) - testLabel == 0)) / numOfTestSamples;
        fprintf('Training data (%d%%), Trial #%d, Accuracy = %.4f.\n',100*ratios(ratio),t,acc(ratio,t) * 100);
    end;
end;