clear all
close all
clc
load data.mat;
data=data;
[n,m]=size(data);
rows=(1:n);
test_count=floor((1/6)*n);
sum_ens=0;sum_result=0;
test_rows=randsample(rows,test_count);
train_rows=setdiff(rows,test_rows);
test=data(test_rows,:);
train=data(train_rows,:);
xtest=test(:,1:m-1);
ytest=test(:,m);
xtrain=train(:,1:m-1);
ytrain=train(:,m);

%-----------svm------------------
svm=svm1(xtest,xtrain,ytrain);

%-------------random forest---------------
rforest=randomforest(xtest,xtrain,ytrain);

%-------------decision tree---------------
DT=DTree(xtest,xtrain,ytrain);

%---------------bayesian---------------------
NBModel = NaiveBayes.fit(xtrain,ytrain, 'Distribution', 'kernel');
Pred = NBModel.predict(xtest);
dt=Pred;

%--------------KNN----------------
knnModel=fitcknn(xtrain,ytrain,'NumNeighbors',4);
     pred=knnModel.predict(xtest);
     sk=pred;


% Get the size of x
[m, n] = size(x);

% Display the number of rows and columns in A
fprintf('Number of rows in x: %d\n', m);
fprintf('Number of columns in x: %d\n', n);

% Get the size of t
[a, b] = size(t);

% Display the number of rows and columns in A
fprintf('Number of rows in t: %d\n', a);
fprintf('Number of columns in t: %d\n', b);


