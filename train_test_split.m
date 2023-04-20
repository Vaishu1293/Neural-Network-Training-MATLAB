function [X_train, y_train, X_test, y_test] = train_test_split(x, t) 
p = randperm(size(x, 2));
numTrain = round(0.8 * size(x, 2));
numTest = size(x, 2) - numTrain;
X_train = x(:, p(1:numTrain));
y_train = t(:, p(1:numTrain));
X_test = x(:, p(numTest+1:end));
y_test = t(:, p(numTest+1:end));