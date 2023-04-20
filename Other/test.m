[x,t] = cancer_dataset;
x = x; 
t = t;

numIterations = 1;
numBaseClassifiers = randperm(23) + 2; % Generate 12 random numbers in the range of 3 to 25

trainAccuracies = cell(1, length(numBaseClassifiers));
testAccuracies = cell(1, length(numBaseClassifiers));

optimalEpoch = 1; % optimal epoch
optimalHiddenLayers = 32; % optimal hidden layers

for i = 1:length(numBaseClassifiers) % i.e 12
    baseClassifiers = cell(1, numBaseClassifiers(i));
    trainAccuracies_ind = zeros(1, length(numBaseClassifiers(i)));
    testAccuracies_ind = zeros(1, length(numBaseClassifiers(i)));
    for iter = 1:numIterations
        p = randperm(size(x, 2)); 
        numTrain = round(0.8 * size(x, 2)); 
        X_train = x(:, p(1:numTrain));
        y_train = t(:, p(1:numTrain));
        X_test = x(:, p(numTrain+1:end));
        y_test = t(:, p(numTrain+1:end));

        for j = 1:numBaseClassifiers(i)
            baseClassifier = build_model_function(X_train, y_train, optimalHiddenLayers, optimalEpoch, j);        
            [trainAccuracies_ind(j), testAccuracies_ind(j)] = calculate_accuracy(baseClassifier, X_train, y_train, X_test, y_test);
        end
    end
    trainAccuracies{i} = trainAccuracies_ind; % Update using curly braces
    testAccuracies{i} = testAccuracies_ind; % Update using curly braces
end

disp(trainAccuracies)
