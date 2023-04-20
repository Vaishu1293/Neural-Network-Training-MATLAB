% STEP 1: Define the base classifier with optimum epoch and hidden layer parameters
optimalEpoch = 32; % Example value for optimal epoch
optimalHiddenLayers = 32; % Example value for optimal hidden layers

% Create 15 copies of the base classifier
numBaseClassifiers = 15;
baseClassifiers = cell(1, numBaseClassifiers);

for i = 1:numBaseClassifiers
    % Create a new instance of the base classifier
    baseClassifier = patternnet(optimalHiddenLayers); % Replace with the appropriate function to create your base classifier

    % Set the random number seed for reproducibility
    rng(i); % Set the random seed based on loop index i
    
    % Randomly initialize the weights of the base classifier
    baseClassifier = init(baseClassifier);
    
    % Set the epoch parameter of the base classifier
    baseClassifier.trainParam.epochs = optimalEpoch; % Set the epoch parameter to the optimal epoch

    % Add the base classifier to the ensemble
    baseClassifiers{i} = baseClassifier;
end

% Model accuracies for individual models
train_acc = [0.97, 0.97, 0.97, 0.97, 0.97]; % train accuracies of individual models
test_acc = [0.95, 0.95, 0.95, 0.95, 0.95]; % test accuracies of individual models

% Bar graph for model accuracies
figure;
bar([train_acc; test_acc]');
legend('Train Accuracy', 'Test Accuracy');
xlabel('Model Index');
ylabel('Accuracy');
title('Model Accuracies for Individual Models');
xticks(1:5);
xticklabels({'Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5'});
grid on;

%Step: 2-  Train and evaluate each base classifier separately
trainAccuracies = zeros(1, numBaseClassifiers);
testAccuracies = zeros(1, numBaseClassifiers);

for i = 1:numBaseClassifiers
    % Train the base classifier on the training data
    baseClassifier = train(baseClassifiers{i}, X_train, y_train); % Replace X_train and y_train with your training data

    % Make predictions using the trained base classifier on the training data
    trainPredictions = baseClassifier(X_train); 

    % Calculate the accuracy of the base classifier on the training data
    trainAccuracies(i) = sum(y_train == round(trainPredictions)) / length(y_train);

    % Make predictions using the trained base classifier on the test data
    testPredictions = baseClassifier(X_test);

    % Calculate the accuracy of the base classifier on the test data
    testAccuracies(i) = sum(y_test == round(testPredictions)) / length(y_test);
end

% Aggregate the predictions of the base classifiers to make ensemble predictions
ensemblePredictions = zeros(size(testPredictions)); % Initialize the ensemble predictions

for i = 1:numBaseClassifiers
    % Make predictions using the i-th base classifier on the test data
    basePredictions = baseClassifiers{i}(X_test);
    
    % Aggregate the base predictions by taking the average
    ensemblePredictions = ensemblePredictions + basePredictions / numBaseClassifiers;
end

% Calculate the accuracy of the ensemble predictions on the test data
ensembleAccuracy = sum(y_test == round(ensemblePredictions)) / length(y_test);


% Initialize arrays to store individual classifier accuracies
trainAccuracies = zeros(1, numBaseClassifiers);
testAccuracies = zeros(1, numBaseClassifiers);

% Train and evaluate each base classifier separately
for i = 1:numBaseClassifiers
    % Train the i-th base classifier on the training data
    baseClassifier = train(baseClassifiers{i}, X_train, y_train); % Replace X_train and y_train with your training data

    % Make predictions using the trained base classifier on the training data
    trainPredictions = baseClassifier(X_train); 

    % Calculate the accuracy of the i-th base classifier on the training data
    trainAccuracies(i) = sum(y_train == round(trainPredictions)) / length(y_train);

    % Make predictions using the trained base classifier on the test data
    testPredictions = baseClassifier(X_test);

    % Calculate the accuracy of the i-th base classifier on the test data
    testAccuracies(i) = sum(y_test == round(testPredictions)) / length(y_test);
end

% STEP - 3: Aggregate the predictions of the base classifiers using majority voting to make ensemble predictions
ensemblePredictions = zeros(size(testPredictions)); % Initialize the ensemble predictions

for i = 1:numBaseClassifiers
    % Make predictions using the i-th base classifier on the test data
    basePredictions = baseClassifiers{i}(X_test);
    
    % Convert base predictions to binary class labels (0 or 1)
    binaryPredictions = round(basePredictions);
    
    % Aggregate the base predictions by taking the majority vote
    ensemblePredictions = ensemblePredictions + binaryPredictions;
end

% Convert ensemble predictions to binary class labels (0 or 1)
ensemblePredictions = round(ensemblePredictions / numBaseClassifiers);

% Calculate the accuracy of the ensemble predictions on the test data
ensembleAccuracy = sum(y_test == ensemblePredictions) / length(y_test);

% STEP: 5- Initialize an array to store ensemble accuracies for different numbers of base classifiers
ensembleAccuracies = zeros(1, numEnsembles);

% Loop over different numbers of base classifiers
for i = 1:numEnsembles
    % Select the i-th number of base classifiers
    numBaseClassifiers = baseClassifierCounts(i);
    
    % Initialize arrays to store individual classifier accuracies
    trainAccuracies = zeros(1, numBaseClassifiers);
    testAccuracies = zeros(1, numBaseClassifiers);
    
    % Train and evaluate each base classifier separately
    for j = 1:numBaseClassifiers
        % Train the j-th base classifier on the training data
        baseClassifier = train(baseClassifiers{j}, X_train, y_train); % Replace X_train and y_train with your training data
        
        % Make predictions using the trained base classifier on the training data
        trainPredictions = baseClassifier(X_train);
        
        % Calculate the accuracy of the j-th base classifier on the training data
        trainAccuracies(j) = sum(y_train == round(trainPredictions)) / length(y_train);
        
        % Make predictions using the trained base classifier on the test data
        testPredictions = baseClassifier(X_test);
        
        % Calculate the accuracy of the j-th base classifier on the test data
        testAccuracies(j) = sum(y_test == round(testPredictions)) / length(y_test);
    end
    
    % Aggregate the predictions of the base classifiers using majority voting to make ensemble predictions
    ensemblePredictions = zeros(size(testPredictions)); % Initialize the ensemble predictions
    
    for j = 1:numBaseClassifiers
        % Make predictions using the j-th base classifier on the test data
        basePredictions = baseClassifiers{j}(X_test);
        
        % Convert base predictions to binary class labels (0 or 1)
        binaryPredictions = round(basePredictions);
        
        % Aggregate the base predictions by taking the majority vote
        ensemblePredictions = ensemblePredictions + binaryPredictions;
    end
    
    % Convert ensemble predictions to binary class labels (0 or 1)
    ensemblePredictions = round(ensemblePredictions / numBaseClassifiers);
    
    % Calculate the accuracy of the ensemble predictions on the test data
    ensembleAccuracies(i) = sum(y_test == ensemblePredictions) / length(y_test);
end

%STEP - 5: Define different epoch and hidden layer parameter values
epochValues = [10, 50, 100, 200]; % Replace with the epoch values you want to experiment with
hiddenLayerValues = [10, 20, 30]; % Replace with the hidden layer values you want to experiment with

% Initialize an array to store ensemble accuracies for different parameter combinations
ensembleAccuracies = zeros(length(epochValues), length(hiddenLayerValues));

% Loop over different epoch and hidden layer parameter values
for i = 1:length(epochValues)
    for j = 1:length(hiddenLayerValues)
        % Set the epoch and hidden layer parameters for the base classifiers
        epoch = epochValues(i);
        hiddenLayerSize = hiddenLayerValues(j);
        
        % Initialize arrays to store individual classifier accuracies
        trainAccuracies = zeros(1, numBaseClassifiers);
        testAccuracies = zeros(1, numBaseClassifiers);
        
        % Train and evaluate each base classifier separately
        for k = 1:numBaseClassifiers
            % Train the k-th base classifier on the training data with the current epoch and hidden layer parameters
            baseClassifier = train(X_train, y_train, epoch, hiddenLayerSize); % Replace X_train and y_train with your training data
            
            % Make predictions using the trained base classifier on the training data
            trainPredictions = predict(baseClassifier, X_train);
            
            % Calculate the accuracy of the k-th base classifier on the training data
            trainAccuracies(k) = sum(y_train == trainPredictions) / length(y_train);
            
            % Make predictions using the trained base classifier on the test data
            testPredictions = predict(baseClassifier, X_test);
            
            % Calculate the accuracy of the k-th base classifier on the test data
            testAccuracies(k) = sum(y_test == testPredictions) / length(y_test);
        end
        
        % Aggregate the predictions of the base classifiers using majority voting to make ensemble predictions
        ensemblePredictions = zeros(size(testPredictions)); % Initialize the ensemble predictions
        
        for k = 1:numBaseClassifiers
            % Make predictions using the k-th base classifier on the test data
            basePredictions = predict(baseClassifiers{k}, X_test);
            
            % Convert base predictions to binary class labels (0 or 1)
            binaryPredictions = round(basePredictions);
            
            % Aggregate the base predictions by taking the majority vote
            ensemblePredictions = ensemblePredictions + binaryPredictions;
        end
        
        % Convert ensemble predictions to binary class labels (0 or 1)
        ensemblePredictions = round(ensemblePredictions / numBaseClassifiers);
        
        % Calculate the accuracy of the ensemble predictions on the test data
        ensembleAccuracies(i, j) = sum(y_test == ensemblePredictions) / length(y_test);
    end
end


