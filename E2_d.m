[x, t] = cancer_dataset;
x = x;
t = t;

optimalEpoch = 16; % optimal epoch
optimalHiddenLayers = 32; % optimal hidden layers
num_combinations = 10; % Number of random combinations to generate

% Initialize cell array to store the random combinations as pairs
random_combinations = cell(1, num_combinations);

% Generate random combinations of nodes and epochs
for i = 1:num_combinations
    % Generate random number for nodes
    random_nodes = randi([optimalHiddenLayers, 100]);
    
    % Generate random number for epochs
    random_epochs = randi([optimalEpoch, 100]);
    
    % Store the random combination as a pair in the cell array
    random_combinations{i} = [random_nodes, random_epochs];
end

disp(random_combinations);

numIterations = 30;
numBaseClassifiers = 15;

ensemblePrediction = cell(1, num_combinations);
ensembleTrainPrediction = cell(1, num_combinations);
ensembleTestAccuracy = zeros(1, num_combinations);
ensembleTrainAccuracy = zeros(1, num_combinations);

for i = 1:num_combinations % i.e 12
    baseClassifiers = cell(1, numBaseClassifiers);
    y_pred_ind = cell(1, numBaseClassifiers);
    y_pred_train_ind = cell(1, numBaseClassifiers);
    trainAccuracies_ind = zeros(1, numBaseClassifiers);
    testAccuracies_ind = zeros(1, numBaseClassifiers);
    
    nodes = random_combinations{i}(1,1);
    epoch = random_combinations{i}(1,2);

    for iter = 1:numIterations
        % Get Train Test split
        [X_train, y_train, X_test, y_test] = train_test_split(x, t);

        % Build Classifiers
        for j = 1:numBaseClassifiers
            baseClassifier = build_model_function(X_train, y_train, nodes, epoch, j);
            baseClassifiers{j} = baseClassifier; % Store base classifier in cell array
            [trainAccuracies_ind(j), testAccuracies_ind(j), y_pred_ind{j}, y_pred_train_ind{j}] = calculate_accuracy(baseClassifier, X_train, y_train, X_test, y_test);
        end

        % Train Accuracy of the ensemble
        ensembleTrainAccuracy(i) = mean(trainAccuracies_ind);

        % Call Majority Voting Function
        [ensembleTrainPrediction{i}] = majority_vote(y_pred_train_ind, X_train);
        [ensemblePrediction{i}] = majority_vote(y_pred_ind, X_test);
        
        %Calculate accuracy of ensemble:
        [ensembleTrainAccuracy(i)] = calculate_accuracy_ensemble(ensembleTrainPrediction{i}, y_train);
        [ensembleTestAccuracy(i)] = calculate_accuracy_ensemble(ensemblePrediction{i}, y_test);
    end
end

% Overall Accuracy of ensemble:
disp(random_combinations);
disp(ensembleTestAccuracy);
disp(ensembleTrainAccuracy);

TrainAccuracy = mean(ensembleTrainAccuracy)
TestAccuracy = mean(ensembleTestAccuracy)

% Convert cell array to numeric array for x-axis values
x_values = cellfun(@(x) x(1), random_combinations);

% Plot x_values vs. train and test accuracy
figure;
plot(x_values, ensembleTrainAccuracy, 'bo-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Train Accuracy');
hold on;
plot(x_values, ensembleTestAccuracy, 'rx-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Test Accuracy');
xlabel('Number of Base Classifiers');
ylabel('Accuracy');
title('Ensemble Accuracy vs. Number of Base Classifiers');
legend('Train Accuracy', 'Test Accuracy');
grid on;
