[x, t] = cancer_dataset;
x = x;
t = t;
%x = x(:, 1:25);
%t = t(:, 1:25);

optimalEpoch = 16; % optimal epoch
optimalHiddenLayers = 8; % optimal hidden layers
num_combinations = 10; % Number of random combinations to generate

% Initialize cell array to store the random combinations as pairs
random_combinations = cell(1, num_combinations);

% Generate random combinations of nodes and epochs
for i = 1:num_combinations
    % Generate random number for nodes
    random_nodes = randi([1, optimalHiddenLayers]);
    
    % Generate random number for epochs
    random_epochs = randi([1, optimalEpoch]);
    
    % Store the random combination as a pair in the cell array
    random_combinations{i} = [random_nodes, random_epochs];
end

disp(random_combinations);

numIterations = 30;
numBaseClassifiers = 15;

ensemblePrediction = cell(1, length(random_nodes));
ensembleTestAccuracy = zeros(1, length(random_nodes));
ensembleTrainAccuracy = zeros(1, length(random_nodes));

for i = 1:length(random_nodes) % i.e 12
    baseClassifiers = cell(1, numBaseClassifiers);
    y_pred_ind = cell(1, numBaseClassifiers);
    trainAccuracies_ind = zeros(1, numBaseClassifiers);
    testAccuracies_ind = zeros(1, numBaseClassifiers);
    
    nodes = random_nodes(i);
    epoch = random_epochs(i);

    for iter = 1:numIterations
        % Get Train Test split
        [X_train, y_train, X_test, y_test] = train_test_split(x, t);

        % Build Classifiers
        for j = 1:numBaseClassifiers
            baseClassifier = build_model_function(X_train, y_train, nodes, epoch, j);
            baseClassifiers{j} = baseClassifier; % Store base classifier in cell array
            [trainAccuracies_ind(j), testAccuracies_ind(j), y_pred_ind{j}] = calculate_accuracy(baseClassifier, X_train, y_train, X_test, y_test);
        end

        % Train Accuracy of the ensemble
        ensembleTrainAccuracy(i) = mean(trainAccuracies_ind);

        % Call Majority Voting Function
        [ensemblePrediction{i}] = majority_vote(y_pred_ind, X_test);
        % encode y_test
        y_test_en = encode_data(y_test);
        %Calculate accuracy of ensemble:
        [ensembleTestAccuracy(i)] = calculate_accuracy_ensemble(ensemblePrediction{i}, y_test_en, y_test);
    end
end

disp(ensembleTestAccuracy);
disp(ensembleTrainAccuracy);

% Overall Accuracy of ensemble:

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


