[x, t] = cancer_dataset;
x = x;
t = t;

optimalEpoch = randi([1, 100], 1, 10); % optimal epoch
optimalEpoch = sort(optimalEpoch);
optimalHiddenLayers = 32; % optimal hidden layers

disp(optimalEpoch);
disp(length(optimalEpoch));

numIterations = 30;
numBaseClassifiers = 15;

ensemblePrediction = cell(1, length(optimalEpoch));
ensembleTrainPrediction = cell(1,length(optimalEpoch));
ensembleTestAccuracy = zeros(1, length(optimalEpoch));
ensembleTrainAccuracy = zeros(1, length(optimalEpoch));

for i = 1:length(optimalEpoch) % i.e 12
    baseClassifiers = cell(1, numBaseClassifiers);
    y_pred_ind = cell(1, numBaseClassifiers);
    y_pred_train_ind = cell(1, numBaseClassifiers);
    trainAccuracies_ind = zeros(1, numBaseClassifiers);
    testAccuracies_ind = zeros(1, numBaseClassifiers);
    
    nodes = optimalHiddenLayers;
    epoch = optimalEpoch(i);

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
disp(optimalEpoch);
disp(ensembleTestAccuracy);
disp(ensembleTrainAccuracy);

TrainAccuracy = mean(ensembleTrainAccuracy)
TestAccuracy = mean(ensembleTestAccuracy)

% Plot x_values vs. train and test accuracy
figure;
plot(optimalEpoch, ensembleTrainAccuracy, 'bo-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Train Accuracy');
hold on;
plot(optimalEpoch, ensembleTestAccuracy, 'rx-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Test Accuracy');
xlabel('Number of Base Classifiers');
ylabel('Accuracy');
title('Ensemble Accuracy vs. Number of Base Classifiers');
legend('Train Accuracy', 'Test Accuracy');
grid on;