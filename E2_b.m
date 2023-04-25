[x, t] = cancer_dataset;
x = x;
t = t;

numIterations = 30;

numBaseClassifiers = randperm(23,10) + 2; % Generate 10 random numbers in the range of 3 to 25
numBaseClassifiers = sort(numBaseClassifiers);

ensemblePrediction = cell(1, length(numBaseClassifiers));
ensembleTrainPrediction = cell(1, length(numBaseClassifiers));
ensembleTestAccuracy = zeros(1, length(numBaseClassifiers));
ensembleTrainAccuracy = zeros(1, length(numBaseClassifiers));

optimalEpoch = 16; % optimal epoch
optimalHiddenLayers = 32; % optimal hidden layers

for i = 1:length(numBaseClassifiers) % i.e 12
    baseClassifiers = cell(1, numBaseClassifiers(i));
    y_pred_ind = cell(1, numBaseClassifiers(i));
    y_pred_train_ind = cell(1, numBaseClassifiers(i));

    trainAccuracies_ind = zeros(1, numBaseClassifiers(i));
    testAccuracies_ind = zeros(1, numBaseClassifiers(i));

    for iter = 1:numIterations
        % Get Train Test split
        [X_train, y_train, X_test, y_test] = train_test_split(x, t);

        % Build Classifiers
        for j = 1:numBaseClassifiers(i)
            baseClassifier = build_model_function(X_train, y_train, optimalHiddenLayers, optimalEpoch, j);
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

TrainAccuracy = mean(ensembleTrainAccuracy);
TestAccuracy = mean(ensembleTestAccuracy);
disp(numBaseClassifiers);
disp("Train");
disp(ensembleTrainAccuracy);
disp("test");
disp(ensembleTestAccuracy);
disp(['Ensemble Train Accuracy: ', num2str(TrainAccuracy), '%']);
disp(['Ensemble Test Accuracy: ', num2str(TestAccuracy), '%']);

% Plot numBaseClassifiers count array vs. train and test accuracy
figure;
plot(numBaseClassifiers, ensembleTrainAccuracy, 'bo-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Train Accuracy');
hold on;
plot(numBaseClassifiers, ensembleTestAccuracy, 'rx-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Test  Accuracy');
xlabel('Number of Base Classifiers');
ylabel('Accuracy');
title('Ensemble Accuracy vs. Number of Base Classifiers');
legend('Train Accuracy', 'Test Accuracy');
grid on;

