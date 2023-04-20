[x, t] = cancer_dataset;
%x = x;
%t = t;
x = x(:, 1:25);
t = t(:, 1:25);

numIterations = 1;

%numBaseClassifiers = randperm(23,10) + 2; % Generate 10 random numbers in the range of 3 to 25
%numBaseClassifiers = sort(numBaseClassifiers);
numBaseClassifiers = [3];

ensemblePrediction = cell(1, length(numBaseClassifiers));
ensembleTestAccuracy = zeros(1, length(numBaseClassifiers));
ensembleTrainAccuracy = zeros(1, length(numBaseClassifiers));

optimalEpoch = 1; % optimal epoch
optimalHiddenLayers = 32; % optimal hidden layers

for i = 1:length(numBaseClassifiers) % i.e 12
    baseClassifiers = cell(1, numBaseClassifiers(i));
 
    for iter = 1:numIterations
        % Get Train Test split
        [X_train, y_train, X_test, y_test] = train_test_split(x, t);

        % Build Classifiers
        for j = 1:numBaseClassifiers(i)
            [baseClassifiers{j}] = build_model_function(X_train, y_train, optimalHiddenLayers, optimalEpoch, j);
        end

        % Train Accuracy of the ensemble
        [ensembleTrainAccuracy(i), ensembleTestAccuracy(i)] = majority_voting(baseClassifiers, X_train, X_test, y_train, y_test);
    end
end

disp(ensembleTestAccuracy);
disp(ensembleTrainAccuracy);

% Overall Accuracy of ensemble:

TrainAccuracy = mean(ensembleTrainAccuracy)
TestAccuracy = mean(ensembleTestAccuracy)

% Plot numBaseClassifiers count array vs. train and test accuracy
figure;
plot(numBaseClassifiers, ensembleTrainAccuracy, 'bo-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Train Accuracy');
hold on;
plot(numBaseClassifiers, ensembleTestAccuracy, 'rx-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Test Accuracy');
xlabel('Number of Base Classifiers');
ylabel('Accuracy');
title('Ensemble Accuracy vs. Number of Base Classifiers');
legend('Train Accuracy', 'Test Accuracy');
grid on;

