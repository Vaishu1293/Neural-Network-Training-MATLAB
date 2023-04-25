[x,t] = cancer_dataset;
x = x; 
t = t;

numIterations = 30;

numBaseClassifiers = 15;
baseClassifiers = cell(1, numBaseClassifiers);

trainAccuracies = zeros(1, numBaseClassifiers);
testAccuracies = zeros(1, numBaseClassifiers); % Store test accuracies for each base classifier

for iter = 1:numIterations
    [X_train, y_train, X_test, y_test] = train_test_split(x, t);

    optimalEpoch = 3;
    optimalHiddenLayers = 5;

    for i = 1:numBaseClassifiers
        baseClassifier = build_model_function(X_train, y_train, optimalHiddenLayers, optimalEpoch, i);
        [trainAccuracies(i), testAccuracies(i)] = calculate_accuracy(baseClassifier, X_train, y_train, X_test, y_test);
        baseClassifiers{i} = baseClassifier;

        disp(['Base Classifier ', num2str(i), ' Accuracy:']);
        disp(['Train Accuracy: ', num2str(trainAccuracies(i)), '%']);
        disp(['Test Accuracy: ', num2str(testAccuracies(i)), '%']);
    end
    avgMeanTrainAccuracy = mean(trainAccuracies);
    avgMeanTestAccuracy = mean(testAccuracies);

    disp(['Iteration: ', num2str(iter)]);
    disp('Individual Model Accuracies:');
    disp(['Train Accuracies: ', num2str(trainAccuracies)]);
    disp(['Test Accuracies: ', num2str(testAccuracies)]);

    disp('Average Mean Train and Test Accuracies:');
    disp(['Average Mean Train Accuracy: ', num2str(avgMeanTrainAccuracy), '%']);
    disp(['Average Mean Test Accuracy: ', num2str(avgMeanTestAccuracy), '%']);
    disp(['Average Mean Train Accuracy: ', num2str(avgMeanTrainAccuracy), '%']);
    disp(['Average Mean Test Accuracy: ', num2str(avgMeanTestAccuracy), '%']);
end

% Bar graph for


% Bar graph for model accuracies
figure;
bar([trainAccuracies; testAccuracies]');
legend('Train Accuracy', 'Test Accuracy');
xlabel('Model Index');
ylabel('Accuracy (%)');
title('Model Accuracies for Individual Models');
xticks(1:numBaseClassifiers);
xticklabels({'Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Model 6', 'Model 7', 'Model 8', 'Model 9', 'Model 10', 'Model 11', 'Model 12', 'Model 13', 'Model 14', 'Model 15'});
grid on;
