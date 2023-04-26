function [ensemble_train_accuracy, ensemble_test_accuracy, ensembleTrainPrediction, ensembleTestPrediction] = get_ensemble_accuracy(X_train,y_train,X_test,y_test, optimalEpoch, optimalHiddenLayers) 
numIterations = 30;

numBaseClassifiers = 15;

y_pred_ind = cell(1, numBaseClassifiers);
y_pred_train_ind = cell(1, numBaseClassifiers);

trainAccuracies_ind = zeros(1, numBaseClassifiers);
testAccuracies_ind = zeros(1, numBaseClassifiers);

for iter = 1:numIterations
    % Build Classifiers
    for j = 1:numBaseClassifiers
        baseClassifier = build_model_function(X_train, y_train, optimalHiddenLayers, optimalEpoch, j);
        [trainAccuracies_ind(j), testAccuracies_ind(j), y_pred_ind{j}, y_pred_train_ind{j}] = calculate_accuracy(baseClassifier, X_train, y_train, X_test, y_test);
    end
    % Call Majority Voting Function
    [ensembleTrainPrediction] = majority_vote(y_pred_train_ind, X_train);
    [ensembleTestPrediction] = majority_vote(y_pred_ind, X_test);
    
    %Calculate accuracy of ensemble:
    [ensemble_train_accuracy] = calculate_accuracy_ensemble(ensembleTrainPrediction, y_train);
    [ensemble_test_accuracy] = calculate_accuracy_ensemble(ensembleTestPrediction, y_test);
end
