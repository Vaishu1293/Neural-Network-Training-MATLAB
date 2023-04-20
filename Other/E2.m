% Load the cancer_dataset (assuming it's already loaded)

% Step 1: Choose optimum node-epoch combination from Experiment 1
nodes = 32;
epochs = 32;

% Step 2: Choose an odd number of ensemble classifiers between 3 and 25
num_classifiers = 15; % Example: using 15 ensemble classifiers

% Step 3: Modify the 'advanced script' from Experiment 1 to train multiple individual classifiers
classifiers = cell(num_classifiers, 1);
for i = 1:num_classifiers
    % Create a neural network with random initial weights
    net = patternnet(nodes);
    net = configure(net, X_train, y_train);
    net.initFcn = 'initlay';
    net.layers{1}.initFcn = 'initnw';
    net.layers{2}.initFcn = 'initnw';
    net.layers{1}.initnw = 'initwb';
    net.layers{2}.initnw = 'initwb';
    net.trainFcn = 'traingdx';
    net.divideFcn = 'dividerand';
    net.divideParam.trainRatio = 0.7;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.15;
    net.performFcn = 'crossentropy';
    net.trainParam.epochs = epochs;
    
    % Train the neural network
    [net, tr] = train(net, X_train, y_train);
    
    % Store the trained model
    classifiers{i} = net;
end

% Step 4: Implement majority vote technique to predict class labels for test set
y_pred = zeros(size(y_test));
for i = 1:numel(y_test)
    pred_labels = zeros(num_classifiers, 1);
    for j = 1:num_classifiers
        pred = sim(classifiers{j}, X_test(:, i));
        [~, pred_labels(j)] = max(pred);
    end
    y_pred(i) = mode(pred_labels);
end

% Step 5: Repeat the ensemble process at least thirty times with different train/test splits
num_runs = 30; % Example: repeating 30 times
accuracies = zeros(num_runs, 1);
for i = 1:num_runs
    % Randomly split the dataset into training and test sets with 50/50 split
    cv = cvpartition(size(X, 2), 'Holdout', 0.5);
    X_train = X(:, training(cv));
    y_train = y(:, training(cv));
    X_test = X(:, test(cv));
    y_test = y(:, test(cv));

    % Train and predict with the ensemble of individual classifiers
    y_pred = zeros(size(y_test));
    for j = 1:num_classifiers
        % Train the neural network
        [net, tr] = train(classifiers{j}, X_train, y_train);
        
        % Predict class labels for test set
        pred_labels = sim(net, X_test);
        [~, pred_labels] = max(pred_labels);
        y_pred = y_pred + pred_labels;
    end
    
    % Implement majority vote technique
    y_pred = round(y_pred / num_classifiers);
    
    % Calculate accuracy for this run
    accuracies(i) = sum(y_pred == y_test) / numel(y_test);
end

% Step 6: Calculate average classification accuracy for the ensemble and plot it against the number of base classifiers
ensemble_avg_accuracy_train = mean(ensemble_accuracy_train); % Calculate average ensemble accuracy on training data
ensemble_avg_accuracy_test = mean(ensemble_accuracy_test); % Calculate average ensemble accuracy on test data

% Plot ensemble accuracy on training and test data using bar graph
figure;
bar(num_base_classifiers, [ensemble_avg_accuracy_train; ensemble_avg_accuracy_test]', 'grouped');
xlabel('Number of Base Classifiers');
ylabel('Accuracy');
title('Ensemble Accuracy vs Number of Base Classifiers');
legend('Training Data', 'Test Data');
grid on;
