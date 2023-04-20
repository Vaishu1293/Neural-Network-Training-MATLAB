% Step 1: Choose optimal values of nodes and epochs from Experiment 1
nodes = 32; % Number of neurons in the hidden layers
epochs = 32; % Number of times the entire dataset is passed through the network during training

% Step 2: Choose number of base classifiers (15 for scenarios 1, 3, 4, and 5)
num_base_classifiers = 15; % Number of base classifiers in the ensemble

% Step 3: Define parameters for different scenarios
% Scenario 1: 15 base classifiers with the same epoch and hidden layers
epoch_scenario1 = epochs; % Epochs for scenario 1
nodes_scenario1 = nodes; % Nodes in the hidden layers for scenario 1

% Scenario 2: Change of accuracy across different ensemble classifier counts (same as previously given code)

% Scenario 3: All base classifiers have different epochs and hidden layers lower than optimal layers and epoch combination (15 combinations)
epoch_scenario3 = [16, 24, 16, 24, 20, 28, 20, 28, 18, 26, 18, 26, 22, 30]; % Epochs for scenario 3
nodes_scenario3 = [16, 16, 24, 24, 20, 20, 28, 28, 18, 18, 26, 26, 22, 22, 30]; % Nodes in the hidden layers for scenario 3

% Scenario 4: All base classifiers have different epochs and hidden layers higher than optimal layers and epoch combination (any 15 combinations)
epoch_scenario4 = [40, 36, 38, 34, 42, 44, 46, 48, 52, 54, 56, 58, 60, 62, 64]; % Epochs for scenario 4
nodes_scenario4 = [36, 38, 34, 40, 42, 44, 46, 48, 52, 54, 56, 58, 60, 62, 64]; % Nodes in the hidden layers for scenario 4

% Scenario 5: All base classifiers have different epochs but same hidden layers [15]
epoch_scenario5 = [16, 24, 20, 28, 18, 26, 22, 30, 32, 34, 36, 38, 40, 42, 44]; % Epochs for scenario 5
nodes_scenario5 = repmat(nodes, 1, num_base_classifiers); % Nodes in the hidden layers for scenario 5

% Step 4: Modify the 'advanced script' from Experiment 1 to train multiple individual classifiers
% ... (same as previously given code)

% Step 5: Implement majority vote technique to determine predicted class labels for test set
% ... (same as previously given code)

% Step 6: Repeat ensemble of individual classifiers at least thirty times with different 50/50 train/test splits
num_runs = 30; % Number of times to repeat the experiment

% Initialize arrays to store ensemble accuracy for different scenarios
ensemble_accuracy_scenario1_train = zeros(num_runs, 1);
ensemble_accuracy_scenario1_test = zeros(num_runs, 1);
ensemble_accuracy_scenario3_train = zeros(num_runs, 1);
ensemble_accuracy_scenario3_test = zeros(num_runs, 1);
ensemble_accuracy_scenario4_train = zeros(num_runs, 1);
ensemble_accuracy_scenario4_test = zeros(num_runs

ensemble_accuracy_scenario5_train = zeros(num_runs, 1);
ensemble_accuracy_scenario5_test = zeros(num_runs, 1);

for run = 1:num_runs
% Generate random train/test split
rng(run); % Set random seed for reproducibility
train_indices = randperm(num_samples, floor(0.5*num_samples));
test_indices = setdiff(1:num_samples, train_indices);

% Train and evaluate ensemble for scenario 1
ensemble_accuracy_scenario1_train(run) = get_ensemble_accuracy(ensemble_labels_scenario1_train, train_indices, train_labels, num_classes);
ensemble_accuracy_scenario1_test(run) = get_ensemble_accuracy(ensemble_labels_scenario1_test, test_indices, test_labels, num_classes);

% Train and evaluate ensemble for scenario 3
ensemble_accuracy_scenario3_train(run) = get_ensemble_accuracy(ensemble_labels_scenario3_train, train_indices, train_labels, num_classes);
ensemble_accuracy_scenario3_test(run) = get_ensemble_accuracy(ensemble_labels_scenario3_test, test_indices, test_labels, num_classes);

% Train and evaluate ensemble for scenario 4
ensemble_accuracy_scenario4_train(run) = get_ensemble_accuracy(ensemble_labels_scenario4_train, train_indices, train_labels, num_classes);
ensemble_accuracy_scenario4_test(run) = get_ensemble_accuracy(ensemble_labels_scenario4_test, test_indices, test_labels, num_classes);

% Train and evaluate ensemble for scenario 5
ensemble_accuracy_scenario5_train(run) = get_ensemble_accuracy(ensemble_labels_scenario5_train, train_indices, train_labels, num_classes);
ensemble_accuracy_scenario5_test(run) = get_ensemble_accuracy(ensemble_labels_scenario5_test, test_indices, test_labels, num_classes);

end

% Step 7: Plot the ensemble accuracy for different scenarios using bar graph
figure;
bar([mean(ensemble_accuracy_scenario1_train), mean(ensemble_accuracy_scenario1_test);
mean(ensemble_accuracy_scenario3_train), mean(ensemble_accuracy_scenario3_test);
mean(ensemble_accuracy_scenario4_train), mean(ensemble_accuracy_scenario4_test);
mean(ensemble_accuracy_scenario5_train), mean(ensemble_accuracy_scenario5_test)], 'grouped');
xlabel('Scenario');
ylabel('Ensemble Accuracy');
legend('Train', 'Test');
set(gca, 'xticklabels', {'Scenario 1', 'Scenario 3', 'Scenario 4', 'Scenario 5'});
title('Ensemble Accuracy for Different Scenarios');