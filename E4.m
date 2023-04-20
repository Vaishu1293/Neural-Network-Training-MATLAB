%% STEP 01:

% Parameters for Class 1
mean1 = [0, 0];
variance1 = eye(2); % Variance is identity matrix

% Parameters for Class 2
mean2 = [2, 0];
variance2 = [4, 0; 0, 4]; % Variance is diagonal matrix

% Generate random variables for Class 1 and Class 2
num_samples = 1650; % Number of samples for each class
class1_samples = mvnrnd(mean1, variance1, num_samples);
class2_samples = mvnrnd(mean2, variance2, num_samples);

% Concatenate the samples from both classes
data = [class1_samples; class2_samples];

% Generate labels for the samples
labels = [repmat([1, 0], num_samples, 1); repmat([0, 1], num_samples, 1)];

% Randomly shuffle the data and labels
shuffled_indices = randperm(size(data, 1));
data = data(shuffled_indices, :);
labels = labels(shuffled_indices, :);

%% STEP 02:

% Split the data into training and testing sets
train_data = data(1:300, :);
train_labels = labels(1:300, :);
test_data = data(301:end, :);
test_labels = labels(301:end, :);

% Set the number of hidden units and epochs for the neural network
hidden_units = 60;
epochs = 38;

% Create the neural network model
baseClassifier = patternnet(hidden_units);

% Set the training parameters for the model
baseClassifier.trainParam.epochs = epochs;
baseClassifier.trainFcn = 'trainscg';

% Train the neural network
baseClassifier = train(baseClassifier, train_data', train_labels');

% Test the neural network
train_predictions = baseClassifier(train_data');
train_accuracy = sum(train_predictions' == train_labels) / size(train_labels, 1);
test_predictions = baseClassifier(test_data');
test_accuracy = sum(test_predictions' == test_labels) / size(test_labels, 1);

fprintf('Train Accuracy: %.2f%%\n', train_accuracy * 100);
fprintf('Test Accuracy: %.2f%%\n', test_accuracy * 100);

%% STEP 03:

% Set the number of base classifiers in the ensemble
num_base_classifiers = 10;

% Initialize arrays to store ensemble predictions
ensemble_train_predictions = zeros(size(train_labels));
ensemble_test_predictions = zeros(size(test_labels));

% Train and predict with multiple base classifiers
for i = 1:num_base_classifiers
    % Create a new instance of the base classifier with the optimal architecture
    baseClassifier = patternnet(hidden_units);
    baseClassifier.trainParam.epochs = epochs;
    baseClassifier.trainFcn = 'trainscg';
    
    % Train the base classifier
    baseClassifier = train(baseClassifier, train_data', train_labels');
    % Predict with the base classifier on training and testing data
    train_predictions = baseClassifier(train_data');
    test_predictions = baseClassifier(test_data');
    
    % Update ensemble predictions with the current base classifier's predictions
    ensemble_train_predictions = ensemble_train_predictions + train_predictions';
    ensemble_test_predictions = ensemble_test_predictions + test_predictions';
end

% Normalize the ensemble predictions by dividing by the number of base classifiers
ensemble_train_predictions = ensemble_train_predictions / num_base_classifiers;
ensemble_test_predictions = ensemble_test_predictions / num_base_classifiers;

% Round the ensemble predictions to obtain the final ensemble predictions
ensemble_train_predictions = round(ensemble_train_predictions);
ensemble_test_predictions = round(ensemble_test_predictions);

% Calculate the accuracy of the ensemble on training and testing data
ensemble_train_accuracy = sum(ensemble_train_predictions == train_labels, 'all') / numel(train_labels);
ensemble_test_accuracy = sum(ensemble_test_predictions == test_labels, 'all') / numel(test_labels);

fprintf('Ensemble Train Accuracy: %.2f%%\n', ensemble_train_accuracy * 100);
fprintf('Ensemble Test Accuracy: %.2f%%\n', ensemble_test_accuracy * 100);




