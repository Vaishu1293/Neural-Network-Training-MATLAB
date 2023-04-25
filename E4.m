%% STEP 01:

% Parameters for Class 1
mean1 = [0, 0];
variance1 = eye(2); % Variance is identity matrix

% Parameters for Class 2
mean2 = [2, 0];
variance2 = [2, 0; 0, 2]; % Variance is diagonal matrix

% Generate random variables for Class 1 and Class 2
% num_samples = 1650; % Number of samples for each class
num_samples = 10; % Number of samples for each class
split_idx = 0.1*2*num_samples;
start = split_idx + 1;
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

train_data = data(1:split_idx, :);
train_labels = labels(1:split_idx, :);
test_data = data(start:end, :);
test_labels = labels(start:end, :);

% Set the number of hidden units and epochs for the neural network
[hidden_units, epochs, train_accuracy, test_accuracy] = estimate_node_epoch(train_data', train_labels', test_data', test_labels');

fprintf('Train Accuracy: %.2f%%\n', train_accuracy);
fprintf('Test Accuracy: %.2f%%\n', test_accuracy);

%% STEP 03:

[ensemble_train_accuracy, ensemble_test_accuracy, y_pred_train, y_pred_test] = get_ensemble_accuracy(train_data', train_labels', test_data', test_labels', hidden_units, epochs);

fprintf('Ensemble Train Accuracy: %.2f%%\n', ensemble_train_accuracy);
fprintf('Ensemble Test Accuracy: %.2f%%\n', ensemble_test_accuracy);

%% STEP 04 PLOT DECISION BOUNDARY

% Create a grid of points
x_min = min(data(:,1)) - 1;
x_max = max(data(:,1)) + 1;
y_min = min(data(:,2)) - 1;
y_max = max(data(:,2)) + 1;
[x, y] = meshgrid(x_min:0.1:x_max, y_min:0.1:y_max);
xy = [x(:), y(:)];
disp(size(x));
disp(size(y));
% Get predictions for the grid of points using the trained neural network or ensemble
y_pred_test = encode_data(y_pred_test);
disp(size(y_pred_test));
[~, y_pred_test] = max(y_pred_test);

% Plot the predictions as a contour plot
figure;
contourf(x, y, reshape(y_pred_test, size(x)), 'LineStyle', 'none');
hold on;

% Plot the Bayes boundary as a circle
theta = linspace(0, 2*pi, 100);
x_circle = -2/3 + 2.34*cos(theta);
y_circle = 2.34*sin(theta);
plot(x_circle, y_circle, 'r', 'LineWidth', 2);

% Set the axis limits and labels
xlim([x_min, x_max]);
ylim([y_min, y_max]);
xlabel('Feature 1');
ylabel('Feature 2');
title('Decision Boundary and Bayes Boundary');

% Show the legend
legend('Decision Boundary', 'Bayes Boundary');



