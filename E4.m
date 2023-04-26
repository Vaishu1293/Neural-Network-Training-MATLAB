%% STEP 01:

% Parameters for Class 1
mean1 = [0, 0];
variance1 = eye(2); 

% Parameters for Class 2
mean2 = [2, 0];
variance2 = [2, 0; 0, 2];

% Generate random variables for Class 1 and Class 2
num_samples = 1650; % Number of samples for each class
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

%% STEP 04 PLOT DECISION BOUNDARY (modified code)
% Find the indices for the label with value 1
idx = find(y_pred_test(:, 1) > 0);
% Get the features corresponding to the label with value 1
label1 = test_data(idx, :);
% Create a scatter plot with predicted labels as colors
figure(1);
hold on;
gscatter(test_data(:, 1), test_data(:, 2), y_pred_test', 'rb', '.', 15);
hold on;
% Calculate the center of the circle
x = mean(label1(:, 1))
y = mean(label1(:, 2))
x_var = cov(label1(:, 1))
y_var = cov(label1(:, 2))

% Calculate the radius of the circle
r = sqrt(var(label1(:, 1))) * 2.5
% Generate a range of angles for the circle
th = 0:pi/50:2*pi;

% Calculate the x-coordinates and y-coordinates of the circle
x_circle = r*cos(th) + x;
y_circle = r*sin(th) + y;

% Plot the decision boundary
plot(x_circle, y_circle, 'y', 'LineWidth', 3);

% Add a legend
legend('Class 1', 'Class 2', 'Decision Boundary');
hold off;

%% STEP 05 PLOT DECISION BOUNDARY

%Plot the samples
figure(2);
hold on;
scatter(class1_samples(:, 1), class1_samples(:, 2), 'MarkerFaceColor', 'blue');
scatter(class2_samples(:, 1), class2_samples(:, 2), 'MarkerFaceColor', 'red');

% Calculate the mean and variance of each class
mu1 = mean(class1_samples)
sigma1 = cov(class1_samples)
mu2 = mean(class2_samples)
sigma2 = cov(class2_samples)

% Calculate the decision boundary as a circle
center = (mu2 - mu1) * 2/3 + mu1; % center of the circle
radius = 2.34; % radius of the circle
theta = linspace(0, 2*pi, 100);
x_circle = center(1) + radius*cos(theta);
y_circle = center(2) + radius*sin(theta);

% Plot the decision boundary
plot(x_circle, y_circle, 'y', 'LineWidth', 2);

% Add a legend
legend('Class 1', 'Class 2', 'Decision Boundary');
