% Load the cancer dataset
[x,t]=cancer_dataset;
x = x; 
t = t;

% Define X_train as the training data
X_train = x; % or X_train = x'; depending on the data structure

% Definition of nodes and epochs
nodes = [2,8,32];
epochs = [ 1, 2, 4, 8, 16, 32,64];

% Set the number of runs
num_runs = 30;

% Initialize arrays to store results
final_train_error_mean = zeros(length(nodes), length(epochs));
final_test_error_mean = zeros(length(nodes), length(epochs));
final_sd_train = zeros(length(nodes), length(epochs));
final_sd_test = zeros(length(nodes), length(epochs));
final_train_loss = zeros(length(nodes), length(epochs));

% Loop over each node/epoch combination
for i = 1:length(nodes)
    for j = 1:length(epochs)
        train_error_rates = zeros(1, num_runs);
        test_error_rates = zeros(1, num_runs);
        train_losses = zeros(1, num_runs);

        % Loop over each run
        for k = 1:num_runs
            [X_train, y_train, X_test, y_test] = train_test_split(x, t);

            % Train the neural network on the training set
            net = patternnet(nodes(i));
            net.trainParam.epochs = epochs(j);
            net.trainFcn = 'trainscg';
            net.trainParam.lr = 0.01; 
            net=configure(net,X_train,y_train); 

            % Perform prediction on the training and test sets
            net = train(net, X_train, y_train);
            y_train_pred = net(X_train);
            y_test_pred = net(X_test);

            % Calculate the classification error rate for train and test sets
            train_error_rates(k) = sum(round(y_train_pred) ~= y_train, 'all') / numel(y_train); % Update here
            test_error_rates(k) = sum(round(y_test_pred) ~= y_test, 'all') / numel(y_test); % Update here

            % Calculate the train loss
            train_losses(k) = perform(net, y_test, y_train_pred);
        end

        % Calculate average and standard deviation of train error rates, test error rates, and train losses
        final_train_error_mean(i, j) = mean(train_error_rates);
        final_test_error_mean(i, j) = mean(test_error_rates);
        final_sd_train(i, j) = std(train_error_rates);
        final_sd_test(i, j) = std(test_error_rates);
        final_train_loss(i, j) = mean(train_losses);
    end
end

% Find the index of the minimum test error rate
disp(min(final_test_error_mean(:)))
[min_test_error, min_test_error_idx] = min(final_test_error_mean(:));
[optimal_node_idx, optimal_epoch_idx] = ind2sub(size(final_test_error_mean), min_test_error_idx);

% Obtain the optimal node and epoch values
optimal_node = nodes(optimal_node_idx);
optimal_epoch = epochs(optimal_epoch_idx);

% Report the optimal test error rate and its associated node/epoch values
fprintf('Optimal Test Error Rate: %.4f\n', min_test_error);
fprintf('Optimal Node Value: %d\n', optimal_node);
fprintf('Optimal Epoch Value: %d\n', optimal_epoch);


% Plotting results
for i = 1:length(nodes)
    % Create a new figure with figure number set to i
    figure(i);
    
    % Create a subplot for the current node
    subplot(1, 1, 1);
    
    % Plot the train and test error rates with error bars
    errorbar(epochs, final_train_error_mean(i, :), final_sd_train(i, :), 'b-', 'LineWidth', 1, 'MarkerSize', 12);
    hold on;
    errorbar(epochs, final_test_error_mean(i, :), final_sd_test(i, :), 'r-', 'LineWidth', 1, 'MarkerSize', 12);
    
    % Add labels and title
    xlabel('Epochs');
    ylabel('Classification Error Rate');
    title(['Nodes = ' num2str(nodes(i))]);
    
    % Add legend
    legend('Train', 'Test', 'Location', 'best');
end

% Define colors for each node
colors = {'r', 'g', 'b'}; % Add more colors if needed
% Plotting results
figure(4);

for i = 1:length(nodes)
    errorbar(epochs, final_train_error_mean(i, :), final_sd_train(i, :), [colors{i} '-'], 'LineWidth', 1, 'MarkerSize', 12);
    hold on;
end

xlabel('Epochs');
ylabel('Classification Error Rate');
title('Train Classification Error Rate (Std) for All Nodes');
legend(cellstr(num2str(nodes')), 'Location', 'best');

% Plotting results
figure;

for i = 1:length(nodes)
    errorbar(epochs, final_test_error_mean(i, :), final_sd_test(i, :), [colors{i} '-'], 'LineWidth', 1, 'MarkerSize', 12);
    hold on;
end

xlabel('Epochs');
ylabel('Classification Error Rate');
title('Test Classification Error Rate (Std) for All Nodes');
legend(cellstr(num2str(nodes')), 'Location', 'best');



