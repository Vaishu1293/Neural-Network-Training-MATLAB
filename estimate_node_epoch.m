function [optimal_node, optimal_epoch, train_accuracy, test_accuracy] = estimate_node_epoch(X_train,y_train,X_test,y_test) 
% Definition of nodes and epochs
% nodes = randi([1 100],1,10);
% epochs = randi([1 100],1,10);
% nodes = sort(nodes)
% epochs = sort(epochs)
nodes = 100;
epochs = 85;
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
            % Train the neural network on the training set
            net = patternnet(nodes(i), 'trainrp');
            net.trainParam.epochs = epochs(j);
            net.trainParam.lr = 0.01; 
            net = train(net, X_train, y_train);

            % Perform prediction on the training and test sets
            y_train_pred = net(X_train);
            y_test_pred = net(X_test);

            % Calculate the classification error rate for train and test sets
            train_error_rates(k) = sum(round(y_train_pred) ~= y_train, 'all') / numel(y_train); % Update here
            test_error_rates(k) = sum(round(y_test_pred) ~= y_test, 'all') / numel(y_test); % Update here

            % Calculate the train loss
            train_losses(k) = perform(net, y_train, y_train_pred);
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
[min_test_error, min_test_error_idx] = min(final_test_error_mean(:));
[optimal_node_idx, optimal_epoch_idx] = ind2sub(size(final_test_error_mean), min_test_error_idx);

% Obtain the optimal node and epoch values
optimal_node = nodes(optimal_node_idx);
optimal_epoch = epochs(optimal_epoch_idx);

train_accuracy = 100 * (1-final_train_error_mean);
test_accuracy = 100 * (1-final_test_error_mean);