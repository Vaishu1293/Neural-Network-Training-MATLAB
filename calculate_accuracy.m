function [train_acc,test_acc, y_pred] = calculate_accuracy(mod,x_train,y_train,x_test,y_test) 
[mod, tr]=train(mod,x_train,y_train); 
y_train_pred = mod(x_train);
y_pred = mod(x_test); 
train_acc = 100 * sum(~round(y_train_pred) ~= y_train, 'all') / numel(y_train);
test_acc = 100 * sum(~round(y_pred) ~= y_test, 'all') / numel(y_test);

% [mod, tr]=train(mod,x_train,y_train); 
% y_train_pred = mod(x_train);
% y_train_pred_binary = round(y_train_pred); 
% % Calculate train accuracy
% correctPredictions = 0;
% for k = 1:size(y_train, 2)
%     if y_train_pred_binary(k) == y_train(k)
%         correctPredictions = correctPredictions + 1;
%     end
% end
% tra = 100 * (1 - tr.best_tperf);
% %disp(tra);
% train_acc = 100 * (correctPredictions / size(y_train, 2)); 
% 
% y_pred = mod(x_test); 
% y_pred_binary = round(y_pred); 
% % Calculate test accuracy
% correctPredictions = 0;
% for k = 1:size(y_test, 2)
%     if y_pred_binary(k) == y_test(k)
%         correctPredictions = correctPredictions + 1;
%     end
% end
% test_acc = 100 * (correctPredictions / size(y_test, 2)); 