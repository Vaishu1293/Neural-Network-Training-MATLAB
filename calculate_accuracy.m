function [train_acc,test_acc, y_pred, y_train_pred] = calculate_accuracy(mod,x_train,y_train,x_test,y_test) 
[mod, tr]=train(mod,x_train,y_train); 
y_train_pred = mod(x_train);
y_pred = mod(x_test); 
train_acc = 100 * sum(~round(y_train_pred) ~= y_train, 'all') / numel(y_train);
test_acc = 100 * sum(~round(y_pred) ~= y_test, 'all') / numel(y_test);