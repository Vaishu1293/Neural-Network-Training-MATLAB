function [percentErrors_train,percentErrors_test,mod] = calculate_performance(mod,xtrain,ytrain,xtest,ytest) 
mod=train(mod,xtrain,ytrain); 
ypred_train = mod(xtrain); 
ypred_test = mod(xtest); 
tind_train = vec2ind(ytrain); 
yind_train = vec2ind(ypred_train); 
percentErrors_train = sum(tind_train ~= yind_train)/numel(tind_train); 
tind_test = vec2ind(ytest); yind_test = vec2ind(ypred_test); 
percentErrors_test = sum(tind_test ~= yind_test)/numel(tind_test);