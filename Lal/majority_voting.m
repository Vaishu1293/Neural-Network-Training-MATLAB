function [train_error,test_error]=majority_voting(models,xtrain,xtest,ytrain,ytest) 
model_train_outs=[]; 
model_test_outs=[]; 
for a=1:length(models) 
    model=models{a}; 
    ypred_train=model(xtrain); 
    ypred_test=model(xtest); 
    yind_train = vec2ind(ypred_train); 
    yind_test = vec2ind(ypred_test); 
    model_train_outs=[model_train_outs,reshape(yind_train,length(yind_train),1)]; 
    model_test_outs=[model_test_outs,reshape(yind_test,length(yind_test),1)]; 
end 
train_error=transpose(model_train_outs); 
test_error=transpose(model_test_outs); 
ensemble_y_train=mode(train_error); 
ensemble_y_test=mode(test_error); 
tind_train=vec2ind(ytrain); 
tind_test=vec2ind(ytest); 
train_error = sum(tind_train ~= ensemble_y_train)/numel(tind_train); 
test_error = sum(tind_test ~= ensemble_y_test)/numel(tind_test); 
%train_error=vec2ind(ypred_train); 
%test_error=vec2ind(ypred_test);