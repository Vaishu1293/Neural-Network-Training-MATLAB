[x,t]=cancer_dataset; 
for a=1:15 
    hl_ep=[]; 
    train_error_mod{a}=[]; 
    test_error_mod{a}=[]; 
    hidden=randi([33 100],1,1); 
    ep=randi([17 100],1,1); 
%     [hidden,ep]
    mod{a}=model_function(x,t,2,3); 
end 
train_error_ensemble=[]; 
test_error_ensemble=[];

train_error_ensemble_mean=[]; 
test_error_ensemble_mean=[]; 

for iter=1:30 
    random=randperm(length(x)); 
    xtrain=x(1:9,random(1:round(length(x)*0.5))); 
    ytrain=t(1:2,random(1:round(length(x)*0.5))); 
    xtest=x(1:9,random(round(length(x)*0.5:end))); 
    ytest=t(1:2,random(round(length(x)*0.5):end)); 
    
    for a=1:15 
        [train_error_mod_temp,test_error_mod_temp,mod1{a}] = calculate_performance(mod{a},xtrain,ytrain,xtest,ytest); 
        train_error_mod{a}=[train_error_mod{a},train_error_mod_temp]; 
        test_error_mod{a}=[test_error_mod{a},test_error_mod_temp]; 
    end 
    
    [train_error,test_error]=majority_voting(mod1,xtrain,xtest,ytrain,ytest); 
    train_error_ensemble=[train_error_ensemble,train_error]; 
    test_error_ensemble=[test_error_ensemble,test_error]; 
end 

final_train_mean=[]; 
final_test_mean=[]; 

for a=1:15 
    final_train_mean=[final_train_mean,mean(train_error_mod{a})]; 
    final_test_mean=[final_test_mean,mean(test_error_mod{a})]; 
end

disp(final_train_mean);
disp(final_test_mean);