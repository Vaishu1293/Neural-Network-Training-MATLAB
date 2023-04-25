[x,t]=cancer_dataset; 
mod = cell(1,25);
for a=1:25 
    train_error_mod{a}=[]; 
    test_error_mod{a}=[]; 
    mod{a}=model_function(x,t,32,16); 
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
    
    for a=1:25 
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

for a=1:25 
    final_train_mean=[final_train_mean,mean(train_error_mod{a})]; 
    final_test_mean=[final_test_mean,mean(test_error_mod{a})]; 
end 

cum_train_error_var=[]; 
cum_test_error_var=[]; 

for bla=3:25 
    temp_var=rem(bla,2); 
    if temp_var ~= 0 
        disp(bla) 
        [train_error_var,test_error_var]=majority_voting(mod1(1:bla),xtrain,xtest,ytrain,ytest); 
        cum_train_error_var=[cum_train_error_var,1-train_error_var]; 
        cum_test_error_var=[cum_test_error_var,1-test_error_var]; 
    end 
end

disp(cum_test_error_var);
disp(cum_train_error_var);
