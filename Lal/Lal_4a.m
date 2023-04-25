[x1,y1] = mvnrnd([0,0],[1,1],1650); 
y1=1*ones(length(x1),1); 
[x2,y2] = mvnrnd([2,0],[2,2],1650); 
y2=2*ones(length(x2),1); 
x=vertcat(x2,x1); 
y=vertcat(y2,y1); 
labels = categorical(y); 
y = onehotencode(labels,2); 
[row_idx,cols]=size(x); 
rowidx = randperm(row_idx); x=transpose(x(rowidx, : )); 
y=transpose(y(rowidx,:)); 
final_train_eror_mean=[]; 
final_test_eror_mean=[]; 
final_sd_train=[]; 
final_sd_test=[]; 
final_train_loss=[]; 

for a=1:15 
    train_error_mod{a}=[];
    test_error_mod{a} = [];
    hidden=randi([1 50],1,1) 
    ep=randi([1 50],1,1) 
    mod{a}=model_function(x,y,hidden,ep); 
end 

train_error_ensemble=[]; 
test_error_ensemble=[]; 
train_error_ensemble_mean=[]; 
test_error_ensemble_mean=[]; 

for iter=1:30 
    random=randperm(length(x)); 
    xtrain=x(1:2,random(1:round(length(x)*0.0909))); 
    ytrain=y(1:2,random(1:round(length(x)*0.0909))); 
    xtest=x(1:2,random(301:end)); 
    ytest=y(1:2,random(301:end)); 
    
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

ensemble_y_test=majority_voting_testing(mod,xtest); 
indeces_for_label=find(ensemble_y_test==1); 
label1=xtest(1:2,indeces_for_label); 
gscatter(xtest(1:1,:),xtest(2:2,:),ensemble_y_test,'rgb','osd') 
hold on 
r = var(label1(1:1,:)); 
x = mean(label1(1:1,:)); 
y = mean(label1(2:2,:)); 
th = 0:pi/50:2*pi; 
xunit = r * cos(th) + x; 
yunit = r * sin(th) + y; 
plot(xunit, yunit,'LineWidth',3) 
hold off