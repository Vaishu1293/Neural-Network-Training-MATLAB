function mod = model_function(xtrain,ytrain,hidden,epochs) 
trainFcn = 'trainscg'; 
hiddenLayerSize = hidden; 
net = patternnet(hiddenLayerSize, trainFcn); 
net=configure(net,xtrain,ytrain); 
net.trainParam.epochs=epochs; 
[net,tr] = train(net,xtrain,ytrain); 
mod=net; 
end