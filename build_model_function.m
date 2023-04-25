function baseClassifier = build_model_function(x_train,y_train,nodes,epochs, j) 
baseClassifier = patternnet(nodes);
rng(j);
baseClassifier = init(baseClassifier);
baseClassifier.trainParam.epochs = epochs; 
% baseClassifier.trainFcn = 'trainscg'; 
% baseClassifier.trainFcn = 'trainlm';
baseClassifier.trainFcn = 'trainrp';
baseClassifier=configure(baseClassifier,x_train,y_train); 