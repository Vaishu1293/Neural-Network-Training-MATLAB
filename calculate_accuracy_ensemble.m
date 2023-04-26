function [ensembleAccuracy] = calculate_accuracy_ensemble(ensemblePrediction,  y_test) 
ensembleAccuracy = 100 * sum(~round(ensemblePrediction) ~= y_test, 'all') / numel(y_test);
