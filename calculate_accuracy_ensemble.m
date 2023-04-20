function [ensembleAccuracy] = calculate_accuracy_ensemble(ensemblePrediction,  y_test) 
ensembleAccuracy = 100 * sum(~round(ensemblePrediction) ~= y_test, 'all') / numel(y_test); % Update here


% correctPredictions = 0;
% for k = 1:length(y_test_en)
%     if ensemblePrediction(k) == y_test_en(k)
%         correctPredictions = correctPredictions + 1;
%     end
% end
% ensembleAccuracy = 100 * (correctPredictions / size(y_test, 2));