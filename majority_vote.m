function ensemblePrediction = majority_vote(Prediction, X_test)
    ensemblePrediction = zeros(2,size(X_test,2));
    all_results = [0,1]; %possible outcomes
    i = 1;
    for col = 1:2:size(X_test,2)*2
        election_array = zeros(1, length(all_results));
        label_zero = zeros(1, length(Prediction));
        label_ones = zeros(1, length(Prediction));
        for row = 1:length(Prediction) 
            temp = Prediction{row};
            label_zero(row) = temp(col);
            label_ones(row) = temp(col+1);
            if max(temp(col), temp(col+1))==temp(col)
                index = 1;
            else
                index = 2;
            end
            election_array(index) = election_array(index) + 1;
        end
        [~,I] = max(election_array);
        avg_label_zero = mean(label_zero);
        avg_label_one = mean(label_ones);
        ensemblePrediction(1, i) = avg_label_zero;
        ensemblePrediction(2, i) = avg_label_one;
        i = i+1;
    end
end