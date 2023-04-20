function [y_test_en] = encode_data(y_test) 
y_test_en = zeros(1,size(y_test,2));
all_results = [0,1]; %possible outcomes
z = 1; 
for col = 1:2:size(y_test,2)*2
    if max(y_test(col), y_test(col+1))==y_test(col)
        index = 1;
    else
        index = 2;
    end
    y_test_en(z) = all_results(index);
    z = z + 1;
end