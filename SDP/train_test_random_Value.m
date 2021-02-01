function [indexes]=train_test_random_Value(y)
K = max(y);indexes = [];indexes_c=[];
Value=[20,20,20,20,20,20,20,20,20,20,20,20,20]';
for i=1:K
    index1 = find(y == i);
    per_index1 = randperm(length(index1));
    Number=per_index1(1:Value(i));
    indexes_c=[indexes_c index1(Number)];
end
indexes = indexes_c(:);




                  