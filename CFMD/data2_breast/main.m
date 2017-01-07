clear;
clc;
load data;
tic;
train_num = size(train_input,1);
test_num = size(test_input,1);
similarity_matrix = zeros(test_num,train_num);
for i=1:test_num
    i
    for j=1:train_num
        similarity_matrix(i,j) = compute_similarity(test_input(i,:),train_input(j,:));
    end
end
computed_test_output = zeros(size(test_output));
for i=1:test_num
    computed_test_output(i) = mean(test_input(i,:));
end

%tham so k
k=0.0005;
for i=1:test_num
    i
    for j=1:train_num
        computed_test_output(i) = computed_test_output(i) +k *similarity_matrix(i,j)*(train_output(j)-mean(train_input(j,:)));
    end
end
error=mse(test_output-computed_test_output)
toc;