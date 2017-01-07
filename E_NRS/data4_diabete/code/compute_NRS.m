%NRS_71
function NRS_matrix=compute_NRS(similarity_matrix)
    test_num = size(similarity_matrix,1);
    train_num = size(similarity_matrix,2);
    NRS_matrix = zeros(test_num, train_num);
    for i=1:test_num
        for j=1:train_num
            NRS_matrix(i,j)=similarity_matrix(i,j,1);
        end
    end
end