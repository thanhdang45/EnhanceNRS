%compute_similarity_67
function similarity_matrix=compute_similarity(train_T_component,train_I_component,train_F_component,test_T_component,test_I_component,test_F_component)
    train_num=size(train_T_component,1);
    test_num=size(test_T_component,1);
    test_attri_num = size(test_T_component,2);
    term_num = size(test_T_component,3);
    similarity_matrix = zeros(test_num,train_num,test_attri_num+1);
    for i=1:test_num
        for j=1:train_num
            for k=1:test_attri_num
                similarity_matrix(i,j,k)=1;
                for m=1:term_num
                    similarity_matrix(i,j,k)=similarity_matrix(i,j,k)*max(abs([train_T_component(j,k,m)-test_T_component(i,k,m);
                                                                              train_I_component(j,k,m)-test_I_component(i,k,m);
                                                                              train_F_component(j,k,m)-test_F_component(i,k,m)]));
                end
                similarity_matrix(i,j,k)=similarity_matrix(i,j,k)/8;
            end
        end
    end
    for i=1:test_num
        for j=1:train_num
            for m=1:term_num
                similarity_matrix(i,j,test_attri_num+1)=0;
            end
        end
    end
end