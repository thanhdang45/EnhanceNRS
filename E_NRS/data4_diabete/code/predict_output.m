%train_T_component : train_num X term_num
%train_I_component : train_num X term_num
%train_F_component : train_num X term_num
%test_T_component : test_num X term_num
%test_I_component : test_num X term_num
%test_F_component : test_num X term_num
function [test_output_T_component,test_output_I_component,test_output_F_component]=predict_output(NRS_matrix,train_T_component,train_I_component,train_F_component)
    test_num=size(NRS_matrix,1);
    train_num=size(NRS_matrix,2);
    term_num =size(train_T_component,2);
    test_output_T_component = zeros(test_num,term_num);
    test_output_I_component = zeros(test_num,term_num);
    test_output_F_component = zeros(test_num,term_num);
    denominator = zeros(train_num,1);
    load cluster_result.mat;
    for i=1:test_num
        for j=1:train_num
            if cluster_result(i+train_num)==cluster_result(j)
                denominator(i)=denominator(i)+NRS_matrix(i,j);
            end
        end
    end
    for i=1:test_num
        for j=1:train_num
            if cluster_result(i+train_num)==cluster_result(j)
                for k=1:term_num
                    test_output_T_component(i,k)=test_output_T_component(i,k)+NRS_matrix(i,j)*train_T_component(j,3,k);
                    test_output_I_component(i,k)=test_output_I_component(i,k)+NRS_matrix(i,j)*train_I_component(j,3,k);
                    test_output_F_component(i,k)=test_output_F_component(i,k)+NRS_matrix(i,j)*train_F_component(j,3,k);
                end
            end
        end
    end
    for i=1:test_num
        for k=1:term_num
            if cluster_result(i+train_num)==cluster_result(j)
                test_output_T_component(i,k)=test_output_T_component(i,k)/denominator(i);
                test_output_I_component(i,k)=test_output_I_component(i,k)/denominator(i);
                test_output_F_component(i,k)=test_output_F_component(i,k)/denominator(i);
            end
        end
    end
    test_output_I_component=test_output_I_component+test_output_T_component;
    test_output_F_component=test_output_F_component+test_output_I_component;
end