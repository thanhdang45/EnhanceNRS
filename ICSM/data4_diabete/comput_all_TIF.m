%T_para is a attri_num X term_num X 4 matrix
%I_para is a attri_num X term_num X 4 matrix
%F_para is a attri_num X term_num X 4 matrix
% data is a data_num X attri_num matrix
function [T_component,I_component,F_component]=comput_all_TIF(data,T_para,I_para,F_para)
    data_num = size(data,1);
    term_num = size(T_para,2);
    attri_num = size(data,2);
    T_component = zeros(data_num,attri_num,term_num);
    I_component = zeros(data_num,attri_num,term_num);
    F_component = zeros(data_num,attri_num,term_num);
    for i=1:data_num
        for j=1:attri_num
            for k=1:term_num
                [T,I,F]=compute_TIF(data(i,j),T_para(j,k,:),I_para(j,k,:),F_para(j,k,:));
                T_component(i,j,k) = T;
                I_component(i,j,k) = I;
                F_component(i,j,k) = F;
            end
        end
    end
end