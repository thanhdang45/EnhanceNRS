clear;
clc;
load data;
tic;
T_para=zeros(1,3,4);
T_para(1,:,:)=make_increasing_matrix(3,4);

I_para=zeros(1,3,4);
I_para(1,:,:)=make_increasing_matrix(3,4);

F_para=zeros(1,3,4);
F_para(1,:,:) = make_increasing_matrix(3,4);

[s_T_component,s_I_component,s_F_component]= comput_all_TIF(input_data,T_para,I_para,F_para);
 
unique_output= unique(output_data);
disease_num = length(unique_output);
[d_T_component,d_I_component,d_F_component] = comput_all_TIF(unique_output,T_para,I_para,F_para);
similarity_matrix = compute_similarity(d_T_component,d_I_component,d_F_component,s_T_component,s_I_component,s_F_component);
patient_num = size(input_data,1);
output_label = zeros(patient_num,1);
for i=1:patient_num
    temp_max=0;
    for j=1:disease_num
        if temp_max<similarity_matrix(i,j)
            temp_max=similarity_matrix(i,j);
            output_label=unique_output(j);
        end
    end
end
error = mse(output_label-output_data)
toc;