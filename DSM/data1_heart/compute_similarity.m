function similarity_matrix = compute_similarity(d_T_component,d_I_component,d_F_component,s_T_component,s_I_component,s_F_component)
    disease_num = size(d_T_component,1);
    patient_num = size(s_T_component,1);
    similarity_matrix=zeros(patient_num,disease_num);
    for i=1:patient_num
        for j=1:disease_num
            for k=1:3
                numerator = s_T_component(i,1,k)*d_T_component(j,1,k)+s_T_component(i,1,k+3)*d_T_component(j,1,k+3)+s_I_component(i,1,k)*d_I_component(j,1,k)+s_I_component(i,1,k+3)*d_I_component(j,1,k+3)+s_F_component(i,1,k)*d_F_component(j,1,k)+s_F_component(i,1,k+3)*d_F_component(j,1,k+3);
                denominator = 0.5*(s_T_component(i,1,k)^2+d_T_component(j,1,k)^2+s_T_component(i,1,k+3)^2+d_T_component(j,1,k+3)^2+s_I_component(i,1,k)^2+d_I_component(j,1,k)^2+s_I_component(i,1,k+3)^2+d_I_component(j,1,k+3)^2+s_F_component(i,1,k)^2+d_F_component(j,1,k)^2+s_F_component(i,1,k+3)^2+d_F_component(j,1,k+3)^2);
                similarity_matrix(i,j) = similarity_matrix(i,j) + numerator/denominator ;
            end
        end
    end
    similarity_matrix=similarity_matrix/3;
end