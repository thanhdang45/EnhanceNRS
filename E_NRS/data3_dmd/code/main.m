%main function
clear;
clc;
load data;
label_num = 3;
mses  = zeros(1,20);
for t=1:20
    tic;
    T_para = zeros(3,3,4);
    I_para = zeros(3,3,4);
    F_para = zeros(3,3,4);
    T_para(1,:,:) = make_increasing_matrix(3,4);
    T_para(2,:,:) = T_para(1,:,:);
    T_para(3,:,:) = T_para(1,:,:);
    I_para(1,:,:) = make_increasing_matrix(3,4);
    I_para(2,:,:) = I_para(1,:,:);
    I_para(3,:,:) = I_para(1,:,:);
    F_para(1,:,:) = make_increasing_matrix(3,4);
    F_para(2,:,:) = F_para(1,:,:);
    F_para(3,:,:) = F_para(1,:,:);
    [train_T_component,train_I_component,train_F_component] = comput_all_TIF([train_input train_output],T_para,I_para,F_para);
    [test_T_component,test_I_component, test_F_component] = comput_all_TIF(test_input,T_para,I_para,F_para);
    similarity_matrix=compute_similarity(train_T_component,train_I_component,train_F_component,test_T_component,test_I_component,test_F_component);
    NRS_matrix=compute_NRS(similarity_matrix);
    [test_output_T_component,test_output_I_component,test_output_F_component]=predict_output(NRS_matrix,train_T_component,train_I_component,train_F_component);
    a=0.3;
    b=0.2;
    c=0.5;
    defuzz_value = [0 0.5 1];
    predicted_output = deneutro(test_output_T_component,test_output_I_component,test_output_F_component,a,c,b,defuzz_value);
    error = test_output-predicted_output;
    
    time=toc;
    fprintf('label1 a1=%f, a2=%f, a3=%f, a4=%f\n',T_para(1,1,1),T_para(1,1,2),T_para(1,1,3),T_para(1,1,4));
    fprintf('label1 b1=%f, b2=%f, b3=%f, b4=%f\n',I_para(1,1,1),I_para(1,1,2),I_para(1,1,3),I_para(1,1,4));
    fprintf('label1 c1=%f, c2=%f, c3=%f, c4=%f\n',F_para(1,1,1),F_para(1,1,2),F_para(1,1,3),F_para(1,1,4));
    
    fprintf('label2 a1=%f, a2=%f, a3=%f, a4=%f\n',T_para(1,2,1),T_para(1,2,2),T_para(1,2,3),T_para(1,2,4));
    fprintf('label2 b1=%f, b2=%f, b3=%f, b4=%f\n',I_para(1,2,1),I_para(1,2,2),I_para(1,2,3),I_para(1,2,4));
    fprintf('label2 c1=%f, c2=%f, c3=%f, c4=%f\n',F_para(1,2,1),F_para(1,2,2),F_para(1,2,3),F_para(1,2,4));
    
    fprintf('label3 a1=%f, a2=%f, a3=%f, a4=%f\n',T_para(1,3,1),T_para(1,3,2),T_para(1,3,3),T_para(1,3,4));
    fprintf('label3 b1=%f, b2=%f, b3=%f, b4=%f\n',I_para(1,3,1),I_para(1,3,2),I_para(1,3,3),I_para(1,3,4));
    fprintf('label3 c1=%f, c2=%f, c3=%f, c4=%f\n',F_para(1,3,1),F_para(1,3,2),F_para(1,3,3),F_para(1,3,4));
    mses(t) = mse(error);
    fprintf('mse %f\n', mses(t));
    fprintf('time %f\n',time);
end
fprintf('mean of mse %f\n',mean(mses));
fprintf('varience of mse %f\n',mse(mses-mean(mses)));