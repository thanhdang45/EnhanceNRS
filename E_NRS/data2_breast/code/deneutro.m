function test_output = deneutro(test_output_T_component,test_output_I_component,test_output_F_component,a,b,c,defuzz_value)
    test_num=size(test_output_T_component,1);
    term_num=size(test_output_T_component,2);
    fuzzy_output = zeros(test_num,term_num);
    for i=1:test_num
        for j=1:term_num
            fuzzy_output(i,j)=a*test_output_T_component(i,j)+b*test_output_I_component(i,j)/2+c*test_output_F_component(i,j)/4;
        end
    end
    test_output = zeros(test_num,1);
    for i=1:test_num
        temp = 0;
        for j=1:term_num
            test_output(i)=test_output(i)+defuzz_value(j)*fuzzy_output(i,j);
            temp = temp + fuzzy_output(i,j);
        end
        test_output(i)=test_output(i)/temp;
    end
end