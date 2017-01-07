function increasing_matrix = make_increasing_matrix(row_n, col_n)
    increasing_matrix = zeros(row_n, col_n);
    for i=1:row_n
        random_row = rand(1,col_n+1);
        sum_row = sum(random_row);
        temp_sum=0;
        for j=1:col_n
            temp_sum = temp_sum+random_row(j);
            increasing_matrix(i,j) = temp_sum/ sum_row;
        end
    end    
end