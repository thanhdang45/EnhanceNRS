function new_mat=normalize(old_mat)
    row_n=size(old_mat,1);
    col_n=size(old_mat,2);
    new_mat=zeros(row_n,col_n);
    max_val=max(old_mat);
    min_val=min(old_mat);
    dif_val=max_val-min_val;
    for i=1:row_n
        for j=1:col_n
            new_mat(i,j)=(old_mat(i,j)-min_val(j))/dif_val(j);
        end
    end
end