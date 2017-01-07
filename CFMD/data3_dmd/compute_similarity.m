function similarity = compute_similarity(vector1, vector2)
    attri_num = length(vector1);
    mean_vec1= mean(vector1);
    mean_vec2= mean(vector2);
    numerator = 0;
    for i=1:attri_num
        numerator = numerator + (vector1(i)-mean_vec1)*(vector2(i)-mean_vec2);
    end
    denominator = sqrt(mse(vector1-mean_vec1)*mse(vector2-mean_vec2));
    similarity = numerator/denominator;
end