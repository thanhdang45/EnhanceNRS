function similarity = compute_similarity(vector1, vector2)
    attri_num = length(vector1);
    numerator = 0;
    denominator1 = 0;
    denominator2 = 0;
    for i=1:attri_num
        numerator = numerator + vector1(i)*vector2(i);
        denominator1 = denominator1 + vector1(i)^2;
        denominator2 = denominator2 + vector2(i)^2;
    end
    similarity = numerator/sqrt(denominator1*denominator2);
end