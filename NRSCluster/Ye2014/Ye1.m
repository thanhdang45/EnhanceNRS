function VALUE = Ye1()
filename = 'Diabetes2.xlsx';
A = get_data(filename);
Data_normal(:,1) = normalization(A(:,1),min(A(:,1)),max(A(:,1)));
Data_normal(:,2) = normalization(A(:,2),min(A(:,2)),max(A(:,2)));
Data_normal(:,3) = normalization(A(:,3),min(A(:,3)),max(A(:,3)));
matrix = creatmatrixtif(Data_normal);
%matrix=test();
n = size (matrix,2);
b = CalcMatrixM(matrix,1/n);
c = cutting(b,0.92);
clust = cluster(c);
numcluster = max(clust);
figure;
scatter3(Data_normal(:,1),Data_normal(:,2),Data_normal(:,3),30,clust,'filled');
title('Ye14');
DBvalue = DB(Data_normal, findcentroid(clust,Data_normal),clust);
SSWCvalue = SWC(Data_normal, findcentroid(clust,Data_normal),clust);
IFVvalue = IFV(Data_normal, findcentroid(clust,Data_normal),clust);
PBMvalue = PBM(Data_normal, findcentroid(clust,Data_normal),clust);
VALUE = [DBvalue,SSWCvalue,IFVvalue,PBMvalue];
sprintf('DB NRS: %f',DBvalue);
sprintf('SWC NRS: %f',SSWCvalue);
sprintf('IFV NRS: %f',IFVvalue);
sprintf('PBM NRS: %f',PBMvalue);
end

function matrix = get_data(file,sheet,xlRange)
    % Set option
    import ValidityIndex.*;
    filename = file;
    sheet = 'Data';
    xlRange = 'A:C';
    % Load database
    [num,txt,raw]  = xlsread(filename,sheet,xlRange,'basic');
    matrix = num2cell(num);
    matrix = cell2mat(matrix);
end

% Normalization data
function nomar_value = normalization(array,min,max)
   [n,m] = size(array);
   for i=1:n
       nomar_value(i,1) = (array(i,1)- min)/(max- min);
   end
end

% Created matrix TIF
function [Matrix_data] = creat_data(num,field)
    Matrix_data = cell(num,field);
%     Matrix_data{1,1} = {0.3,0.2,0.5};
%     Matrix_data{1,2} = {0.6,0.3,0.1};
%     Matrix_data{1,3} = {0.4,0.3,0.3};
%     Matrix_data{1,4} = {0.8,0.1,0.1};
%     Matrix_data{1,5} = {0.1,0.3,0.6};
%     Matrix_data{1,6} = {0.5,0.2,0.4};
%     Matrix_data{2,1} = {0.6,0.3,0.3};
%     Matrix_data{2,2} = {0.5,0.4,0.2};
%     Matrix_data{2,3} = {0.6,0.2,0.1};
%     Matrix_data{2,4} = {0.7,0.2,0.1};
%     Matrix_data{2,5} = {0.3,0.1,0.6};
%     Matrix_data{2,6} = {0.4,0.3,0.3};
%     Matrix_data{3,1} = {0.4,0.2,0.4};
%     Matrix_data{3,2} = {0.8,0.2,0.1};
%     Matrix_data{3,3} = {0.5,0.3,0.1};
%     Matrix_data{3,4} = {0.6,0.1,0.2};
%     Matrix_data{3,5} = {0.4,0.1,0.5};
%     Matrix_data{3,6} = {0.3,0.2,0.2};
%     Matrix_data{4,1} = {0.2,0.4,0.4};
%     Matrix_data{4,2} = {0.4,0.5,0.1};
%     Matrix_data{4,3} = {0.9,0.2,0.0};
%     Matrix_data{4,4} = {0.8,0.2,0.1};
%     Matrix_data{4,5} = {0.2,0.3,0.5};
%     Matrix_data{4,6} = {0.7,0.3,0.1};
%     Matrix_data{5,1} = {0.5,0.3,0.2};
%     Matrix_data{5,2} = {0.3,0.2,0.6};
%     Matrix_data{5,3} = {0.6,0.1,0.3};
%     Matrix_data{5,4} = {0.7,0.1,0.1};
%     Matrix_data{5,5} = {0.6,0.2,0.2};
%     Matrix_data{5,6} = {0.5,0.2,0.3};
end

% Created matrix TIF with data
function matrix_TIF = creatmatrixtif(data)
    [n,m] = size(data);
    matrix_TIF = creat_data(n,m);
    for i=1:n
        for j=1:m
            matrix_TIF{i,j}{1} = calcT(data(i,j), 0.02, 0.25, 0.47, 0.7);
            matrix_TIF{i,j}{2} = calcI(data(i,j), 0.16, 0.28, 0.41, 0.53);
            matrix_TIF{i,j}{3} = calcF(data(i,j), 0.15, 0.31, 0.48, 0.65);
        end
    end        
end

% calculator T-I-F
function T_value = calcT(x, a1, a2, a3, a4)
    if x>=a4 
        Tmp = 0;
    end
    if x<a1
        Tmp = 0;
    end
    if (x<a4)&&(a3<=x)
        Tmp = (x-a3)/(a4-a3);
    end
    
    if (x<a3)&&(x>=a2)
       Tmp = (a3-x)/(a3-a2);
    end
    if (x<a2)&&(x>=a1)
        Tmp = (x-a1)/(a2-a1);
    end
         T_value   = Tmp;        
end
    
function I_value = calcI(x, b1, b2, b3, b4)
    if  x>=b4 
        Tmp = 1;
    end
    if x<b1
        Tmp = 1;
    end
    if (b4>x)&&(x>=b3)
        Tmp = (b4-x)/(b4-b3);
    end
    if (b3>x)&&(x>=b2)
        Tmp = (x-b2)/(b3-b2);
    end
    if (b2>x)&&(x>=b1)
        Tmp = (b2-x)/(b2-b1);
    end
   I_value = Tmp;
end
    
function F_value = calcF(x, c1, c2, c3, c4)
    if  x>=c4 
        Tmp = 1;
    end
    if x<c1
        Tmp = 1;
    end
    if  (c4>x)&&(x>=c3)
        Tmp = (c4-x)/(c4-c3);
    end
    if (c3>x)&&(x>=c2)
        Tmp = x/c3;
    end
    if (c2>x)&&(x>=c1)
        Tmp = (c2-x)/(c2-c1);
    end
   F_value = Tmp;
end

function MatrixM_value = CalcMatrixM(matrix,w)
    [m,n] = size(matrix);
    for i=1:m
        for j=i:m
           MatrixM_value(j,i) = CalcM(i,j,matrix,w);
        end
    end
end

function M_value = CalcM(Si,Sj,matrix,w)
    [m,n] = size(matrix(Si,:));
    Sum = 0;
    for i=1:n
        A = min(matrix{Si,i}{1,1},matrix{Sj,i}{1,1})+min(matrix{Si,i}{1,2},matrix{Sj,i}{1,2})+min(matrix{Si,i}{1,3},matrix{Sj,i}{1,3});
        B = double (matrix{Si,i}{1,1}+matrix{Sj,i}{1,1}+matrix{Si,i}{1,2}+matrix{Sj,i}{1,2}+matrix{Si,i}{1,3}+matrix{Sj,i}{1,3})*0.5;
        Sum = Sum + (w*A/B);
    end
    M_value = Sum;

end

function Matrix_cuting  = cutting(matrix,p)
 [m,n] = size(matrix);
  for i=1:m
        for j=i:m
           if matrix(j,i)<p
               Matrix_cuting(j,i) = 0;
           else
               Matrix_cuting(j,i) = 1;
           end
        end
  end
    for i=1:m
        for j=i:m
           Matrix_cuting(i,j) = Matrix_cuting(j,i);
        end
    end
end

function friends = findDirectFriends(k,Matrix_cuting)
    [m,n] = size(Matrix_cuting);
    friends = zeros(m,1);
    friends(k) = 1;
    for j=1:m
        if (Matrix_cuting(k,j) == 1)
            friends(j)=1;
        end
    end
end
function isDirectFriend = isDirectFriend(i,j,Matrix_cuting)
    friendsOfI = findDirectFriends(i,Matrix_cuting);
    isDirectFriend = 0;
    if friendsOfI(j) == 1
        isDirectFriend = 1;
    end
end
function isRelatedFriend = isRelatedFriend(i,j,Matrix_cuting)
    [m,n] = size(Matrix_cuting);
    sum = zeros(m,1);
    sum = findDirectFriends(i,Matrix_cuting) + findDirectFriends(j,Matrix_cuting);
    isRelatedFriend = 0;
    for k=1:m
        if (sum(k) >=2)
            isRelatedFriend = 1;
            break;
        end
    end
end
function clust = cluster(cutting_matrix)
[m,n] = size(cutting_matrix);
friends = cutting_matrix;
    for i=1:n
        initialFriends = zeros(m,1);
        changedFriends = ones(m,1);
        while isequal(initialFriends,changedFriends) == 0
            initialFriends = friends(:,i);
            for j=1:m
                if(friends(j,i)==1)
                    for k=1:m
                        friends(k,i) = max(friends(k,i),friends(k,j));
                        friends(k,j) = friends(k,i);
                    end
                end
            end
            changedFriends = friends(:,i);
        end
    end
    clust = zeros(m,1);
    uniquerows = unique(friends,'rows');
    [p,q] = size(uniquerows);
    k = 0;
    for j=1:p
        k = k+1;
        for i=1:q
            if (uniquerows(j,i)==1)
               clust(i) = k;
            end
        end
    end
end
function centroid_array = findcentroid(clust,data)
    numclust = max(clust);
    [n,m] = size(clust);
    for i=1:numclust
        member=0;
        sum1=0;sum2=0;sum3=0;
        for j=1:n
            if clust(j,1)==i
                sum1=sum1+data(j,1);
                sum2=sum2+data(j,2);
                sum3=sum3+data(j,3);
                member = member+1;
            end
        end
        centroid_array(i,1) = sum1/member;
        centroid_array(i,2) = sum2/member;
        centroid_array(i,3) = sum3/member;
    end
end

function DB_value = DB(data, center, clust)
    numClust = max(clust);
    S = zeros(1, numClust);
    for i = 1 : numClust
        index = find(clust(:,1) == i);
        [n,m] = size(index);
        for j = 1:n
            S(i) = S(i) + norm(data(j,:) - center(i,:)) ^ 2;
        end
        S(i) = sqrt(S(i)/size(index, 2));
    end
    DB_value = 0;
    for i = 1:numClust
        maxSM = 0;
        for j = 1:numClust
           if j ~= i
               if norm(center(i, :) - center(j, :)) ~=0
               temp = (S(i) + S(j))/norm(center(i, :) - center(j, :));
               maxSM = max(maxSM, temp);
               end
           end
        end
        DB_value = DB_value + maxSM;
    end
    
    DB_value = DB_value/numClust;
end

function t = calcSumDistDataPoint2X(data, X)
    temp = data - X(ones(size(data, 1), 1), :);
    temp = temp.^2;
    temp = sum(temp, 2);
    temp = sqrt(temp);
    t = sum(temp);
end

function SWC_value = SWC(data, center,clust)
    numClust = max(clust);
    SWC_value = 0;
    epsilon = 10;
    for i = 1 : numClust
        index = find(clust(:,1) == i);
        clustData = data(index, :);

        for j = index
            a_i_j = calcSumDistDataPoint2X(clustData, data(j, :)) / size(index, 2);
            b_i_j = 10^6;

            for k = 1 : numClust
                if k ~= i
                    index_k = find(clust(k,:) == i);

                    clustData_k = data(index_k, :);

                    d_k_j = calcSumDistDataPoint2X(clustData_k, data(j, :)) / size(index_k, 2);
                     if d_k_j~=0
                    b_i_j = min(b_i_j, d_k_j);
                     end
                end
            end
            SWC_value = SWC_value + (b_i_j - a_i_j) / max(a_i_j, b_i_j);
           
        end
    end
    SWC_value = SWC_value / (epsilon*size(data, 1));
end


function PBM_value = PBM(data, center, clust)
E_1 = calcSumDistDataPoint2X(data, mean(data));

numClust = max(clust);
E_k = 0;

for i = 1 : numClust
    index = find(data(i,:) == i);
    clustData = data(index, :);
    E_k = E_k +  calcSumDistDataPoint2X(clustData, center(i, :));
end

D_k = 0;
for i = 1:numClust-1
    for j = i+1:numClust
        D_k = max(D_k, norm(center(i, :) - center(j, :)));
    end
end
    
PBM_value = (E_1 * D_k / (numClust * E_k)) ^ 2;
end


function IFV_value = IFV(data, center, clust)
    numClust = max(clust);
    sigmaD = 0;
    sum = 0;
    sizeData = size(data,1);
    epsilon = 100;
    for i = 1:numClust
        tg1 = 0;
        tg2 = 0;
        for j = 1:sizeData
            if clust(j,1) == i 
                clust(j,1) = 1 - eps;
            end
            
            tg1 = tg1 + log(clust(i, 1))/log(2);
            tg2 = tg2 + clust(i, 1)^2;
            sigmaD = sigmaD + norm(data(j, :) - center(i, :))^2;
        end
        
        tg = (log(numClust)/log(2) - tg1/sizeData)^2;
        tg2 = tg2/sizeData;
        
        sum = sum + tg * tg2;
    end
    sigmaD = epsilon*sigmaD/(numClust * sizeData);
    
    calcSDmax = 0;
    for i = 1:numClust-1
        for j = i+1:numClust
            calcSDmax = max(calcSDmax, norm(center(i, :) - center(j, :))^2);
        end
    end
    
    IFV_value  = (sum * calcSDmax) / (sigmaD * numClust);
end