function VALUE = Sahin()
filename = 'dmd12.xlsx';
data = get_data(filename);
Data_normal(:,1) = normalization(data(:,1),min(data(:,1)),max(data(:,1)));
Data_normal(:,2) = normalization(data(:,2),min(data(:,2)),max(data(:,2)));
Data_normal(:,3) = normalization(data(:,3),min(data(:,3)),max(data(:,3)));
matrix = creatmatrixtif(Data_normal);
[m,n] = size(matrix);
constM = m;
h = 1;
while m > 1
    D_matrix = calDmatrix(matrix);
    C{h} = getClusCells(D_matrix);
    new_matrix = getNewmatrix(C, matrix ,h);
    matrix = new_matrix;
    m = floor (m/2);
    h = h + 1;
end
clust = convertC(C,constM);
max(clust);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
scatter3(Data_normal(:,1),Data_normal(:,2),Data_normal(:,3),30,clust,'filled');
title('Sahin');
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
% function convert cluster
function Cluster_Conv = convert(matrix)
    [n,m] = size(matrix);
    for i=1:m
        [matrix_n,matrix_m] = size(matrix{1,i});
        for j=1:matrix_m
            Cluster_Conv(matrix{1,i}{1,j},1) = i;
        end
    end
end
% Get data from excel
function matrix = get_data(file)
    % Set option
    import ValidityIndex.*
    filename = file;
    sheet = 'Data';
    xlRange = 'A:C';
%     xlRange = 'A:D';
    % Load database
    [num,txt,raw]  = xlsread(filename,sheet,xlRange,'basic');
    matrix = num2cell(num);
%     [m,n] = size(matrix);
%     for i=1:m 
%        matrix{i,3} = matrix{i,3}/matrix{i,4};
%     end
%     matrix(:,4) = [];
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
end

% Created matrix TIF with data
function matrix_TIF = creatmatrixtif(data)
    [n,m] = size(data);
    matrix_TIF = creat_data(n,m);
    for i=1:n
        for j=1:m
           matrix_TIF{i,j}{1,1} = calcT(data(i,j), 0.2,0.4,0.6,0.8);
           matrix_TIF{i,j}{1,2} = calcI(data(i,j), 0.2,0.4,0.6,0.8);
           matrix_TIF{i,j}{1,3} = calcF(data(i,j), 0.2,0.4,0.6,0.8);
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
    T_value = Tmp;        
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


% Cluster
%%% ma tran khoang cach phu thuoc ma tran dau vao %%%
function D = calDmatrix(matrix)
[m,n] = size(matrix); 
    D = zeros(m);
    for i=1:m
        for j=1:i
            temp = 0;
            for p=1:3
                for q=1:3
                    temp = temp + (matrix{i,p}{q} - matrix{j,p}{q})^2;
                end
            end
            temp = temp^(1/2);
            D(i,j) = temp/9;
        end
    end
    for i=1:m
        D(i,i) = 100;
        for j=1:i
            D(j,i)= D(i,j);
        end
    end
end
%%% ma tran khoang cach phu thuoc ma tran dau vao %%%
function C = getClusCells(D_matrix)
    [m,n] = size(D_matrix); 
    C = cell(1,ceil(m/2)); 
    for i=1:(ceil(m/2))
        if (i< ceil(m/2))
            [minD,ind] = min(D_matrix(:));
            [p,q] = ind2sub(size(D_matrix),ind);
            delrc = [p q];
            D_matrix(:,delrc) = inf;
            D_matrix(delrc,:) = inf; 
            C{i}{1} = p; 
            C{i}{2} = q; 
        else
            [minD,ind] = min(D_matrix(:));
            [p,q] = ind2sub(size(D_matrix),ind);
            delrc = [p q];
            D_matrix(:,delrc) = inf;
            D_matrix(delrc,:) = inf; 
            C{i}{1} = p; 
            C{i}{2} = q; 
        end
    end
end
function new_matrix = getNewmatrix(C,matrix,h)
    [m,n] = size(matrix);
    [p,q] = size(C{h});
    new_matrix = cell(ceil(m/2),3);
    for i = 1:ceil(m/2)
      new_matrix(i,:) = matrix(C{h}{i}{1},:);
    end
end
function noOfCluster = findCluster (i, myCells, m , iter)
    %find i in myCells
    for k=1:ceil(m/2)
        if (i == myCells{iter}{k}{1} || i == myCells{iter}{k}{2})
            break;
        end
    end
    if (m > 8)% change m
        noOfCluster = findCluster(k,myCells,ceil(m/2),iter+1);
    else
        noOfCluster = k;
        return;
    end
    
end
function U = convertC(myCells, m)
    U = zeros(m,1);
    for i=1:m
        U(i) = findCluster(i, myCells, m, 1);
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
    epsilon=1;
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
    epsilon=1;
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
