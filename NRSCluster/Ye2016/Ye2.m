function VALUE = Ye2()
filename = 'dmd12.xlsx';
A = get_data(filename);
Data_normal(:,1) = normalization(A(:,1),min(A(:,1)),max(A(:,1)));
Data_normal(:,2) = normalization(A(:,2),min(A(:,2)),max(A(:,2)));
Data_normal(:,3) = normalization(A(:,3),min(A(:,3)),max(A(:,3)));
%
%
matrix = creatmatrixtif(Data_normal);
%matrix = creat_data(5,6);
simMatrix = calSimMatrix(matrix);
EquiMatrix = calEquiMatrix(simMatrix);
%Q_TMP = NMS(Q_ECA);
while isequal(EquiMatrix,simMatrix) == 0
    simMatrix = EquiMatrix;
    EquiMatrix = calEquiMatrix(simMatrix);
end
cuttingMatrix = cutting_matrix(EquiMatrix,0.95);
clust = clusterECA(cuttingMatrix);
% 
%figure;
figure;
scatter3(Data_normal(:,1),Data_normal(:,2),Data_normal(:,3),30,clust,'filled');
title('Ye16');
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

% Get data from excel
function matrix = get_data(file)
% Set option
import ValidityIndex.*
filename = file;
sheet = 'Data';
xlRange = 'A:C';
% Load database
[num,txt,raw]  = xlsread(filename,sheet,xlRange,'basic');
matrix = num2cell(num);
% [m,n] = size(matrix);
%   for i=1:m 
%       s=double(raw{i,3});
%       a = sum(s);
%       matrix{i,3} = [a];
%   end
matrix = cell2mat(matrix);
end
function [Matrix_data] = creat_data(num,field)
    Matrix_data = cell(num,field);
    Matrix_data{1,1} = {0.3,0.2,0.5};
    Matrix_data{1,2} = {0.6,0.3,0.1};
    Matrix_data{1,3} = {0.4,0.3,0.3};
    Matrix_data{1,4} = {0.8,0.1,0.1};
    Matrix_data{1,5} = {0.1,0.3,0.6};
    Matrix_data{1,6} = {0.5,0.2,0.4};
    Matrix_data{2,1} = {0.6,0.3,0.3};
    Matrix_data{2,2} = {0.5,0.4,0.2};
    Matrix_data{2,3} = {0.6,0.2,0.1};
    Matrix_data{2,4} = {0.7,0.2,0.1};
    Matrix_data{2,5} = {0.3,0.1,0.6};
    Matrix_data{2,6} = {0.4,0.3,0.3};
    Matrix_data{3,1} = {0.4,0.2,0.4};
    Matrix_data{3,2} = {0.8,0.2,0.1};
    Matrix_data{3,3} = {0.5,0.3,0.1};
    Matrix_data{3,4} = {0.6,0.1,0.2};
    Matrix_data{3,5} = {0.4,0.1,0.5};
    Matrix_data{3,6} = {0.3,0.2,0.2};
    Matrix_data{4,1} = {0.2,0.4,0.4};
    Matrix_data{4,2} = {0.4,0.5,0.1};
    Matrix_data{4,3} = {0.9,0.2,0.0};
    Matrix_data{4,4} = {0.8,0.2,0.1};
    Matrix_data{4,5} = {0.2,0.3,0.5};
    Matrix_data{4,6} = {0.7,0.3,0.1};
    Matrix_data{5,1} = {0.5,0.3,0.2};
    Matrix_data{5,2} = {0.3,0.2,0.6};
    Matrix_data{5,3} = {0.6,0.1,0.3};
    Matrix_data{5,4} = {0.7,0.1,0.1};
    Matrix_data{5,5} = {0.6,0.2,0.2};
    Matrix_data{5,6} = {0.5,0.2,0.3};
end
% Normalization data
function nomar_value = normalization(array,min,max)
   [n,m] = size(array);
   for i=1:n
       nomar_value(i,1) = (array(i,1)- min)/(max- min);
   end
end
function simMatrix = calSimMatrix(matrix)
    [m,n] = size(matrix);
    simMatrix = zeros(m);
    for i=1:m
        for j=1:m
            d = 0;
            for p=1:3 % example p=1:6
                Ti = matrix{i,p}{1};
                Ii = matrix{i,p}{2};
                Fi = matrix{i,p}{3};
                Tj = matrix{j,p}{1};
                Ij = matrix{j,p}{2};
                Fj = matrix{j,p}{3};
                d = d + (abs(Ti-Tj))^2 + (abs(Ii-Ij))^2 + (abs(Fi-Fj))^2;
            end
            simMatrix(i,j) = 1- (d/9) ^ (1/2); % example d/18
        end
    end
end
% Created matrix TIF
% function [Matrix_data] = creat_data(num,field)
%     Matrix_data = cell(num,field);
% end

% Created matrix TIF with data
function matrix_TIF = creatmatrixtif(data)
    [n,m] = size(data);
    matrix_TIF = creat_data(n,m);
    for i=1:n
        for j=1:m
            matrix_TIF{i,j}{1,1} = calcT(data(i,j), 0.2, 0.4, 0.6, 0.8);
            matrix_TIF{i,j}{1,2} = calcI(data(i,j), 0.2, 0.4, 0.6, 0.8);
            matrix_TIF{i,j}{1,3} = calcF(data(i,j), 0.2, 0.4, 0.6, 0.8);
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
        Tmp = (x-c2)/(c3-c2);
    end
    if (c2>x)&&(x>=c1)
        Tmp = (c2-x)/(c2-c1);
    end
   F_value = Tmp;
end

% Calculator Matrix ECA
function Q_ECA = ECA(matrix_data,l1,e1,e2,e3)
    [n,m]=size(matrix_data);
    Q_ECA = creat_data(n,n);
    for i=1:n
        for j=i:n
            [P1,P2]= PVALUE(matrix_data,i,j,l1,e1,e2,e3);
            Q_ECA{i,j}{1,1} = 1- nthroot(P1,l1);
            Q_ECA{i,j}{1,2} = nthroot(P2,l1);
            Q_ECA{i,j}{1,3} = Q_ECA{i,j}{1,2};
        end
    end
    
     for i=1:n
        for j=1:i
              Q_ECA{i,j} =  Q_ECA{j,i};
        end
     end
     
end

% Calculator Matrix ECA
function EquiMatrix = calEquiMatrix(simMatrix)
    [n] = size(simMatrix,1);
    EquiMatrix = zeros(n);
    for i=1:n
        for j=i:n
            EquiMatrix(i,j) = FindMaxNM(i,j,simMatrix);
        end
    end
    for i=1:n
        for j=1:i
            EquiMatrix(i,j) = EquiMatrix(j,i);
        end
    end
end

function Equi_value = FindMaxNM(i,j,EquiMatrix)
    [n] = size(EquiMatrix,1);
    max = 0;
    for k=1:n
        tmp = min(EquiMatrix(i,k),EquiMatrix(k,j));
        if tmp> max
            max = tmp;
        end
    end
    Equi_value = max;
end

% Cutting Matrix
function cuttingMatrix = cutting_matrix(EquiMatrix,lambda)
   [m] = size(EquiMatrix,1);  
   cuttingMatrix = zeros(m);
   for i=1:m
       for j=1:m
           if EquiMatrix(i,j) >= lambda
               cuttingMatrix(i,j) = 1;
           else
               cuttingMatrix(i,j) = 0;
           end
       end
   end
end

% Cluster
function C = clusterECA(L_matrix)
    C = cell(1); 
    C{1}{1} = 1; 
    [m,n] = size(L_matrix); 
    for i=2:m % 
        [c_m,c_n] = size(C); 
        for k=1:c_n 
            check = 1; 
            if isequal(L_matrix(:,i),L_matrix(:,C{k}{1}))==1 
                [ck_m,ck_n] = size(C{k}); 
                C{k}{ck_n+1} = i; 
                check = 0; 
                break;
            end
        end
        if check ==1 
           C{c_n+1}{1} = i;
        end
    end
    C = convert(C);
end

function Check_value = compare(col1,col2)
    Check_value = 1;
n = size(col1,1)
k = 0;
    for i=1:n
        if cell2mat(col1(i,1))~= cell2mat(col2(i,1))
            if k<(n/15) 
                k=k+1;
            else
                Check_value = 0;
            end
        end
    end
end
% Cluster convert matlab
function Cluster_Conv = convert(matrix)
    [n,m] = size(matrix);
    for i=1:m
        [matrix_n,matrix_m] = size(matrix{1,i});
        for j=1:matrix_m
            Cluster_Conv(matrix{1,i}{1,j},1) = i;
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
    epsilon = 1;
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
    epsilon = 2;
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
