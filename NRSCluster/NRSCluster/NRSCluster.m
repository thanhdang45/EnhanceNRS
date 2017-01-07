function VALUE = NRSCluster()
filename = 'Diabetes2.xlsx';
data = get_data(filename);
matrix = creatmatrixtif(data);
[M, S, P] = NRS(matrix,1,0.4,0.3,0.3);
M_TMP = NMS(M);
S_TMP = NMS(S);
P_TMP = NMS(P);  
iter = 0;
while isequal(M_TMP,M) == 0
    M = M_TMP;
    M_TMP = NMS(M);
    iter = iter + 1;
    disp(iter);
end
disp('End First Iteration');
while isequal(S_TMP,S) == 0
    S = S_TMP;
    S_TMP = NMS(S);
    iter = iter + 1;
    disp(iter);
end
disp('End Second Iteration');
while isequal(P_TMP,P) == 0
    P = P_TMP;
    P_TMP = NMS(P);
    iter = iter + 1;
    disp(iter);
end
disp('End Last Iteration');
M_star = cutting_lamda(M_TMP,0.92,0,1/7,1/2,1/3,1);
S_star = cutting_lamda(S_TMP,0.92,0,1/6,1/4,1/2,1);
P_star = cutting_lamda(P_TMP,0.92,0,1/8,1/4,1/2,1);
B = [M_star,S_star,P_star];
T = cell2table(B);
writetable(T,'ketquafull.xlsx');
clust = cluster(B);
max(clust)
figure;

Data_normal(:,1) = normalization(data(:,1),min(data(:,1)),max(data(:,1)));
Data_normal(:,2) = normalization(data(:,2),min(data(:,2)),max(data(:,2)));
Data_normal(:,3) = normalization(data(:,3),min(data(:,3)),max(data(:,3)));
scatter3(Data_normal(:,1),Data_normal(:,2),Data_normal(:,3),30,clust,'filled');
title('Our Method');
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
    %Diabetes1
    params =  [29 37 45 53 61; 160 182 204 226 248; 0.81 0.845 0.88 0.915 0.95];
    %Breast1
    %params =  [26	36	41	46	56; 12	20	23	27	35; 0	0.5	1	1.5	2];
    %RHC1
    %params =  [35	52	61	70	88; 26	61	78	95	130; 0	0.25	0.5	0.75	1];
    %dmd11
    %params = [23 40	48	56	72; 141	186	208	231	275; 29	35	38	41	47];

    for i=1:n
        for j=1:m
           matrix_TIF{i,j}{1,1} = T_low(data(i,j), params(j,1), params(j,2), params(j,3), params(j,4), params(j,5));
           matrix_TIF{i,j}{1,2} = I_low(data(i,j), params(j,1), params(j,2), params(j,3), params(j,4), params(j,5));
           matrix_TIF{i,j}{1,3} = F_low(data(i,j), params(j,1), params(j,2), params(j,3), params(j,4), params(j,5));
           
           matrix_TIF{i,j}{2,1} = T_middle(data(i,j), params(j,1), params(j,2), params(j,3), params(j,4), params(j,5));
           matrix_TIF{i,j}{2,2} = I_middle(data(i,j), params(j,1), params(j,2), params(j,3), params(j,4), params(j,5));
           matrix_TIF{i,j}{2,3} = F_middle(data(i,j), params(j,1), params(j,2), params(j,3), params(j,4), params(j,5));
           
           matrix_TIF{i,j}{3,1} = T_high(data(i,j), params(j,1), params(j,2), params(j,3), params(j,4), params(j,5));
           matrix_TIF{i,j}{3,2} = I_high(data(i,j), params(j,1), params(j,2), params(j,3), params(j,4), params(j,5));
           matrix_TIF{i,j}{3,3} = F_high(data(i,j), params(j,1), params(j,2), params(j,3), params(j,4), params(j,5));
        end
    end        
end
% Calculator Matrix NRS
function [M, S, P] = NRS(matrix_data,l1,e1,e2,e3)
    [n,m]=size(matrix_data);
    M = creat_data(n,n);
    for i=1:n
        for j=i:n
            [MAX,MIN]= PVALUE(matrix_data,i,j,l1,e1,e2,e3,1);
            M{i,j}{1,1} = 1- nthroot(MAX,l1);
            M{i,j}{1,2} = nthroot(MIN,l1);
            M{i,j}{1,3} = M{i,j}{1,2};
        end
    end
    S = creat_data(n,n);
    for i=1:n
        for j=i:n
            [MAX,MIN]= PVALUE(matrix_data,i,j,l1,e1,e2,e3,2);
            S{i,j}{1,1} = 1- nthroot(MAX,l1);
            S{i,j}{1,2} = nthroot(MIN,l1);
            S{i,j}{1,3} = S{i,j}{1,2};
        end
    end
    P = creat_data(n,n);
    for i=1:n
        for j=i:n
            [MAX,MIN]= PVALUE(matrix_data,i,j,l1,e1,e2,e3,3);
            P{i,j}{1,1} = 1- nthroot(MAX,l1);
            P{i,j}{1,2} = nthroot(MIN,l1);
            P{i,j}{1,3} = P{i,j}{1,2};
        end
    end
     for i=1:n
        for j=1:i
              M{i,j} =  M{j,i};
              S{i,j} =  S{j,i};
              P{i,j} =  P{j,i};
        end
     end
end

function [maxValue, minValue] = PVALUE(matrix_data,ai,ak,l1,e1,e2,e3,i)
    matrix_tmp = matrix_data;
    P = zeros(1,3);
    Ti_low = matrix_tmp{ai,i}{1,1}; Ii_low = matrix_tmp{ai,i}{1,2}; Fi_low = matrix_tmp{ai,i}{1,3};
    Ti_middle = matrix_tmp{ai,i}{2,1}; Ii_middle = matrix_tmp{ai,i}{2,2}; Fi_middle = matrix_tmp{ai,i}{2,3};
    Ti_high = matrix_tmp{ai,i}{3,1}; Ii_high = matrix_tmp{ai,i}{3,2}; Fi_high = matrix_tmp{ai,i}{3,3};

    Tk_low = matrix_tmp{ak,i}{1,1}; Ik_low = matrix_tmp{ak,i}{1,2}; Fk_low = matrix_tmp{ak,i}{1,3};
    Tk_middle = matrix_tmp{ak,i}{2,1}; Ik_middle = matrix_tmp{ak,i}{2,2}; Fk_middle = matrix_tmp{ak,i}{2,3};
    Tk_high = matrix_tmp{ak,i}{3,1}; Ik_high = matrix_tmp{ak,i}{3,2}; Fk_high = matrix_tmp{ak,i}{3,3};

    P(1) = e1*((abs(Ti_low-Tk_low))^l1) +  e2*((abs(Ii_low-Ik_low))^l1)+ e3*((abs(Fi_low-Fk_low))^l1);
    P(2) = e1*((abs(Ti_middle-Tk_middle))^l1) +  e2*((abs(Ii_middle-Ik_middle))^l1)+ e3*((abs(Fi_middle-Fk_middle))^l1);
    P(3) = e1*((abs(Ti_high-Tk_high))^l1) +  e2*((abs(Ii_high-Ik_high))^l1)+ e3*((abs(Fi_high-Fk_high))^l1);
    minValue = min(P);
    maxValue = max(P);
    
end

% Calculator Matrix NRS
function Q_NMS = NMS(Q_NRS)
    [n] = size(Q_NRS,1);
    Q_NMS = creat_data(n, n);
    for i=1:n
        for j=i:n
            Q_NMS{i,j}{1,1} = FindMaxNM(i,j,Q_NRS,1);
            Q_NMS{i,j}{1,2} = FindMinNM(i,j,Q_NRS,2);
            Q_NMS{i,j}{1,3} = FindMinNM(i,j,Q_NRS,3);
        end
    end
    for i=1:n
        for j=1:i
            Q_NMS{i,j} = Q_NMS{j,i};
        end
    end
end

function Q_NMS_value = FindMaxNM(i,j,Q_NMS,tif)
    [n] = size(Q_NMS,1);
    max = 0;
    for k=1:n
        tmp = min(Q_NMS{i,k}{1,tif},Q_NMS{k,j}{1,tif});
        if tmp> max 
            max = tmp;
        end
    end
    Q_NMS_value = max;
end


function Q_NMS_value = FindMinNM(i,j,Q_NMS,tif)
    [n] = size(Q_NMS,1);
    min = 1;
    for k=1:n
       tmp = max(Q_NMS{i,k}{1,tif},Q_NMS{k,j}{1,tif});
       if tmp < min 
            min = tmp;
       end
    end
    Q_NMS_value = min;
end

% Check neutrosophic matrix
function check = check_neutr(Q_NRS)
    check = 1;
    [n] = size(Q_NRS,1);
   for i=1:n
       for j=1:n
          if i==j
              if isequal(Q_NRS{i,j},{1,0,0})
              else check = 0;
              end
          else 
               if isequal(Q_NRS{i,j},Q_NRS{j,i})
              else check = 0;
               end
          end
       end
   end
end

% Cutting Matrix
% function L_matrix = cutting_lamda(Q_matrix,l1,a1,a2,a3,a4,a5)
%    [m,n] = size(Q_matrix);
%    L_matrix = cell(m,n);
%    for j=1:m
%        for k=1:n
%             if l1 > (1- Q_matrix{j,k}{1,3})
%                 L_matrix{j,k} = a1;
%             elseif l1 > (1-Q_matrix{j,k}{1,2})
%                 L_matrix{j,k} = a2;
%             elseif (l1 > Q_matrix{j,k}{1,1}) && (l1 < (1- Q_matrix{j,k}{1,3}))
%                 L_matrix{j,k} = a3;
%             elseif (l1 > Q_matrix{j,k}{1,1}) && (l1 > ((1-Q_matrix{j,k}{1,2})/2))
%                 L_matrix{j,k} = a4;
%             elseif Q_matrix{j,k}{1,1} > l1
%                 L_matrix{j,k} = a5;
%             end
%        end
%    end            
% end
function L_matrix = cutting_lamda(Q_matrix,l1,a1,a2,a3,a4,a5)
   [m,n] = size(Q_matrix);
   L_matrix = cell(m,n);
   for j=1:m
       for k=1:n
            if l1 > (1- Q_matrix{j,k}{1,3})
                L_matrix{j,k} = a1;
%             elseif l1 > (1-Q_matrix{j,k}{1,2})
%                 L_matrix{j,k} = a2;
            elseif (l1 > Q_matrix{j,k}{1,1}) && (l1 < (1- Q_matrix{j,k}{1,3}))
                L_matrix{j,k} = a3;
%             elseif (l1 > Q_matrix{j,k}{1,1}) && (l1 > ((1-Q_matrix{j,k}{1,2})/2))
%                 L_matrix{j,k} = a4;
            elseif Q_matrix{j,k}{1,1} > l1
                L_matrix{j,k} = a5;
            end
       end
   end            
end

% Cluster
function C = cluster(L_matrix)
    C = cell(1); 
    C{1}{1} = 1; 
    [m,n] = size(L_matrix); 
    for i=2:m
        [c_m,c_n] = size(C); %% compare two rows
        for k=1:c_n 
            check = 1; 
            if sumsquarerow(L_matrix(i,:),L_matrix(C{k}{1}, :)) == 0
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
function sum = sumsquarerow(row1,row2)
    sum = 0;
    [m, n] = size(row1);
    for i=1:n
        sum = sum + sqrt((cell2mat(row1(1,i)) - cell2mat(row2(1,i)))^2);
    end
    %disp(sum);
end

function Check_value = compare(col1,col2)
    Check_value = 1;
n = size(col1,1)
    for i=1:n
        if cell2mat(col1(i,1))~= cell2mat(col2(i,1))
        Check_value = 0;
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
    epsilon = 1;
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%Calculator T-I-F%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function T_value = T_low(x, low,l_middle, middle, h_middle, high)
    if (x <= low) 
        T_value = 1;
    elseif (x > high) 
        T_value = 0;
    else 
        T_value = (high - x)/(high - low);
    end      
end
    
function I_value = I_low(x, low,l_middle, middle, h_middle, high)
    if (x <= low) 
        I_value = 0;
    elseif (x > high) 
        I_value = 0;
    elseif (low <x && x <= middle)
        I_value = (x - low)/(middle - low);
    elseif(middle <x && x <= high)
        I_value = (high - x)/(high - middle);
    end      
end
    
function F_value = F_low(x, low,l_middle, middle, h_middle, high)
    if (x <= low) 
        F_value = 0;
    elseif (x > high) 
        F_value = 1;
    else 
        F_value = (x - low)/(high - low);
    end     
end
%%%
function T_value = T_middle(x, low,l_middle, middle, h_middle, high)
    if (x <= low) 
        T_value = 0;
    elseif (x > high) 
        T_value = 0;
    elseif (low <x && x <= l_middle)
        T_value = (x - low)/(l_middle - low);
    elseif (l_middle <x && x <= h_middle)
        T_value = 1;
    elseif (h_middle <x && x <= high)
        T_value = (high - x)/(high - h_middle);
    end      
end
    
function I_value = I_middle(x, low,l_middle, middle, h_middle, high)
    if (x <= low) 
        I_value = 1;
    elseif (x > high) 
        I_value = 1;
    elseif (low <x && x <= l_middle)
        I_value = (l_middle - x)/(l_middle - low);
    elseif (l_middle <x && x <= h_middle)
        I_value = 0;
    elseif(h_middle <x && x <= high)
        I_value = (x - h_middle)/(high - h_middle);
    end      
end
    
function F_value = F_middle(x, low,l_middle, middle, h_middle, high)
    if (x <= low) 
        F_value = 1;
    elseif (x > high) 
        F_value = 1;
    elseif (low <x && x <= l_middle)
        F_value = (l_middle - x)/(l_middle - low);
    elseif (l_middle <x && x <= h_middle)
        F_value = 0;
    elseif(h_middle <x && x <= high)
        F_value = (high - x)/(high - h_middle);
    end      
end
%%%
function T_value = T_high(x, low,l_middle, middle, h_middle, high)
    if (x <= low) 
        T_value = 0;
    elseif (x > high) 
        T_value = 1;
    else 
        T_value = (x - low)/(high - low);
    end      
end
    
function I_value = I_high(x, low,l_middle, middle, h_middle, high)
    if (x <= low) 
        I_value = 0;
    elseif (x > high) 
        I_value = 0;
    elseif (low <x && x <= middle)
        I_value = (x - low)/(middle - low);
    elseif(middle <x && x <= high)
        I_value = (high - x)/(high - middle);
    end      
end
    
function F_value = F_high(x, low,l_middle, middle, h_middle, high)
    if (x <= low) 
        F_value = 1;
    elseif (x > high) 
        F_value = 0;
    else 
        F_value = (high - x)/(high - low);
    end     
end