function new
    [data,txt,raw]  = xlsread('diabetes.xlsx','Data','A:C','basic');
    data_normal(:,1) = normalization(data(:,1),min(data(:,1)),max(data(:,1)));
    data_normal(:,2) = normalization(data(:,2),min(data(:,2)),max(data(:,2)));
    data_normal(:,3) = normalization(data(:,3),min(data(:,3)),max(data(:,3)));
    data_normal = data_normal(:,1:2);
    cluster_result = kmeans(data_normal,20);
    save('cluster_result.mat','cluster_result');
%     predicted_output = deneutro();
%     load data;
%     error = test_output - predicted_output;
%     fprintf('mse %f\n', mse(error));
     scatter(data_normal(:,1),data_normal(:,2),30,cluster_result,'filled');
end
function nomar_value = normalization(array,min,max)
   [n,m] = size(array);
   for i=1:n
       nomar_value(i,1) = (array(i,1)- min)/(max- min);
   end
end
function test_output = deneutro()
   load data;
   load idx3.mat;
   m = size(test_input,1);
   n = size(train_input,1);
   test_output = zeros(m,1);
   train_data = [train_input train_output];
   tempCell = cell(1);
   for i=1:m
       k=0;
       for j=1:n
           if idx3(i+n) == idx3(j)
               k=k+1;
               tempCell{k} = train_data(j,3);
           end
       end
       tempM = cell2mat(tempCell);
       test_output(i) = mode(tempM);
   end
end
function test_output = deneutro2() 
    load data;
    data = [train_input train_output];
    m = size(test_input,1);
    n = size(train_input,1);
    test_output = zeros(m,1);
    denominator = zeros(m,1);
    load cluster.mat;
    for i=1:m
        for j=1:n
            if idx3(i+216) == idx3(j)
                test_output(i) = test_output(i) + 1 * data(j,3);
                denominator(i) = denominator(i) + 1;
            end
        end
        test_output(i) = test_output(i)/denominator(i);
    end
end