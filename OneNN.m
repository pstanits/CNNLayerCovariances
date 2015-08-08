% 1NN for feature covariances
% Panagiotis Stanitsas
% University of Minnesota 

% September 2015

clear all
clc

load('C:\Users\nikos\Desktop\data\Covariance1.mat');
load('C:\Users\nikos\Desktop\data\Covariance2.mat');
load('C:\Users\nikos\Desktop\data\Covariance3.mat');
load('C:\Users\nikos\Desktop\data\Labels_test.mat');



%% Split the data into training and testing
accuracy = zeros(5,1);
for cases = 1:5
    train_percent = cases * 0.1;
    indices = 1:1:10000;
    ind_train = randsample(indices,ceil(train_percent * size(Layer2,1)));
    ind_test = setdiff(indices,ind_train);

    Cov_train = Layer2(ind_train,:,:);
    Cov_test = Layer2(ind_test,:,:);
    labtrain = labels(ind_train);
    labtest = labels(ind_test);




    %% Compute distances between covariances
    Cov_Distance = zeros(size(Cov_test,1),size(Cov_train,1));
    index_match = zeros(size(Cov_test,1),1);
    for i = 1:size(Cov_test,1)
        disp([i,cases])
        C1 = reshape(Cov_test(i,:,:),[size(Cov_train,3),size(Cov_train, 3)]);
        for j = 1:size(Cov_train,1)
            C2 = reshape(Cov_train(j,:,:),[size(Cov_train,3),size(Cov_train, 3)]);
            Cov_Distance(i,j) = logEucl(C1,C2);
        end
        [~,index_match(i)] = min(Cov_Distance(i,:)); 

    end


    %% Error Computation 
    error_buffer = 0;
    for i = 1:size(index_match,1)
        if labtest(i) ~= labtrain(index_match(i))
            error_buffer = error_buffer + 1;
        end
    end
    accuracy(cases) = (size(index_match,1) - error_buffer) / size(index_match,1);
end

save AccuracyLayer2LogEucl.mat accuracy