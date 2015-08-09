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
cases = 1;
% for cases = 1:5
    train_percent = cases * 0.1;
    indices = 1:1:10000;
    ind_train = randsample(indices,ceil(train_percent * size(Layer2,1)));
    ind_test = setdiff(indices,ind_train);

    Cov_train = Layer2(ind_train,:,:);
    Cov_test = Layer2(ind_test,:,:);
    labtrain = labels(ind_train);
    labtest = labels(ind_test);


%     % Offline logarithm computations
%     for i = 1:size(Cov_train,1)
%         Cov_train(i,:,:) = logm(squeeze(Cov_train(i,:,:)) + eps * eye(size(Cov_train,2)));
%     end
%     disp('Cov_train logm computed...')
%     for i = 1:size(Cov_test,1)
%         Cov_test(i,:,:) = logm(squeeze(Cov_test(i,:,:)) + eps * eye(size(Cov_train,2)));
%     end
%     disp('Cov_test logm computed...')
%     
%     % Reshape covariances
%     Cov_train = reshape(Cov_train,[(size(Layer2,2))^2,size(Cov_train,1)])';
%     Cov_test = reshape(Cov_test,[(size(Layer2,2))^2,size(Cov_test,1)])';
%     
%     
%     % Distance computation
%     dist_matrix = pdist2(Cov_test, Cov_train, 'euclidean');
%     disp('Distance calculated...')
%     % Nearest neighboe
%     [~, index_match] = min(dist_matrix, [], 2); 
%     disp('Nearest Neighbors retrieved...')
    %% Compute distances between covariances
    Cov_Distance = zeros(size(Cov_test,1),size(Cov_train,1));
    index_match = zeros(size(Cov_test,1),1);
    for i = 1:size(Cov_test,1)
        disp([i,cases])
        C1 = reshape(Cov_test(i,:,:),[size(Cov_train,3),size(Cov_train, 3)]);
        for j = 1:size(Cov_train,1)
            C2 = reshape(Cov_train(j,:,:),[size(Cov_train,3),size(Cov_train, 3)]);
            Cov_Distance(i,j) = Stein(C1,C2);
        end
        [~,index_match(i)] = min(Cov_Distance(i,:)); 

    end


    %% Error Computation 
    error_buffer = 0;
    for i = 1:200
        if labtest(i) ~= labtrain(index_match(i))
            error_buffer = error_buffer + 1;
        end
    end
    accuracy = (size(index_match,1) - error_buffer) / size(index_match,1);
% end

% save AccuracyLayer2LogEucl.mat accuracy