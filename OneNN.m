% 1NN for feature covariances
% Panagiotis Stanitsas
% University of Minnesota 

% September 2015

clear all
clc


% stream = RandStream(6387);
% reset(stream);

load('C:\Users\nikos\Desktop\data\Covariance1.mat');
load('C:\Users\nikos\Desktop\data\Covariance2.mat');
load('C:\Users\nikos\Desktop\data\Covariance3.mat');
load('C:\Users\nikos\Desktop\data\Labels_test.mat');
load('C:\Users\nikos\Desktop\data\BaselineCov.mat');

variables = {Layer1,Layer2,Layer3,Cov_Baseline};
% variables = {Cov_Baseline};
%% Split the data into training and testing
accuracy = zeros(5,4);
for cases = 1:5
    stream = RandStream('twister','Seed',0);
    train_percent = cases * 0.1;
    indices = randsample(1:1:10000,2000);
    ind_train = randsample(indices,ceil(train_percent * 2000));
    ind_test = setdiff(indices,ind_train);
    for vr = 1:length(variables);
        Dum = variables{vr};
        Cov_train = Dum(ind_train,:,:);
        Cov_test = Dum(ind_test,:,:);
        labtrain = labels(ind_train);
        labtest = labels(ind_test);


%     %     Offline logarithm computations
%         for i = 1:size(Cov_train,1)
%             Cov_train(i,:,:) = my_logm(squeeze(Cov_train(i,:,:)));
%         end
%         disp('Cov_train logm computed...')
%         for i = 1:size(Cov_test,1)
%             Cov_test(i,:,:) = my_logm(squeeze(Cov_test(i,:,:)));
%         end
%         disp('Cov_test logm computed...')
%         
% %         Reshape covariances
%         Cov_train = reshape(Cov_train,[(size(Dum,2))^2,size(Cov_train,1)])';
%         Cov_test = reshape(Cov_test,[(size(Dum,2))^2,size(Cov_test,1)])';
%         
%         
% %         Distance computation
%         dist_matrix = pdist2(Cov_test, Cov_train, 'euclidean');
%         disp('Distance calculated...')
%     %     Nearest neighboe
%         [~, index_match] = min(dist_matrix, [], 2); 
%         disp('Nearest Neighbors retrieved...')
        %% Compute distances between covariances
        Cov_Distance = zeros(size(Cov_test,1),size(Cov_train,1));
        index_match = zeros(size(Cov_test,1),1);
        for i = 1:size(Cov_test,1)
            disp([i,cases,vr])
            C1 = reshape(Cov_test(i,:,:),[size(Cov_train,3),size(Cov_train, 3)]);
            for j = 1:size(Cov_train,1)
                C2 = reshape(Cov_train(j,:,:),[size(Cov_train,3),size(Cov_train, 3)]);
                Cov_Distance(i,j) = Stein(C1,C2);
            end
            [~,index_match(i)] = min(Cov_Distance(i,:)); 
        end

% 
%         %% Error Computation 
        error_buffer = 0;
        for i = 1:length(labtest)
            if labtest(i) ~= labtrain(index_match(i))
                error_buffer = error_buffer + 1;
            end
        end
        accuracy(cases,vr) = (size(index_match,1) - error_buffer) / size(index_match,1);
    end
end

% save AccuracyStein.mat accuracy