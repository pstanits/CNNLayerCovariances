% Performs the experiments on the covariance comparisons 
% Input: Covariance matrices for the test batch of Cifar10 for the 3
%        convolutional layers and Covariances derived from directional
%        derivatives and pixel intensities.
% Output: Perormance plots comparing the performance of a 1NN classifier on
%         the 4 different inputs.

% Panagiotis Stanitsas and Anoop Cherian
% Univerisy of Minnesota
% September 2015

clear all
clc

%% Data import

load('C:\Users\nikos\Desktop\data\Covariance1.mat');
load('C:\Users\nikos\Desktop\data\Covariance2.mat');
load('C:\Users\nikos\Desktop\data\Covariance3.mat');
load('C:\Users\nikos\Desktop\data\Labels_test.mat');
load('C:\Users\nikos\Desktop\data\BaselineCov.mat');
labels = double(labels);
labels = labels + 1;
variables = {Layer1,Layer2,Layer3,Cov_Baseline};
split = 0.8;
train_percent = [0.3 0.4 0.5 0.6];
Metrics = input('Define the Similarity function to be used:     ');
accuracy = zeros(size(variables,2),numel(train_percent));
%% Prepare n folds for more robust validations
distinctlab =  unique(labels);
NumClass = size(distinctlab,1);
I_train = [];
I_test = [];
testlab = [];
train_lab = [];

for l = 1:NumClass
        indclass = find(labels == l);
        indtrain = randsample(indclass,floor(split * length(indclass)));
        indtest = setdiff(indclass,indtrain);
        I_train = [I_train indtrain];
        I_test = [I_test indtest];
        testlab = [testlab; labels(indtest)];
        train_lab = [train_lab; labels(indtrain)];
end

%% Iterate over the percentages
for prc = 1:length(train_percent)
    ind_train_perc = randsample(1:1:numel(I_train),floor(train_percent(prc) * size(I_train,1)));
    % Iterate over different covariances
    for vr = 1:length(variables)
        X = variables{vr};
        % Split between training and test
        Cov_test = X(I_test,:,:);
        Cov_train = X(I_train,:,:); 
        % Downsample train based on ind_train_perc
        Cov_train = Cov_train(ind_train_perc,:,:);
        sampled_train_labels = train_lab(ind_train_perc);
       
        % Compute distance between covariance matrices
        Cov_Distances = zeros(size(Cov_train,1),size(Cov_test,1));
        switch Metrics
            case 'Stein'
                % Iterate with double for between test (rows) and train
                % (column) covariance matrices.
                for testmat = 1:size(Cov_test,1)
                    for trainmat = 1:size(Cov_train,1)
                        Cov_Distances(testmat,trainmat) = sqrt(log(det(squeeze(Cov_train(trainmat,:,:)) + squeeze(Cov_test(testmat,:,:)) / 2)) ...
                                                          - 0.5 * log(det(squeeze(Cov_train(trainmat,:,:)) * squeeze(Cov_test(testmat,:,:)))));
                    end 
                end  
                                
            case 'LE'
                % Compute matrix logarithms based on my_logm
                for covtrain = 1:size(Cov_train,1)
                    Cov_train(covtrain,:,:) = my_logm(squeeze(Cov_train,:,:));
                end
                
                for covtest = 1:size(Cov_test,1)
                    Cov_test(covtest,:,:) = my_logm(squeeze(Cov_test,:,:));
                end
                
                % Reshape covariance and compute euclidean distances
                Cov_train = reshape(Cov_train,[(size(Cov_train,2))^2,size(Cov_train,1)])';
                Cov_test = reshape(Cov_test,[(size(Cov_test,2))^2,size(Cov_test,1)])';
                Cov_distances = pdist2(Cov_test, Cov_train, 'euclidean');
        end
        
        %% Predict labels based on 1NN
        [~, index_match] = min(Cov_Distances, [], 2); 
        
        %% Compute Accuracy
        error = 0;
        for testsam = 1:length(testlab)
             if testlab(testsam) ~= sampled_train_labels(index_match(testsam))
                 error = error + 1;
             end
        end  
        accuracy(vr,prc) = 1 - error / numel(testlab);
    end   
end
     
    
    
    
    
    
    
    
    
    
    