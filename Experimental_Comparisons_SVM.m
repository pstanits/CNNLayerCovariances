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
load('C:\Users\nikos\Desktop\data\Comb_Cov_12.mat');
load('C:\Users\nikos\Desktop\data\Comb_Cov_23.mat');
load('C:\Users\nikos\Desktop\data\Comb_Cov_123.mat');

labels = double(labels);
labels = labels + 1;

% Comb_Cov_12 = zeros(10000,64,64);
% Comb_Cov_23 = zeros(10000,96,96);
% Comb_Cov_123 = zeros(10000,128,128);
% for i = 1:size(Layer1,1)
%     disp(i)
%     Comb_Cov_12(i,:,:) = blkdiag(squeeze(Layer1(i,:,:)),squeeze(Layer2(i,:,:)));
%     Comb_Cov_23(i,:,:) = blkdiag(squeeze(Layer2(i,:,:)),squeeze(Layer3(i,:,:)));
%     Comb_Cov_123(i,:,:) = blkdiag(squeeze(Layer1(i,:,:)),squeeze(Layer2(i,:,:)),squeeze(Layer3(i,:,:)));
% end


variables = {Layer1,Layer2,Layer3, Comb_Cov_12, Comb_Cov_23, Comb_Cov_123, Cov_Baseline};
split = 0.8;
alpha = 1000;
train_percent = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];
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
        disp([prc vr])
        % Split between training and test
        Cov_test = X(I_test,:,:);
        Cov_train = X(I_train,:,:); 
        % Downsample train based on ind_train_perc
        Cov_train = Cov_train(ind_train_perc,:,:);
        sampled_train_labels = train_lab(ind_train_perc);
       
        % Compute distance between covariance matrices
        
        switch Metrics
            case 'Stein'
                Test_kernel = zeros(size(Cov_test,1),size(Cov_train,1));
                Train_kernel = zeros(size(Cov_train,1),size(Cov_train,1));
                % Iterate with double for between test (rows) and train
                % (column) covariance matrices.
                for testmat = 1:size(Cov_test,1)
                    for trainmat = 1:size(Cov_train,1)
                        Test_kernel(testmat,trainmat) = log(det(squeeze(Cov_train(trainmat,:,:)) + squeeze(Cov_test(testmat,:,:)) / 2)) ...
                                                          - 0.5 * log(det(squeeze(Cov_test(testmat,:,:)) * squeeze(Cov_train(trainmat,:,:))));
                    end 
                end  
                Test_kernel = [(1:size(Cov_test,1))' exp( - Test_kernel.^2 / alpha)];
                
                % Compute Train kernel
                b = 0;
                for index1 = 1:size(Cov_train,1)
                    b = b + 1;
                    for index2 = b:size(Cov_train,1)
                        Train_kernel(index1,index2) = log(det(squeeze(Cov_train(index1,:,:)) + squeeze(Cov_train(index2,:,:)) / 2)) ...
                                                          - 0.5 * log(det(squeeze(Cov_train(index1,:,:)) * squeeze(Cov_train(index2,:,:))));
                        
                    end
                end
                % Symmetrize to avoid computing entries twice
                Train_kernel = Train_kernel + Train_kernel' - 1/2 * diag(diag(Train_kernel));
                Train_kernel = [(1:size(Cov_train))' exp( - Train_kernel / alpha)];
                if sum(sum(Train_kernel)) == 0
                    error('Bad Kernel')
                end
                   
                
                
            case 'LE'
                % Compute matrix logarithms based on my_logm
                for covtrain = 1:size(Cov_train,1)
                    Cov_train(covtrain,:,:) = my_logm(squeeze(Cov_train(covtrain,:,:)));
                end
                
                for covtest = 1:size(Cov_test,1)
                    Cov_test(covtest,:,:) = my_logm(squeeze(Cov_test(covtest,:,:)));
                end
                
                % Reshape covariance and compute euclidean distances
                Cov_train = reshape(Cov_train,[(size(Cov_train,1)),size(Cov_train,2)^2]);
                Cov_test = reshape(Cov_test,[(size(Cov_test,1)),size(Cov_test,2)^2]);
                Test_kernel = pdist2(Cov_test, Cov_train, 'euclidean');
                Test_kernel = [(1:size(Cov_test,1))' exp( - Test_kernel.^2 / alpha)];
                Train_kernel = pdist2(Cov_train, Cov_train, 'euclidean');
                Train_kernel = [(1:size(Cov_train))' exp( - Train_kernel.^2 / alpha)];
                
                
                
            case 'AIRM'
                Test_kernel = zeros(size(Cov_test,1),size(Cov_train,1));
                Train_kernel = zeros(size(Cov_train,1),size(Cov_train,1));
                % Iterate with double for between test (rows) and train
                % (column) covariance matrices.
                for testmat = 1:size(Cov_test,1)
                    for trainmat = 1:size(Cov_train,1)
                        Test_kernel(testmat,trainmat) = norm(my_logm(squeeze(Cov_test(testmat,:,:)).^(0.5) * squeeze(Cov_train(trainmat,:,:)) * ...
                            squeeze(Cov_test(testmat,:,:)).^(0.5)),'fro');
                    end 
                end  
                Test_kernel = [(1:size(Cov_test,1))' exp( - Test_kernel.^2 / alpha)];
                
                % Compute Train kernel
                b = 0;
                for index1 = 1:size(Cov_train,1)
                    for index2 = 1:size(Cov_train,1)
                        Train_kernel(index1,index2) = norm(my_logm(squeeze(Cov_train(index1,:,:)).^(0.5) * squeeze(Cov_train(index2,:,:)) * ...
                            squeeze(Cov_train(index1,:,:)).^(0.5)),'fro');
                        
                    end
                end
                Train_kernel = [(1:size(Cov_train))' exp( - Train_kernel.^2 / alpha)];
                
        end
        
        %% Train SVM based on the precomputed kernels
        trained_model = svmtrain(sampled_train_labels, Train_kernel,'-s 0 -t 4');
        
        %% Predict based on the model
        [predClass, acc, decVals] = svmpredict(testlab, Test_kernel, trained_model);
        accuracy(vr,prc) = acc(1);
        sprintf('Accuracy for variable %d  and train percent %1.1f%% is %10.1f%%',vr, 100 * train_percent(prc),acc(1))
    end   
end
     
%% Plot Results
col = {'g','r','m','y','c','k', 'b'};

for vr = 1:size(accuracy,1)
    plot(train_percent,accuracy(vr,:),'color',col{vr},'Linewidth',2);
    hold on
end

grid on
axis('tight')
title(sprintf('Performance Results using the %s metric',Metrics))