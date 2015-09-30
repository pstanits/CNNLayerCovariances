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
close all
clc
rng shuffle
%% Data import

load('C:\Users\nikos\Desktop\data\Labels_test.mat');
load('C:\Users\nikos\Desktop\data\fullBlock.mat');
load('C:\Users\nikos\Desktop\data\LastLayer.mat');
labels = double(labels);
labels = labels + 1;

mixing = [0.7 0.3];
variables = {fullBlockDiag, LastLayer};
split = 0.8;
alpha1 = 1000;
alpha2 = 10000;
train_percent = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];
accuracy_test = zeros(4,numel(train_percent));
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
        I_train = [I_train; indtrain];
        I_test = [I_test; indtest];
        testlab = [testlab; labels(indtest)];
        train_lab = [train_lab; labels(indtrain)];
end


%% Iterate over the percentages
for prc = 1:length(train_percent)
    ind_train_perc = randsample(1:1:numel(I_train),floor(train_percent(prc) * size(I_train,1)));
    % Iterate over different covariances
    X1 = variables{1};
    X2 = variables{2};
    disp(prc)
    
    % Split between training and test
    Cov_test = X1(I_test,:,:);
    Cov_train = X1(I_train,:,:); 
    Feat_test = X2(I_test,:,:);
    Feat_train = X2(I_train,:,:); 
    

    % Downsample train based on ind_train_perc
    Cov_train = Cov_train(ind_train_perc,:,:);
    Feat_train = Feat_train(ind_train_perc,:,:);
    sampled_train_labels = train_lab(ind_train_perc);


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
    
    % Test kernel components
    Test_kernel_Cov = pdist2(Cov_test, Cov_train, 'euclidean');
    Test_kernel_Cov = [(1:size(Cov_test,1))' exp( - Test_kernel_Cov.^2 / alpha1)];
    Test_kernel_Feat = pdist2(Feat_test, Feat_train, 'euclidean');
    Test_kernel_Feat = [(1:size(Feat_test,1))' exp( - Test_kernel_Feat.^2 / alpha2)];
    
    % Train kernel componenets
    Train_kernel_Cov = pdist2(Cov_train, Cov_train, 'euclidean');
    Train_kernel_Cov = [(1:size(Cov_train))' exp( - Train_kernel_Cov.^2 / alpha1)];
    Train_kernel_Feat = pdist2(Feat_train, Feat_train, 'euclidean');
    Train_kernel_Feat = [(1:size(Feat_train))' exp( - Train_kernel_Feat.^2 / alpha2)];
    
    % Mixing kernels
    Train_kernel_Fuse = mixing(1) * Train_kernel_Cov + mixing(2) + Train_kernel_Feat;
    Test_kernel_Fuse = mixing(1) * Test_kernel_Cov + mixing(2) + Test_kernel_Feat;
    Train_kernel_Fuse = [(1:size(Feat_train))' Train_kernel_Fuse(:,2:end)];
    Test_kernel_Fuse = [(1:size(Feat_test,1))' Test_kernel_Fuse(:,2:end)];
    
    %%Train and test on Feature Kernel
    trained_model = svmtrain(sampled_train_labels, Train_kernel_Cov,'-s 0 -c 100 -t 4');
    [~, acc, decVals] = svmpredict(testlab, Test_kernel_Cov, trained_model);
    accuracy_test(1,prc) = acc(1);
    sprintf('Test accuracy for variable %d  and train percent %1.1f%% is %10.1f%%',1, 100 * train_percent(prc),acc(1))
    
    %%Train and test on Block kernel
    trained_model = svmtrain(sampled_train_labels, Train_kernel_Feat,'-s 0 -c 100 -t 4');
    [~, acc, decVals] = svmpredict(testlab, Test_kernel_Feat, trained_model);
    accuracy_test(2,prc) = acc(1);
    sprintf('Test accuracy for variable %d  and train percent %1.1f%% is %10.1f%%',2, 100 * train_percent(prc),acc(1))
    
    %% Train SVM based on Fused kernel
    trained_model = svmtrain(sampled_train_labels, Train_kernel_Fuse,'-s 0 -c 100 -t 4');
    [~, acc, decVals] = svmpredict(testlab, Test_kernel_Fuse, trained_model);
    accuracy_test(3,prc) = acc(1);
    sprintf('Test accuracy for variable %d  and train percent %1.1f%% is %10.1f%%',3, 100 * train_percent(prc),acc(1))
        
    %% Train and Test nonKernel
    trained_model = svmtrain(sampled_train_labels, Feat_train,'-s 0 -c 0.1 -t 0');
    [~, acc, decVals] = svmpredict(testlab, Feat_test, trained_model);
    accuracy_test(4,prc) = acc(1);
    sprintf('Test accuracy for variable %d  and train percent %1.1f%% is %10.1f%%',4, 100 * train_percent(prc),acc(1))
end
     
%% Plot Results
figure(1);
col = {'g','r','m','b'};
for vr = 1:size(accuracy_test,1)
    hold on
    plot(train_percent,accuracy_test(vr,:),'color',col{vr},'Linewidth',2);  
end
grid on
axis('tight')
title(sprintf('MultiKernel Test Performance Results using the %s metric','LE'))
legend('Covariances Kernel', 'Feature Kernel', 'Fused Kernels', '10D Non-kernel');
hold off
