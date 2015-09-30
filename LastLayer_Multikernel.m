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

%% Data import

load('C:\Users\nikos\Desktop\data\Labels_test.mat');
load('C:\Users\nikos\Desktop\data\fullBlock.mat');
load('C:\Users\nikos\Desktop\data\LastLayer.mat');
labels = double(labels);
labels = labels + 1;

variables = {fullBlockDiag, LastLayer};
split = 0.8;
alpha = 1000;
train_percent = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];
Metrics = 'LE';
accuracy_test = zeros(size(variables,2),numel(train_percent));
accuracy_train = zeros(size(variables,2),numel(train_percent));
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
    
    Feat_train = reshape(Feat_train,[(size(Feat_train,1)),size(Feat_train,2)^2]);
    Feat_test = reshape(Feat_test,[(size(Feat_test,1)),size(Feat_test,2)^2]);
    
    % Test kernel components
    Test_kernel1 = pdist2(Cov_test, Cov_train, 'euclidean');
    Test_kernel1 = [(1:size(Cov_test,1))' exp( - Test_kernel1.^2 / alpha)];
    Test_kernel2 = pdist2(Feat_test, Feat_train, 'euclidean');
    Test_kernel2 = [(1:size(Feat_test,1))' exp( - Test_kernel2.^2 / alpha)];
    Test_kernel3 = pdist2(Cov_test3, Cov_train3, 'euclidean');
    Test_kernel3 = [(1:size(Cov_test3,1))' exp( - Test_kernel3.^2 / alpha)];
    
    % Train kernel componenets
    Train_kernel1 = pdist2(Cov_train, Cov_train, 'euclidean');
    Train_kernel1 = [(1:size(Cov_train))' exp( - Train_kernel1.^2 / alpha)];
    Train_kernel2 = pdist2(Feat_train, Feat_train, 'euclidean');
    Train_kernel2 = [(1:size(Feat_train))' exp( - Train_kernel2.^2 / alpha)];
    Train_kernel3 = pdist2(Cov_train3, Cov_train3, 'euclidean');
    Train_kernel3 = [(1:size(Cov_train3))' exp( - Train_kernel3.^2 / alpha)];
    
    % Mixing kernels
    Train_kernel = mixing(1) * Train_kernel1 + mixing(2) + Train_kernel2 + mixing(3) * Train_kernel3;
    Test_kernel = mixing(1) * Test_kernel1 + mixing(2) + Test_kernel2 + mixing(3) * Test_kernel3;
    Train_kernel = [(1:size(Cov_train3))' Train_kernel(:,2:end)];
    Test_kernel = [(1:size(Cov_test3,1))' Test_kernel(:,2:end)];

    %% Train SVM based on the precomputed kernels
    trained_model = svmtrain(sampled_train_labels, Train_kernel,'-s 0 -c 3 -t 4');

    %% Predict based on the model
    [~, acc, decVals] = svmpredict(testlab, Test_kernel, trained_model);
    accuracy_test(1,prc) = acc(1);
    sprintf('Test accuracy for variable %d  and train percent %1.1f%% is %10.1f%%',1, 100 * train_percent(prc),acc(1))

    [~, acc, decVals] = svmpredict(sampled_train_labels, Train_kernel, trained_model);
    accuracy_train(1,prc) = acc(1);
    sprintf('Train accuracy for variable %d  and train percent %1.1f%% is %10.1f%%',1, 100 * train_percent(prc),acc(1))
        
end
     
%% Plot Results
figure(1);
col = {'g','r','m','y','c','k', 'b'};
for vr = 1:size(accuracy_test,1)
    hold on
    plot(train_percent,accuracy_test(vr,:),'color',col{vr},'Linewidth',2);
    
end
grid on
axis('tight')
title(sprintf('Test Performance Results using the %s metric','LE'))
hold off

figure(2)
for vr = 1:size(accuracy_train,1)
    hold on
    plot(train_percent,accuracy_train(vr,:),'color',col{vr},'Linewidth',2);
    
end
grid on
axis('tight')
title(sprintf('Train Performance Results using the %s metric','LE'))