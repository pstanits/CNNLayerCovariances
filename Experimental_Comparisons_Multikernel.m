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

load('C:\Users\nikos\Desktop\data\Covariance1.mat');
load('C:\Users\nikos\Desktop\data\Covariance2.mat');
load('C:\Users\nikos\Desktop\data\Covariance3.mat');
load('C:\Users\nikos\Desktop\data\Labels_test.mat');
load('C:\Users\nikos\Desktop\data\BaselineCov.mat');
load('C:\Users\nikos\Desktop\data\Comb_Cov_12.mat');
load('C:\Users\nikos\Desktop\data\Comb_Cov_23.mat');
load('C:\Users\nikos\Desktop\data\Comb_Cov_123.mat');
load('C:\Users\nikos\Desktop\data\Fused_Cov_1.mat');

labels = double(labels);
labels = labels + 1;


variables = {Layer1,Layer2,Layer3, Comb_Cov_12, Comb_Cov_23, Comb_Cov_123, Cov_Baseline};
mixing = [0.2 0.3 0.5];
split = 0.8;
alpha = 1000;
train_percent = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];
accuracy_test = zeros(1,numel(train_percent));
accuracy_train = zeros(1,numel(train_percent));
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
    X1 = variables{1};
    X2 = variables{2};
    X3 = variables{3};
    disp(prc)
    % Split between training and test
    Cov_test1 = X1(I_test,:,:);
    Cov_train1 = X1(I_train,:,:); 
    Cov_test2 = X2(I_test,:,:);
    Cov_train2 = X2(I_train,:,:); 
    Cov_test3 = X3(I_test,:,:);
    Cov_train3 = X3(I_train,:,:); 

    % Downsample train based on ind_train_perc
    Cov_train1 = Cov_train1(ind_train_perc,:,:);
    Cov_train2 = Cov_train2(ind_train_perc,:,:);
    Cov_train3 = Cov_train3(ind_train_perc,:,:);
    sampled_train_labels = train_lab(ind_train_perc);


    % Compute matrix logarithms based on my_logm
    for covtrain = 1:size(Cov_train1,1)
        Cov_train1(covtrain,:,:) = my_logm(squeeze(Cov_train1(covtrain,:,:)));
        Cov_train2(covtrain,:,:) = my_logm(squeeze(Cov_train2(covtrain,:,:)));
        Cov_train3(covtrain,:,:) = my_logm(squeeze(Cov_train3(covtrain,:,:)));

    end

    for covtest = 1:size(Cov_test1,1)
        Cov_test1(covtest,:,:) = my_logm(squeeze(Cov_test1(covtest,:,:)));
        Cov_test2(covtest,:,:) = my_logm(squeeze(Cov_test2(covtest,:,:)));
        Cov_test3(covtest,:,:) = my_logm(squeeze(Cov_test3(covtest,:,:)));
    end

    % Reshape covariance and compute euclidean distances
    Cov_train1 = reshape(Cov_train1,[(size(Cov_train1,1)),size(Cov_train1,2)^2]);
    Cov_test1 = reshape(Cov_test1,[(size(Cov_test1,1)),size(Cov_test1,2)^2]);
    Cov_train2 = reshape(Cov_train2,[(size(Cov_train2,1)),size(Cov_train2,2)^2]);
    Cov_test2 = reshape(Cov_test2,[(size(Cov_test2,1)),size(Cov_test2,2)^2]);
    Cov_train3 = reshape(Cov_train3,[(size(Cov_train3,1)),size(Cov_train3,2)^2]);
    Cov_test3 = reshape(Cov_test3,[(size(Cov_test3,1)),size(Cov_test3,2)^2]);
    
    % Test kernel components
    Test_kernel1 = pdist2(Cov_test1, Cov_train1, 'euclidean');
    Test_kernel1 = [(1:size(Cov_test1,1))' exp( - Test_kernel1.^2 / alpha)];
    Test_kernel2 = pdist2(Cov_test2, Cov_train2, 'euclidean');
    Test_kernel2 = [(1:size(Cov_test2,1))' exp( - Test_kernel2.^2 / alpha)];
    Test_kernel3 = pdist2(Cov_test3, Cov_train3, 'euclidean');
    Test_kernel3 = [(1:size(Cov_test3,1))' exp( - Test_kernel3.^2 / alpha)];
    
    % Train kernel componenets
    Train_kernel1 = pdist2(Cov_train1, Cov_train1, 'euclidean');
    Train_kernel1 = [(1:size(Cov_train1))' exp( - Train_kernel1.^2 / alpha)];
    Train_kernel2 = pdist2(Cov_train2, Cov_train2, 'euclidean');
    Train_kernel2 = [(1:size(Cov_train2))' exp( - Train_kernel2.^2 / alpha)];
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
