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
close all

%% Data import
load('C:\Users\nikos\Desktop\data\Labels_test.mat');
load('C:\Users\nikos\Desktop\data\Mean1.mat');
load('C:\Users\nikos\Desktop\data\Mean2.mat');
load('C:\Users\nikos\Desktop\data\Mean3.mat');

labels = double(labels);
labels = labels + 1;

variables = {mean1, mean2, mean3};
%variables = {Fused_Cov_1};
split = 0.8;
train_percent = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];
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
        Mean_test = X(I_test,:);
        Mean_train = X(I_train,:); 
        % Downsample train based on ind_train_perc
        Mean_train = Mean_train(ind_train_perc,:);
        sampled_train_labels = train_lab(ind_train_perc);
            
        
        
        %% Train SVM based on the precomputed kernels
        trained_model = svmtrain(sampled_train_labels, Mean_train, '-s 1 -c 100 -t 0');
        
        %% Predict based on the model
        [~, acc, ~] = svmpredict(testlab, Mean_test, trained_model);
        accuracy_test(vr,prc) = acc(1);
        sprintf('Test accuracy for variable %d  and train percent %1.1f%% is %10.1f%%',vr, 100 * train_percent(prc),acc(1))
        
        [~, acc, decVals] = svmpredict(sampled_train_labels, Mean_train, trained_model);
        accuracy_train(vr,prc) = acc(1);
        sprintf('Train accuracy for variable %d  and train percent %1.1f%% is %10.1f%%',vr, 100 * train_percent(prc),acc(1))
        
        
    end   
end
     
%% Plot Results
figure(1);
col = {'g','r','m'};
for vr = 1:size(accuracy_test,1)
    hold on
    plot(train_percent,accuracy_test(vr,:),'color',col{vr},'Linewidth',2);
    
end
grid on
axis('tight')
title('Test Performance Results using the Layer means')
hold off

figure(2)
for vr = 1:size(accuracy_train,1)
    hold on
    plot(train_percent,accuracy_train(vr,:),'color',col{vr},'Linewidth',2);
    
end
grid on
axis('tight')
title('Train Performance Results using Layer means')
