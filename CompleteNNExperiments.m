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
load('C:\Users\nikos\Desktop\data\Comb_Cov_123.mat');
load('C:\Users\nikos\Desktop\data\Labels_test.mat');


labels = double(labels);
labels = labels + 1;

variables = {Comb_Cov_123};
split = 0.8;
alpha = 1000;
train_percent = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];
NumNN = [1 2 3 4 5];
accuracy_test = zeros(size(variables,2),numel(train_percent),numel(NumNN));
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
        % Split between training and 
        Cov_test = X(I_test,:,:);
        Cov_train = X(I_train,:,:); 
        % Downsample train based on ind_train_perc
        Cov_train = Cov_train(ind_train_perc,:,:);
        sampled_train_labels = train_lab(ind_train_perc);

        % Compute distance between covariance matrices
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
%             Dist_mat = pdist2(Cov_test,Cov_train,'euclidean');                

        %% Find nearest neighbors 
        for num = 1:numel(NumNN)
            [NearNeigh,~] = knnsearch(Cov_train,Cov_test,'k',num);
            errortest = zeros(numel(testlab),1);
            
            % Quantify error
            for testsam = 1:numel(testlab)
                errorbuf = 1;
                for f = 1:num
                    if testlab(testsam) == sampled_train_labels(NearNeigh(testsam,f));
                        errorbuf = 0;
                    end
                end 
                errortest(testsam) = errorbuf;
            end 
            
            accuracy_test(vr, prc, num) = 1 - (sum(errortest) / numel(errortest));
        end
    end   
end
   




%% Plot Results
figure(1);
colors = cell(length(NumNN),1);
for i = 1:length(NumNN)
    colors{i} = rand(1,3);
end

for nNN = 1:size(accuracy_test,1)
    hold on
    plot(train_percent,accuracy_test(vr, : ,nNN),'color',colors{vr},'Linewidth',2);
    
end
grid on
axis('tight')
title(sprintf('Test Performance Results using the %s metric',Metrics))
% legend('Layer1', 'Layer2', 'Layer3', 'Block12', 'Block23','Block123','Fused123','Baseline', 'Scatter1', 'Scatter2','Scatter3')
hold off
