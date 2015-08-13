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
variables = {Layer1,Layer2,Layer3,Cov_Baseline};


%% Prepare n folds for more robust validations
num_crossval = 10;
% Construct the n requested folds
Fold = cell(num_crossval,1);
indexes = randperm(size(X,1));
numperfold = floor(size(X,1)/num_crossval);
indexes = indexes(1:numperfold*num_crossval);
indexes = reshape(indexes,[numperfold num_crossval]);
for i = 1:num_crossval
Fold{i} = X(indexes(:,i),:);
end
% dimensions = size(X,2)-1;
list_feat = 1:1:dim;
% Shift vector
splitvec = 1:1:num_crossval;