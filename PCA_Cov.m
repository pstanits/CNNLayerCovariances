% Reduce the dimensionality of the covariance matrices


clear all
clc
cloase all

load('C:\Users\nikos\Desktop\data\Covariance1.mat');
load('C:\Users\nikos\Desktop\data\Covariance2.mat');
load('C:\Users\nikos\Desktop\data\Covariance3.mat');

mat1 = reshape(Layer1,[(size(Layer1,1)),size(Layer1,2)^2]);
mat2 = reshape(Layer2,[(size(Layer2,1)),size(Layer2,2)^2]);
mat3 = reshape(Layer3,[(size(Layer3,1)),size(Layer3,2)^2]);

mats = {mat1 mat2 mat3};

for i = 1:length(mats)
    



end