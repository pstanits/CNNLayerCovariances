% Computes the covairances for the cifar10 dataset.
% Returns a 3D .mat array of size (n_samples x 8 x 8).

%  Each slice contains an 8 x 8 covariance matrix corresponds to variables
%  [Ir, Ib, Ig, gradx, grady, gradxx, gradyy, gradxy]

clear all
clc

load('C:\Users\nikos\Desktop\Active Silhouette Classification\datasets\cifar-10-batches-mat\test_batch.mat');


Cov_Baseline = zeros(size(data,1), 8, 8);
% Data processing
for i = 1:size(data,1)
    disp(i)
    I = rot90(reshape(data(i,:,:),[32,32,3]),3);
%     imshow(I)
    I_d = im2double(I);
    [gradx, grady] = gradient(im2double(rgb2gray(I)));
    [gradxx, ~] = gradient(gradx);
    [gradxy, gradyy] = gradient(grady);
    gradx = (gradx - min(gradx(:))) ./ (max(gradx(:)) - min(gradx(:)));
    grady = (grady - min(grady(:))) ./ (max(grady(:)) - min(grady(:)));
    gradxx = (gradxx - min(gradx(:))) ./ (max(gradx(:)) - min(gradx(:)));
    gradyy = (gradyy - min(gradyy(:))) ./ (max(gradyy(:)) - min(gradyy(:)));
    gradxy = (gradxy - min(gradxy(:))) ./ (max(gradxy(:)) - min(gradxy(:)));
    feat = [reshape(I_d(:,:,1).',[],1),reshape(I_d(:,:,2).',[],1),reshape(I_d(:,:,3).',[],1),...
        reshape(gradx.',[],1),reshape(grady.',[],1),reshape(gradxx.',[],1),reshape(gradyy.',[],1),reshape(gradxy.',[],1)];
    Cov_Baseline(i,:,:) = cov(feat);
%     pause       
end
