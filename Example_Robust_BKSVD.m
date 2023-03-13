clc,clear all, close all
%Initial reference matrix
load 'MLandini' RM;
D0=RM; %Initial Dictionary (Ruifrok matrix)
thr=0.3; %Threshold value, might depend on your data. A value of 0.3 was found to obtain the best results in our paper.

%% Image


I=imread('02_r_0.5.png');
I=im2uint8(I); %Matlab tends to read what it wants. Make sure your data is in the range [0,255]
imshow(I)

[D2,X2,third,mask,Blood_channel] = Robust_BKSVD_v2(double(I),D0,thr);

%Output:
% D2 --> color vector matrix for hematoxylin and eosin (in that order)
% X2 --> H&E Concentration values 
% Third --> Stain color of the third channel
% mask --> Blood positive pixels
% Blood_channel --> Concentration values of the third (residual/blood) channel

%%
[m,n,c] = size(I)

figure()
titles=["Hematoxylin", "Eosin"]
Xfiltered=X2;
for i=1:2
    subplot(1,3,i);
    Xfiltered(i,mask>0)=0; %Stain separation at blood poxitive pixels is not valid
    X_2d=reshape(Xfiltered(i,:),m,n);
    imshow(X_2d);
    title(titles(i))
end

subplot(1,3,3)
mask2d=reshape(mask,m,n);
% figure(),imshow(mask2d)

PintaMask(I,mask2d)
title('Blood positive pixels')

%%

PintaMatriz(D2)
