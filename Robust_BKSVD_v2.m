function [D2,X2,third,mask,Blood_channel] = Robust_BKSVD_v2(I,D0,thr)
%Input:
% I --> image in the range [0,255] but in type double
% D0 --> initial dictionary in OD space
% thr --> threshold to filter the third channel
%Output:
% D2 --> color vector matrix for hematoxylin and eosin (in that order)
% X2 --> H&E Concentration values 
% Third --> Stain color of the third channel
% mask --> Blood positive pixels
% Blood_channel --> Concentration values of the third (residual/blood) channel
ns=3;
[m,n,c]=size(I);
I_col=reshape(I,(m)*(n),c)';
X0= directDeconvolve(I,D0);
mask0= X0(3,:)>thr;

[D3,X3,iter_T,current_batch] = MB_BKSVD4SD_FI_v3(I,D0,ns);

[Dsort,change_flag] = CheckHEorder(D3);
if change_flag
    D3=Dsort;
    
end
X3=directDeconvolve(I,D3);
%%
mask1= X3(3,:)>thr; %Positivo para todos los que SON anomalias
mask= or(mask0,mask1);
% figure()
% Mask_percent= X3(3,:)>1;%prctile(XR(3,:),90);
%imshow(reshape(Mask_percent,m2,n2))
%%

%%

I_col=reshape(I,(m)*(n),c)';
I_filtered=I_col(:, ~mask);
if size(I_filtered,2)<=3
	disp('WARNING 0 values after removing mask')
	I_filtered=I_col(:,~mask0)
end
ns=2;
D2= BKSVD4SD_FI_col(I_filtered,D0,ns);

[Dsort,change_flag] = CheckHEorder(D2);
if change_flag
    D2=Dsort;
end

X2= directDeconvolve(I,D2);

third=D3(:,3);
Blood_channel=X3(3,:);
end



