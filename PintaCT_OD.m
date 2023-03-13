function [] = PintaCT_OD(CT)

%% Reshape and imshow
% D=M;
X=CT;
[ns,p]=size(X);
c=3;
m=sqrt(p);
n=m;
figure()
for i=1:ns
subplot(1,ns,i);
X_2d=reshape(X(i,:),m,n);

imshow(X_2d);
end
