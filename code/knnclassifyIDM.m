function [Lhat yhat] = knnclassifyIDM(IDM,ys,kvec)
% knn classify using interpoint-distance-matrix 
% 
% INPUT
%   IDM: interpoint distance matrix, non-negative, symmetic, and INFINITY
%   on diagonal (for knn)
%   ys: list of class
%   kvec: list of which k's to use
% 
% OUTPUT
%   Lhat: misclassification rate
%   yhat: predicted classes


disp('knn classifying...')

kn=length(kvec);
s=length(IDM);
yhat=nan(kn,s);
Lhat=nan(kn,1);
j=0;
for k=kvec
    j=j+1;
    
    if mod(j,10)==0, disp('next set of 10 different k'), end
        
    parfor i=1:s
        [~, IX] = sort(IDM(i,:));
        yhat(j,i)=sum(ys(IX(1:k)))>k/2;
    end
    Lhat(j)=mean(yhat(j,:)~=ys');
end



