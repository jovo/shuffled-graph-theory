function IDM = InterpointDistanceMatrix(data)
% compute Interpoint-Distance-Matrix of data
% 
% INPUT:
%   data: an array, either n-by-s or n-by-m-s
% 
% OUTPUT
%   IDM: distance between every pair of matrices.  this matrix is symmetic
%   with infinities on the diagonal

disp('Computing Interpoint-Distance-Matrix...')

siz=size(data);
s=siz(end);

% compute distances
d=zeros(s);
for i=1:s
    
    if mod(i,100)==0, disp('did another 100 rows of the IDM'); end
    
    parfor j=i+1:s
        if ndims(data)==2
            d(i,j)=sum((data(i,:)-data(j,:)).^2);
        else
            d(i,j)=sum(sum((data(:,:,i)-data(:,:,j)).^2));
        end
    end
end

IDM=d'+d;
IDM(1:s+1:end)=inf;