%% simulate independent edge models and classify using or not using the vertex names
% clear; clc

alg.fname   = 'MRconnectome';    % different names will generate different simulations
alg.datadir = '../data/';
alg.figdir  = '../figs/';
alg.save    = 0;                    % whether to save/print results

load('~/Research/data/MRI/BLSA/BLSA_0317/base/BLSA_0317_countMtx.mat')

%%
t=200;
siz=size(AdjMats);
n=siz(1);
s=siz(3);
As=0*AdjMats;
idl=find(tril(ones(n),-1));
idu=find(triu(ones(n),+1));
Avec=nan(s,length(idl));
for i=1:s
    A=(AdjMats(:,:,i));
    A(idu)=0;
    Awei(:,:,i)=A;
    Avec(i,:)=A(idl);
end
Abin=Awei;
Abin(Abin<=t)=0;
Abin(Abin>t)=1;

Avbin=Avec;
Avbin(Avbin<=t)=0;
Avbin(Avbin>t)=1;

%% get invariant representations of the data

glen.count=importdata('~/Research/data/MRI/BLSA/BLSA_0317/base/count_calculated_invariants.csv');
invars_c=glen.count.data(:,1:end-1);
invars_c=invars_c-repmat(mean(invars_c),s,1);
invars_c=invars_c./repmat(std(invars_c),s,1);

glen.all=importdata('~/Research/data/MRI/BLSA/BLSA_0317/base/all_invariants.csv');
data=glen.all.data(:,1:end-1);
data=data-repmat(mean(data),s,1);
data=data./repmat(std(data),s,1);
BWinvars=data;

Binvars=glen.all.data(:,1:10);
Binvars=Binvars-repmat(mean(Binvars),s,1);
Binvars=Binvars./repmat(std(Binvars),s,1);

Winvars=glen.all.data(:,11:end-1);
Winvars=Winvars-repmat(mean(Winvars),s,1);
Winvars=Winvars./repmat(std(Winvars),s,1);


% ignore 4th column
invars_c4=glen.count.data(:,1:end-1);
invars_c4(:,4)=[];
invars_c4=invars_c4-repmat(mean(invars_c4),s,1);
invars_c4=invars_c4./repmat(std(invars_c4),s,1);

data=glen.all.data(:,1:10);
data=data-repmat(mean(data),s,1);
data=data./repmat(std(data),s,1);
Binvars4=data;

%%

datum=BWinvars;

A=datum'*datum;                           % covariance matrix
subplot(223), imagesc(A), colorbar, title('covariance')

[V D]=eig(A);                       % PCA
explained=cumsum(flipud(diag(D)))/sum(D(:)); % relative fraction of variance explained

figure(1), clf
subplot(221), plot(explained), title('frac var'), axis('tight')

% reorder stuff in the "right" way
Ddiag=diag(D);
Ds=diag(flipud(Ddiag));
Vs=fliplr(V);


% keep only eigenvectors that explain 95% of variance
keepers=find(explained>0.95,1);
Dsdiag=diag(Ds);
Dsdiag(keepers+1:end)=[];
Ds2=diag(Dsdiag);
Vs2=Vs(:,1:keepers);

% get new covariance matrix
B2=(Vs2*Ds2*Vs2');
subplot(224), imagesc(B2), colorbar, title('reconstruction')
subplot(222), imagesc(Vs2), colorbar, title('eigenvectors')

denoised=datum*Vs2;



%% compare unweighted graphs
clear yhat, 
clear Lhat
kmax=round(s/2)-1;
algs=[1 5 7 8 9 10]; %1:10;
for alg=algs
    
    switch alg
        case 1
            data=Abin;
        case 2
            data=Awei;
        case 3
            data=BWinvars;
        case 4
            data=denoised;
        case 5
            data=Binvars;
        case 6
            data=Winvars;
        case 7
            data=invars_c;
        case 8
            data=invars_c(:,4);
        case 9
            data=invars_c4;
        case 10
            data=Binvars4;
    end
    
    disp(alg)
    
    % compute distances
    d=zeros(s);
    for i=1:s
        for j=i+1:s
            if ndims(data)==2
                d(i,j)=sum((data(i,:)-data(j,:)).^2);
            else
                d(i,j)=sum(sum((data(:,:,i)-data(:,:,j)).^2));
            end
        end
    end
    d=d'+d;
    d(1:s+1:end)=inf;
    
    % classify
    for k=1:kmax
        for i=1:s
            [foo IX] = sort(d(i,:));
            yhat{alg}(i)=sum(ClassIDs(IX(1:k)))>k/2;
        end
        Lhat{alg}(k)=mean(yhat{alg}~=ClassIDs);
    end
    
end


%% plot stuff
figure(2), clf, hold all
ls{1}='-.';
ls{2}='--';
ls{3}='-';
ls{4}=':';

for alg=algs
    plot([1:24]+rand-0.5,Lhat{alg},ls{mod(alg,3)+1}) %,'-','color',color(alg))
end
grid on
legend('Ab','Binvars','invars_c','invars_c(4)','invars-4','Binvars-4','location','best')
