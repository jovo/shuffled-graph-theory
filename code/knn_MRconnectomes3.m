%% simulate independent edge models and classify using or not using the vertex names
clear; clc

alg.fname   = 'MRconnectome';    % different names will generate different simulations
alg.datadir = '../data/';
alg.figdir  = '../figs/';
alg.save    = 0;                    % whether to save/print results

load('~/Research/data/MRI/BLSA/BLSA_0317/base/BLSA_0317_countMtx.mat')

t=200;
siz=size(AdjMats);
n=siz(1);
s=siz(3);

%% unlabel 

As=AdjMats;
As(As<=t)=0;
As(As>t)=1;

% make data unlabeled
Ss=0*As;
for i=1:s
    q=randperm(n);
    A=As(:,:,i);
    Ss(:,:,i)=A(q,q);
end


%%
yhat=nan(s,1);
for i=1:s % xval
    
    trn_idx=setdiff(1:s,i);

    d=nan(s,1);
    parfor j=1:s-1
        d(j)=sfw(-Ss(:,:,trn_idx(j)),Ss(:,:,i)',15);
    end
    
    [foo idx]=min(d);
    yhat(i)=ClassIDs(idx);
    
    
end

%%
idx=find(tril(ones(n),-1));
Amat=nan(s,length(idx));
for i=1:s
    A=(AdjMats(:,:,i));
    Avec=A(idx);
    Awei(i,:)=Avec;
end
Abin=Awei;
Abin(Abin<=t)=0;
Abin(Abin>t)=1;

%% get invariant representations of the data

glen.count=importdata('~/Research/data/MRI/BLSA/BLSA_0317/base/count_calculated_invariants.csv')
invars_c=glen.count.data(:,1:end-1);
invars_c=invars_c-repmat(mean(invars_c),s,1);
invars_c=invars_c./repmat(std(invars_c),s,1);

%% knnclassify labeled graphs

nAlgs=3;
kmax=20;
for k=1:kmax
    for i=1:s
        
        trn_idx=setdiff(1:s,i);
        
        for alg=1:nAlgs
            
            switch alg
                case 1
                    data=Awei;
                case 2
                    data=Abin;
                case 3
                    data=invars_c;
            end
            
            class{alg}(i)=knnclassify(data(i,:),data(trn_idx,:),ClassIDs(trn_idx),k);
        end
        
    end
    
    for alg=1:nAlgs
        Lhat{alg}(k)=mean(class{alg}~=ClassIDs);
    end
end

%%
figure(1), clf, hold all
for alg=1:nAlgs
    plot(Lhat{alg})
end

legend('weighted','bin','invars')
