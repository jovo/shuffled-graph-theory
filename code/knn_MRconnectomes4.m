%% simulate independent edge models and classify using or not using the vertex names
clear; clc

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

glen.count=importdata('~/Research/data/MRI/BLSA/BLSA_0317/base/count_calculated_invariants.csv');
invars_c=glen.count.data;
invars_c=invars_c-repmat(mean(invars_c),s,1);
invars_c=invars_c./repmat(std(invars_c),s,1);

glen.FA=importdata('~/Research/data/MRI/BLSA/BLSA_0317/base/FA_calculated_invariants.csv');
invars_f=glen.FA.data;
invars_f=invars_f-repmat(mean(invars_f),s,1);
invars_f=invars_f./repmat(std(invars_f),s,1);

glen.weighted=importdata('~/Research/data/MRI/BLSA/BLSA_0317/base/calculated_weighted_invariants_forMark.csv');
invars_w=glen.FA.data;
invars_w=invars_w-repmat(mean(invars_w),s,1);
invars_w=invars_w./repmat(std(invars_w),s,1);

%% algorithm paramaters


% alg(1).name='knn';
% alg(1).params.k=1:kmax;
% alg(1).whiten=0;
% 
% 
% % alg(2).name='discrim';
% % alg(2).whiten=0;
% % alg(3).params.
% 
% kmax=20;
% discrim.LDA=1;
% discrim.QDA=1;
% discrim.dLDA=1;
% discrim.dQDA=1;


%% classify labeled graphs

nAlgs=5;
for alg=1:nAlgs
    
    switch alg
        case 1
            data=Awei;
        case 2
            data=Abin;
        case 3
            data=invars_c;
        case 4
            data=invars_f;
        case 5
            data=invars_w;
    end
    
    for i=1:s
        
        trn_idx=setdiff(1:s,i);
        
        % knn classifiers
        for k=1:kmax
            class{alg}(i)=knnclassify(data(i,:),data(trn_idx,:),ClassIDs(trn_idx,k);
        end
        
        %     % disciminant classifiers
        %
        %     Lhat_wDA = lda_loocv(data, ClassIDs, discrim, 1);
        %     Lhat_DA  = lda_loocv(data, ClassIDs, discrim, 0);
        
        
        for alg=1:nAlgs
            Lhat{alg}(k)=mean(class{alg}~=ClassIDs);
        end
    end
end


[Lhat Lsem] = discriminant_classifiers(preds,ys,params,discrim)

%%
figure(1), clf, hold all
for alg=1:nAlgs*3
    plot(Lhat{alg})
end

legend('weighted','bin','inv_c','inv_w','inv_f')

%% unlabel and don't try to do any assignment; this gives us L_chance

% make data unlabeled
shuffled_graphs=0*As;
for i=1:s
    q=randperm(n);
    A=As(:,:,i);
    shuffled_graphs(:,:,i)=A(q,q);
end

