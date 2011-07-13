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
for i=1:s
    A=(AdjMats(:,:,i));
    A(A<=t)=0;
    A(A>t)=1;
    As(:,:,i)=A;
end

%% unlabel and don't try to do any assignment; this gives us L_chance

% make data unlabeled
shuffled_graphs=0*As;
for i=1:s
    q=randperm(n);
    A=As(:,:,i);
    shuffled_graphs(:,:,i)=A(q,q);
end

% chance
Lhats.chance = xval_SigSub_classifier(shuffled_graphs,ClassIDs,[],'loo');

% optimal
Lhats.star = xval_SigSub_classifier(As,ClassIDs,[],'loo');

% unshuffled
Utrn=0*As;
myp=nan(n,s);
parfor i=1:s
    i
    if ClassIDs(i)==1
        [~, myp(:,i)]=sfw(-shuffled_graphs(:,:,i),shuffled_graphs(:,:,1)',15);
    else
        [~, myp(:,i)]=sfw(-shuffled_graphs(:,:,i),shuffled_graphs(:,:,4)',15);
    end
    Utrn(:,:,i)=shuffled_graphs(myp(:,i),myp(:,i),i);
end


[Lhat incorrects subspace] = xval_SigSub_classifier(Utrn,ClassIDs,[],'loo');
Lhats.unshuffled=Lhat

