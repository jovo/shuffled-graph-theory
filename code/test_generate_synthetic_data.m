%% test QAP+KNN vs # samples
%
% 1. load data
% 2. binarize
% 3. generate synthetic data
% 4. knn classify labeled graphs and plot results
% 5. shuffle graphs
% 6. try to unshuffle using QAP
% 7. knn classify unlabeled graphs and plot results


%% 1. load data

clear, clc
load('~/Research/data/MRI/BLSA/BLSA_0317/base/BLSA_0317_countMtx.mat')

%% 2. binarize

t=200;
siz=size(AdjMats);
Nvertices=siz(1);
s=siz(3);
Awei=0*AdjMats;
idu=find(triu(ones(Nvertices),+1));
for i=1:s
    A=(AdjMats(:,:,i));
    A(idu)=0;
    Awei(:,:,i)=A;
end
Abin=Awei;
Abin(Abin<=t)=0;
Abin(Abin>t)=1;

%% 3. generate synthetic data

const=get_constants(Abin,ClassIDs);
P=get_ind_edge_params(Abin,const,'mle');

Ns=100;
% P.E0 = mean(Abin,3);
% P.E1 = mean(P.E0+P.E0')*ones(Nvertices);

% % (testing using non-synthetic data to make sure code works)
% Nvertices=10;
% P.E0=rand(Nvertices)*0.7;   % params in class 0
% P.E1=P.E0+0.3;              % params in class 1

[Gs_La Ys P_syn] = generate_synthetic_data(P,ClassIDs,Ns);

%% 4. knn classify labeled graphs and plot results

kvec = 1:round(Ns*0.9);

IDM = InterpointDistanceMatrix(Gs_La);

[Lhat yhat] = knnclassifyIDM(IDM,Ys,kvec);

% plot synthetic data
figure(1), clf
subplot(311), imagesc(P_syn.E0-P.E0), colorbar
subplot(312), imagesc(P_syn.E1-P.E1), colorbar
subplot(313), plot(Lhat,'k')

%% 5. shuffle graphs

Gs_Sh=0*Gs_La;
for i=1:Ns
    q=randperm(Nvertices);
    A=Gs_La(:,:,i);
    A=A+A';
    Gs_Sh(:,:,i)=A(q,q);
end


%% 6. try to unshuffle using QAP

Gs_Un=0*Gs_Sh;
myp=nan(Nvertices,Ns);

y0proto=find(Ys==0,1);
y1proto=find(Ys==1,1);

parfor i=1:Ns
    if mod(i,10)==0, disp(['10 more unshufflings done our of ', Ns]), end
    if Ys(i)==1
        [~, myp(:,i)]=sfw(-Gs_Sh(:,:,i),Gs_Sh(:,:,y1proto)',15);
    else
        [~, myp(:,i)]=sfw(-Gs_Sh(:,:,i),Gs_Sh(:,:,y0proto)',15);
    end
    Gs_Un(:,:,i)=Gs_Sh(myp(:,i),myp(:,i),i);
end

%% 7. knn clasify unshuffled graphs

IDM = InterpointDistanceMatrix(Gs_Un);

[Lhat_shuffled] = knnclassifyIDM(IDM,Ys,kvec);

subplot(313), hold all, plot(Lhat_shuffled,'b')

%% 8. naive bayes plugin unlabeled graphs

Phat=get_ind_edge_params(Gs_Un,Ys);

Ntst=1000;
[Gtst_La Ytst] = generate_synthetic_data(P,ClassIDs,Ntst);


Gtst_Sh=0*Gtst_La;
for i=1:Ntst
    q=randperm(Nvertices);
    A=Gtst_La(:,:,i);
    A=A+A';
    Gtst_Sh(:,:,i)=A(q,q);
end

Ytst_hat=nan(Ntst,1);
parfor i=1:Ntst
    [~, myp0]=sfw(-Gtst_Sh(:,:,i),Gs_Un(:,:,y0proto)',15);
    Utst0=Gtst_Sh(myp0,myp0,i);
    
    [~, myp1]=sfw(-Gtst_Sh(:,:,i),Gs_Un(:,:,y1proto)',15);
    Utst1=Gtst_Sh(myp1,myp1,i);
    
    
    post0=sum(Utst0(:).*Phat.lnE0(:)+(1-Utst0(:)).*Phat.ln1E0(:))+Phat.lnprior0;
    post1=sum(Utst1(:).*Phat.lnE1(:)+(1-Utst1(:)).*Phat.ln1E1(:))+Phat.lnprior1;
    
    [~, bar] = sort([post0, post1]); % find the bigger one
    Ytst_hat(i)=bar(2)-1;
end
Lhat_NB = sum(Ytst_hat~=Ytst)/length(Ytst);

disp(['naive bayes performance is: ', Lhat_NB])