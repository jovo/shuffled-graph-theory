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
Ns=siz(3);
Awei=0*AdjMats;
idu=find(triu(ones(Nvertices),+1));
for i=1:Ns
    A=(AdjMats(:,:,i));
    A(idu)=0;
    Awei(:,:,i)=A;
end
Abin=Awei;
Abin(Abin<=t)=0;
Abin(Abin>t)=1;

%% 3. select/generate data

datatype='ind_edge';

if strcmp(datatype,'synthetic')
    const=get_constants(Abin,ClassIDs);
    P=get_ind_edge_params(Abin,const,'mle');
    
    Ns=500;    
    
    P.E0=mean(double(Abin(:,:,ClassIDs==0)),3);
    P.E1=mean(double(Abin(:,:,ClassIDs==1)),3);
    
    Nvertices=10;
    P.E0=P.E0(1:Nvertices,1:Nvertices);
    P.E1=P.E1(1:Nvertices,1:Nvertices);
    
    
    % % (testing using non-synthetic data to make sure code works)
    % Nvertices=10;
    % P.E0=rand(Nvertices)*0.7;   % params in class 0
    % P.E1=P.E0+0.3;              % params in class 1
    
    %     P.E0 = mean(Abin,3);
    %     E0=P.E0+P.E0';
    %     P.E1 = mean(E0(:))*tril(ones(Nvertices),-1);
    
    [Gs_La Ys P_syn] = generate_synthetic_data(P,ClassIDs,Ns);
    
    % plot synthetic data
    figure(2), clf
    subplot(211), imagesc(P_syn.E0-P.E0), colorbar
    subplot(212), imagesc(P_syn.E1-P.E1), colorbar
    
elseif strcmp(datatype,'syntheticweighted')
    const=get_constants(Abin,ClassIDs);
    P=get_ind_edge_params(Abin,const,'mle');
    
    Ns=100;    
    
    P.E0=mean(double(log(Awei(:,:,ClassIDs==0)+1)),3);
    P.E1=mean(double(log(Awei(:,:,ClassIDs==1)+1)),3);

    P.E0(idu)=0; 
    P.E1(idu)=0;
    
    [Gs_La Ys P_syn] = generate_synthetic_data(P,ClassIDs,Ns,'Poisson');
    
    
elseif strcmp(datatype,'weighted')
    Gs_La=Awei;
    Ys=ClassIDs';
    Ns=length(Ys);
elseif strcmp(datatype,'binary')
    Gs_La=Abin;
    Ys=ClassIDs';
    Ns=length(Ys);
elseif strcmp(datatype,'ind_edge')
    
    Ns=500;
    P.E0=rand(Nvertices);
    P.E1=rand(Nvertices);

    P.E0(idu)=0; 
    P.E1(idu)=0;
    
    Nvertices=10;
    P.E0=P.E0(1:Nvertices,1:Nvertices);
    P.E1=P.E1(1:Nvertices,1:Nvertices);
    
    ClassIDs=[zeros(1,Ns/2) ones(1,Ns/2)];
    
    [Gs_La Ys P_syn] = generate_synthetic_data(P,ClassIDs,Ns,'Bernoulli');

    
else
    disp('no data selected')
end


%% 4. knn classify labeled graphs and plot results

kvec = 1:(Ns-1);

labeledIDM = InterpointDistanceMatrix(Gs_La);

[Lhats.knn yhat] = knnclassifyIDM(labeledIDM,Ys,kvec);


%% 5. shuffle graphs

Gs_Sh=0*Gs_La;
for i=1:Ns
    q=randperm(Nvertices);
    A=Gs_La(:,:,i);
    A=A+A';
    Gs_Sh(:,:,i)=A(q,q);
end

% if any(Gs_Sh>1), err('something is horrible'), end

%% 6. compute unlabeled interpoint distance matrix

disp('Computing Interpoint-Distance-Matrix...')


% compute distances
% d=zeros(Ns);
% f=zeros(Ns
for i=1:Ns
    
    if mod(i,10)==0, disp('did another 10 rows of the IDM'); end
    
    parfor j=1:i %1:Ns
        disp('did another 10 elements of the IDM')
        [f(i,j), myp]=sfw(-Gs_Sh(:,:,i),Gs_Sh(:,:,j)',15);
        d(i,j)=sum(sum((Gs_Sh(myp,myp,i)-Gs_Sh(:,:,j)).^2));
    end
end

%% 7. knn clasify unshuffled graphs

IDMd=d'+d;
IDMd(1:Ns+1:end)=inf;
[Lhats.shuffled] = knnclassifyIDM(IDMd,Ys,kvec);


%% 8. knn classify using graph invariants

Gs_LaaL=0*Gs_La;
for i=1:Ns
    A=Gs_La(:,:,i);
    A=A+A';
    Gs_LaaL(:,:,i)=A;
end

x = get_graph_invariants(Gs_LaaL,1:7);
x=x-repmat(mean(x')',1,Ns);
x=x./repmat(std(x,[],2),1,Ns);

GI_IDM = InterpointDistanceMatrix(x);
[Lhats.GI yhat] = knnclassifyIDM(GI_IDM,Ys,kvec(kvec<Ns));

%% 9. plot some stuff

% plot(Lhatf_shuffled),
figure(1), clf, 
plot(Lhats.knn), hold all
plot(Lhats.shuffled)
plot(Lhats.GI)
xlabel('k')
ylabel('misclassification rate')

legend('labeled','unshuffled','invariants')
print('-dpng','../figs/knn_Lhats')


%% 8. naive bayes plugin unlabeled graphs
%
% save(['../data/', datatype, '_classification'])
%
%
% if strcmp(datatype,'true')
%
%     Phat=get_ind_edge_params(Gs_Un,Ys);
%
%     Ntst=Ns;
%     [Gtst_La Ytst] = generate_synthetic_data(P,ClassIDs,Ntst);
%
%
%     Gtst_Sh=0*Gtst_La;
%     for i=1:Ntst
%         q=randperm(Nvertices);
%         A=Gtst_La(:,:,i);
%         A=A+A';
%         Gtst_Sh(:,:,i)=A(q,q);
%     end
%
%     Ytst_hat=nan(Ntst,1);
%     parfor i=1:Ntst
%         [~, myp0]=sfw(-Gtst_Sh(:,:,i),Gs_Un(:,:,y0proto)',15);
%         Utst0=Gtst_Sh(myp0,myp0,i);
%
%         [~, myp1]=sfw(-Gtst_Sh(:,:,i),Gs_Un(:,:,y1proto)',15);
%         Utst1=Gtst_Sh(myp1,myp1,i);
%
%
%         post0=sum(Utst0(:).*Phat.lnE0(:)+(1-Utst0(:)).*Phat.ln1E0(:))+Phat.lnprior0;
%         post1=sum(Utst1(:).*Phat.lnE1(:)+(1-Utst1(:)).*Phat.ln1E1(:))+Phat.lnprior1;
%
%         [~, bar] = sort([post0, post1]); % find the bigger one
%         Ytst_hat(i)=bar(2)-1;
%     end
%     Lhat_NB = sum(Ytst_hat~=Ytst)/length(Ytst);
%
%     disp(['naive bayes performance is: ', num2str(Lhat_NB)])
%
% end

%% 9. save stuff

save(['../data/', datatype, '_classification'])

%%
binLa=labeledIDM;
binLa(binLa<=median(binLa(:)))=0;
binLa(binLa>median(binLa(:)))=1;

binSh=IDMd;
binSh(binSh<=median(binSh(:)))=0;
binSh(binSh>median(binSh(:)))=1;

subplot(231), imagesc(labeledIDM)
subplot(232), imagesc(IDMd)
% subplot(233), imagesc(labe~=binSh)


subplot(234), imagesc(binLa)
subplot(235), imagesc(binSh)
subplot(236), imagesc(binLa~=binSh)



