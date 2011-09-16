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

datatype='binary';

if strcmp(datatype,'synthetic')
    const=get_constants(Abin,ClassIDs);
    P=get_ind_edge_params(Abin,const,'mle');
    
    Ns=100;
    P.E0 = mean(Abin,3);
    E0=P.E0+P.E0';
    P.E1 = mean(E0(:))*tril(ones(Nvertices),-1);
    
    % % (testing using non-synthetic data to make sure code works)
    % Nvertices=10;
    % P.E0=rand(Nvertices)*0.7;   % params in class 0
    % P.E1=P.E0+0.3;              % params in class 1
    
    [Gs_La Ys P_syn] = generate_synthetic_data(P,ClassIDs,Ns);
    
    % plot synthetic data
    figure(2), clf
    subplot(211), imagesc(P_syn.E0-P.E0), colorbar
    subplot(212), imagesc(P_syn.E1-P.E1), colorbar
    
elseif strcmp(datatype,'weighted')
    Gs_La=Awei;
    Ys=ClassIDs';
    Ns=length(Ys);
elseif strcmp(datatype,'binary')
    Gs_La=Abin;
    Ys=ClassIDs';
    Ns=length(Ys);
end


%% 4. knn classify labeled graphs and plot results

kvec = 1:round(Ns*0.9);

labeledIDM = InterpointDistanceMatrix(Gs_La);

[Lhat yhat] = knnclassifyIDM(labeledIDM,Ys,kvec);

figure(1), clf, plot(Lhat), hold all

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
[Lhatd_shuffled] = knnclassifyIDM(IDMd,Ys,kvec);

IDMf=-f'-f;
IDMf(1:Ns+1:end)=inf;
[Lhatf_shuffled] = knnclassifyIDM(IDMf,Ys,kvec);

% plot(Lhatf_shuffled), 
plot(Lhatd_shuffled)
xlabel('k')
ylabel('misclassification rate')

legend('labeled','unshuffled')
print('-dpng','../figs/knn_Lhats')


%% 8. naive bayes plugin unlabeled graphs

save(['../data/', datatype, '_classification'])


if strcmp(datatype,'true')
    
    Phat=get_ind_edge_params(Gs_Un,Ys);
    
    Ntst=Ns;
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
    
    disp(['naive bayes performance is: ', num2str(Lhat_NB)])
    
end

%% 9. save stuff

save(['../data/', datatype, '_classification'])