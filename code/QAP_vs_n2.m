%% simulate independent edge models and classify using or not using the vertex names
clear; clc

n_tst= 500;                           % # of samples
n   = 10;                           % # of vertices
alg.model = 'bern';
fname   = 'hetero_easy';    % different names will generate different simulations

alg.fname   = [fname '_n' num2str(n) '_MC' num2str(n_tst)];
switch fname                    % choose simulation parameters
    case 'homo_kidney_egg'
        
        p   = 0.5;      % prob of connection for kidney
        q0  = 0.2;      % prob of connection for egg
        q1  = 0.8;      % prob of connection for egg
        egg = 2:n-1;      % vertices in egg
        
        E0=p*ones(n);   % params in class 0
        E0(egg,egg)=q0; % egg params in class 0
        
        E1=p*ones(n);   % params in class 1
        E1(egg,egg)=q1; % egg params in class 1
        
        P.p=p; P.q0=q0; P.q1=q1; P.egg=egg;
        
    case 'ER1'
        
        p=0.25;
        q=0.75;
        E0=p*ones(n);     % params in class 0
        E1=q*ones(n);     % params in class 1
        
        alg.fname=[alg.fname '_p' num2str(round(p*100)) '_q' num2str(round(q*100))];
        
    case 'hetero'
        
        E0=rand(n);     % params in class 0
        E1=rand(n);     % params in class 1
                
    case 'hetero_easy'
        
        E0=rand(n)*0.7;     % params in class 0
        E1=E0+0.3;     % params in class 1
                
    case 'structured'
        
        E0=0.1*ones(n);     % params in class 0
        E1=0.1*ones(n);     % params in class 1
        E0ind=[2 n+4 2*n+8 3*n+3 4*n+9 5*n+6 6*n+7 7*n+1 8*n+4 9*n+5];
        E1ind=[4 n+2 2*n+3 3*n+8 4*n+9 5*n+1 6*n+5 7*n+1 8*n+5 9*n+4];
        pp=randperm(n^2); pp=pp(1:10);
        E0([pp E0ind])=0.8;
        E1([pp E1ind])=0.8;
                
    case 'hetero_kidney_egg'
        
        E0=rand(n);     % params in class 0
        E1=rand(n);     % params in class 1
        egg = 1:5;      % vertices in egg
        E1(egg,egg)=E0(egg,egg);
        
    case 'hard_hetero'
        
        E0=rand(n);         % params in class 0
        E1=E0+randn(n)*.005;  % params in class 1
        E1(E1>=1)=1-1e-3;   % don't let prob be >1
        E1(E1<=0)=1e-3;     % or <0
                
    case 'celegans'
        
        load('~/Research/data/EM/c_elegans/c_elegans_chemical_connectome.mat')
        
        n=279;
        m=10;
        thesem = [69,80,82,94,110,127,129,133,134,138];
        eps = 0.05;  % noise parameter "eps"
        sig = 15;     % egg parameter "sig"
        
        E0=A+eps;
        E1=E0;
        egg = sig*rand(m)-sig/2;
        E1(thesem,thesem) = E1(thesem,thesem) + egg;
        E1(E1<0)=eps;
        
        alg.model='poiss';
        
    case 'poiss'
        
        E0=rand(n)*100;     % params in class 0
        E1=rand(n)*100;     % params in class 1
        
        alg.model='poiss';
        
    case 'BLSA0317_synth'
        
        load('~/Research/data/MRI/BLSA/BLSA_0317/base/BLSA_0317_countMtx.mat')

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
        
        constants=get_constants(As,ClassIDs);
        Psynth = get_ind_edge_params(As,constants);

        E0=Psynth.E0;
        E1=Psynth.E1;
        
        
    case 'BLSA0317_sigsub'

        load('~/Research/projects/active/ind_edge_classifier/data/results/BLSA0317_Count_Lhats');
        
        constants=get_constants(As,ClassIDs);
        Psynth = get_ind_edge_params(As,constants);
        totalmean=mean(As,3);
        
        E0=totalmean;
        E0(coherent)=Psynth.E0(coherent);
        
        E1=totalmean;
        E1(coherent)=Psynth.E1(coherent);

        
end

undirected=1;
if undirected==1
   E0=tril(E0,-1);
   E1=tril(E1,-1);
end

P.n=n; P.S=n_tst; P.E0=E0; P.E1=E1;

alg.datadir = '../data/';
alg.figdir  = '../figs/';
alg.save    = 0;                    % whether to save/print results

ytst=repmat([0 1],1,n_tst/2); %[zeros(n_tst/2,1); ones(n_tst/2,1)]; %round(rand(n_tst,1));                   % sample classes iid where P[Y=1]=1/2

ytst1=find(ytst==1);
len1=sum(ytst);

ytst0=find(ytst==0);
len0=n_tst-len1;

% sample testing data
if strcmp(alg.model,'bern')
    Atst=nan(n,n,n_tst);
    Atst(:,:,ytst1)=repmat(E1,[1 1 len1]) > rand(n,n,len1);    % class 0 samples
    Atst(:,:,ytst0)=repmat(E0,[1 1 len0]) > rand(n,n,len0);    % class 0 samples
elseif strcmp(alg.model,'poiss')
    Atst=nan(n,n,n_tst);
    Atst(:,:,ytst0)=poissrnd(repmat(E0,[1 1 len0]));    % class 0 samples
    Atst(:,:,ytst1)=poissrnd(repmat(E1,[1 1 len1]));    % class 0 samples
end
for i=1:n_tst
    A=Atst(:,:,i);
    Atst(:,:,i)=A+A';
end


% parameters for naive bayes classifiers
P.lnE0  = log(P.E0);
P.ln1E0 = log(1-P.E0);
P.lnE1  = log(P.E1);
P.ln1E1 = log(1-P.E1);

P.lnprior0 = log(1/2);
P.lnprior1 = log(1/2);

if alg.save
    save([alg.datadir alg.fname])
end

%% unlabel and don't try to do any assignment; this gives us L_chance

% make data unlabeled
adj_mat=0*Atst;
for i=1:n_tst
    q=randperm(n);
    A=Atst(:,:,i);
    adj_mat(:,:,i)=A(q,q);
end

inds=find(tril(ones(n),-1));
subspace.nb=inds;

% naive bayes classify
Lhat = naive_bayes_classify(adj_mat,ytst,P,subspace);
Lhats.rand  = Lhat.nb;
Lsems.rand  = sqrt(Lhat.nb*(1-Lhat.nb))/sqrt(n_tst);

%% performance using true parameters and labels; this gives us L_*

Lhat = naive_bayes_classify(Atst,ytst,P,subspace);
Lhats.star  = Lhat.nb;
Lsems.star  = sqrt(Lhat.nb*(1-Lhat.nb))/sqrt(n_tst);


%% sample training data
% # samples for testing and training

n_trns=[10:10:100]; % 200:100:500];
tic
for jj=1:length(n_trns)
    jj
    for tt=1:20
        tt
        n_trn=n_trns(jj);
        S0 = n_trn/2; % # of samples in class 0
        S1 = n_trn/2; % # of samples in class 1
        
        if strcmp(alg.model,'bern')
            A0 = repmat(E0,[1 1 S0]) > rand(n,n,S0);    % class 0 samples
            A1 = repmat(E1,[1 1 S1]) > rand(n,n,S1);    % class 1 samples
        elseif strcmp(alg.model,'poiss')
            A0 = poissrnd(repmat(E0,[1 1 S0]));    % class 0 samples
            A1 = poissrnd(repmat(E1,[1 1 S1]));    % class 1 samples
        end
        Atrn = cat(3,A0,A1);                        % concatenate to get all training samples
        for i=1:n_trn
           A=Atrn(:,:,i);
           Atrn(:,:,i)=A+A';
        end
        
        ytrn = [zeros(1,n_trn/2) ones(1,n_trn/2)];
                
        % shuffle training data
        trn_perm=zeros(n_trn,n);
        Strn=0*Atrn;
        for j=1:n_trn
            trn_perm(j,:) = randperm(n);
            Strn(:,:,j)=Atrn(trn_perm(j,:),trn_perm(j,:),j);
        end
        
        % try to unshuffle
        Utrn=0*Atrn;
        myp=nan(n,n_trn);
        for i=1:n_trn/2
            [~, myp(:,i)]=sfw(-Strn(:,:,i),Strn(:,:,1)',15);
            Utrn(:,:,i)=Strn(myp(:,i),myp(:,i),i);
        end
        
        for i=n_trn/2+1:n_trn
            [~, myp(:,i)]=sfw(-Strn(:,:,i),Strn(:,:,n_trn/2+1)',15);
            Utrn(:,:,i)=Strn(myp(:,i),myp(:,i),i);
        end
        
        % estimate parameters given unshuffled graphs
        constants=get_constants(Utrn,ytrn);
        Phat = get_ind_edge_params(Utrn,constants);

        
        % classify using unshuffled graphs
        
        for i=1:n_tst
            [~, myp0]=sfw(-Atst(:,:,i),Strn(:,:,1)',15);
            Utst0=Atst(myp0,myp0,i);
            
            [~, myp1]=sfw(-Atst(:,:,i),Strn(:,:,n_trn/2+1)',15);
            Utst1=Atst(myp1,myp1,i);
            
            post0=sum(Utst0(:).*Phat.lnE0(:)+(1-Utst0(:)).*Phat.ln1E0(:))+Phat.lnprior0;
            post1=sum(Utst1(:).*Phat.lnE1(:)+(1-Utst1(:)).*Phat.ln1E1(:))+Phat.lnprior1;
            
            [~, bar] = sort([post0, post1]); % find the bigger one
            yhat(i)=bar(2)-1;
        end
        
        Lhat_tmp = sum(yhat~=ytst)/length(ytst);
        shuffle_Lhat(jj,tt)= Lhat_tmp;
        shuffle_Lsem(jj,tt)  = sqrt(Lhat_tmp*(1-Lhat_tmp))/sqrt(n_tst);
                
    end
end
toc
if alg.save
    save([alg.datadir alg.fname '_results'])
end



%% plot model
% load([alg.datadir alg.fname '_results'])
%
figure(2), clf
fs=12;  % fontsize
hold all
errorbar(n_trns,mean(shuffle_Lhat,2),std(shuffle_Lhat,[],2),'color','k','linewidth',2)
plot([1 10], [Lhats.star Lhats.star])
plot(n_trns(end)*1.02,Lhats.star,'.','color',0.7*[1 1 1],'markersize',18)
ylabel('misclassification rate','color','k','fontsize',fs)
xlabel('number of training samples','fontsize',fs)
xlim([n_trns(1)*0.8 n_trns(end)*1.05])
set(gca,'XTick',n_trns([1 5:end]))
% title('QAP Classification Performance','fontsize',fs)
% axis([0 alg.QAP_max_iters min(Lhats.QAP-Lsems.QAP)*0.9 max(Lhats.QAP+Lsems.QAP)*1.1])
% 
if alg.save
    wh=[4 2];   %width and height
    set(gcf,'PaperSize',wh,'PaperPosition',[0 0 wh],'Color','w');
    figname=[alg.figdir alg.fname '_QAP_vs_n'];
    print('-dpdf',figname)
    print('-dpng',figname)
    saveas(gcf,figname)
end
