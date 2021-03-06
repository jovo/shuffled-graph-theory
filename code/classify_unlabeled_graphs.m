function [LAP QAP] = classify_unlabeled_graphs(Atrn,Atst,ytst,P,alg)
% this function classifies unlabeled graphs, in the following steps:
%
% (i) permute Atst to be like both a single Atrn in class 0 and a single
% Atrn in class 1
% (ii) compute the likelihood of the permuted Atsts coming from either of
% the classes
%
% we approximate solving the permuation by using LAP and/or QAP
% initial conditions can be set for both
% QAP is iterative, so we can set the number of iterations
%
% INPUT:
%   Atrn:   an n_V x n_V x S/2 array of adjacency matrices for training
%   Atst:   an n_V x n_V x S/2 array of adjacency matrices for testing
%   ytst:   list of classes for test data
%   P:      structure of parameters necessary for naive bayes classification
%   alg:    a structure of settings for the algorithms
%
% OUTPUT:
%   LAP:    a structure containing all the LAP results
%   QAP:    a structure containing all the QAP results

%% initialize stuff

siz = size(Atrn);
n_V   = siz(1);       % # of vertices
n_MC= length(ytst);   % total # of samples

if alg.truth_start == false     % if not starting at the truth, generate permuations for the test graphs
    tst_ind=zeros(n_MC,n_V);
    for j=1:n_MC
        tst_ind(j,:) = randperm(n_V);
    end
end

LAP.do = false;
QAP.do = false;
for i=1:length(alg.names)
    if strcmp(alg.names(i),'LAP'),
        LAP.time    = 0;
        LAP.do      = true;
        LAP.yhat    = NaN(n_MC,1);
        LAP.correct = NaN(n_MC,1);
        LAP.ind0   = NaN(n_MC,n_V);
        LAP.ind1   = NaN(n_MC,n_V);
    end
    
    if strcmp(alg.names(i),'QAP'),
        QAP.time    = 0;
        QAP.do      = true;
        QAP.max_iters = ceil(alg.QAP_max_iters);
        QAP.yhat    = NaN(n_MC,QAP.max_iters);
        QAP.correct = NaN(n_MC,QAP.max_iters);
        QAP.obj0    = NaN(n_MC,QAP.max_iters);
        QAP.obj1    = NaN(n_MC,QAP.max_iters);
        QAP.inds0    = NaN(n_MC,QAP.max_iters,n_V);
        QAP.inds1    = NaN(n_MC,QAP.max_iters,n_V);
    end
end

%% do Monte Carlo classifications
for j=1:n_MC
    
    disp(j)
    
    if alg.truth_start == false
        A = Atst(tst_ind(j,:),tst_ind(j,:),j);
    else
        A = Atst(:,:,j);
    end
    
    % do LAP
    if LAP.do
        starttime = cputime;
        LAP.ind0(j,:) = munkres(-Atrn(:,:,j)*A');       %Note that sum(sum(-A(ind0,:).*Atst))=f0
        LAP.ind1(j,:) = munkres(-Atrn(:,:,j+n_MC)*A');  %Note that sum(sum(-A(ind1,:).*A))=f1
        
        [LAP.yhat(j) LAP.correct(j)] = do_classify(A,LAP.ind0(j,:),LAP.ind1(j,:),ytst(j),'LAP',P,Atrn(:,:,j),Atrn(:,:,j+n_MC));
        LAP.time = LAP.time + cputime-starttime;
    end
    
    % do QAP
    if QAP.do
        
        starttime = cputime;
        
        % QAP A to each graph in class 0
        [~,~,~,iter0,fs0,QAP.inds0(j,:,:)]=sfw(-Atrn(:,:,j),A',alg.QAP_max_iters,alg.QAP_init);
        QAP.obj0(j,1) = sum(sum(Atrn(:,:,j).*A));
        QAP.obj0(j,2:iter0+1)=fs0;
        
        % QAP A to each graph in class 1
        [~,~,~,iter1,fs1,QAP.inds1(j,:,:)]=sfw(-Atrn(:,:,j+n_MC),A',alg.QAP_max_iters,alg.QAP_init);
        QAP.obj1(j,1)  = sum(sum(Atrn(:,:,j+n_MC).*A));
        QAP.obj1(j,2:iter1+1)=fs1;
        
        for ii=1:min(iter0,iter1)
            [QAP.yhat(j,ii) QAP.correct(j,ii)] = do_classify(A,squeeze(QAP.inds0(j,ii,:)),squeeze(QAP.inds1(j,ii,:)),ytst(j),'QAP',P,Atrn(:,:,j),Atrn(:,:,j+n_MC));
        end
        QAP.time = QAP.time + cputime-starttime;
        
    end
end

%% get statistics

if LAP.do
    LAP.Lhat = 1-mean(LAP.correct);
    LAP.Lsem = sqrt(LAP.Lhat*(1-LAP.Lhat))/sqrt(n_MC);
end

if QAP.do
    
    % get stats prior to QAP'ing
    QAP.obj0_avg(1) = mean(QAP.obj0(:,1));
    QAP.obj1_avg(1) = mean(QAP.obj1(:,1));
    QAP.obj0_var(1) = var(QAP.obj0(:,1));
    QAP.obj1_var(1) = var(QAP.obj1(:,1));
    
    QAP.obj_avg(1)  = (QAP.obj0_avg(1)+QAP.obj1_avg(1))/2;
    QAP.obj_sem(1)   = std([QAP.obj0(:,1); QAP.obj1(:,1)])/sqrt(n_MC);
    
    % get stats for each iteration
    for ii=1:QAP.max_iters
        
        % remove nans
        corrects = QAP.correct(:,ii);
        keeper   = ~isnan(corrects);
        corrects = corrects(keeper);
        
        % get means
        QAP.Lhat(ii)        = 1-mean(corrects);
        QAP.obj0_avg(ii+1)  = mean(QAP.obj0(keeper,ii+1));
        QAP.obj1_avg(ii+1)  = mean(QAP.obj1(keeper,ii+1));
        QAP.obj_avg(ii+1)   = (QAP.obj0_avg(ii+1)+QAP.obj1_avg(ii+1))/2;
        
        % get vars/stds/sems
        QAP.num(ii)         = length(corrects);
        QAP.Lsem(ii)        = sqrt(QAP.Lhat(ii)*(1-QAP.Lhat(ii)))/sqrt(QAP.num(ii));
        QAP.obj0_var(ii+1)  = var(QAP.obj0(keeper,ii+1));
        QAP.obj1_var(ii+1)  = var(QAP.obj1(keeper,ii+1));
        QAP.obj_sem(ii+1)   = std([QAP.obj0(keeper,ii+1); QAP.obj1(keeper,ii+1)])/sqrt(QAP.num(ii));
        
    end
end


%% the classifier functions
    function [yhat correct] = do_classify(A,ind0,ind1,ytst,AP,P,Atrn0,Atrn1)
        % implement various classifiers
        % INPUT:
        %   A:      test graph
        %   ind0:   estimated permutation for training graph from class 0
        %   ind1:   estimated permutation for training graph from class 1
        %   ytst:   the true class of the test graph
        %   AP:     which assignment to assume, LAP or QAP
        %   P:      parameters
        %   Atrn0:  training graph from class 0
        %   Atrn1:  test graph from class 1
        %
        % OUTPUT:
        %   yhat:   estimated class of the test graph
        %   correct:whether yhat = y
        
        
        if strcmp(AP,'LAP')
            A0=A(ind0,:);
            A1=A(ind1,:);
        elseif strcmp(AP,'QAP')
            A0=A(ind0,ind0);
            A1=A(ind1,ind1);
        end
        
        if strcmp(alg.classifier,'BPI')
            if strcmp(alg.model,'bern')
                loss0 = sum(sum(A0.*P.lnE0+(1-A0).*P.ln1E0))+P.lnprior0; %get_lik(A0,P.lnE0,P.ln1E0);
                loss1 = sum(sum(A1.*P.lnE1+(1-A1).*P.ln1E1))+P.lnprior1; %get_lik(A1,P.lnE1,P.ln1E1);
            elseif strcmp(alg.model,'poiss')
                loss0=sum(sum(A0.*P.lnE0 - P.E0))+P.lnprior0;
                loss1=sum(sum(A1.*P.lnE1 - P.E1))+P.lnprior1;
            end
        elseif strcmp(alg.classifier,'1NN')
            loss0 = -sum(sum((A0-Atrn0).^2));
            loss1 = -sum(sum((A1-Atrn1).^2));
        end
        
        [~, bar] = sort([loss0, loss1]);
        yhat=bar(2)-1;
        correct=(yhat==ytst);
        
    end

end