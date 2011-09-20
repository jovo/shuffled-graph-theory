function Lhats = shuffled_classifiers(As,Ys)
%% shuffled classifiers

%% 1. labeled Frobenius norm knn on labeled graphs

labeledIDM = InterpointDistanceMatrix(As);

Lhats.knn = knnclassifyIDM(labeledIDM,Ys,1);

%% 2. approximate graph-matched Frobenius norm knn on shuffled graphs

Ashuffled=0*As;
Ns=length(labeledIDM);

% shuffle
for i=1:Ns
    q=randperm(Nvertices);
    A=As(:,:,i);
    A=A+A';
    Ashuffled(:,:,i)=A(q,q);
end

disp('Computing Interpoint-Distance-Matrix...')
for i=1:Ns
    
    if mod(i,10)==0, disp('did another 10 rows of the IDM'); end
    
    parfor j=1:i %1:Ns
        disp('did another 10 elements of the IDM')
        [f(i,j), myp]=sfw(-Ashuffled(:,:,i),Ashuffled(:,:,j)',15);
        d(i,j)=sum(sum((Ashuffled(myp,myp,i)-Ashuffled(:,:,j)).^2));
    end
end

IDMd=d'+d;
IDMd(1:Ns+1:end)=inf;
Lhats.shuffled = knnclassifyIDM(IDMd,Ys,1);


%% 3. knn on graph invariants

x = get_graph_invariants(As);



