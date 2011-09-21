function [As_syn ys_syn P] = generate_synthetic_data(data,ys,nmc,dist)
% generate nmc synthetic data samples from the joint distribution implied by (data,ys)
%
% INPUT
%   data: could be either the collection of graphs or a structure
%           containing E0 and E1
%   ys: list of class labels
%   nmc: # of data samples to generate
%
% OUTPUT
%   As_syn: collection of graphs
%   ys_syn: collection of class labels associated with graphs
%   P:      distribution from which synthetic data were sampled


disp('Generating synthetic data....')

% get likelihood parameters
if ~isstruct(data)
    As=data;
    
    [n , ~, s]=size(As);
    
    constants   = get_constants(As,ys);
    P           = get_ind_edge_params(As,constants);
else
    P=data;
    s=length(ys);
    n=length(P.E0);
end

if nargin==3
    dist='Bernoulli';
    display('assumed binary edges')
end

% priors
s1=round(sum(ys)*nmc/s);
s0=nmc-s1;


ix=randperm(nmc); % arbitrary ordering

% generate graphs
As_syn = nan(n,n,nmc);
if strcmp(dist,'Bernoulli')
    As_syn(:,:,ix(1:s0))    = repmat(P.E0,[1 1 s0]) > rand(n,n,s0);
    As_syn(:,:,ix(s0+1:end))= repmat(P.E1,[1 1 s1]) > rand(n,n,s1);
elseif strcmp(dist,'Poisson')
    for i = ix(1:s0)
        As_syn(:,:,i) = poissrnd(P.E0);
    end
    for i = ix(s0+1:end)
        As_syn(:,:,i) = poissrnd(P.E1);
    end
end


% generate labels
ys_syn=nan(nmc,1);
ys_syn(ix(1:s0))=0;
ys_syn(ix(s0+1:end))=1;
