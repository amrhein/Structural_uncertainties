 function m = pm5(dat)
% 
% Update: Everything done in frequency space upfront!
%
% dat   Dataset, arranged space (rows) by time (cols)
%
% % -- Output -- %
%
% m      Vector of fit metrics
%
% Remove mean
dat = bsxfun(@minus,dat,mean(dat));

% Compute the SVD.
[~, S, V] = svd(dat,'econ');

% Compute the MTM power spectral density estimate of the column vectors of V. The output will be
% another matrix pV consisting of the PSDs. Each column corresponds to a
% singular vector (column of V) and rows are frequencies, nu.
[pV,nu]=pmtm((S^2*V')');

% Discard the last Ld frequencies, which are likely very uncertain. Here
% chosen to be 3.
Ld = 3;
pVc = pV((Ld+1):end,:);

% Truncate for frequency
nuc = nu((Ld+1):end);
nf = length(nuc);

% Compute the (constant) frequency increment
dnu = mode(diff(nuc));
% Make sure it's actually constant!
if ~all(abs(diff(diff(nuc)))<1e-12)
    error('Non-constant frequency increment')
end

%% Compute some metrics!
m = [];

% Bretherton et al. 1998 Eq. 3
m(:,1) = sum(pVc').^2 ./ sum(pVc'.*pVc');

