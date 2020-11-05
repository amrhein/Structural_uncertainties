% function m = pm2(dat0,dat1)
% 
% Update: Everything done in frequency space upfront!
%
% dat   Dtaset, arranged space (rows) by time (cols)
%
% % -- Output -- %
%
% m      Vector of fit metrics
%
% Updated from pattern_metric_all.m
%
% DEA, 26 Sept 2018
% 
% Some interesting choices for thinking about observational noise!
% dat = randn(100,100);
% dat = cumsum(randn(100,100));

function m = pm2(dat)
% Compute the SVD.
[~, S, V1] = svd(dat,'econ');

% Compute the MTM power spectral density estimate of the column vectors of V. The output will be
% another matrix pV consisting of the PSDs. Each column corresponds to a
% singular vector (column of V) and rows are frequencies, nu.
[~, S, V] = svd(dat,'econ');

[pV,nu]=pmtm(S*V);

% Discard the last Ld frequencies, which are likely very uncertain. Here
% chosen to be 3.
Ld = 3;

pVc = pV((Ld+1):end,:);
nuc = nu((Ld+1):end);

nf = length(nuc);

% Compute the (constant) frequency increment
dnu = mode(diff(nuc));
% Make sure it's actually constant!
if ~all(abs(diff(diff(nuc)))<1e-12)
    error('Non-constant frequency increment')
end

% Compute the cumulative integral. Minus sign is necessary to get a
% positive result because the integration is working in the opposite
% direction, and swapping limits changes the sign
% imtm = cumtrapz(-flipud(nuc),flipud(pVc));

% Define the full pattern variance to be the reverse cumulative integral
% down to our truncation frequency
maxpvar = dnu*sum(pVc);
% Should be the same as imtm(:,end)

% Trace of the full field. Used for normalizing
tr = sum(maxpvar);

%% Compute some metrics!
m = [];

% Pattern richness as a function of frequency:
m(:,1) = sum(pVc');

% Normalized (for intermodel comparisons).
m(:,2) = m(:,1)/tr;

% The cumulative sum (previously the "simple metric").
% Because frequencies are evenly spaced, we can use cumsum and multiply by
% dnu.
%m(:,3) = flipud(cumtrapz(-flipud(nuc),flipud(m(:,1))));
m(:,3) = dnu*cumsum(m(:,1),'reverse');

% Normalized. Should be 1 at the end!
m(:,4) = m(:,3)/tr;





