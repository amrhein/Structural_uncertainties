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
% air = ncread('../../../Desktop/CCSM4_ctrl/analyses/BerkeleyEarth/BerkeleyEarth_LandOcean_infilled_1900_2015.nc','tas');
% airtest = air(:,:,1:1200);
% airr = reshape(airtest,[],1200);

function [nuc, m] = pm3(dat)

% Remove mean
dat = bsxfun(@minus,dat,mean(dat));

% Compute the SVD.
[~, S, V] = svd(dat,'econ');

% Compute the MTM power spectral density estimate of the column vectors of V. The output will be
% another matrix pV consisting of the PSDs. Each column corresponds to a
% singular vector (column of V) and rows are frequencies, nu.
[pV,nu]=pmtm((S*V')');

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

% Define the full pattern variance as a function of frequency
maxpvar = dnu*sum(pVc);

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
% NOTE: MIGHT NEED TO TAKE SQUARE ROOTS OF THIS AND THEN SQUARE AFTER THE
% SUM.
m(:,3) = dnu*cumsum(m(:,1),'reverse');

% Normalized. Should be 1 at the end!
m(:,4) = m(:,3)/tr;





