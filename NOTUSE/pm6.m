 function [nuc,m,pVc] = pm6(dat)
% 
% Update: Everything done in frequency space upfront!
%
% dat   Dataset, arranged space (rows) by time (cols)
%
% % -- Output -- %
%
% m      Vector of fit metrics
% nuc    vector of pmtm frequencies (normalized)
% pVc    matrix of PC PSD estimates
%
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
Ld = 0;
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

% Just the trace
m(:,2) = sum(pVc').^2;


%% Suggestions for plotting script

% first compute svd
[~,S,V] = svd(dat,'econ');

%%
close all

% compute variance of PCs
vv = var(V);

% compute the frequency separation from pmtm
dnu = mode(diff(nuc));

% Cumulative sum over pVcc
pVcc = bsxfun(@times,cumsum(flipud(pVc))*dnu,1./vv);

% Sum over the pmtm and normalize by variance of V. This should be very
% similar to the squared singular values diag(S).^2 !
sp = sum(flipud((pVc)))*dnu./vv;

% Pick some indices for showing frequency bins. Here based on logarithmic
% spacing. nl is number of indices.
nl = 10;
inds = fliplr(unique(round(length(nuc)-logspace(0,3,nl))));
% Here just a uniform spacing
inds = 1:200:length(nuc);

% Plot with SV spectrum. The frequency bands should nicely approach the
% final SV spectrum in black.
figure(1)
%plot(sqrt(sp),'c','linewidth',1)
plot(sqrt(pVcc(inds,:))')
hold on
plot(diag(S),'k','LineWidth',2)

% Same thing in log space
figure(2)
loglog(sqrt(pVcc(inds,:))')

% Now the different cumulative contributions from each frequency band
figure(3)
loglog(abs(diff(sqrt(pVcc(inds,:))')))
%plot(abs(bsxfun(@minus,sqrt(pVcc(inds,:))',diag(S))))
%loglog(sqrt(pVcc(inds,:))')

