% function m = pattern_metric(dat0,dat1)
% 
% Computes a measure of how well the data in dat1 approximates the
% distribution and strength of patterns in dat0. The measure is given by
% the sum of squared differences between the singular values for dat0, and
% the weights computed when dat1 is decomposed using the left singular
% vectors (EOFs) of dat0. Note that because dat1 and dat0 may be linearly
% independent, there is no guarantee that pattern_metric(dat0,dat1) will be
% equal to pattern_metric(dat1,dat0).
%
% % -- Inputs -- %
% 
% dat0   Base dataset, arranged space (rows) by time (cols)
% dat1   A second set arranged the same way, having the same spatial
%        dimension
%
% % -- Output -- %
%
% m      Up to five fit metrics.
%
% DEA, 20 Sept 2018
%
% Notes:
% Rank, Kullback-Leibler difference, effective rank could be alternate
% measures
% Might eventually want to normalize by total variance (of the time series) to see how much bang
% for our buck we get in terms of pattern creation by unit variance


function m = pattern_metric(dat0,dat1)

% Simplest:
m(1) = sum(sum(dat1.^2,2));

% Just the rank!
m(2) = rank(dat1);

% % -- Some more complex metrics -- % %

% Compute the SVD of dat0. In a loop over different cases of dat1, this
% could be placed outside to reduce computation.
[U0, S0, ~] = svd(dat0,'econ');

% Project the singular vectors of dat0 onto dat1 using backslash
% (effectively a least squares fit of dat1 to U0). SV1 has dimensions of
% singuar vector index by time.
SV1 = U0\dat1;

% Compute the weights that orthonormalize SV1.
S1  = sqrt(diag(SV1*SV1'));

% One metric is the sum of squared differences between S0 and the diagonal
% elements of S1, normalized by final variance. Approaches 0 for best fit.
m(3) = sum((diag(S0)-S1).^2)/sum(diag(S0).^2);

% To ignore the variance weighting, which makes some patterns more
% important than others, we can normalize out by final variance on a
% pattern-by-pattern basis. This is related to Shannon entropy(?) Approaches
% 0.
m(4) = sum(((S1-diag(S0))./diag(S0)).^2);

% Another is the sum of squared weight amplitudes, again normalized by
% total variance. Approaches 1.
m(5) = sum(S1.^2)/sum(diag(S0).^2);





