function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 
%h θ (x) = θ 0 + θ 1 ∗ (waterLevel) + θ 2 ∗ (waterLevel) 2 + · · · + θ p ∗ (waterLevel) p
%= θ 0 + θ 1 x 1 + θ 2 x 2 + ... + θ p x p .
%fprintf("\nHERE!");
%disp(size(X)); 12 1
%disp(size(p)); 1 1
for i=1:p,
	X_poly(:,i) = X(:,1).^i;
end




% =========================================================================

end
