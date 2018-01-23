function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);
% fprintf('size X  : %i %i \n', size(X)); % size m x 1
% fprintf('size X_poly : %i %i \n', size(X_poly)); %
% numel   Number of elements in an array or subscripted array expression.

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 


%   Returns a new feature array with more features, comprising of 
%   X1, X1.^2, X1.^3 ...X1.^p
for i = 1:p
        X_poly(:, i) = X.^i;
end


% fprintf('size X_poly (al final): %i %i \n', size(X_poly)); %

% size X  : 12 1 
% size X_poly : 12 8 
% size X_poly (al final): 12 8 
% size X  : 21 1 
% size X_poly : 21 8 
% size X_poly (al final): 21 8 
% size X  : 21 1 
% size X_poly : 21 8 
% size X_poly (al final): 21 8 


% when a training set X of size m × 1 is passed into the function, 
% the function should return a m×p matrix X poly
% =========================================================================

end
