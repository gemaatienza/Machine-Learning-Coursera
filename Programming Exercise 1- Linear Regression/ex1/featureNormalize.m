function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       


%My code:
% size(X,2) is the number of the features (#columns of X)

% Calculates mean and std dev for each feature
for i=1:size(X, 2)
%     fprintf('For i = ');
%     fprintf('%i', i);
    mu(i)=mean(X(:,i));
%     fprintf(', mu(i)= ');
%     fprintf('%f', mu(i));
    sigma(i)=std(X(:,i));
%     fprintf(', sigma(i)= ');
%     fprintf('%f', sigma(i));
%     fprintf(' \n');
    X_norm(:,i)=(X(:,i)-mu(i))/sigma(i);
end


% ============================================================

end
