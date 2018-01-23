function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% fprintf('size X: %i %i \n', size(X)); %300x2
% fprintf('size centroids: %i %i \n', size(centroids)); %3x2
dis_centroids = zeros(K, 1);
for i=1:size(X,1),
  for j=1:K,
    distance = X(i,:)' - centroids(j,:)'; %2x1
    %fprintf('size distance: %i %i \n', size(distance)); %3x2
    value  = distance' * distance; %la norma al cuadrado es el producto
    % de una matriz traspuesta por ella misma. Es un real.
    dis_centroids(j) = value; 
  end
  [mini, idxj] = min(dis_centroids); %este es el centroid más cercano a x(i)
  idx(i) = idxj;
end


% =============================================================





% =============================================================

end

