function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% Compute the costJ of a particular choice of theta

thetaLen = length(theta);
predictions=sigmoid(X*theta); % predictions of hypothesis on all m examples
%fprintf('size predictions: %i %i', size(predictions)); 
% predictions = mx1 column vector (100x1)
% y = mx1 column vector (100x1)
% y.*log(predictions) is mx1 column vector (100x1)
% (1-y).*log(1-predictions)) is mx1 column vector (100x1)

% costJ = single number 
%fprintf('Calculo la función de coste \n'); 

J=(-1/m)*(sum(y.*log(predictions)+(1-y).*log(1-predictions)));

% compute the gradient 
% The gradient of the cost is a vector of the same length as theta where
% the j element (for jo=0,1,....,n) is defined as ?J(?)/??j. Cuidado, estás
% calculando la derivada parcial de J, no el vector theta!!!

%fprintf('Calculo el gradiente \n'); 
% predictions = mx1 column vector (100x1)
% y = mx1 column vector (100x1)
% predictions-y = mx1 column vector (100x1)
% X = mx(n+1) matrix (100x3)
% X' =(n+1)xm matrix  (3x100)
% X'*(predictions-y) =(n+1)x1 vector (3x1)
 grad = (1/m)* (X'*(predictions-y));
 % grad gradient = n+1 column vector (3x1)

 % Si quisiéramos calcular theta tendríamos que iterar para todo
 % i=1...m,j=1...thetaLen. No nos lo piden
%  for i=1:m 
%     for j=1:thetaLen
%        %theta(j,1)=theta(j,1)-(1/m)*sum((predictions-y).*X(:,j));
%        theta(j,1)=theta(j,1)-(1/m)*sum((X(:,j))'*(predictions-y));
%     end
%  end
 
 
% =============================================================

end
