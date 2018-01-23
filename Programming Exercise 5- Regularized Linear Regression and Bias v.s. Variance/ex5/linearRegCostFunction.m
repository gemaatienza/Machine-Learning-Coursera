function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%J cost:
thetaLen = length(theta);
predictions=X*theta; % predictions of hypothesis on all m examples
sqrErrors=(predictions-y).^2; %squared errors
J_noReg=1/(2*m)*sum(sqrErrors);
theta_Reg=theta(2:thetaLen);
J=J_noReg + (lambda/(2*m))* sum(theta_Reg.^2) ;


%grad gradientDescent

grad_noReg= (1/m)* (X'*(predictions-y)); % (n+1)x1 vector 
grad_Reg = grad_noReg + (lambda/m)* theta; % (n+1)x1 vector 
grad=[grad_noReg(1);grad_Reg(2:thetaLen)]; % (n+1)x1 vector 

%fprintf('size grad : %i %i \n', size(grad)); %size grad : 2 1 


% =========================================================================

grad = grad(:);

end
