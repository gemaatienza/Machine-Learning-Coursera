function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

% Compute the costJ of a particular choice of theta
J_noReg = 0;
thetaLen = length(theta);
predictions=sigmoid(X*theta); % predictions of hypothesis on all m examples
% fprintf('size predictions: %i %i', size(predictions)); 
% predictions = mx1 column vector (118x1)
% y = mx1 column vector (118x1)
% y.*log(predictions) is mx1 column vector (118x1)
% (1-y).*log(1-predictions)) is mx1 column vector (118x1)


%fprintf('Calculo la función de coste \n'); 

J_noReg=(-1/m)*(sum(y.*log(predictions)+(1-y).*log(1-predictions)));
theta_Reg=theta(2:thetaLen);
J=J_noReg + (lambda/(2*m))* sum(theta_Reg.^2) ;

%fprintf('Coste dentro de la función = %f \n', J ); 
% cost J, J_noReg = single number 


% compute the gradient 
% The gradient of the cost is a vector of the same length as theta where
% the j element (for jo=0,1,....,n) is defined as ?J(?)/??j. Cuidado, estás
% calculando la derivada parcial de J, no el vector theta!!!

%fprintf('Calculo el gradiente \n'); 
% predictions = mx1 column vector (118x1)
% y = mx1 column vector (118x1)
% predictions-y = mx1 column vector (118x1)
% X = mxn matrix (118x28)
% X' =nxm matrix  (28x118)
% X'*(predictions-y) =nx1 vector (28x1)

 grad_noReg= (1/m)* (X'*(predictions-y)); % nx1 vector (28x1)
 grad_Reg = grad_noReg + (lambda/m)* theta; % nx1 vector (28x1)
 grad=[grad_noReg(1);grad_Reg(2:thetaLen)];
 % grad gradient = n column vector (28x1)

% =============================================================

end
