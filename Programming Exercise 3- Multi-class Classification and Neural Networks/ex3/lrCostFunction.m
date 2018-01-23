function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples, 5000 en este caso

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta)); % grad es un vector de 401

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 


% Compute the costJ of a particular choice of theta
J_noReg = 0;
thetaLen = length(theta); %401
%fprintf('size theta: %i \n', thetaLen); 
%fprintf('size X en la función de coste: %i %i \n', size(X));
predictions=sigmoid(X*theta); % predictions of hypothesis on all m examples
%fprintf('size predictions: %i %i \n', size(predictions)); 
% predictions = mx1 column vector (5000x1)
% y = mx1 column vector (5000x1)
% y.*log(predictions) is mx1 column vector (5000x1)
% (1-y).*log(1-predictions)) is mx1 column vector (5000x1)

%fprintf('Calculo la función de coste \n'); 

J_noReg=(-1/m)*(sum(y.*log(predictions)+(1-y).*log(1-predictions)));
theta_Reg=theta(2:thetaLen);
J=J_noReg + (lambda/(2*m))* sum(theta_Reg.^2) ;

%fprintf('Coste calculado en lrCostFunction = %f \n', J ); 
% cost J, J_noReg = single number 

%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

%fprintf('Calculo el gradiente \n'); 
% predictions = mx1 column vector (5000x1)
% y = mx1 column vector (5000x1)
% predictions-y = mx1 column vector (5000x1)
% X = mx(n+1) matrix (5000x401)
% X' =(n+1)xm matrix  (401x5000)
% X'*(predictions-y) =(n+1)x1 vector (401x1)
% theta  (n+1)x1 vector

 grad_noReg= (1/m)* (X'*(predictions-y)); % (n+1)x1 vector (401x1)
 grad_Reg = grad_noReg + (lambda/m)* theta; % (n+1)x1 vector (401x1)
 grad=[grad_noReg(1);grad_Reg(2:thetaLen)]; % (n+1)x1 vector (401x1)

%fprintf('size grad : %i %i \n', size(grad));
% =============================================================

%grad = grad(:);

end
