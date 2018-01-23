function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

%My code:
thetaLen = length(theta);
%tempValueTheta = theta; % temporary variable to store part of the theta values.


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
      
    predictions=X*theta; % predictions of hypothesis on all m examples
    diff=predictions-y;
    
    for j=1:thetaLen
         theta(j,1)=theta(j,1)-alpha*(1/m)*sum(diff.*X(:,j));
         %tempValueTheta(j,1)=sum(diff.*X(:,j));
    end
    
    %theta=theta-alpha*(1/m)*tempValueTheta;
    
    % print theta to screen
    fprintf('If iter = ');
    fprintf('%i', iter);
    fprintf(', Theta = ');
    fprintf('%f ', theta(1,1));
    fprintf(', %f ', theta(2,1));
    fprintf(', %f ', theta(3,1));
 
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
    % print J_history to screen
    fprintf(', Cost Function J(iter)= ');
    fprintf('%f', J_history(iter));
    fprintf(' \n');

end

end
