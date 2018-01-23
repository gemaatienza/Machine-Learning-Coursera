function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% =========================================================================
% Part 1: Feedforward the neural network and return the cost in the variable J.

% Note that you need to add a +1 term to ensure that the vectors of activations 
% for layers a(1) and a(2) also include the bias unit. 
% In Octave/MATLAB, if a 1 is a column vector, adding one corrensponds to a1 = [1 ; a1].

Y=[];
E = eye(num_labels);    
for k=1:num_labels
    Y0 = find(y==k);  
    Y(Y0,:) = repmat(E(k,:),size(Y0,1),1);
end
%fprintf('size Y: %i %i \n', size(Y)); %5000 x 10

% Input Layer::::::
% Add ones to the X data matrix and Theta1
% Set the input layer?s values (a(1)) to the t-th training example x(t).
a1 = [ones(m, 1) X]; % size a1: 5000 401 --> añado una columna de unos a la matriz X
%fprintf('size a1: %i %i \n', size(a1));


% Hidden Layer
%Size Theta1 = 25x401; size a1: 5000 401 
z2 = Theta1 * a1'; % size z2: 25 5000 
%fprintf('size z2: %i %i \n', size(z2)); 
a2=sigmoid(z2); % size a2: 25 5000 
%fprintf('size a2: %i %i \n', size(a2)); 

temp=size(a2,2); %5000
a2=[ones(1,temp); a2]; %size a2: 26 5000 --> añado una fila de unos a la matriz a2

%fprintf('size a2: %i %i \n', size(a2)); 

% Output layer
%Size Theta2 = 10x26 , size a2: 26 5000
z3 = Theta2 * a2; % size z3: 10 5000 
%z3 = a2 * Theta2';   
a3=sigmoid(z3); %size a3: 10 5000 

%fprintf('size y: %i %i \n', size(y)); % 5000x1

%Con los dos bucles:
for i=1:m
    for k=1:num_labels
        J=J-Y(i,k)*log(a3(k,i))-(1-Y(i,k))*log(1-a3(k,i));
    end
end

J=(1/m)*J;

%size a3: 10 5000 , size Y: 5000 10

%Con solo un bucle:
% for i=1:m
%     J=J-Y(i,:)*log(a3(:,i))-(1-Y(i,:))*log(1-a3(:,i));
% end
%  J=(1/m)*J;
 
 %De forma vectorial
  %J= (1/m)*sum(sum((-Y'.*log(a3)-(1-Y').*log(1-a3))));
  %Para usar .* tienen que tener la misma dimensión las dos matrices que
  %multiplico
  
% =========================================================================
% Part 3: Implement regularization with the cost function 

regularization = (lambda/(2*m)) * (sum(sum((Theta1(:,2:end)).^2)) + sum(sum((Theta2(:,2:end)).^2)));
J=J+regularization;

% =========================================================================
%Part 2: Implement the backpropagation algorithm to compute the gradients
%Theta1_grad and Theta2_grad

%size a3: 10 5000, size Y: 5000 10

% For each output unit k in layer 3 (the output layer), set ?(3) = (a(3) ? yk),
% where yk ? {0,1} indicates whether the current training example belongs to class k (yk = 1), 
% or if it belongs to a different class (yk = 0).

delta_3 = a3 - Y';

%For the hidden layer l = 2
%delta_2 = (Theta2'* delta_3) .* a2 .* (1-a2);
delta_2_temp = (Theta2'* delta_3); %size delta_2_temp: 26 5000 
%fprintf('1. size delta_2_temp: %i %i \n', size(delta_2_temp)); 
delta_2=delta_2_temp(2:end,:).* sigmoidGradient(z2);
%fprintf('2. size delta_2: %i %i \n', size(delta_2)); 

%Accumulate the gradient. 
% Note that you should skip or remove ?0(2). In Octave/MATLAB,
% removing ?0(2) corresponds to delta 2 = delta 2(2:end).

%size delta_3= 10 5000  size a2: 26 5000
%size delta_2=  25 5000  size a1:  5000 401
delta_capital2 = delta_3 * a2'; %size delta_capital2: 10 26
%fprintf('size delta_capital2: %i %i \n', size(delta_capital2)); 
delta_capital1 = delta_2 * a1; % size delta_capital1: 25 401 
%fprintf('size delta_capital1: %i %i \n', size(delta_capital1)); 

%SIN REGULARIZACIÓN:
Theta1_grad = (1/m) * delta_capital1; % 25X401
Theta2_grad = (1/m) * delta_capital2; % 10X26

% CON REGULARIZACIÓN:

Theta1_grad = Theta1_grad + ((lambda/m) * Theta1);
Theta2_grad = Theta2_grad + ((lambda/m) * Theta2);

% fprintf('size Theta1_grad: %i %i \n', size(Theta1_grad)); %25x401
% fprintf('size Theta2_grad: %i %i \n', size(Theta2_grad)); %10x26
 
 Theta1_grad(:,1) = (1/m) * delta_capital1(:,1);
 Theta2_grad(:,1) = (1/m) * delta_capital2(:,1);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
