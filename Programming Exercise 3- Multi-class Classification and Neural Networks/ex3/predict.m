function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1); %size p = 5000x1

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


%% unregularized Feedforward cost function lambda=0
% Input Layer
% Add ones to the X data matrix and Theta1
a1 = [ones(m, 1) X]; % size a1: 5000 401
% fprintf('size a1: %i %i \n', size(a1));
%Size Theta1 = 25x401
z2 = Theta1 * a1'; % size z2: 25 5000 
%fprintf('size z2: %i %i \n', size(z2)); 

% Hidden Layer
a2=sigmoid(z2); % size a2: 25 5000 
%Size Theta2 = 10x26
%fprintf('size a2: %i %i \n', size(a2)); 
temp=size(a2,2); %5000
a2=[ones(1,temp); a2]; %size a2: 26 5000

%fprintf('size a2: %i %i \n', size(a2)); 

% =========================================================================
% Output layer
z3 = Theta2 * a2; % size z3: 10 5000 
%fprintf('size z3: %i %i \n', size(z3)); 
a3=sigmoid(z3); %size z3: 10 5000 

[x, p] = max(a3', [], 2);

end
