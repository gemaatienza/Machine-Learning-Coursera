function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%Your task is to use the cross validation set Xval, yval to determine the 
%best C and ? parameter to use. 


%For both C and ?, trying values:
set = [0.01,0.03,0.1,0.3,1,3,10,30]; %ex6.pdf example
dimension=length(set);
sgm=zeros(dimension);
c=zeros(dimension);
predictions_error=zeros(dimension, dimension);

for i= 1:dimension
    c(i) = set(i);
    for j = 1:dimension
       sgm(j) = set(j);  
       % Train the SVM 
       model = svmTrain(X, y, c(i), @(x1, x2) gaussianKernel(x1, x2, sgm(j)));
       %You can use the svmPredict function to generate the predictions for the cross validation set.
       predictions  = svmPredict(model,Xval);
       predictions_error(i,j) = mean(double(predictions~=yval));
    end
end


min_error = min(min(predictions_error));
for i =1:dimension
    for j =1:dimension
        if predictions_error(i,j) == min_error
            C = set(i);
            sigma = set(j);
            break;
        end
    end
end

fprintf('Best value C, sigma = [%f %f] with prediction error = %f\n', C, sigma, min_error);


% =========================================================================

end
