function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;
return;
values = [0.01 0.03 0.1 0.3 1 3 10 30];
error_best = 99999999;
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
%One method is to use two nested for-loops - each one iterating over the range of C or sigma values given in the ex6.pdf file.
for C = values
  for sigma = values
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
    predict = svmPredict(model, Xval);
    err = mean(double(predict ~= yval));
    if(err < error_best)
      C_best=C;
      sigma_best=sigma;
      error_best = err;
      fprintf('New low C, sigma = [%f %f] error = %f\n', C, sigma, error_best);
    end
  endfor
endfor
C=C_best;
sigma=sigma_best;
 fprintf('New low C, sigma = [%f %f] error = %f\n', C, sigma, error_best);
%[C, sigma] = dataset3Params(X, y, Xval, yval)
%Inside the inner loop:
%
%    Train the model using svmTrain with X, y, a value for C, and the gaussian kernel using a value for sigma. 
%        See ex6.m at line 108 for an example of the correct syntax to use in calling svmTrain() and the gaussian kernel.
%    Compure the predictions for the validation set using svmPredict() with model and Xval.
%    Compute the error between your predictions and yval.
%    When you find a new minimum error, save the C and sigma values that were used.






% =========================================================================

end
