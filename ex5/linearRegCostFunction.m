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
h = X * theta;
error = h - y;
error_s = error.^2;
J =  1/(2*m) * sum(error_s);

%compute the sum of all of the theta values squared. One handy way to do 
%this is sum(theta.^2). Since theta(1) has been forced to zero, it doesn't 
%add to the regularization term. Now scale this value by lambda / (2*m), and add it to the unregularized cost.

theta(1) = 0;
J = J + sum(theta.^2) * lambda / (2*m);

%For the gradient regularization:
% The regularized gradient term is theta scaled by (lambda / m). Again, since theta(1) has been set to zero, 
% it does not contribute to the regularization term.    Add this vector to the unregularized portion.
% =========================================================================

    % Updating the parameters

    grad = ( X' * (h - y) + lambda*theta ) / m;
end

