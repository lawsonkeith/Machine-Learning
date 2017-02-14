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
disp("PAUSING!!!");
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
%tep 1 - Expand the 'y' output values into a matrix of single values (see ex4.pdf Page 5). This is most 
%easily done using an eye() matrix of size num_labels, with vectorized indexing by 'y', as in "eye(num_labels)(y,:)". 
Y=eye(num_labels)(y,:);
%tep 2 - perform the forward propagation:a1 equals the X input matrix with a column of 1's added (bias units)z2 equals the product 
%of a1 and Θ1a2 is the result of passing z2 through g()a2 then has a column of 1st added (bias units)z3 equals the product of a2 and 
%Θ2a3 is the result of passing z3 through g()Cost Function, non-regularized
a1 = [ones(m,1) X];
z2 = a1 * Theta1';
a2 = [ones(size(z2,1), 1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
#Step 3 - Compute the unregularized cost according to ex4.pdf (top of Page 5), (I had a hard time understanding this equation mainly 
#that I had a misconception that y(i)k is a vector, instead it is just simply one number) using a3, your ymatrix, and m (the number of 
#training examples). Cost should be a scalar value. If you get a vector of cost values, you can sum that vector to get the cost.Remember 
#to use element-wise multiplication with the log() function.Now you can run ex4.m to check the unregularized cost is correct, then you can
#submit Part 1 to the grader.
J = (1/m)*sum(sum((-Y).*log(a3)-(1-Y).*log(1-a3),2));
reg = lambda/(2*m) * (sum(sum(Theta1(:, 2:end).^2, 2)) + sum(sum(Theta2(:,2:end).^2, 2)));
J = J + reg;
%4 - Compute the regularized component of the cost according to ex4.pdf Page 6, using Θ1 and Θ2 (excluding the Theta columns for the
% bias units), along with λ, and m. The easiest method to do this is to compute the regularization terms separately, then add them to 
%the unregularized cost from Step 3.
Sigma3 = a3 - Y;

Sigma2 = (Sigma3*Theta2 .* sigmoidGradient([ones(size(z2, 1), 1) z2]))(:, 2:end);


Delta_1 = Sigma2'*a1;
Delta_2 = Sigma3'*a2;


Theta1_grad = Delta_1./m + (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)];
Theta2_grad = Delta_2./m + (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)];

%pause(111111110); 
%grad = (1/m)*(sigmoid(X*theta)-y)'*X

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
