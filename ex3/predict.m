function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

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
% Add ones to the X data matrix
X = [ones(m, 1) X];
z2 = X * Theta1';
a2 = [ones(m,1) sigmoid(z2)];    
%Multiply by Theta2, compute the sigmoid() and it becomes 'a3'.
a3 = sigmoid(a2 * Theta2');
%Now use the max(a3, [], 2) function to return two vectors - one of the highest value for each row, 
%and one with its index. Ignore the highest values. Keep the vector of the indexes where the highest 
%values were found. These are your predictions
[ max_value, max_index ] = max( a3, [], 2 )
p=max_index;
% =========================================================================
%predict = sigmoid(X * all_theta');
%disp(predict);
%[ max_value, max_index ] = max( predict, [], 2 )
%   

end
