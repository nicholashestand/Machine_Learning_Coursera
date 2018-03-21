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


% Calculate hx
% There is an input layer, a hidden layer and an output layer

% get the activation for the input layer
% take the transpose to have the examples in columns
a1 = [ones(m,1), X]';

% calculate z2 for the hidden layer
z2 = Theta1 * a1;

% calculate activation for hidden layer
a2 = sigmoid(z2);
% add the unit row
a2 = [ones(1,m); a2];

% calculate z3 for the output layer
z3 = Theta2 * a2;

%calculate the activation for the outputl layer 
a3 = sigmoid(z3); % and set to hx
hx = a3;

% now calculate the cost based on hx, first calculate the unregularized cost
% convert the m dimensional y column vector into a num_label by m matrix
ymat = zeros(num_labels, m);
for i = 1:m;
    nx = y(i);
    ymat(nx, i) = 1;
end;

% it is confusing how I do it, but it works. The basic idea is that from 
% above we have a matrix of predictions hx that has dimension num_label by m
% and true labels as a num_label by m matrix that we just created. We 
% can find the full matrix of terms for the cost function by just taking
% element by element multiplication of the matrices and then summing over both
% dimensions.
%J = (ymat .* log(hx) + ( 1 - ymat ) .* ( log( 1- hx ) ))*(-1/m);
%J = sum(J(:));

% is there a better way to do it though? I think we can just unroll the ymat
% and hmat and take the dot product. This works, and I imagine it will be faster
% so lets comment out the one above
J = (ymat(:)' * log(hx(:)) + ( 1 - ymat(:) )' * log( 1 - hx(:) )) * (-1/m);

% now add regularization term, we want to omit the bias term which is in the first column
J += lambda/(2*m) * ( sum( Theta1(:,2:end)(:).^2) + sum( Theta2(:,2:end)(:).^2));



% Now implement the backpropigation

% compute delta 3 using our a3 and ymat caclulated above
d3 = a3 - ymat;

% now compute delta2 using d3, Theta3 and sigmoid of z2
d2 = (Theta2'*d3)(2:end,:).*sigmoidGradient(z2);

% now accumulate the error in Theta2_grad and Theta1_grad
Theta2_grad = d3*a2'/m;
Theta1_grad = d2*a1'/m;

% finally, add regularization term to the gradient. Nothing gets added to the bias term, which is in the
% first column
Theta2_grad(:,2:end) += (lambda/m) * Theta2(:,2:end);
Theta1_grad(:,2:end) += (lambda/m) * Theta1(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
