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

% Add columns of 1s to the X values and set to a1. 
% Take the transpose to get as a column vector
a1 = [ones(m,1) X]';

% compute z2 for layer 2 ( the hidden layer )
% in z2, the rows correspond to different nodes
% and the columns correspond to different inputs
z2 = Theta1 * a1;

% add a row of ones to a2 and compute sigmoid of z2
a2 = [ones(1,m); sigmoid(z2)];

% compute z3 for the output layer
z3 = Theta2 * a2;

% compute probability of each example being in each class. 
% proba has the nodes as rows and the inputs as columns
proba = sigmoid( z3 );

% calculate the class by finding the maximum probability in each column
% max returns the maximum value in each column as a row vector
[maxproba, maxprobaix] = max(proba);

% the prediction corresponds to th emax probabaix transposed because
% we want it as a column vector
p = maxprobaix';

% =========================================================================


end
