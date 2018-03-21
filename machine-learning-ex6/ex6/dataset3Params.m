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
sigma = 0.3;

C = 1;
sigma = 0.1;
return
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

%C = 0.01;
%sigma = 0.1;
%return

% search the phase space for the best combination of sigma and c
Cs =     [0.01 0.03 0.1 0.3 1 1.3 3 10 30];
sigmas = [0.01 0.03 0.1 0.3 1 1.3 3 10 30];
minerr = 1E10;
best_c = 0;
best_sigma = 0;

for c = Cs;
    for sigma = sigmas;
        % print info
        fprintf('C: %f sigma: %f\n',c, sigma);
        % train the model
        model = svmTrain(X, y, c, @(x1, x2) gaussianKernel( x1, x2, sigma));
        % calculate the cost
        % get predictions
        predictions = svmPredict( model, Xval );
        err = mean( double( predictions ~= yval ) );
        fprintf('err %f\n', err );
        if err <= minerr;
            minerr = err;
            best_c = c;
            best_sigma = sigma;
        end 
    end 
end

% set C and sigma to the best
fprintf('Best C: %f Best sigma: %f\n', best_c, best_sigma)
C = best_c;
sigma = best_sigma;

% =========================================================================

end
