function g = sigmoid(z)
% Comput the sigmoid function of z: g = 1/ (1+e^-z)

d = 1 + e.^(-z);
g = 1./d;

end
